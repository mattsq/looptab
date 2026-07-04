"""Real schedule-history bridge: versioned flight-state snapshots -> the `disruption` task's
(X, y) contract (a real-tabular analog of the M20 bridge, `make_disruption`/`trm_mixer` compatible).

Operational systems typically log a flight's schedule state as a VERSIONED HISTORY table — one row
per change event, each row a FULL SNAPSHOT of that flight's state after the event (not a diff).
This module bridges that shape to the fixed-width per-component feature contract `make_disruption`
(``generators.py``) defines and that ``trm`` / ``trm_mixer`` consume unchanged:

  X: per-flight blocks ``[severe_0, is_head, bank_onehot]`` (``2 + n_banks`` each), flattened, for
     exactly ``w`` canonically-ordered (tail, then scheduled departure) flights per row.
  y: the settled per-flight severe outcome, ``(n, w)`` in {0,1}.

Only TWO rows per flight matter to this contract: its FIRST recorded snapshot (``severe_0`` — the
state right after the initial disruption trigger, i.e. t0) and its LAST recorded snapshot (the
settled outcome, ``y`` — mirroring ``make_disruption``'s ``severe_0 -> s_inf``). Everything in
between is not used; if a caller wants a different "as of" cutoff for t0 (e.g. predicting before
the disruption has fully played out), pre-filter ``history`` to end at that cutoff before calling
this function — keeping the transform itself a single, order-independent aggregation (first/last
row per flight), not a business-logic decision about what counts as "the" decision time.

Unlike the synthetic generator, real data has no fixed ``task_seed``-derived coupling matrix — the
grouping of flights into a "disruption component" (which ones are jointly modeled as one row) is an
upstream business-logic decision (e.g. a shared irregular-ops event id), so it is taken as an
explicit input column (``component_id``), not inferred from the history. Components that do not
have EXACTLY ``w`` flights are dropped (their ids reported) rather than padded/truncated, since
padding would corrupt the canonical positional ordering the mixer's cross-cell token-mixing relies
on (and `trm_mixer` requires ``in_features % out_features == 0`` — no distractor columns here).

``bank_categories`` fixes the one-hot vocabulary. Derive it ONCE (``fit_bank_vocabulary``, e.g. on
a training split) and reuse it for every subsequent split — exactly as ``real.py`` fits
standardization stats on train only — so train/test feature columns line up and an unseen bank
value at transform time is a loud error, not a silently-wrong all-zero one-hot.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Sequence

import numpy as np


@dataclass
class ScheduleHistory:
    """One row per ``(flight_id, version)`` snapshot — a versioned flight-state history log.

    ``version`` need only be *orderable* per ``flight_id`` (an integer sequence number or a
    sortable timestamp both work) — ascending ``version`` means "later." ``tail_number`` /
    ``scheduled_departure`` / ``station_bank`` are read off the flight's FIRST snapshot (i.e. the
    state of the world as of t0, matching what would actually be known at decision time), so a
    flight that is reassigned to a different tail or bank later in its history is handled
    correctly — the structural features never peek at post-t0 information.
    """

    flight_id: np.ndarray
    component_id: np.ndarray
    version: np.ndarray
    tail_number: np.ndarray
    scheduled_departure: np.ndarray
    station_bank: np.ndarray
    severe: np.ndarray

    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, np.asarray(getattr(self, f.name)))
        lengths = {len(getattr(self, f.name)) for f in fields(self)}
        if len(lengths) != 1:
            raise ValueError(
                f"ScheduleHistory columns have mismatched lengths: "
                f"{ {f.name: len(getattr(self, f.name)) for f in fields(self)} }"
            )


def fit_bank_vocabulary(history: ScheduleHistory) -> tuple[Any, ...]:
    """Fix a deterministic bank -> one-hot-index vocabulary from a reference history (e.g. train).

    Reuse the SAME tuple for every subsequent ``transform_schedule_history`` call (train, test,
    ...) so bank one-hot columns line up across splits; an unseen bank value at transform time
    raises rather than silently collapsing to an all-zero one-hot (§5-style loud guard).
    """
    return tuple(sorted(set(history.station_bank.tolist())))


def _endpoints_by_flight(history: ScheduleHistory) -> dict[Any, tuple[int, int]]:
    """Per flight_id: (row index of its earliest-version snapshot, row index of its latest)."""
    by_flight: dict[Any, list[int]] = defaultdict(list)
    for i, fid in enumerate(history.flight_id.tolist()):
        by_flight[fid].append(i)
    endpoints = {}
    for fid, idxs in by_flight.items():
        idxs.sort(key=lambda i: history.version[i])
        endpoints[fid] = (idxs[0], idxs[-1])
    return endpoints


def transform_schedule_history(
    history: ScheduleHistory,
    w: int,
    bank_categories: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """Transform a versioned flight-state history into ``make_disruption``'s ``(X, y)`` contract.

    Returns ``(X, y, dropped_component_ids)``:
      - ``X`` is ``(n, w * (2 + len(bank_categories)))`` float32 — per-flight blocks
        ``[severe_0, is_head, bank_onehot]``, flattened, bit-for-bit the layout ``make_disruption``
        produces (no distractor columns: ``trm_mixer`` requires exact ``in_features //
        out_features`` divisibility).
      - ``y`` is ``(n, w)`` int64 in {0,1} — each flight's LAST recorded ``severe`` value.
      - ``dropped_component_ids`` lists components whose flight count was not exactly ``w`` (they
        contribute nothing to ``X``/``y``), so data loss is auditable rather than silent.

    ``n`` = the number of components with exactly ``w`` flights, emitted in order of first
    appearance in ``history``. Raises if that is zero (nothing to train/eval on) — most likely
    ``w`` does not match the real component sizes.

    Within each accepted component, flights are placed in CANONICAL ORDER — sorted by
    ``tail_number``, then ``scheduled_departure`` (ties broken by ``flight_id``) — mirroring
    ``_build_disruption_weights``'s convention, so ``is_head`` (no rotation predecessor: the first
    flight of a contiguous same-tail run in that order) matches the synthetic generator's
    semantics. Note ``is_head`` is scoped to the component: a tail whose rotation is split across
    two components will show ``is_head=1`` in both — the synthetic generator has the same
    per-instance scoping, since it never models an infinite chain across rows either.
    """
    n_banks = len(bank_categories)
    bank_index = {b: i for i, b in enumerate(bank_categories)}
    endpoints = _endpoints_by_flight(history)

    components: dict[Any, list[Any]] = defaultdict(list)
    seen: set[Any] = set()
    for fid in history.flight_id.tolist():
        if fid in seen:
            continue
        seen.add(fid)
        t0_idx, _ = endpoints[fid]
        components[history.component_id[t0_idx]].append(fid)

    rows_X: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    dropped: list[Any] = []

    for comp_id, flight_ids in components.items():
        if len(flight_ids) != w:
            dropped.append(comp_id)
            continue

        recs = []
        for fid in flight_ids:
            t0_idx, settle_idx = endpoints[fid]
            recs.append(
                (
                    history.tail_number[t0_idx],
                    history.scheduled_departure[t0_idx],
                    history.station_bank[t0_idx],
                    bool(history.severe[t0_idx]),
                    bool(history.severe[settle_idx]),
                )
            )
        order = sorted(range(w), key=lambda k: (recs[k][0], recs[k][1], flight_ids[k]))

        block = np.zeros((w, 2 + n_banks), dtype=np.float32)
        y_row = np.empty(w, dtype=np.int64)
        prev_tail = None
        for pos, k in enumerate(order):
            tail, _, bank, severe0, severe_settle = recs[k]
            if bank not in bank_index:
                raise ValueError(
                    f"transform_schedule_history: station_bank {bank!r} (flight "
                    f"{flight_ids[k]!r}, component {comp_id!r}) is not in bank_categories; refit "
                    "the vocabulary with fit_bank_vocabulary on data that includes it."
                )
            is_head = pos == 0 or tail != prev_tail
            prev_tail = tail
            block[pos, 0] = float(severe0)
            block[pos, 1] = float(is_head)
            block[pos, 2 + bank_index[bank]] = 1.0
            y_row[pos] = int(severe_settle)

        rows_X.append(block.reshape(-1))
        rows_y.append(y_row)

    if not rows_X:
        raise ValueError(
            f"transform_schedule_history: no component had exactly w={w} flights "
            f"({len(dropped)} components dropped); check `w` against the real component sizes."
        )

    X = np.stack(rows_X).astype(np.float32)
    y = np.stack(rows_y).astype(np.int64)
    return X, y, dropped

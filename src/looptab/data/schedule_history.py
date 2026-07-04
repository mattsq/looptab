"""Real schedule-history bridge: versioned flight-state snapshots -> the `disruption` task's
(X, y) contract (a real-tabular analog of the M20 bridge, `make_disruption`/`trm_mixer` compatible).

Operational systems typically log a flight's schedule state as a VERSIONED HISTORY table — one row
per change event, each row a FULL SNAPSHOT of that flight's state after the event (not a diff). But
"as it would actually be stored" also means: arbitrary, source-specific column names (never
`severe`/`station_bank`/`tail_number` verbatim), a handful of the roles this bridge needs are
usually DERIVED rather than stored at all (a "severe" flag is a judgement call over delay/
cancellation/diversion columns; a "bank" is a judgement call over station + time window; a
"disruption component" grouping essentially never exists as a literal column — ops systems log
per-flight events, not pre-grouped incidents), and the table itself might arrive as row-records
(a SQL/API/event-log query result) rather than ready-made column arrays.

The workflow this module is built around, in order:

  1. ``describe_roles()`` / ``suggest_column_mapping(raw_columns(raw))`` — figure out which of
     YOUR raw columns (or which small derivation) fills each semantic role this bridge needs.
  2. Build a ``ColumnMapping`` from that — each field is either a raw column NAME, or a CALLABLE
     that derives the role from one or more raw columns (``derive_severe_from_delay``/
     ``derive_bank`` cover the common cases).
  3. ``mapping_report(raw, mapping)`` — sanity-check the mapping resolves the way you expect
     (dtype + a value preview per role) before running anything downstream.
  4. ``apply_column_mapping(raw, mapping)`` -> a normalized ``ScheduleHistory``.
  5. ``transform_schedule_history(history, w, bank_categories)`` -> ``(X, y, dropped)``, the exact
     per-flight-block layout ``make_disruption`` produces and ``trm``/``trm_mixer`` already consume:

       X: per-flight blocks ``[severe_0, is_head, bank_onehot]`` (``2 + n_banks`` each), flattened,
          for exactly ``w`` canonically-ordered (tail, then scheduled departure) flights per row.
       y: the settled per-flight severe outcome, ``(n, w)`` in {0,1}.

Only TWO rows per flight matter to the final contract: its FIRST recorded snapshot (``severe_0`` —
the state right after the initial disruption trigger, i.e. t0) and its LAST recorded snapshot (the
settled outcome, ``y`` — mirroring ``make_disruption``'s ``severe_0 -> s_inf``). If a caller wants a
different "as of" cutoff for t0 (e.g. predicting before the disruption has fully played out),
pre-filter the raw table to end at that cutoff before step 4 — keeping the transform itself a
single, order-independent aggregation (first/last row per flight), not a business-logic decision
about what counts as "the" decision time.

Disruption-COMPONENT membership (which flights are jointly modeled as one row) is either supplied
directly (rare — only if an upstream event-management system already tags one, e.g. an IROPS event
id) or, the common case, left unset and reconstructed by ``derive_components`` from the same two
coupling families ``_build_disruption_weights`` encodes synthetically: rotation-chain adjacency
(LOCAL — shared tail, adjacent in canonical order) and shared-bank cliques (NON-LOCAL). Components
that do not have EXACTLY ``w`` flights are dropped (their ids reported) rather than padded/
truncated, since padding would corrupt the canonical positional ordering the mixer's cross-cell
token-mixing relies on (and `trm_mixer` requires ``in_features % out_features == 0`` — no
distractor columns here).

``bank_categories`` fixes the one-hot vocabulary. Derive it ONCE (``fit_bank_vocabulary``, e.g. on
a training split) and reuse it for every subsequent split — exactly as ``real.py`` fits
standardization stats on train only — so train/test feature columns line up and an unseen bank
value at transform time is a loud error, not a silently-wrong all-zero one-hot.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# A raw, as-stored table, in whichever of the two shapes a real source hands back: a dict of
# columns (a dataframe/columnar export) or a list of row-records (the more common shape for a
# versioned history log queried out of a database or read off an event stream).
RawColumns = Mapping[str, Sequence[Any]]
RawRecords = Sequence[Mapping[str, Any]]
RawSource = RawColumns | RawRecords

# A ColumnMapping field is either the NAME of an existing raw column, or a derivation callable
# taking the normalized columnar table and returning the resolved array.
ColumnOrDerivation = str | Callable[[dict[str, np.ndarray]], Any]


def _to_columnar(raw: RawSource) -> dict[str, np.ndarray]:
    """Normalize either raw shape into one dict of numpy column arrays."""
    if isinstance(raw, Mapping):
        return {k: np.asarray(v) for k, v in raw.items()}
    records = list(raw)
    if not records:
        raise ValueError("raw schedule-history table is empty (0 records).")
    columns = list(records[0].keys())
    return {c: np.array([r[c] for r in records]) for c in columns}


def raw_columns(raw: RawSource) -> list[str]:
    """Column names present in `raw`, regardless of whether it's columnar or row-records.

    Use this to get candidate names to feed ``suggest_column_mapping`` before you've built a
    ``ColumnMapping`` at all.
    """
    if isinstance(raw, Mapping):
        return list(raw.keys())
    records = list(raw)
    return list(records[0].keys()) if records else []


# ---------------------------------------------------------------------------------------------
# Schema help (b): what each role means, whether it's normally a literal stored column or
# something you compute, and common raw-column aliases — for a user who has never seen this
# bridge before and needs to figure out what their own table's columns correspond to.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _RoleSpec:
    description: str
    aliases: tuple[str, ...]
    stored_directly: bool  # False = typically DERIVED, not a literal column in the raw store


ROLE_SPECS: dict[str, _RoleSpec] = {
    "flight_id": _RoleSpec(
        "Stable identifier for one flight, constant across all of its versioned snapshots.",
        ("flight_id", "flt_id", "flight_key", "leg_id", "flight_leg_id"),
        stored_directly=True,
    ),
    "version": _RoleSpec(
        "Anything that orders a flight's snapshots — a sequence number or a timestamp both "
        "work, it only needs to be orderable per flight_id (ascending = later).",
        ("version", "event_seq", "seq_nbr", "event_ts", "updated_at", "effective_ts", "valid_from"),
        stored_directly=True,
    ),
    "tail_number": _RoleSpec(
        "The aircraft assigned to the flight as of that snapshot (defines rotation chains — "
        "the LOCAL coupling in _build_disruption_weights).",
        ("tail_number", "tail_nbr", "tail", "aircraft_reg", "registration", "ac_reg"),
        stored_directly=True,
    ),
    "scheduled_departure": _RoleSpec(
        "A sortable scheduled-departure value, used only to order same-tail flights within a "
        "rotation (does not need to be wall-clock accurate, just consistently orderable).",
        ("scheduled_departure", "std", "sked_dep", "sched_dep_utc", "dep_time_sched"),
        stored_directly=True,
    ),
    "station_bank": _RoleSpec(
        "The shared-resource group coupling otherwise-unrelated flights (e.g. a hub connection "
        "bank: a station + arrival/departure time window — the NON-LOCAL coupling in "
        "_build_disruption_weights). RARELY a literal stored column — usually DERIVED from a "
        "station code + a time bucket; see `derive_bank`.",
        ("bank", "gate_bank", "connection_bank", "station_group", "hub_wave"),
        stored_directly=False,
    ),
    "severe": _RoleSpec(
        "Binary flag: is this flight in a severe disruption state as of this snapshot? RARELY "
        "stored directly — usually DERIVED from delay minutes / cancellation / diversion flags; "
        "see `derive_severe_from_delay`.",
        ("severe", "irops_flag", "delay_min", "delay_minutes", "cx_flag", "cancelled",
         "divert_flag"),
        stored_directly=False,
    ),
    "component_id": _RoleSpec(
        "Which group of jointly-affected flights this snapshot belongs to (the unit that "
        "becomes one row of X/y). ALMOST NEVER a stored column in a raw schedule-history table "
        "— operational systems log per-flight events, not pre-grouped disruption components. "
        "Either supply one if your source event-management system already tags one (e.g. an "
        "IROPS event id), or leave it unmapped (`ColumnMapping.component_id = None`) and "
        "`transform_schedule_history` will reconstruct it via `derive_components` from "
        "tail-rotation + bank connectivity — the same coupling `make_disruption`'s synthetic W "
        "encodes.",
        ("component_id", "irops_event_id", "disruption_id", "event_group_id"),
        stored_directly=False,
    ),
}


def describe_roles() -> str:
    """Human-readable explanation of every semantic role this bridge needs — what it means,
    whether real tables usually store it directly or you need to derive it, and common raw
    column names it goes by. Read this (or `ROLE_SPECS` directly) before building a
    `ColumnMapping` for a schema you haven't wired up before.
    """
    lines = []
    for role, spec in ROLE_SPECS.items():
        kind = "usually a stored column" if spec.stored_directly else "usually DERIVED — see below"
        lines.append(
            f"- {role} ({kind}): {spec.description}\n"
            f"    common raw names: {', '.join(spec.aliases)}"
        )
    return "\n".join(lines)


def suggest_column_mapping(columns: Sequence[str]) -> dict[str, list[str]]:
    """For each semantic role, list which of `columns` plausibly maps to it.

    A starting point for hand-building a `ColumnMapping` — substring-matches raw names against
    known aliases (case/underscore/hyphen-insensitive). Matches are SUGGESTIONS to confirm, never
    auto-applied: a wrong guess on `severe` or `component_id` would silently produce a wrong label,
    so this function only narrows the search, it doesn't make the call for you.
    """

    def norm(s: str) -> str:
        return s.lower().replace("_", "").replace("-", "")

    normed = {c: norm(c) for c in columns}
    out: dict[str, list[str]] = {}
    for role, spec in ROLE_SPECS.items():
        alias_norms = [norm(a) for a in spec.aliases]
        out[role] = [c for c, nc in normed.items() if any(a in nc or nc in a for a in alias_norms)]
    return out


# ---------------------------------------------------------------------------------------------
# Column mapping: raw table -> normalized ScheduleHistory
# ---------------------------------------------------------------------------------------------


@dataclass
class ColumnMapping:
    """Maps this bridge's semantic roles onto YOUR raw, as-stored schedule-history table.

    Each field accepts either the NAME of an existing raw column, or a CALLABLE
    ``normalized_raw_table -> array-like`` for roles that are usually computed rather than stored
    (`station_bank`, `severe` — see `derive_bank`/`derive_severe_from_delay` for common recipes,
    e.g. via ``functools.partial``). `component_id` is optional: leave it `None` to reconstruct
    components automatically with `derive_components` (real event-sourced tables almost never
    pre-group flights into disruption incidents). Call `describe_roles()` first if it's unclear
    what a role means, or `suggest_column_mapping(raw_columns(raw))` for name candidates.
    """

    flight_id: ColumnOrDerivation
    version: ColumnOrDerivation
    tail_number: ColumnOrDerivation
    scheduled_departure: ColumnOrDerivation
    station_bank: ColumnOrDerivation
    severe: ColumnOrDerivation
    component_id: ColumnOrDerivation | None = None


def _resolve(cols: dict[str, np.ndarray], spec: ColumnOrDerivation, role: str) -> np.ndarray:
    if callable(spec):
        return np.asarray(spec(cols))
    if spec not in cols:
        hits = suggest_column_mapping(list(cols.keys())).get(role, [])
        hint = f" Close candidates in this table: {hits}." if hits else ""
        raise KeyError(
            f"ColumnMapping.{role} = {spec!r} not found among this table's columns "
            f"({sorted(cols.keys())}).{hint} Run describe_roles() to see what {role!r} means, or "
            "suggest_column_mapping(raw_columns(raw)) for automatic candidates."
        )
    return np.asarray(cols[spec])


def apply_column_mapping(raw: RawSource, mapping: ColumnMapping) -> "ScheduleHistory":
    """Resolve a `ColumnMapping` against a raw, as-stored table into the normalized
    `ScheduleHistory` `transform_schedule_history` operates on.

    Raises a clear, per-role `KeyError` (with close-name suggestions) if a mapped column name
    isn't actually present in `raw`. Derivation callables (for `station_bank`/`severe`/
    `component_id`) receive the normalized columnar table (a `dict[str, np.ndarray]`), regardless
    of whether `raw` itself was columns or row-records.
    """
    cols = _to_columnar(raw)
    component_id = (
        _resolve(cols, mapping.component_id, "component_id")
        if mapping.component_id is not None
        else None
    )
    return ScheduleHistory(
        flight_id=_resolve(cols, mapping.flight_id, "flight_id"),
        version=_resolve(cols, mapping.version, "version"),
        tail_number=_resolve(cols, mapping.tail_number, "tail_number"),
        scheduled_departure=_resolve(cols, mapping.scheduled_departure, "scheduled_departure"),
        station_bank=_resolve(cols, mapping.station_bank, "station_bank"),
        severe=_resolve(cols, mapping.severe, "severe"),
        component_id=component_id,
    )


def mapping_report(raw: RawSource, mapping: ColumnMapping, n_preview: int = 3) -> str:
    """Human-readable summary of what a `ColumnMapping` actually resolves to against `raw`.

    For each semantic role: its source (a raw column name, or "derived"), the resolved dtype, and
    a small value preview — or the exact error if the mapping is wrong. Run this BEFORE
    `apply_column_mapping`/`transform_schedule_history` to sanity-check a mapping you're not yet
    sure about, especially for the usually-derived roles (`station_bank`, `severe`,
    `component_id`) where a silent mistake would corrupt the label rather than crash.
    """
    cols = _to_columnar(raw)
    lines = []
    for role in (
        "flight_id",
        "version",
        "tail_number",
        "scheduled_departure",
        "station_bank",
        "severe",
        "component_id",
    ):
        spec = getattr(mapping, role)
        if spec is None:
            note = (
                " -> will be auto-derived from tail/bank connectivity (see derive_components)"
                if role == "component_id"
                else ""
            )
            lines.append(f"- {role}: <not provided>{note}")
            continue
        try:
            values = _resolve(cols, spec, role)
        except KeyError as exc:
            lines.append(f"- {role}: ERROR -- {exc}")
            continue
        source = "derived (callable)" if callable(spec) else f"raw column {spec!r}"
        preview = values[:n_preview].tolist()
        lines.append(f"- {role}: {source}, dtype={values.dtype}, preview={preview}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Common derivation recipes for the roles real tables rarely store directly.
# ---------------------------------------------------------------------------------------------


def derive_severe_from_delay(
    cols: Mapping[str, np.ndarray],
    delay_minutes_col: str,
    threshold_minutes: float = 15.0,
    cancelled_col: str | None = None,
    diverted_col: str | None = None,
) -> np.ndarray:
    """Common recipe for `severe`: delayed past a threshold, OR cancelled, OR diverted.

    Bind the column names with ``functools.partial`` and pass the result as
    ``ColumnMapping.severe`` when the raw table has no literal severity flag — the typical case,
    since "severe" is a judgement call made from raw delay/disposition data, not something ops
    systems log as a single boolean.
    """
    delay = np.asarray(cols[delay_minutes_col], dtype=np.float64)
    severe = delay >= threshold_minutes
    if cancelled_col is not None:
        severe = severe | np.asarray(cols[cancelled_col]).astype(bool)
    if diverted_col is not None:
        severe = severe | np.asarray(cols[diverted_col]).astype(bool)
    return severe.astype(np.int64)


def derive_bank(
    cols: Mapping[str, np.ndarray],
    station_col: str,
    time_col: str,
    window_minutes: float = 60.0,
) -> np.ndarray:
    """Common recipe for `station_bank`: a hub connection "bank" = a station + a rolling
    departure/arrival time window (flights connecting through the same station in the same wave
    share downstream congestion — the NON-LOCAL coupling `_build_disruption_weights`'s bank
    cliques model). `time_col` must already be numeric (e.g. minutes/seconds since epoch). Bind
    the column names with ``functools.partial`` and pass the result as
    ``ColumnMapping.station_bank`` when the raw table only has a station code and a timestamp.
    """
    station = np.asarray(cols[station_col])
    t = np.asarray(cols[time_col], dtype=np.float64)
    bucket = np.floor(t / window_minutes).astype(np.int64)
    return np.array([f"{s}#{b}" for s, b in zip(station.tolist(), bucket.tolist())])


# ---------------------------------------------------------------------------------------------
# Normalized history + the (X, y) transform
# ---------------------------------------------------------------------------------------------


@dataclass
class ScheduleHistory:
    """One row per ``(flight_id, version)`` snapshot — a versioned flight-state history log,
    normalized to the roles this bridge needs (build this via `apply_column_mapping`, not by
    hand-slicing your own arrays, unless your source already happens to use these exact names).

    ``version`` need only be *orderable* per ``flight_id`` (a sequence number or a sortable
    timestamp both work) — ascending ``version`` means "later." ``tail_number`` /
    ``scheduled_departure`` / ``station_bank`` are read off the flight's FIRST snapshot (i.e. the
    state of the world as of t0, matching what would actually be known at decision time), so a
    flight reassigned to a different tail or bank later in its history is handled correctly — the
    structural features never peek at post-t0 information.

    ``component_id`` may be `None` (the common case for real data — see `ROLE_SPECS`); if so,
    `transform_schedule_history` derives it automatically via `derive_components`.
    """

    flight_id: np.ndarray
    version: np.ndarray
    tail_number: np.ndarray
    scheduled_departure: np.ndarray
    station_bank: np.ndarray
    severe: np.ndarray
    component_id: np.ndarray | None = None

    def __post_init__(self):
        for f in fields(self):
            if f.name == "component_id" and self.component_id is None:
                continue
            setattr(self, f.name, np.asarray(getattr(self, f.name)))
        lengths = {
            len(getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) is not None
        }
        if len(lengths) != 1:
            sizes = {
                f.name: len(getattr(self, f.name))
                for f in fields(self)
                if getattr(self, f.name) is not None
            }
            raise ValueError(f"ScheduleHistory columns have mismatched lengths: {sizes}")


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


def derive_components(history: ScheduleHistory, w: int) -> np.ndarray:
    """Reconstruct disruption-component membership when the raw source doesn't provide one (the
    common case — real event-sourced schedule-history logs are per-flight, not pre-grouped).

    Mirrors `_build_disruption_weights`'s two coupling families: rotation chains (LOCAL — flights
    sharing a `tail_number`, adjacent in canonical (tail, scheduled_departure) order, are linked)
    and bank cliques (NON-LOCAL — flights sharing a `station_bank` value are linked). Union-find
    over BOTH edge types gives connected "disruption neighbourhoods"; each is walked in canonical
    order and CHUNKED into consecutive groups of exactly `w` flights (a leftover remainder smaller
    than `w` still gets an id — it is simply dropped downstream by `transform_schedule_history`'s
    width filter, exactly like any other odd-sized component). Chunk ids are assigned by sorting
    all chunks on their canonically-first flight, so they are a deterministic function of the data
    (not of arbitrary union-find root choice).

    Returns a component-id array PER ROW of `history` (every snapshot of a flight shares its id).
    """
    endpoints = _endpoints_by_flight(history)
    flight_ids = list(endpoints.keys())
    parent: dict[Any, Any] = {fid: fid for fid in flight_ids}

    def find(x: Any) -> Any:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Any, b: Any) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tail: dict[Any, Any] = {}
    dep: dict[Any, Any] = {}
    bank: dict[Any, Any] = {}
    for fid in flight_ids:
        t0_idx, _ = endpoints[fid]
        tail[fid] = history.tail_number[t0_idx]
        dep[fid] = history.scheduled_departure[t0_idx]
        bank[fid] = history.station_bank[t0_idx]

    by_bank: dict[Any, list[Any]] = defaultdict(list)
    for fid in flight_ids:
        by_bank[bank[fid]].append(fid)
    for group in by_bank.values():
        for fid in group[1:]:
            union(group[0], fid)

    by_tail: dict[Any, list[Any]] = defaultdict(list)
    for fid in flight_ids:
        by_tail[tail[fid]].append(fid)
    for group in by_tail.values():
        group.sort(key=lambda fid: (dep[fid], fid))
        for a, b in zip(group, group[1:]):
            union(a, b)

    by_root: dict[Any, list[Any]] = defaultdict(list)
    for fid in flight_ids:
        by_root[find(fid)].append(fid)

    chunks: list[list[Any]] = []
    for members in by_root.values():
        members.sort(key=lambda fid: (tail[fid], dep[fid], fid))
        for i in range(0, len(members), w):
            chunks.append(members[i : i + w])
    chunks.sort(key=lambda chunk: (tail[chunk[0]], dep[chunk[0]], chunk[0]))

    comp_of_flight: dict[Any, str] = {}
    for idx, chunk in enumerate(chunks):
        for fid in chunk:
            comp_of_flight[fid] = f"auto_{idx}"

    return np.array([comp_of_flight[fid] for fid in history.flight_id.tolist()])


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

    If ``history.component_id`` is ``None`` (the common case for real data), components are first
    reconstructed via ``derive_components(history, w)``.

    ``n`` = the number of components with exactly ``w`` flights, emitted in order of first
    appearance. Raises if that is zero (nothing to train/eval on) — most likely ``w`` does not
    match the real component sizes.

    Within each accepted component, flights are placed in CANONICAL ORDER — sorted by
    ``tail_number``, then ``scheduled_departure`` (ties broken by ``flight_id``) — mirroring
    ``_build_disruption_weights``'s convention, so ``is_head`` (no rotation predecessor: the first
    flight of a contiguous same-tail run in that order) matches the synthetic generator's
    semantics. Note ``is_head`` is scoped to the component: a tail whose rotation is split across
    two components will show ``is_head=1`` in both — the synthetic generator has the same
    per-instance scoping, since it never models an infinite chain across rows either.
    """
    if history.component_id is None:
        history = replace(history, component_id=derive_components(history, w))

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

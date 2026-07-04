"""Tests for the real schedule-history -> `disruption` (X, y) bridge helper.

Covers both the "clean parallel arrays" entry point (`ScheduleHistory` built directly, explicit
`component_id`) and the "as it would actually be stored" entry point (an arbitrary raw table with
source-specific column names, `severe`/`station_bank`/`component_id` derived rather than mapped
1:1) plus the discoverability helpers (`describe_roles`, `suggest_column_mapping`,
`mapping_report`) meant to help a user figure out their own column mapping.
"""

from functools import partial

import numpy as np
import pytest

from looptab.data.schedule_history import (
    ROLE_SPECS,
    ColumnMapping,
    ScheduleHistory,
    apply_column_mapping,
    derive_bank,
    derive_components,
    derive_severe_from_delay,
    describe_roles,
    fit_bank_vocabulary,
    mapping_report,
    raw_columns,
    suggest_column_mapping,
    transform_schedule_history,
)


def _history_with_explicit_components():
    # Component "A": w=3, two tails (T1 has 2 flights, T2 has 1).
    #   F1 (T1, dep=100, BankX): severe 1 -> 0 (two snapshots)
    #   F2 (T1, dep=200, BankY): severe 1 (one snapshot only; t0 == settle)
    #   F3 (T2, dep=150, BankX): severe 0 -> 1 -> 1 (three snapshots)
    # Component "B": only 2 flights -> must be dropped (wrong width for w=3).
    rows = [
        ("F1", "A", 0, "T1", 100, "BankX", 1),
        ("F1", "A", 1, "T1", 100, "BankX", 0),
        ("F2", "A", 0, "T1", 200, "BankY", 1),
        ("F3", "A", 0, "T2", 150, "BankX", 0),
        ("F3", "A", 1, "T2", 150, "BankX", 1),
        ("F3", "A", 2, "T2", 150, "BankX", 1),
        ("F4", "B", 0, "T3", 10, "BankX", 1),
        ("F5", "B", 0, "T4", 20, "BankX", 0),
    ]
    cols = list(zip(*rows))
    return ScheduleHistory(
        flight_id=cols[0],
        component_id=cols[1],
        version=cols[2],
        tail_number=cols[3],
        scheduled_departure=cols[4],
        station_bank=cols[5],
        severe=cols[6],
    )


# --- Direct ScheduleHistory construction (explicit component_id) ---------------------------------


def test_fit_bank_vocabulary_deterministic():
    h = _history_with_explicit_components()
    assert fit_bank_vocabulary(h) == ("BankX", "BankY")
    assert fit_bank_vocabulary(h) == fit_bank_vocabulary(h)


def test_transform_matches_hand_computed_block():
    h = _history_with_explicit_components()
    banks = fit_bank_vocabulary(h)
    X, y, dropped = transform_schedule_history(h, w=3, bank_categories=banks)

    assert dropped == ["B"]
    assert X.shape == (1, 3 * (2 + 2))
    assert y.shape == (1, 3)
    assert X.dtype == np.float32 and y.dtype == np.int64

    # Canonical order: (tail, dep) -> F1(T1,100), F2(T1,200), F3(T2,150).
    # Per-flight block = [severe_0, is_head, bank_onehot(BankX, BankY)].
    expected_X = np.array(
        [
            1, 1, 1, 0,  # F1: severe_0=1, is_head=1 (start of T1 run), BankX
            1, 0, 0, 1,  # F2: severe_0=1, is_head=0 (still T1),        BankY
            0, 1, 1, 0,  # F3: severe_0=0, is_head=1 (new tail T2),     BankX
        ],
        dtype=np.float32,
    )
    expected_y = np.array([0, 1, 1], dtype=np.int64)  # last-recorded severe per flight
    np.testing.assert_array_equal(X[0], expected_X)
    np.testing.assert_array_equal(y[0], expected_y)


def test_wrong_width_components_are_dropped_not_padded():
    h = _history_with_explicit_components()
    banks = fit_bank_vocabulary(h)
    # w=2 flips which component survives: "B" (2 flights) now fits, "A" (3 flights) is dropped.
    X, y, dropped = transform_schedule_history(h, w=2, bank_categories=banks)
    assert dropped == ["A"]
    assert X.shape == (1, 2 * (2 + 2))
    assert y.shape == (1, 2)


def test_no_matching_component_raises():
    h = _history_with_explicit_components()
    banks = fit_bank_vocabulary(h)
    with pytest.raises(ValueError):
        transform_schedule_history(h, w=99, bank_categories=banks)


def test_unseen_bank_raises_loudly():
    h = _history_with_explicit_components()
    with pytest.raises(ValueError, match="not in bank_categories"):
        transform_schedule_history(h, w=3, bank_categories=("BankY",))  # BankX missing


def test_transform_is_deterministic():
    h = _history_with_explicit_components()
    banks = fit_bank_vocabulary(h)
    X1, y1, d1 = transform_schedule_history(h, w=3, bank_categories=banks)
    X2, y2, d2 = transform_schedule_history(h, w=3, bank_categories=banks)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert d1 == d2


def test_mismatched_column_lengths_raise():
    with pytest.raises(ValueError):
        ScheduleHistory(
            flight_id=["F1", "F2"],
            component_id=["A", "A"],
            version=[0, 0],
            tail_number=["T1", "T1"],
            scheduled_departure=[100, 200],
            station_bank=["BankX"],  # too short
            severe=[1, 0],
        )


# --- Discoverability helpers (b): does a user get enough help to build their own mapping? --------


def test_describe_roles_covers_every_role():
    text = describe_roles()
    for role in ROLE_SPECS:
        assert role in text


def test_suggest_column_mapping_finds_plausible_aliases():
    columns = ["FLT_ID", "TAIL_NBR", "SKED_DEP", "STATION", "EVENT_TS", "DELAY_MIN", "CX_FLAG"]
    suggestions = suggest_column_mapping(columns)
    assert "FLT_ID" in suggestions["flight_id"]
    assert "TAIL_NBR" in suggestions["tail_number"]
    assert "SKED_DEP" in suggestions["scheduled_departure"]
    assert "EVENT_TS" in suggestions["version"]
    assert {"DELAY_MIN", "CX_FLAG"} <= set(suggestions["severe"])
    # No raw column looks like a disruption-component id -- nothing suggested (the expected case).
    assert suggestions["component_id"] == []


def test_raw_columns_helper_agrees_for_both_shapes():
    columnar = {"a": [1, 2], "b": [3, 4]}
    records = [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    assert raw_columns(columnar) == ["a", "b"]
    assert raw_columns(records) == ["a", "b"]


# --- Raw, as-stored table + ColumnMapping (the realistic entry point) ----------------------------


def _raw_records():
    # Cryptic, source-specific column names; `severe` and `station_bank` are NOT stored directly
    # (this is the realistic case -- see ROLE_SPECS). No component/incident id column at all.
    return [
        {"FLT_ID": "F1", "TAIL_NBR": "T1", "SKED_DEP": 100, "STATION": "ORD",
         "EVENT_TS": 0, "DELAY_MIN": 20, "CX_FLAG": 0},
        {"FLT_ID": "F1", "TAIL_NBR": "T1", "SKED_DEP": 100, "STATION": "ORD",
         "EVENT_TS": 1, "DELAY_MIN": 0, "CX_FLAG": 0},
        {"FLT_ID": "F2", "TAIL_NBR": "T1", "SKED_DEP": 200, "STATION": "DFW",
         "EVENT_TS": 0, "DELAY_MIN": 30, "CX_FLAG": 0},
        {"FLT_ID": "F3", "TAIL_NBR": "T2", "SKED_DEP": 150, "STATION": "ORD",
         "EVENT_TS": 0, "DELAY_MIN": 0, "CX_FLAG": 0},
        {"FLT_ID": "F3", "TAIL_NBR": "T2", "SKED_DEP": 150, "STATION": "ORD",
         "EVENT_TS": 1, "DELAY_MIN": 0, "CX_FLAG": 0},
        {"FLT_ID": "F3", "TAIL_NBR": "T2", "SKED_DEP": 150, "STATION": "ORD",
         "EVENT_TS": 2, "DELAY_MIN": 20, "CX_FLAG": 0},
        {"FLT_ID": "F4", "TAIL_NBR": "T3", "SKED_DEP": 300, "STATION": "LAX",
         "EVENT_TS": 0, "DELAY_MIN": 0, "CX_FLAG": 0},
    ]


def _raw_columnar():
    records = _raw_records()
    return {k: [r[k] for r in records] for k in records[0]}


def _mapping():
    return ColumnMapping(
        flight_id="FLT_ID",
        version="EVENT_TS",
        tail_number="TAIL_NBR",
        scheduled_departure="SKED_DEP",
        station_bank=partial(derive_bank, station_col="STATION", time_col="SKED_DEP",
                              window_minutes=1000),
        severe=partial(derive_severe_from_delay, delay_minutes_col="DELAY_MIN",
                        threshold_minutes=15),
        # component_id left unmapped -> auto-derived from tail/bank connectivity.
    )


def test_derive_severe_from_delay_thresholds_and_overrides():
    cols = {"delay": [10, 20], "cx": [0, 0], "div": [0, 1]}
    out = derive_severe_from_delay(cols, "delay", threshold_minutes=15)
    np.testing.assert_array_equal(out, [0, 1])
    out2 = derive_severe_from_delay(cols, "delay", threshold_minutes=15, diverted_col="div")
    np.testing.assert_array_equal(out2, [0, 1])  # row0 still not severe (no delay/cx/divert)
    out3 = derive_severe_from_delay({"delay": [1], "cx": [1]}, "delay", cancelled_col="cx")
    np.testing.assert_array_equal(out3, [1])  # cancelled overrides low delay


def test_derive_bank_buckets_station_and_time():
    cols = {"station": ["ORD", "ORD", "DFW"], "t": [0, 999, 0]}
    banks = derive_bank(cols, "station", "t", window_minutes=1000)
    assert banks[0] == banks[1]  # same station, same 1000-wide bucket
    assert banks[0] != banks[2]  # different station


def test_apply_column_mapping_missing_column_raises_with_hint():
    mapping = ColumnMapping(
        flight_id="NOT_A_COLUMN",
        version="EVENT_TS",
        tail_number="TAIL_NBR",
        scheduled_departure="SKED_DEP",
        station_bank="STATION",
        severe="DELAY_MIN",
    )
    with pytest.raises(KeyError, match="not found among this table's columns"):
        apply_column_mapping(_raw_records(), mapping)


def test_mapping_report_shows_source_dtype_preview_and_unmapped_component():
    report = mapping_report(_raw_records(), _mapping())
    assert "raw column 'FLT_ID'" in report
    assert "derived (callable)" in report
    assert "component_id: <not provided>" in report
    assert "auto-derived" in report


@pytest.mark.parametrize("raw_fn", [_raw_records, _raw_columnar])
def test_transform_from_raw_table_with_auto_derived_components(raw_fn):
    raw = raw_fn()
    history = apply_column_mapping(raw, _mapping())
    assert history.component_id is None  # not yet derived -- happens inside the transform

    banks = fit_bank_vocabulary(history)
    X, y, dropped = transform_schedule_history(history, w=3, bank_categories=banks)

    # F1-F2 share a tail (rotation edge); F1-F3 share a bank ("ORD" within the same time bucket);
    # union-find should merge {F1, F2, F3} into one component of size 3. F4 is an isolated island
    # (different tail, different bank) and gets left in a leftover chunk of size 1 -> dropped.
    assert X.shape == (1, 3 * (2 + len(banks)))
    assert y.shape == (1, 3)
    assert len(dropped) == 1

    bank_index = {b: i for i, b in enumerate(banks)}

    def onehot(bank_value):
        v = [0.0] * len(banks)
        v[bank_index[bank_value]] = 1.0
        return v

    # Canonical order: (tail, dep) -> F1(T1,100), F2(T1,200), F3(T2,150).
    expected = np.array(
        [1.0, 1.0, *onehot("ORD#0"),  # F1: severe_0=1 (delay 20>=15), is_head=1, bank ORD#0
         1.0, 0.0, *onehot("DFW#0"),  # F2: severe_0=1 (delay 30>=15), is_head=0 (still tail T1)
         0.0, 1.0, *onehot("ORD#0")],  # F3: severe_0=0 (delay 0), is_head=1 (new tail T2)
        dtype=np.float32,
    )
    np.testing.assert_array_equal(X[0], expected)
    np.testing.assert_array_equal(y[0], [0, 1, 1])  # F1 settles 0, F2 settles 1, F3 settles 1


def test_derive_components_is_deterministic():
    history = apply_column_mapping(_raw_records(), _mapping())
    a = derive_components(history, w=3)
    b = derive_components(history, w=3)
    np.testing.assert_array_equal(a, b)


def test_derive_components_respects_explicit_component_id_when_given():
    # If the caller DOES supply a component_id, transform_schedule_history must use it as-is
    # rather than silently re-deriving one.
    h = _history_with_explicit_components()
    assert h.component_id is not None
    X, y, dropped = transform_schedule_history(h, w=3, bank_categories=fit_bank_vocabulary(h))
    assert dropped == ["B"]

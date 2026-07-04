"""Tests for the real schedule-history -> `disruption` (X, y) bridge helper."""

import numpy as np
import pytest

from looptab.data.schedule_history import (
    ScheduleHistory,
    fit_bank_vocabulary,
    transform_schedule_history,
)


def _history():
    # Component "A": w=3, two tails (T1 has 2 flights, T2 has 1).
    #   F1 (T1, dep=100, BankX): severe 1 -> 0 (two snapshots)
    #   F2 (T1, dep=200, BankY): severe 1 (one snapshot only; t0 == settle)
    #   F3 (T2, dep=150, BankX): severe 0 -> 1 -> 1 (three snapshots)
    # Component "B": only 2 flights -> must be dropped (wrong width for w=3).
    rows = [
        # flight_id, component_id, version, tail_number, scheduled_departure, station_bank, severe
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


def test_fit_bank_vocabulary_deterministic():
    h = _history()
    assert fit_bank_vocabulary(h) == ("BankX", "BankY")
    assert fit_bank_vocabulary(h) == fit_bank_vocabulary(h)


def test_transform_matches_hand_computed_block():
    h = _history()
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
    h = _history()
    banks = fit_bank_vocabulary(h)
    # w=2 flips which component survives: "B" (2 flights) now fits, "A" (3 flights) is dropped.
    X, y, dropped = transform_schedule_history(h, w=2, bank_categories=banks)
    assert dropped == ["A"]
    assert X.shape == (1, 2 * (2 + 2))
    assert y.shape == (1, 2)


def test_no_matching_component_raises():
    h = _history()
    banks = fit_bank_vocabulary(h)
    with pytest.raises(ValueError):
        transform_schedule_history(h, w=99, bank_categories=banks)


def test_unseen_bank_raises_loudly():
    h = _history()
    with pytest.raises(ValueError, match="not in bank_categories"):
        transform_schedule_history(h, w=3, bank_categories=("BankY",))  # BankX missing


def test_transform_is_deterministic():
    h = _history()
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

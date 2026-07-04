"""M24f Stage 0 — screen the distractors=0 `disruption` instance for ff-hardness.

The mixer needs ``in_features % out_features == 0`` ⇒ we must set ``distractors: 0`` (with the
M22 distractors=8, 152 % 24 = 8 ✗; with 0, 144 % 24 = 0 ✓, cell_dim = 2+n_banks = 6). Distractor
columns are appended to X AFTER y is built (generators.py), so the TARGET is byte-identical to M22
— only ff/linear separability can shift. This confirms the task stays ff-HARD (a linear baseline
must fall well short) before we commit the 8-seed runs, and prints the auto-derived minimal gamma
at w=32.

Run: .venv/bin/python scratchpad/screen_disruption_nodist.py
"""

import numpy as np

from looptab.data.generators import make_disruption

BASE = dict(n_banks=4, w_rot=6, w_bank=3, min_tail=2, max_tail=8)
N_TR, N_TE = 4000, 1000
TASK_SEED, TR_SEED, TE_SEED = 42, 1, 2


def ridge_linear_em(Xtr, ytr, Xte, yte, lam=1.0):
    """Closed-form per-cell ridge readout; threshold at 0.5; report EM + per-cell acc."""
    Xa_tr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1), np.float64)], axis=1)
    Xa_te = np.concatenate([Xte, np.ones((Xte.shape[0], 1), np.float64)], axis=1)
    A = Xa_tr.T @ Xa_tr + lam * np.eye(Xa_tr.shape[1])
    B = np.linalg.solve(A, Xa_tr.T @ ytr.astype(np.float64))
    pred = (Xa_te @ B > 0.5).astype(np.int64)
    em = (pred == yte).all(axis=1).mean()
    cell_acc = (pred == yte).mean()
    return em, cell_acc


def screen(w, gamma):
    params = dict(w=w, task_seed=TASK_SEED, distractors=0, **BASE)
    Xtr, ytr = make_disruption(n=N_TR, sample_seed=TR_SEED, gamma=gamma, **params)
    Xte, yte = make_disruption(n=N_TE, sample_seed=TE_SEED, gamma=gamma, **params)

    # gamma actually used (auto-derived if gamma=None): re-derive to print it.
    if gamma is None:
        from looptab.data.generators import _build_disruption_weights

        W, *_ = _build_disruption_weights(
            w, BASE["n_banks"], BASE["w_rot"], BASE["w_bank"],
            BASE["min_tail"], BASE["max_tail"], TASK_SEED,
        )
        lam_min = float(np.linalg.eigvalsh(W.astype(np.float64)).min())
        gamma_used = max(int(np.ceil(-lam_min - 1e-9)) + 1, 0)  # gamma_margin default = 1
        gamma_min0 = max(int(np.ceil(-lam_min - 1e-9)), 0)      # margin 0 (the M22 convention)
    else:
        gamma_used = gamma
        gamma_min0 = gamma

    cell_dim = 2 + BASE["n_banks"]
    assert Xtr.shape[1] == w * cell_dim, (Xtr.shape, w * cell_dim)
    assert Xtr.shape[1] % w == 0, "mixer divisibility broken"

    severe0 = Xte.reshape(N_TE, w, cell_dim)[:, :, 0].astype(np.int64)
    copy_frac = (yte == severe0).mean()
    flips_per_row = (yte != severe0).sum(axis=1).mean()
    # modal whole-row frequency on test = EM-baseline (majority predictor)
    rows, counts = np.unique(yte, axis=0, return_counts=True)
    em_baseline = counts.max() / N_TE
    balance = yte.mean()

    lin_em, lin_cell = ridge_linear_em(Xtr, ytr, Xte, yte)

    print(f"\n=== w={w} (gamma passed={gamma}) ===")
    print(f"  gamma auto (margin1)={gamma_used}  gamma auto (margin0/M22)={gamma_min0}")
    print(f"  in_features={Xtr.shape[1]}  cell_dim={cell_dim}  {Xtr.shape[1]}%{w}={Xtr.shape[1]%w}")
    print(f"  copy_frac={copy_frac:.3f} (M22 ~0.82)  flips/row={flips_per_row:.2f} (M22 ~4.4)")
    print(f"  em_baseline={em_baseline:.3f} (M22 ~0.036)  cell balance={balance:.3f} (~0.5)")
    print(f"  LINEAR ridge: EM={lin_em:.3f}  per-cell acc={lin_cell:.3f}  (M22 linear EM ~0.23)")
    print("  -> ff-hard if LINEAR EM stays well below Stage-1 ff EM (M22 ff ~0.34)")


if __name__ == "__main__":
    screen(w=24, gamma=14)          # M22 locked minimal-PSD int for task_seed=42
    screen(w=32, gamma=None)        # print the auto-derived minimal gamma at w=32 (expect ~15)

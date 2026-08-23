#!/usr/bin/env python3
"""Ground-truth simulation for the interference estimands S(s), D(s), B(s).

Four mechanisms whose spillover behaviour is known before running anything:

    lookup   memorises exact rows, falls back to a frozen constant --
             spillover is IMPOSSIBLE by construction: S(s) = 0 exactly,
             per draw, not just in expectation. The zero anchor.
    knn1     1-nearest-neighbour -- spillover is LOCAL: an uninserted row
             moves only if an inserted row becomes its nearest neighbour.
    ridge    penalised linear fit -- spillover is GLOBAL: every coefficient
             shifts, every prediction moves.
    tree     unbounded decision tree -- spillover is ADAPTIVE: splits move,
             some regions change and others do not.

Two data regimes, because the size of the spillover is a property of what
the inserted rows can TEACH, not of the mechanism alone:

    iid      evaluation rows drawn from the training law -- a well-specified
             model has nothing to learn from them beyond their own answers,
             so S sits near zero even for the global mechanism, and the
             no-spillover assumption of itemwise corrections is nearly true.
    drift    a constant shift separates evaluation from training (the
             temporal-panel situation) -- inserted rows teach the shift and
             the global mechanism converts a few of them into improvement
             on EVERY row. This is the regime the real panels live in.

What is validated, per (regime, mechanism, saturation):

  1. TRUTH   S, D by dense Monte Carlo over NDRAWS independent fixed-size
             draws (the design's own expectation, reported with its MC SE).
  2. EXACT   lookup's S_hat is 0.0 in EVERY draw (float-exact), and the
             identity B = S + share*D holds per draw to float precision.
  3. MOVED   share of uninserted rows whose prediction changed at all --
             the mechanism fingerprint (lookup 0%, ridge 100%, knn/tree
             in between, growing with s).
  4. COVER   partition the pool into groups of R draws (the audit's
             replicate count); the t interval over each group should cover
             the pool truth ~95% of the time. This is the calibration of
             the intervals the real audit reports.
  5. CORRECT the three corrections against the clean score, in improvement
             scale: drop-the-contaminated misses by S(s); perfect itemwise
             correction (each inserted row's loss restored to its clean
             value -- an oracle no real method beats) still misses by
             (1-s)*S(s); the spillover-aware correction -- one audit's drop
             score corrected with S estimated from an INDEPENDENT audit's
             replicates (disjoint group pairs) -- centres on zero with a
             reported SE. Lemma 3 of the formal section, with ground truth.
             (A within-group leave-one-out average residual would print
             0.0 for ANY data -- an arithmetic identity, not evidence --
             as would any circular pairing; hence disjoint pairs.)
  6. RATIO   sign disagreement per draw between the group-relative
             (ratio-of-sums) direct reading D_rel = L_rel - G_rel and the
             additive D_hat. The disagreement concentrates where D is near
             zero -- in this homogeneous-loss DGP the two agree in sign in
             expectation everywhere -- so the column measures sorteio-level
             sign instability of the relative reading in the pure-spillover
             corner, not a systematic sign reversal. The load-bearing case
             against ratio-of-sums is the lemma's counterexample under
             heterogeneous per-row losses; this column shows where the
             instability bites in practice (ridge under drift).

Environment knobs: SIM_DRAWS (default 6000), SIM_R (default 40),
SIM_SATURATIONS (default 0.05,0.10,0.30), SIM_SEED (default 42).

Run: .venv/bin/python scripts/validation/sim_interference_groundtruth.py
CPU-only, a few minutes; no network, no panels.
"""

import os

import numpy as np

NDRAWS = int(os.environ.get('SIM_DRAWS', '6000'))
R = int(os.environ.get('SIM_R', '40'))
SATURATIONS = tuple(float(x) for x in
                    os.environ.get('SIM_SATURATIONS', '0.05,0.10,0.30').split(','))
SEED = int(os.environ.get('SIM_SEED', '42'))

N_TRAIN, N_EVAL, P = 240, 120, 6
BETA = np.array([2.0, -1.0, 1.5, 0.0, 0.5, -2.0])
NOISE = 1.0

HEADER = (f"{'mech':>7} {'s':>5} {'S_true(MC se)':>18} {'D_true(MC se)':>18} "
          f"{'moved%':>7} {'covS%':>6} {'covD%':>6} "
          f"{'drop':>8} {'itemw':>8} {'sp-aware':>9} {'relFlip%':>9}")


class Lookup:
    """Memorises exact rows; frozen fallback, so spillover cannot happen."""

    def fit(self, X, y):
        self.table = {row.tobytes(): t for row, t in zip(np.asarray(X), y)}
        return self

    def predict(self, X):
        return np.array([self.table.get(row.tobytes(), 0.0)
                         for row in np.asarray(X)])


def make_models():
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    return {
        'lookup': Lookup,
        'knn1': lambda: KNeighborsRegressor(n_neighbors=1),
        'ridge': lambda: Ridge(alpha=10.0),
        'tree': lambda: DecisionTreeRegressor(random_state=SEED),
    }


def t_half(n, level=0.95):
    from scipy import stats
    return stats.t.ppf(0.5 + level / 2, n - 1)


def run_regime(shift, regime_code, tally):
    rng = np.random.default_rng(SEED)
    X_train = rng.standard_normal((N_TRAIN, P))
    y_train = X_train @ BETA + NOISE * rng.standard_normal(N_TRAIN)
    X_eval = rng.standard_normal((N_EVAL, P))
    y_eval = X_eval @ BETA + shift + NOISE * rng.standard_normal(N_EVAL)

    print(f"--- regime: {'iid' if shift == 0 else f'drift (+{shift:g} shift)'} ---")
    print(HEADER)
    th = t_half(R)
    for code, (name, make) in enumerate(make_models().items()):
        clean = make().fit(X_train, y_train)
        clean_pred = np.asarray(clean.predict(X_eval), dtype=float)
        clean_loss = (y_eval - clean_pred) ** 2

        for s in SATURATIONS:
            k = max(1, int(round(s * N_EVAL)))
            share = k / N_EVAL
            S_r = np.empty(NDRAWS)
            D_r = np.empty(NDRAWS)
            moved = np.empty(NDRAWS)
            item_r = np.empty(NDRAWS)
            rel_flip = 0
            for d in range(NDRAWS):
                # Numeric seed parts only: str hash() is per-process salted.
                # round() before int(): plain truncation collides saturations
                # (0.29 -> 28) and silently shares streams between cells.
                drw = np.random.default_rng(
                    (SEED, regime_code, code, int(round(s * 100)), d))
                handed = drw.choice(N_EVAL, size=k, replace=False)
                mask = np.zeros(N_EVAL, dtype=bool)
                mask[handed] = True
                Xa = np.vstack([X_train, X_eval[handed]])
                ya = np.concatenate([y_train, y_eval[handed]])
                pred = np.asarray(make().fit(Xa, ya).predict(X_eval), dtype=float)
                loss = (y_eval - pred) ** 2
                gain = clean_loss - loss
                S_r[d] = gain[~mask].mean()
                D_r[d] = gain[mask].mean() - S_r[d]
                B = gain.mean()
                tally['identity'] = max(tally['identity'],
                                        abs(B - (S_r[d] + share * D_r[d])))
                moved[d] = (np.abs(pred - clean_pred)[~mask] > 1e-12).mean()
                # Perfect itemwise correction: inserted rows restored to their
                # clean losses; residual miss vs the clean score, gain scale.
                item_r[d] = (1 - share) * S_r[d]
                # Group-relative reading, the old channels' scale.
                l_rel = 1 - loss[mask].sum() / clean_loss[mask].sum()
                g_rel = 1 - loss[~mask].sum() / clean_loss[~mask].sum()
                if np.sign(l_rel - g_rel) != np.sign(D_r[d]) and D_r[d] != 0:
                    rel_flip += 1
            if name == 'lookup' and np.abs(S_r).max() > 0.0:
                tally['lookup_exact'] = False

            S_true, D_true = S_r.mean(), D_r.mean()
            se_S = S_r.std(ddof=1) / np.sqrt(NDRAWS)
            se_D = D_r.std(ddof=1) / np.sqrt(NDRAWS)
            groups = NDRAWS // R
            cov_S = cov_D = 0
            group_S = np.empty(groups)
            for g in range(groups):
                sl = slice(g * R, (g + 1) * R)
                m, sd = S_r[sl].mean(), S_r[sl].std(ddof=1)
                group_S[g] = m
                cov_S += abs(m - S_true) <= th * sd / np.sqrt(R)
                m, sd = D_r[sl].mean(), D_r[sl].std(ddof=1)
                cov_D += abs(m - D_true) <= th * sd / np.sqrt(R)
            # Degenerate cell (lookup: every S_hat identical): the coverage
            # comparison is 0 <= 0 and prints 100% vacuously -- say so.
            s_degenerate = np.all(S_r == S_r[0])
            # Cross-audit spillover-aware correction over DISJOINT group
            # pairs: audit 2p's drop bias corrected with S estimated by
            # audit 2p+1. Within-group LOO averages and circular shifts are
            # both identically zero for any data; disjoint pairs are not.
            pairs = group_S[:(groups // 2) * 2].reshape(-1, 2)
            aware = pairs[:, 0] - pairs[:, 1]
            aware_se = (aware.std(ddof=1) / np.sqrt(len(aware))
                        if len(aware) > 1 else float('nan'))

            cov_s_txt = 'exact' if s_degenerate else f'{100 * cov_S / groups:.1f}%'
            print(f'{name:>7} {s:>5.2f} '
                  f'{S_true:>+9.4f}({se_S:.4f}) {D_true:>+9.4f}({se_D:.4f}) '
                  f'{100 * moved.mean():>6.1f}% '
                  f'{cov_s_txt:>6} {100 * cov_D / groups:>5.1f}% '
                  f'{S_true:>+8.4f} {item_r.mean():>+8.4f} '
                  f'{aware.mean():>+8.4f}({aware_se:.4f}) '
                  f'{100 * rel_flip / NDRAWS:>8.1f}%', flush=True)
    print()


def main():
    print(f'ground-truth simulation: n_train={N_TRAIN}, n_eval={N_EVAL}, '
          f'{NDRAWS} draws per cell, replicate groups of {R}, seed {SEED}')
    print(f'saturations {list(SATURATIONS)}; mechanisms lookup/knn1/ridge/tree\n')

    tally = {'identity': 0.0, 'lookup_exact': True}
    for regime_code, shift in enumerate((0.0, 1.5)):
        run_regime(shift, regime_code, tally)

    print(f"identity check: max |B - (S + share*D)| over every draw = "
          f"{tally['identity']:.2e}")
    print(f"lookup exactness: S_hat == 0.0 in every single draw: "
          f"{'YES' if tally['lookup_exact'] else 'NO -- BUG'}")
    print('\ncolumns: drop = bias of dropping contaminated rows (= S, Lemma 3);')
    print('itemw = bias of PERFECT itemwise correction (= (1-s)S -- an oracle')
    print('no calibration-based method beats); sp-aware = mean(SE) of the')
    print('cross-audit corrected residual over disjoint replicate-group pairs')
    print('(stochastically ~0 when the correction works -- a real test, unlike')
    print('a within-group LOO average, which is 0 for any data by identity);')
    print('relFlip% = draws where the ratio-of-sums direct reading disagrees')
    print('in SIGN with the additive one (concentrates where D is near zero).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

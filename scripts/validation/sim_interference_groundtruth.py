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

A fifth mechanism is opt-in via RAMPART_SIM_MECHS (see knobs below):

    mlp      one-hidden-layer network (16 units), lbfgs truncated at a
             fixed iteration budget -- a non-convex GLOBAL learner. The
             truncation is deliberate: the fit is a deterministic map of
             (data, SEED), never a converged optimum, which is exactly
             the regime P-F1.2 below registers a prediction about.

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

Two INDEPENDENT Monte Carlo streams per (regime, mechanism, saturation)
cell, each of NDRAWS fixed-size draws, seeded from disjoint numeric tuples:

    TRUTH    (seed, regime, mech, round(100 s), 0, d) -- its means define
             S_true and D_true and every other expectation of the DGP that
             the receipt prints (MC SE, moved share, drop and itemwise
             biases, sign-flip rate).
    EVAL     (seed, regime, mech, round(100 s), 1, d) -- split into groups
             of R draws that play the real audit's replicates; coverage
             and the cross-audit correction are computed on these groups
             against the TRUTH-stream means.

The split makes coverage a genuine out-of-sample check. Scoring a group
of R draws against the mean of a pool that CONTAINS those R draws is
self-referential: the group pulls the truth towards itself by R/NDRAWS of
its own deviation, so the interval has slightly less to cover than a
group scored against an independent truth, and the printed coverage is
optimistic by construction. With independent streams the only slack left
is the truth's own MC error, which is reported next to it.

What is validated, per (regime, mechanism, saturation):

  1. TRUTH   S, D by dense Monte Carlo over the NDRAWS draws of the TRUTH
             stream (the design's own expectation, reported with its MC SE
             from the same stream).
  2. EXACT   lookup's S_hat is 0.0 in EVERY draw of BOTH streams
             (float-exact), and the identity B = S + share*D holds per
             draw to float precision, again over both streams.
  3. MOVED   share of uninserted rows whose prediction changed at all --
             the mechanism fingerprint (lookup 0%, ridge 100%, knn/tree
             in between, growing with s). TRUTH stream.
  4. COVER   partition the EVAL stream into groups of R draws (the audit's
             replicate count); the t interval over each group should cover
             the TRUTH-stream mean ~95% of the time. This is the
             calibration of the intervals the real audit reports, scored
             out of sample.
  5. CORRECT the three corrections against the clean score, in improvement
             scale: drop-the-contaminated misses by S(s); exact restoration
             (each inserted row's loss restored to its clean value) still
             misses by (1-s)*S(s), the spillover component shared by
             every correction that edits only inserted rows (a partial restoration can land
             closer by chance, so this is not a bound); the spillover-aware correction -- one audit's drop
             score corrected with S estimated from an INDEPENDENT audit's
             replicates (disjoint group pairs of the EVAL stream) -- centres
             on zero with a reported SE. Lemma 3 of the formal section,
             with ground truth.
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
  7. SCALE   cleanL, the mechanism's mean clean squared loss on the
             evaluation rows, so S can be read as a share of the
             mechanism's own clean loss (S / cleanL) rather than in
             absolute units that differ by an order of magnitude across
             mechanisms.
  8. ANALYTIC (ridge only) the first-order intercept-absorption prediction
             S_pred = 2*rbar*delta_c - delta_c^2, where delta_c =
             shift*k/(n_train+k) is the intercept displacement induced by
             k inserted rows carrying +shift (the unpenalised intercept
             absorbs their mean) and rbar is the mean clean residual on
             the evaluation rows; every uninserted row's squared loss goes
             from r^2 to (r - delta_c)^2. Printed with its % gap to S_true.
             Under iid shift = 0, so delta_c = 0 and the prediction is
             identically zero: the gap is printed as n/a and the residual
             S_true (~1e-3) is the second-order coefficient effect the
             first-order formula omits.

Environment knobs: SIM_DRAWS (default 6000, per stream), SIM_R (default 40),
SIM_SATURATIONS (default 0.05,0.10,0.30), SIM_SEED (default 42),
RAMPART_SIM_MECHS (default lookup,knn1,ridge,tree -- the four-mechanism
receipt is byte-identical to the pre-mlp script; add mlp for the neural
mechanism, whose seed tuples use the canonical code 4 so the other four
cells' streams do not move).

Run: .venv/bin/python scripts/validation/sim_interference_groundtruth.py
CPU-only, a few minutes (two streams, so 2*NDRAWS fits per cell); no
network, no panels. With mlp active the run adds ~45 minutes single-process
(measured 36 ms per lbfgs fit at the worst-case cell, 72,000 fits total).

REGISTERED PREDICTION (F1.2, 30 Aug 2026): P-F1.2: the neural mechanism
behaves as a global learner (moved share near 100%); under drift S > 0 and
under iid S near 0; the open question is the sign and dispersion of S on
small frames, where a non-convex fit at a fixed seed may move without
carrying signal. Integration decision, pre-committed: if the real-data
multi-seed fleet (P-F1.1) reads the World Bank negative spillover as seed
noise, the simulated mechanism gets one sentence in Section 7 and its rows
go to the appendix table only; if it reads it as real, the mechanism gets
two to four sentences in Section 7 explaining the pattern. The sim tables
gain two rows (iid/drift); nothing else changes.
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

#: Numeric stream tags appended to every cell's seed tuple. TRUTH defines
#: S_true/D_true; EVAL is split into the replicate groups scored against
#: them. Distinct tags make the two streams independent by construction.
TRUTH_STREAM, EVAL_STREAM = 0, 1

#: Canonical mechanism codes: a mechanism's position HERE (not in the
#: active set) feeds the seed tuple, so activating mlp -- or any future
#: mechanism appended at the end -- never moves the other cells' streams.
MECH_ORDER = ('lookup', 'knn1', 'ridge', 'tree', 'mlp')
MECHS = tuple(m.strip() for m in os.environ.get(
    'RAMPART_SIM_MECHS', 'lookup,knn1,ridge,tree').split(','))
_unknown = [m for m in MECHS if m not in MECH_ORDER]
if _unknown:
    raise SystemExit(f"RAMPART_SIM_MECHS: unknown mechanism(s) {_unknown}; "
                     f"known: {','.join(MECH_ORDER)}")

HEADER = (f"{'mech':>7} {'s':>5} {'S_true(MC se)':>18} {'D_true(MC se)':>18} "
          f"{'moved%':>7} {'covS%':>6} {'covD%':>6} "
          f"{'drop':>8} {'itemw':>8} {'sp-aware':>9} {'relFlip%':>9} "
          f"{'cleanL':>8} {'S_pred':>8} {'gap%':>7}")


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
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    registry = {
        'lookup': Lookup,
        'knn1': lambda: KNeighborsRegressor(n_neighbors=1),
        'ridge': lambda: Ridge(alpha=10.0),
        'tree': lambda: DecisionTreeRegressor(random_state=SEED),
        # The non-convex global learner (P-F1.2). lbfgs with a fixed
        # iteration budget: rarely converged, always the same deterministic
        # map of (data, SEED). random_state is the numeric SEED, the same
        # derivation the tree uses -- never hash() of a string, which is
        # per-process salted. ~36 ms per fit at n_train=240, k=36.
        'mlp': lambda: MLPRegressor(hidden_layer_sizes=(16,), solver='lbfgs',
                                    max_iter=200, random_state=SEED),
    }
    return {name: registry[name] for name in MECHS}


def t_half(n, level=0.95):
    from scipy import stats
    return stats.t.ppf(0.5 + level / 2, n - 1)


def draw_stream(stream, cell_seed, make, data, clean_pred, clean_loss, k,
                share, tally):
    """One Monte Carlo stream of NDRAWS fixed-size draws for one cell.

    ``stream`` is the numeric tag (TRUTH_STREAM or EVAL_STREAM) appended to
    ``cell_seed`` so the two streams of a cell never share a draw. Returns
    per-draw S_hat, D_hat, the moved share of uninserted rows, and the count
    of draws where the ratio-of-sums direct reading disagrees in sign with
    the additive D_hat. Updates the identity tally as a side effect.
    """
    X_train, y_train, X_eval, y_eval = data
    S_r = np.empty(NDRAWS)
    D_r = np.empty(NDRAWS)
    moved = np.empty(NDRAWS)
    rel_flip = 0
    for d in range(NDRAWS):
        # Numeric seed parts only: str hash() is per-process salted.
        # round() before int() (done in cell_seed): plain truncation collides
        # saturations (0.29 -> 28) and silently shares streams between cells.
        drw = np.random.default_rng(cell_seed + (stream, d))
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
        # Group-relative reading, the old channels' scale.
        l_rel = 1 - loss[mask].sum() / clean_loss[mask].sum()
        g_rel = 1 - loss[~mask].sum() / clean_loss[~mask].sum()
        if np.sign(l_rel - g_rel) != np.sign(D_r[d]) and D_r[d] != 0:
            rel_flip += 1
    return S_r, D_r, moved, rel_flip


def run_regime(shift, regime_code, tally):
    rng = np.random.default_rng(SEED)
    X_train = rng.standard_normal((N_TRAIN, P))
    y_train = X_train @ BETA + NOISE * rng.standard_normal(N_TRAIN)
    X_eval = rng.standard_normal((N_EVAL, P))
    y_eval = X_eval @ BETA + shift + NOISE * rng.standard_normal(N_EVAL)
    data = (X_train, y_train, X_eval, y_eval)

    print(f"--- regime: {'iid' if shift == 0 else f'drift (+{shift:g} shift)'} ---")
    print(HEADER)
    th = t_half(R)
    for name, make in make_models().items():
        # Seed code from the canonical order, not the active set's index:
        # the four original cells keep their streams when mlp is added.
        code = MECH_ORDER.index(name)
        clean = make().fit(X_train, y_train)
        clean_pred = np.asarray(clean.predict(X_eval), dtype=float)
        clean_loss = (y_eval - clean_pred) ** 2
        # Scale of the mechanism's own clean loss, and the mean clean
        # residual on the (possibly shifted) evaluation rows that feeds the
        # ridge intercept-absorption prediction. Both deterministic.
        clean_mean = clean_loss.mean()
        rbar = (y_eval - clean_pred).mean()

        for s in SATURATIONS:
            k = max(1, int(round(s * N_EVAL)))
            share = k / N_EVAL
            cell_seed = (SEED, regime_code, code, int(round(s * 100)))
            # TRUTH stream: defines S_true/D_true and every DGP expectation
            # the row prints. EVAL stream: independent draws that play the
            # audit's replicate groups and are scored against the TRUTH
            # means -- out of sample, never against a pool containing them.
            S_t, D_t, moved, rel_flip = draw_stream(
                TRUTH_STREAM, cell_seed, make, data, clean_pred, clean_loss,
                k, share, tally)
            S_e, D_e, _, _ = draw_stream(
                EVAL_STREAM, cell_seed, make, data, clean_pred, clean_loss,
                k, share, tally)
            if name == 'lookup' and (np.abs(S_t).max() > 0.0
                                     or np.abs(S_e).max() > 0.0):
                tally['lookup_exact'] = False

            S_true, D_true = S_t.mean(), D_t.mean()
            se_S = S_t.std(ddof=1) / np.sqrt(NDRAWS)
            se_D = D_t.std(ddof=1) / np.sqrt(NDRAWS)
            # Perfect itemwise correction: inserted rows restored to their
            # clean losses; residual miss vs the clean score, gain scale.
            # Equals the mean of (1-share)*S_hat over the TRUTH stream.
            item_bias = (1 - share) * S_true
            groups = NDRAWS // R
            cov_S = cov_D = 0
            group_S = np.empty(groups)
            for g in range(groups):
                sl = slice(g * R, (g + 1) * R)
                m, sd = S_e[sl].mean(), S_e[sl].std(ddof=1)
                group_S[g] = m
                cov_S += abs(m - S_true) <= th * sd / np.sqrt(R)
                m, sd = D_e[sl].mean(), D_e[sl].std(ddof=1)
                cov_D += abs(m - D_true) <= th * sd / np.sqrt(R)
            # Degenerate cell (lookup: every S_hat identical): the coverage
            # comparison is 0 <= 0 and prints 100% vacuously -- say so.
            s_degenerate = np.all(S_e == S_e[0])
            # Cross-audit spillover-aware correction over DISJOINT group
            # pairs: audit 2p's drop bias corrected with S estimated by
            # audit 2p+1. Within-group LOO averages and circular shifts are
            # both identically zero for any data; disjoint pairs are not.
            pairs = group_S[:(groups // 2) * 2].reshape(-1, 2)
            aware = pairs[:, 0] - pairs[:, 1]
            aware_se = (aware.std(ddof=1) / np.sqrt(len(aware))
                        if len(aware) > 1 else float('nan'))

            # Analytic first-order prediction for the global mechanism: the
            # unpenalised intercept absorbs the k inserted rows' +shift, every
            # uninserted row's loss moves from r^2 to (r - delta_c)^2.
            if name == 'ridge':
                delta_c = shift * k / (N_TRAIN + k)
                S_pred = 2 * rbar * delta_c - delta_c ** 2
                pred_txt = f'{S_pred:+.4f}'
                gap_txt = (f'{100 * (S_pred - S_true) / S_true:+.1f}%'
                           if S_pred != 0 and S_true != 0 else 'n/a')
            else:
                pred_txt = gap_txt = '-'

            cov_s_txt = 'exact' if s_degenerate else f'{100 * cov_S / groups:.1f}%'
            print(f'{name:>7} {s:>5.2f} '
                  f'{S_true:>+9.4f}({se_S:.4f}) {D_true:>+9.4f}({se_D:.4f}) '
                  f'{100 * moved.mean():>6.1f}% '
                  f'{cov_s_txt:>6} {100 * cov_D / groups:>5.1f}% '
                  f'{S_true:>+8.4f} {item_bias:>+8.4f} '
                  f'{aware.mean():>+8.4f}({aware_se:.4f}) '
                  f'{100 * rel_flip / NDRAWS:>8.1f}% '
                  f'{clean_mean:>8.4f} {pred_txt:>8} {gap_txt:>7}', flush=True)
    print()


def main():
    print(f'ground-truth simulation: n_train={N_TRAIN}, n_eval={N_EVAL}, '
          f'{NDRAWS} draws per cell, replicate groups of {R}, seed {SEED}')
    print(f"saturations {list(SATURATIONS)}; mechanisms {'/'.join(MECHS)}")
    print(f'truth and evaluation draw streams are independent: seeds '
          f'(seed, regime, mech, round(100*s), {TRUTH_STREAM}, d) define '
          f'S_true/D_true; (seed, regime, mech, round(100*s), {EVAL_STREAM}, d) '
          f'are split into the coverage groups')
    print('extra columns: cleanL = mean clean squared loss of the mechanism on '
          'the evaluation rows (S/cleanL reads S as a share of it); S_pred '
          '(ridge only) = 2*rbar*delta_c - delta_c^2 with delta_c = '
          f'shift*k/({N_TRAIN}+k) and rbar the mean clean residual on the '
          'evaluation rows; gap% = 100*(S_pred - S_true)/S_true, n/a when '
          'S_pred = 0 (iid: shift = 0)\n')

    if 'mlp' in MECHS:
        # The fixed lbfgs budget rarely satisfies gtol on a non-convex
        # surface; the truncation is deliberate and deterministic, and the
        # warning would otherwise fire once per fit (~72k times per full
        # run). stderr only -- the stdout receipt is unaffected either way.
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    tally = {'identity': 0.0, 'lookup_exact': True}
    for regime_code, shift in enumerate((0.0, 1.5)):
        run_regime(shift, regime_code, tally)

    print(f"identity check: max |B - (S + share*D)| over every draw = "
          f"{tally['identity']:.2e}")
    print(f"lookup exactness: S_hat == 0.0 in every single draw: "
          f"{'YES' if tally['lookup_exact'] else 'NO -- BUG'}")
    print('\ncolumns: drop = bias of dropping contaminated rows (= S, Lemma 3);')
    print('itemw = bias of exact restoration (= (1-s)S, the spillover component')
    print('every inserted-rows-only correction carries; net bias adds its mean')
    print('restoration error); sp-aware = mean(SE)')
    print('of the split-half corrected residual over disjoint replicate-group pairs')
    print('(stochastically ~0 when the correction works -- a real test, unlike')
    print('a within-group LOO average, which is 0 for any data by identity);')
    print('relFlip% = draws where the ratio-of-sums direct reading disagrees')
    print('in SIGN with the additive one (concentrates where D is near zero).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

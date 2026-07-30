#!/usr/bin/env python3
"""Exercise the whole in-context path locally, at a cap low enough for a laptop.

Written after seven cloud jobs died one at a time, each costing a submission to
reveal what this file reveals in two minutes. Every one of them was the same
shape: a piece built correctly and not wired into the path that actually runs.
The cheapest way to find that is to run the path.

It touches, in order: the panel loader, the chronological-order contract the cap
depends on, the factory (which must never hand back a bare estimator), a clean
fit, the absorption measurement that appends rows to an already-capped frame, an
injected arm that appends more, and the registered sensitivity rule.

Run before submitting anything to a cluster:

    .venv/bin/python scripts/validation/check_icl_path.py
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'scripts' / 'validation'))
from core.scientific_config import SCIENTIFIC_CONFIG
# Teto baixo: dispara com folga, e mantem o TabPFN barato em CPU.
SCIENTIFIC_CONFIG['in_context_models']['context_cap_rows'] = 300

import numpy as np, pandas as pd
from core.models.icl import MODELS, ContextCapped
from core.models.absorption import absorption_coefficient
from probe_harness import entity_subsample, folds, panel, prepared
from core.models.ladder import entity_effect_frames

df, cols, cfg = panel('inep_censo')
df = entity_subsample(df, 120)
a, b, ts, te = folds(cfg)[3]
X, y, e, yr, Xt, yt, et, yrt = prepared(df, cols, a, b, ts, te)
print(f"fold: treino {len(X)} linhas (teto 300), avaliacao {len(Xt)}")
assert (yr.values == np.sort(yr.values)).all(), "harness devolveu fora de ordem"
print("ordem cronologica: OK")

fit, ev, _m, _g = entity_effect_frames(X, Xt, y, e, et)
model = MODELS['icl_tabpfn']
est = model.make()
assert isinstance(est, ContextCapped), "estimador cru escapou da fabrica"
est.fit(fit, y)
print(f"fit limpo: {est.context}")
clean = np.asarray(est.predict(ev), dtype=float)

# absorcao: anexa sondas AO quadro ja no teto -- foi exatamente o que estourou
r = absorption_coefficient(model.make, fit, y, ev, yt, baseline=clean)
print(f"absorcao: {r['absorption']:+.4f} em {r['probes_used']} sondas")

# braco de vazamento: anexa 30% das linhas de avaliacao
n = int(0.30 * len(Xt))
picked = np.sort(np.random.default_rng(0).choice(len(Xt), size=n, replace=False))
arm = entity_effect_frames(
    pd.concat([X, Xt.iloc[picked]], ignore_index=True), Xt,
    pd.concat([y, yt.iloc[picked]], ignore_index=True),
    pd.concat([e, et.iloc[picked]], ignore_index=True), et)
est2 = model.make()
est2.fit(arm[0], pd.concat([y, yt.iloc[picked]], ignore_index=True))
print(f"braco de vazamento ({len(arm[0])} linhas oferecidas): {est2.context}")
assert est2.context['context_rows'] <= 300

os.environ['RAMPART_CONTEXT_RULE'] = 'random'
est3 = model.make(); est3.fit(fit, y)
print(f"braco de sensibilidade: {est3.context['rule'][:38]}")
print("\nTODO O CAMINHO PASSA.")

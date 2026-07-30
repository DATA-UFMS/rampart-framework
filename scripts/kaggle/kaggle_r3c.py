# r3c on Kaggle: absorption at matched context size.
#
# One notebook cell, pushed and run by scripts/kaggle/push_and_run.sh. The settings
# it needs live in scripts/kaggle/kernel-metadata.json, not in the web UI:
#   enable_gpu / machine_shape  (TabPFN on CPU is roughly ten times slower)
#   enable_internet             (pip, and TabPFN fetches its weights)
#   dataset_sources             (the bundle built by scripts/kaggle/bundle.sh)
#
# What it answers. On the INEP panel the in-context models read a context capped
# at ten thousand rows while the classical models read all thirty-eight thousand,
# so the absorption column is measured at two different sample sizes and the
# comparison across families is confounded. `RAMPART_CAP_ALL=1` caps every model
# to the same ten thousand, which turns a declared caveat into a result.
#
# It does not affect the conclusion that in-context absorption is not invariant in
# n -- that one compares the in-context models against themselves across panels,
# 400 rows to 10,000, and needs no matching.

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Pinned to the lockfile, because these numbers get compared against runs made
# elsewhere and a different scikit-learn would put estimator changes into the
# comparison.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'numpy==2.2.1', 'pandas==2.3.1', 'scikit-learn==1.5.2',
                'scipy==1.14.1', 'pyarrow==18.1.0',
                'tabpfn>=8,<9', 'tabicl>=2,<3'], check=True)

# Kaggle expands an uploaded zip into the dataset, so the bundle arrives either as
# the archive or as the tree already expanded, depending on how it was created.
# Accept both. Either way it lands on the writable disk, because inputs are mounted
# read-only and TabPFN caches its weights next to the code it is asked to run from.
# The two roots are overridable so this resolution can be exercised off Kaggle,
# against a simulated input tree, before a run is spent finding out. It cannot be
# imported from src/ instead: this block is what puts src/ on the path.
# BUNDLE RESOLUTION BEGIN -- sliced by tests/test_kaggle_bundle_resolution.py
INPUTS = Path(os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input'))
WORK = Path(os.environ.get('KAGGLE_WORKING_DIR', '/kaggle/working'))
root = WORK / 'rampart'
if not root.exists():
    archive = next(INPUTS.rglob('rampart-bundle.zip'), None)
    if archive is not None:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(WORK)
    else:
        expanded = next((p for p in INPUTS.rglob('rampart')
                         if (p / 'src').is_dir() and (p / 'panels').is_dir()), None)
        if expanded is None:
            raise SystemExit(
                f'no bundle under {INPUTS}: attach the dataset built by '
                'scripts/kaggle/bundle.sh as an input to this notebook')
        shutil.copytree(expanded, root)

panels = sorted(root.glob('panels/*/collection/*/complete_data.parquet'))
if len(panels) < 2:
    raise SystemExit(f'expected at least two panel parquets, found {len(panels)}: '
                     f'{panels}')
for panel in panels:                     # a truncated upload reads as a clean run
    if panel.stat().st_size < 1024:
        raise SystemExit(f'{panel} is {panel.stat().st_size} bytes, so it is a stub')
print('bundle at', root, '--',
      sum(1 for p in root.rglob('*') if p.is_file()), 'files,',
      len(panels), 'panels')
# BUNDLE RESOLUTION END

os.chdir(root)
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'scripts' / 'validation'))
os.environ['RAMPART_PANEL_DIR'] = str(root / 'panels')
# The arm. Capped is why this cell exists, so it is the default; a preceding cell
# set by push_and_run.sh --cap-all 0 asks for the uncapped one instead, which is
# how the same measurement is obtained at both sample sizes and the difference
# attributed to the cap rather than to the panel.
os.environ.setdefault('RAMPART_CAP_ALL', '1')
ARM = ('every model reads the same 10,000 rows'
       if os.environ['RAMPART_CAP_ALL'] == '1'
       else 'in-context capped at 10,000, classical reading all 38,000')

import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

# The TabPFN v3 weights are gated, and the code reads TABPFN_TOKEN from the
# environment. Kaggle does not put its Secrets there -- they come through an API of
# its own -- so the bridge is here rather than in src/, where it would be a Kaggle
# detail in a module that knows nothing about Kaggle. Absence is reported and not
# fatal: the v2 arm needs no account, and a run that only wanted v2 should not fail
# because the secret is missing.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['TABPFN_TOKEN'] = UserSecretsClient().get_secret('TABPFN_TOKEN')
    print('TABPFN_TOKEN: read from Kaggle Secrets, v3 arm available')
except Exception as exc:                      # not attached, or not on Kaggle
    print(f'TABPFN_TOKEN: unavailable ({type(exc).__name__}); the v3 arm will be '
          f'skipped and the v2 arm runs as usual')

# The two-minute guard before the thirty-minute run. It exercises the panel
# loader, the chronological-order contract the cap depends on, the factory, a
# clean fit, the absorption measurement appending rows to an already-capped
# frame, an injected arm appending more, and the registered sensitivity rule.
# Seven cloud jobs died one at a time before this existed.
print('\n--- guard ---', flush=True)
subprocess.run([sys.executable, 'scripts/validation/check_icl_path.py'], check=True)

# Which panels, in one run. Two World Bank variants measured on the same GPU with
# the same package versions is the only way the calibration difference between them
# means the imputation and not the platform.
PANELS = [p for p in os.environ.get('RAMPART_PROBE_PANELS', 'inep_censo').split(',')
          if p.strip()]
for _switch in ('RAMPART_PROBES', 'RAMPART_PROBE_FRACTION'):
    if os.environ.get(_switch, '').strip():
        print(f'{_switch} = {os.environ[_switch]}')
for panel in PANELS:
    print(f'\n--- r3c on {panel.strip()}: {ARM} ---', flush=True)
    subprocess.run([sys.executable, 'scripts/validation/probe_leakage_channels.py',
                    panel.strip()], check=True)

# r3c on Kaggle: absorption at matched context size.
#
# Paste as one notebook cell. Requires, in the notebook settings:
#   Accelerator = GPU T4 x1      (TabPFN on CPU is roughly ten times slower)
#   Internet    = On             (pip, and TabPFN fetches its weights)
#   Add Input   -> the dataset holding rampart-bundle.zip
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

# Kaggle mounts inputs read-only, so the bundle is unpacked to the writable disk.
bundle = next(Path('/kaggle/input').rglob('rampart-bundle.zip'))
root = Path('/kaggle/working/rampart')
if not root.exists():
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall('/kaggle/working')
print('unpacked to', root)

os.chdir(root)
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'scripts' / 'validation'))
os.environ['RAMPART_PANEL_DIR'] = str(root / 'panels')
os.environ['RAMPART_CAP_ALL'] = '1'          # the arm this run exists for

import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

# The two-minute guard before the thirty-minute run. It exercises the panel
# loader, the chronological-order contract the cap depends on, the factory, a
# clean fit, the absorption measurement appending rows to an already-capped
# frame, an injected arm appending more, and the registered sensitivity rule.
# Seven cloud jobs died one at a time before this existed.
print('\n--- guard ---', flush=True)
subprocess.run([sys.executable, 'scripts/validation/check_icl_path.py'], check=True)

print('\n--- r3c: every model reads the same 10,000 rows ---', flush=True)
subprocess.run([sys.executable, 'scripts/validation/probe_leakage_channels.py',
                'inep_censo'], check=True)

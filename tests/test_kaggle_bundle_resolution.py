"""The Kaggle cell has to find the bundle however Kaggle chose to deliver it.

Uploading `rampart-bundle.zip` as a Dataset expands it, so the archive the first
version of the cell searched for does not exist on the other side and the run died
on `StopIteration`. That is the same class of failure that cost seven Camber
submissions, found one at a time, each after a queue and a boot. So both arrivals
are exercised here, along with the two ways the bundle can arrive damaged.

The block cannot be imported: it is what puts `src/` on the path. It is sliced out
of the cell between sentinels instead, so this test runs the shipped text rather
than a copy of it that can drift.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

CELL = Path(__file__).resolve().parents[1] / 'scripts' / 'kaggle' / 'kaggle_r3c.py'
BEGIN = '# BUNDLE RESOLUTION BEGIN'
END = '# BUNDLE RESOLUTION END'


def resolution_block():
    text = CELL.read_text()
    assert BEGIN in text and END in text, f'sentinels missing from {CELL}'
    # The sentinel line carries a trailing note, so the slice starts after it.
    after_begin = text.split(BEGIN, 1)[1].split('\n', 1)[1]
    block = after_begin.split(END, 1)[0]
    # Guard the slice itself: if the cell is restructured so the block no longer
    # holds both arrivals, this test would silently stop testing them.
    assert 'zipfile.ZipFile' in block, 'archive arrival not in the sliced block'
    assert 'shutil.copytree' in block, 'expanded arrival not in the sliced block'
    return compile(block, str(CELL), 'exec')


def run_resolution(inputs, working):
    """Execute the block against these roots, returning the resolved bundle root."""
    env = {'KAGGLE_INPUT_DIR': str(inputs), 'KAGGLE_WORKING_DIR': str(working)}
    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        scope = {'os': os, 'shutil': shutil, 'sys': sys, 'zipfile': zipfile,
                 'Path': Path, 'subprocess': subprocess}
        exec(resolution_block(), scope)
        return scope['root']
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_tree(root, panel_bytes=4096):
    """A bundle with the shape bundle.sh produces, small enough to build inline."""
    (root / 'src' / 'core').mkdir(parents=True)
    (root / 'src' / 'core' / 'models.py').write_text('# placeholder\n')
    (root / 'scripts' / 'validation').mkdir(parents=True)
    (root / 'scripts' / 'validation' / 'probe.py').write_text('# placeholder\n')
    for panel, collection in (('azure_results_v7_inep', 'inep_raw'),
                              ('azure_results_v7_wb', 'raw_data')):
        target = root / 'panels' / panel / 'collection' / collection
        target.mkdir(parents=True)
        (target / 'complete_data.parquet').write_bytes(b'\x00' * panel_bytes)
    return root


def test_finds_the_tree_kaggle_expanded_from_the_zip(tmp_path):
    """What the uploaded Dataset actually looks like: expanded, no archive left."""
    inputs, working = tmp_path / 'input', tmp_path / 'working'
    make_tree(inputs / 'rampart-bundle' / 'rampart')
    working.mkdir()

    root = run_resolution(inputs, working)

    assert root == working / 'rampart'
    assert (root / 'src').is_dir() and (root / 'panels').is_dir()
    assert len(list(root.glob('panels/*/collection/*/complete_data.parquet'))) == 2


def test_copies_onto_the_writable_disk_because_inputs_are_read_only(tmp_path):
    """Kaggle mounts inputs read-only; the resolved root must not be under them."""
    inputs, working = tmp_path / 'input', tmp_path / 'working'
    make_tree(inputs / 'rampart-bundle' / 'rampart')
    working.mkdir()

    root = run_resolution(inputs, working)

    assert inputs not in root.parents and root != inputs


def test_finds_the_archive_when_it_survives_upload(tmp_path):
    """The other arrival, kept working so a zip-shaped Dataset is not a dead end."""
    inputs, working = tmp_path / 'input', tmp_path / 'working'
    staged = tmp_path / 'staged'
    make_tree(staged / 'rampart')
    (inputs / 'rampart-bundle').mkdir(parents=True)
    with zipfile.ZipFile(inputs / 'rampart-bundle' / 'rampart-bundle.zip', 'w') as z:
        for path in sorted(p for p in (staged / 'rampart').rglob('*') if p.is_file()):
            z.write(path, path.relative_to(staged.parent).relative_to(staged.name))
    working.mkdir()

    root = run_resolution(inputs, working)

    assert len(list(root.glob('panels/*/collection/*/complete_data.parquet'))) == 2


def test_refuses_when_no_dataset_is_attached(tmp_path):
    """Forgetting the input should name the fix, not raise StopIteration."""
    inputs, working = tmp_path / 'input', tmp_path / 'working'
    inputs.mkdir()
    working.mkdir()

    with pytest.raises(SystemExit, match='no bundle under'):
        run_resolution(inputs, working)


def test_refuses_a_truncated_panel(tmp_path):
    """An eight-byte fixture in place of a panel would otherwise read as a run."""
    inputs, working = tmp_path / 'input', tmp_path / 'working'
    make_tree(inputs / 'rampart-bundle' / 'rampart', panel_bytes=8)
    working.mkdir()

    with pytest.raises(SystemExit, match='so it is a stub'):
        run_resolution(inputs, working)


def test_refuses_when_a_panel_is_missing(tmp_path):
    """Both panels travel together; one of them alone is a partial upload."""
    inputs, working = tmp_path / 'input', tmp_path / 'working'
    tree = make_tree(inputs / 'rampart-bundle' / 'rampart')
    shutil.rmtree(tree / 'panels' / 'azure_results_v7_wb')
    working.mkdir()

    with pytest.raises(SystemExit, match='expected two panel parquets'):
        run_resolution(inputs, working)

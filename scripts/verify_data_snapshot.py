#!/usr/bin/env python3
"""Verifies and installs an immutable snapshot of the input data.

Why it exists: collection depends on an external API whose values change with
revisions of the source. A run that collects again does not reproduce the
previous one, and the difference is indistinguishable from a code change. A
hashed snapshot separates the two things -- and allows running without network.

The hash is of the content, not of the name: renaming a file must not pass
verification, and one file more or one file less is divergence too.

Usage:
    # record the manifest of an already collected data directory
    python scripts/verify_data_snapshot.py --snapshot path/ --dataset worldbank --record

    # verify (and optionally install into outputs/<dataset>/collection)
    python scripts/verify_data_snapshot.py --snapshot path/ --dataset worldbank [--install]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))

from core.config import get_absolute_output_path  # noqa: E402

MANIFEST_NAME = 'snapshot_manifest.json'
CHUNK = 1 << 20


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as handler:
        for block in iter(lambda: handler.read(CHUNK), b''):
            sha.update(block)
    return sha.hexdigest()


def _inventory(root: Path) -> dict:
    """Hash of each file, by relative path, in deterministic order."""
    files = {}
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        files[str(path.relative_to(root))] = _digest(path)
    return files


def record(snapshot: Path, dataset: str) -> Path:
    files = _inventory(snapshot)
    manifest = {
        'dataset': dataset,
        'file_count': len(files),
        # A digest over the whole inventory: catches a file added, removed or
        # renamed, which per-file hashes alone would not catch.
        'inventory_sha256': hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest(),
        'files': files,
    }
    target = snapshot / MANIFEST_NAME
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"  manifest recorded: {target}")
    print(f"  {len(files)} files, inventory "
          f"{manifest['inventory_sha256'][:16]}...")
    return target


def verify(snapshot: Path, dataset: str) -> dict:
    manifest_path = snapshot / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Snapshot without a manifest: {manifest_path}. Record one with "
            f"--record before using it as verified input."
        )
    manifest = json.loads(manifest_path.read_text())

    if manifest['dataset'] != dataset:
        raise ValueError(
            f"Snapshot is of dataset {manifest['dataset']!r}, and "
            f"{dataset!r} was requested: the panels are not interchangeable."
        )

    observed = _inventory(snapshot)
    expected = manifest['files']

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    changed = sorted(name for name in set(expected) & set(observed)
                     if expected[name] != observed[name])
    if missing or extra or changed:
        raise ValueError(
            f"Snapshot diverges from the manifest — missing={missing[:5]} "
            f"extra={extra[:5]} changed={changed[:5]} "
            f"(total {len(missing)}/{len(extra)}/{len(changed)})"
        )

    digest = hashlib.sha256(
        json.dumps(observed, sort_keys=True).encode()).hexdigest()
    if digest != manifest['inventory_sha256']:
        raise ValueError(
            f"Inventory does not match: {digest} != {manifest['inventory_sha256']}"
        )

    print(f"  verified: {len(observed)} files, inventory "
          f"{digest[:16]}...")
    return manifest


def install(snapshot: Path, dataset: str) -> Path:
    os.environ['DATASET_NAME'] = dataset
    destination = Path(get_absolute_output_path('collection'))
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. Remove it before installing a "
            f"snapshot, so there is no doubt about the origin of the data used."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The manifest goes along: it is what tells the collector that this data
    # comes from a verified snapshot and must not be re-downloaded due to age.
    shutil.copytree(snapshot, destination)
    print(f"  installed at {destination}")
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--dataset', required=True,
                        choices=['worldbank', 'inep_censo'])
    parser.add_argument('--record', action='store_true',
                        help='Records the manifest instead of verifying')
    parser.add_argument('--install', action='store_true',
                        help='Copies to outputs/<dataset>/collection after verifying')
    args = parser.parse_args(argv)

    if not args.snapshot.is_dir():
        parser.error(f"{args.snapshot} is not a directory")

    if args.record:
        record(args.snapshot, args.dataset)
        return 0

    verify(args.snapshot, args.dataset)
    if args.install:
        install(args.snapshot, args.dataset)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

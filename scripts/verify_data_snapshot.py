#!/usr/bin/env python3
"""Verifica e instala um snapshot imutável dos dados de entrada.

Por que existe: a coleta depende de uma API externa cujos valores mudam com
revisões da fonte. Uma execução que colete de novo não reproduz a anterior, e a
diferença é indistinguível de uma mudança de código. Um snapshot com hash separa
as duas coisas -- e permite rodar sem rede.

O hash é do conteúdo, não do nome: renomear um arquivo não deve passar por
verificação, e um arquivo a mais ou a menos também é divergência.

Uso:
    # gravar o manifesto de um diretório de dados já coletado
    python scripts/verify_data_snapshot.py --snapshot caminho/ --dataset worldbank --record

    # verificar (e opcionalmente instalar em outputs/<dataset>/collection)
    python scripts/verify_data_snapshot.py --snapshot caminho/ --dataset worldbank [--install]
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
    """Hash de cada arquivo, por caminho relativo, em ordem determinística."""
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
        # Um digest sobre o inventário inteiro: pega arquivo a mais, a menos ou
        # renomeado, que hashes por arquivo isolados não pegariam.
        'inventory_sha256': hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest(),
        'files': files,
    }
    target = snapshot / MANIFEST_NAME
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"  manifesto gravado: {target}")
    print(f"  {len(files)} arquivos, inventário "
          f"{manifest['inventory_sha256'][:16]}...")
    return target


def verify(snapshot: Path, dataset: str) -> dict:
    manifest_path = snapshot / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Snapshot sem manifesto: {manifest_path}. Grave um com --record "
            f"antes de usá-lo como entrada verificada."
        )
    manifest = json.loads(manifest_path.read_text())

    if manifest['dataset'] != dataset:
        raise ValueError(
            f"Snapshot é do dataset {manifest['dataset']!r}, e foi pedido "
            f"{dataset!r}: os painéis não são intercambiáveis."
        )

    observed = _inventory(snapshot)
    expected = manifest['files']

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    changed = sorted(name for name in set(expected) & set(observed)
                     if expected[name] != observed[name])
    if missing or extra or changed:
        raise ValueError(
            f"Snapshot divergente do manifesto — ausentes={missing[:5]} "
            f"extras={extra[:5]} alterados={changed[:5]} "
            f"(total {len(missing)}/{len(extra)}/{len(changed)})"
        )

    digest = hashlib.sha256(
        json.dumps(observed, sort_keys=True).encode()).hexdigest()
    if digest != manifest['inventory_sha256']:
        raise ValueError(
            f"Inventário não confere: {digest} != {manifest['inventory_sha256']}"
        )

    print(f"  verificado: {len(observed)} arquivos, inventário "
          f"{digest[:16]}...")
    return manifest


def install(snapshot: Path, dataset: str) -> Path:
    os.environ['DATASET_NAME'] = dataset
    destination = Path(get_absolute_output_path('collection'))
    if destination.exists():
        raise FileExistsError(
            f"{destination} já existe. Remova-o antes de instalar um snapshot, "
            f"para que não haja dúvida sobre a origem dos dados usados."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # O manifesto vai junto: é ele que diz ao coletor que estes dados vêm de um
    # snapshot verificado e não devem ser re-baixados por idade.
    shutil.copytree(snapshot, destination)
    print(f"  instalado em {destination}")
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--dataset', required=True,
                        choices=['worldbank', 'inep_censo'])
    parser.add_argument('--record', action='store_true',
                        help='Grava o manifesto em vez de verificar')
    parser.add_argument('--install', action='store_true',
                        help='Copia para outputs/<dataset>/collection após verificar')
    args = parser.parse_args(argv)

    if not args.snapshot.is_dir():
        parser.error(f"{args.snapshot} não é um diretório")

    if args.record:
        record(args.snapshot, args.dataset)
        return 0

    verify(args.snapshot, args.dataset)
    if args.install:
        install(args.snapshot, args.dataset)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

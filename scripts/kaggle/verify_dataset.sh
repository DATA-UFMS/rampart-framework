#!/usr/bin/env bash
# Does the uploaded Dataset match the bundle on disk? Refuse to guess.
#
# Two runs were spent finding out that it did not. The bundle had been rebuilt before
# a later edit, uploaded, and then verified by checking ONE file -- which happened to
# be the one that was already current. The file that carried the change was 500 bytes
# short on the other side, the environment switch it read did not exist, and both arms
# silently ran the default. The verification was real and pointed at the wrong artifact.
#
# So this compares every file, by size, over BOTH hops of the chain:
#
#   working tree  ->  rampart-bundle.zip  ->  Kaggle Dataset
#
# The first hop is the one that broke, and checking only the second says "matches"
# while both sides are equally stale -- which is what the first version of this script
# did, and it passed on exactly the state that had already cost the two runs.
#
#   scripts/kaggle/verify_dataset.sh            # after datasets version, before pushing a kernel
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../.." && pwd)"
ZIP="${ZIP:-$(cd "$RAIZ/.." && pwd)/kaggle-rampart/rampart-bundle.zip}"
SLUG="${SLUG:-eosxavier2/rampart-bundle}"
: "${KAGGLE_API_TOKEN:?exporte KAGGLE_API_TOKEN=KGAT_... antes de rodar}"
[[ -s "$ZIP" ]] || { echo "bundle ausente: $ZIP"; exit 1; }

"$RAIZ/.venv/bin/python" - "$ZIP" "$SLUG" "$RAIZ" <<'PY'
import sys, zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

caminho, slug, raiz = sys.argv[1], sys.argv[2], Path(sys.argv[3])

# Hop one: does the zip carry what the working tree has? This is the hop that broke.
with zipfile.ZipFile(caminho) as z:
    empacotado = {i.filename.split('rampart/', 1)[-1]: i.file_size
                  for i in z.infolist() if not i.is_dir()}
desatualizado = []
for sub in ('src', 'scripts'):
    for p in sorted((raiz / sub).rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        rel = str(p.relative_to(raiz))
        if rel not in empacotado:
            desatualizado.append(f"  AUSENTE no bundle       {rel}")
        elif empacotado[rel] != p.stat().st_size:
            desatualizado.append(f"  TAMANHO DIFERE          {rel}: arvore "
                                 f"{p.stat().st_size}, bundle {empacotado[rel]}")
if desatualizado:
    print("arvore de trabalho contra o bundle:")
    print('\n'.join(desatualizado))
    print("\nO bundle esta' defasado em relacao ao repositorio. Rode bundle.sh.")
    sys.exit(1)
print(f"arvore de trabalho contra o bundle: {len(empacotado)} arquivos, iguais")

with zipfile.ZipFile(caminho) as z:
    # The Dataset carries an extra leading component, so compare on the suffix that
    # starts at the bundle root -- that is what the notebook resolves against anyway.
    local = {i.filename.split('rampart/', 1)[-1]: i.file_size
             for i in z.infolist() if not i.is_dir()}

api = KaggleApi(); api.authenticate()
remoto, token = {}, None
while True:
    r = api.dataset_list_files(slug, page_token=token, page_size=300)
    for f in r.files:
        if '/rampart/' in f.name or f.name.startswith('rampart/'):
            remoto[f.name.split('rampart/', 1)[-1]] = f.total_bytes
    token = getattr(r, 'next_page_token', None)
    if not token:
        break

faltando = sorted(set(local) - set(remoto))
sobrando = sorted(set(remoto) - set(local))
difere = sorted(n for n in set(local) & set(remoto) if local[n] != remoto[n])

print(f"local {len(local)} arquivos, dataset {len(remoto)}")
for nome in faltando:
    print(f"  AUSENTE no dataset      {nome}")
for nome in sobrando:
    print(f"  so no dataset           {nome}")
for nome in difere:
    print(f"  TAMANHO DIFERE          {nome}: local {local[nome]}, dataset {remoto[nome]}")

if faltando or difere:
    print("\nO dataset esta' defasado. Rode bundle.sh e datasets version antes do kernel.")
    sys.exit(1)
print("\nO dataset corresponde ao bundle." + (" (arquivos extras sao inofensivos)"
                                              if sobrando else ""))
PY

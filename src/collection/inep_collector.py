#!/usr/bin/env python3
"""
INEP Educational Indicators collector — School Performance Rates.

Downloads XLSX spreadsheets from the INEP portal with pass, fail
and dropout rates by municipality/year, and produces a Parquet in the framework
schema (entity_id/year/features) for direct consumption by the processors.

Source: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais
Citation: BRASIL. Instituto Nacional de Estudos e Pesquisas Educacionais
          Anísio Teixeira (Inep). Indicadores Educacionais. Brasília: Inep.

Usage:
    python src/collection/inep_collector.py [--years 2019 2020] [--output-dir DIR]
"""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "collection", "inep_raw")

# ============================================================================
# Download URLs — vary by year (mapped manually from the INEP portal)
# ============================================================================
INEP_URLS = {
    2007: "https://download.inep.gov.br/informacoes_estatisticas/2011/indicadores_educacionais/taxa_rendimento/2007/tx_rendimento_municipios_2007.zip",
    2008: "https://download.inep.gov.br/informacoes_estatisticas/2011/indicadores_educacionais/taxa_rendimento/2008/tx_rendimento_municipios_2008.zip",
    2009: "https://download.inep.gov.br/informacoes_estatisticas/2011/indicadores_educacionais/taxa_rendimento/2009/tx_rendimento_municipios_2009.zip",
    2010: "https://download.inep.gov.br/informacoes_estatisticas/2011/indicadores_educacionais/taxa_rendimento/2010/tx_rendimento_municipios_2010.zip",
    2011: "https://download.inep.gov.br/informacoes_estatisticas/2011/indicadores_educacionais/taxa_rendimento/2011/tx_rendimento_municipios_2011_2.zip",
    2012: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2012/taxas_rendimento/tx_rendimento_municipios_2012.zip",
    2013: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2013/taxa_rendimento/tx_rendimento_municipios_2013.zip",
    2014: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2014/taxa_rendimento/tx_rendimento_municipios_2014.zip",
    2015: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2015/taxa_rendimento/tx_rendimento_municipios_2015.zip",
    2016: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2016/TAXA_REND_2016_MUNICIPIOS.zip",
    2017: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2017/TAXA_REND_2017_MUNICIPIOS.zip",
    2018: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2018/TX_REND_MUNICIPIOS_2018.zip",
    2019: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2019/tx_rend_municipios_2019.zip",
    2020: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2020/tx_rend_municipios_2020.zip",
    2021: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2021/tx_rend_municipios_2021.zip",
    2022: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2022/tx_rend_municipios_2022.zip",
    2023: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2023/tx_rend_municipios_2023.zip",
    2024: "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2024/tx_rend_municipios_2024.zip",
}

# Positional column names (61 columns in the post-2013 XLSX)
COL_NAMES = [
    'ano', 'regiao', 'uf', 'cod_municipio', 'nome_municipio',
    'localizacao', 'dependencia',
    # Pass rate (18 cols)
    'aprov_ef', 'aprov_ef_ai', 'aprov_ef_af',
    'aprov_ef_1', 'aprov_ef_2', 'aprov_ef_3', 'aprov_ef_4', 'aprov_ef_5',
    'aprov_ef_6', 'aprov_ef_7', 'aprov_ef_8', 'aprov_ef_9',
    'aprov_em', 'aprov_em_1', 'aprov_em_2', 'aprov_em_3', 'aprov_em_4', 'aprov_em_ns',
    # Fail rate (18 cols)
    'reprov_ef', 'reprov_ef_ai', 'reprov_ef_af',
    'reprov_ef_1', 'reprov_ef_2', 'reprov_ef_3', 'reprov_ef_4', 'reprov_ef_5',
    'reprov_ef_6', 'reprov_ef_7', 'reprov_ef_8', 'reprov_ef_9',
    'reprov_em', 'reprov_em_1', 'reprov_em_2', 'reprov_em_3', 'reprov_em_4', 'reprov_em_ns',
    # Dropout rate (18 cols)
    'abandono_ef', 'abandono_ef_ai', 'abandono_ef_af',
    'abandono_ef_1', 'abandono_ef_2', 'abandono_ef_3', 'abandono_ef_4', 'abandono_ef_5',
    'abandono_ef_6', 'abandono_ef_7', 'abandono_ef_8', 'abandono_ef_9',
    'abandono_em', 'abandono_em_1', 'abandono_em_2', 'abandono_em_3', 'abandono_em_4', 'abandono_em_ns',
]

# Candidate features, excluding metadata and the target.
#
# The target is the upper-secondary (Ensino Medio) dropout rate, and INEP's
# rendimento rates partition each level exactly: aprovacao + reprovacao +
# abandono = 100. Every upper-secondary rate is therefore an algebraic
# component of the target -- aprov_em and reprov_em reconstruct it outright,
# and the per-grade rates reconstruct it by weighted combination. Measured over
# the full panel, regressing the target on the upper-secondary rates yields
# R2 = 0.97; on the lower-secondary rates alone, R2 = 0.33.
#
# Only lower-secondary (Ensino Fundamental) rates are kept: they describe a
# different stage of schooling and carry predictive signal rather than an
# identity.
FEATURE_COLS = [
    'aprov_ef', 'aprov_ef_ai', 'aprov_ef_af',
    'reprov_ef', 'reprov_ef_ai', 'reprov_ef_af',
    'abandono_ef', 'abandono_ef_ai', 'abandono_ef_af',
]


# ============================================================================
# Download and parsing
# ============================================================================
def download_year(year: int, cache_dir: str) -> str:
    """Downloads the INEP ZIP. Uses the cache if it already exists."""
    if year not in INEP_URLS:
        raise ValueError(f"URL not mapped for year {year}")

    url = INEP_URLS[year]
    zip_path = os.path.join(cache_dir, f"tx_rend_mun_{year}.zip")

    if os.path.exists(zip_path):
        print(f"   Cache: {zip_path}")
        return zip_path

    os.makedirs(cache_dir, exist_ok=True)
    print(f"   Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=300, verify=False)
    r.raise_for_status()

    with open(zip_path + ".tmp", "wb") as f:
        for chunk in r.iter_content(chunk_size=131072):
            f.write(chunk)
    os.rename(zip_path + ".tmp", zip_path)
    return zip_path


def parse_year(zip_path: str, year: int) -> pd.DataFrame:
    """
    Extracts and parses the XLSX of performance rates for one year.

    Returns 1 row per municipality (filter: localizacao=Total, dependencia=Total).
    """
    # Extract the spreadsheet from the ZIP (XLSX for 2012+, XLS for 2007-2011)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        excel_files = [n for n in zf.namelist()
                       if (n.lower().endswith('.xlsx') or n.lower().endswith('.xls'))
                       and not n.startswith('__') and not n.startswith('~')]
        if not excel_files:
            raise FileNotFoundError(f"No Excel spreadsheet found in the ZIP for {year}")
        excel_name = excel_files[0]
        zf.extract(excel_name, os.path.dirname(zip_path))
        xlsx_path = os.path.join(os.path.dirname(zip_path), excel_name)

    # Read the XLSX — skip the multi-line header (varies across years: 8-10 lines)
    for skiprows in [9, 8, 10, 7]:
        try:
            ncols = len(COL_NAMES)
            df = pd.read_excel(xlsx_path, skiprows=skiprows, header=None,
                               usecols=range(min(ncols, 61)))
            # Check whether the first column looks like a year
            first_val = df.iloc[0, 0]
            if pd.notna(first_val) and str(first_val).strip().isdigit():
                val = int(str(first_val).strip())
                if 2000 <= val <= 2030:
                    break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not detect the header for {year}")

    # Name the columns (adjust if the file has fewer)
    actual_cols = min(len(df.columns), len(COL_NAMES))
    df.columns = COL_NAMES[:actual_cols]

    # Filter: Total/Total (1 row per municipality)
    total = df[(df['localizacao'] == 'Total') & (df['dependencia'] == 'Total')].copy()

    # Convert numeric columns (-- -> NaN)
    num_cols = [c for c in total.columns if c not in
                ['regiao', 'uf', 'nome_municipio', 'localizacao', 'dependencia']]
    for col in num_cols:
        total[col] = pd.to_numeric(total[col], errors='coerce')

    total['ano'] = year
    return total


def adapt_to_framework_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps INEP columns -> framework schema (entity_id/year/etc.).

    The framework computes dropout_rate = 100 - target_source_rate.
    Therefore: target_source_rate = 100 - abandono_em.
    """
    adapted = df.copy()

    # Rename the entity
    adapted = adapted.rename(columns={
        'cod_municipio': 'entity_id',
        'nome_municipio': 'entity_name',
    })

    # Create entity_stratum from the UF (state)
    adapted['entity_stratum'] = adapted['uf']

    # Target: invert abandono EM -> completion rate
    adapted['target_source_rate'] = 100 - adapted['abandono_em']

    # Select the final columns
    keep_cols = ['entity_id', 'entity_name', 'entity_stratum', 'year']
    keep_cols += [c for c in FEATURE_COLS if c in adapted.columns]
    keep_cols += ['target_source_rate']

    # Remove duplicates
    keep_cols = list(dict.fromkeys(keep_cols))

    adapted = adapted.rename(columns={'ano': 'year'})
    result = adapted[[c for c in keep_cols if c in adapted.columns]].copy()

    # entity_id as a string
    result['entity_id'] = result['entity_id'].astype(float).astype(int).astype(str)
    result['year'] = result['year'].astype(int)

    return result


# ============================================================================
# Main pipeline
# ============================================================================
def collect_inep_data(output_dir: str, years: Optional[List[int]] = None,
                      cache_dir: Optional[str] = None) -> Dict:
    """
    Complete collection pipeline for INEP Educational Indicators.

    Downloads, parses and adapts to the framework schema.
    """
    if years is None:
        years = sorted(INEP_URLS.keys())

    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "..", "inep_cache")

    os.makedirs(output_dir, exist_ok=True)

    print(f"INEP collection - Performance Rates, {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"Output: {output_dir}")

    all_years = []
    metadata = {
        'dataset': 'inep_censo',
        'source': 'INEP Indicadores Educacionais - Taxas de Rendimento',
        'collection_start': datetime.now().isoformat(),
        'years': {},
    }

    for year in years:
        print(f"\n--- {year} ---")
        t0 = time.time()
        try:
            zip_path = download_year(year, cache_dir)
            df = parse_year(zip_path, year)
            elapsed = time.time() - t0
            print(f"   {len(df)} municipalities, mean abandono EM: "
                  f"{df['abandono_em'].mean():.1f}% ({elapsed:.1f}s)")
            all_years.append(df)
            metadata['years'][str(year)] = {
                'status': 'ok', 'municipalities': len(df),
                'abandono_em_mean': round(float(df['abandono_em'].mean()), 2),
                'elapsed_s': round(elapsed, 1),
            }
        except Exception as e:
            print(f"   ERROR: {e}")
            metadata['years'][str(year)] = {'status': 'error', 'error': str(e)}

    if not all_years:
        print("\nNo data collected!")
        return metadata

    # Concatenate and adapt
    complete = pd.concat(all_years, ignore_index=True)
    adapted = adapt_to_framework_schema(complete)

    # Filter out municipalities without EM data (NaN in the target)
    before = len(adapted)
    adapted = adapted.dropna(subset=['target_source_rate'])
    dropped = before - len(adapted)
    if dropped > 0:
        print(f"   Filtered out {dropped} records without EM data (NaN in the target)")

    # NaN in numeric features preserved on purpose:
    # municipalities without a 3rd year of EM have aprov_em_3=NaN (a legitimate
    # absence, not a rate of zero). Imputation by the training median happens in
    # the models (P5: preprocessing scope restricted to the training set).

    # Save
    parquet_path = os.path.join(output_dir, "complete_data.parquet")
    adapted.to_parquet(parquet_path, index=False)

    with open(parquet_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    n_mun = adapted['entity_id'].nunique()
    n_years = adapted['year'].nunique()

    metadata.update({
        'collection_end': datetime.now().isoformat(),
        'total_rows': len(adapted),
        'total_municipalities': n_mun,
        'total_years': n_years,
        'columns': list(adapted.columns),
        'sha256': sha256,
    })

    meta_path = os.path.join(output_dir, "scientific_collection_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCollection complete: {len(adapted):,} obs "
          f"({n_mun} municipalities x {n_years} years)")
    print(f"Parquet: {parquet_path}")
    print(f"SHA-256: {sha256}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description='INEP Educational Indicators collection')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--years', nargs='+', type=int, default=None)
    parser.add_argument('--cache-dir', default=None)
    args = parser.parse_args()

    collect_inep_data(args.output_dir, args.years, args.cache_dir)


if __name__ == '__main__':
    main()

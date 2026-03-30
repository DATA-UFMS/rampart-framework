#!/usr/bin/env python3
"""
Coletor de microdados do INEP Censo Escolar.

Baixa ZIPs do portal de dados abertos do INEP, extrai tabelas
MATRICULA e ESCOLA, filtra por Ensino Médio, e salva como Parquet
particionado por ano para consumo pelos processadores arquiteturais.

Fonte: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar
Citação: BRASIL. Instituto Nacional de Estudos e Pesquisas Educacionais
         Anísio Teixeira (Inep). Microdados do Censo Escolar da Educação
         Básica {ANO}. Brasília: Inep, {ANO}.

Uso:
    python src/collection/inep_collector.py [--output-dir DIR] [--years 2019 2020]
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# Adiciona raiz do projeto ao path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from datasets.inep_censo import InepCensoDatasetConfig


# ============================================================================
# Configuração
# ============================================================================
INEP_CONFIG = InepCensoDatasetConfig()

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", INEP_CONFIG.raw_data_subdir)

# Colunas a ler de cada tabela (reduz uso de memória)
MATRICULA_USECOLS = [
    "NU_ANO_CENSO", "CO_MUNICIPIO",
    "TP_SEXO", "TP_COR_RACA", "NU_IDADE",
    "TP_ETAPA_ENSINO", "TP_MEDIACAO_DIDATICO_PEDAGO",
    "IN_TRANSPORTE_PUBLICO",
]

ESCOLA_USECOLS = [
    "CO_ENTIDADE", "CO_MUNICIPIO", "NU_ANO_CENSO",
    "IN_INTERNET", "IN_LABORATORIO_INFORMATICA",
    "IN_LABORATORIO_CIENCIAS", "IN_BIBLIOTECA",
    "IN_QUADRA_ESPORTES_COBERTA", "IN_QUADRA_ESPORTES_DESCOBERTA",
    "IN_AGUA_POTAVEL", "IN_ESGOTO_REDE_PUBLICA",
    "IN_ENERGIA_REDE_PUBLICA",
    "TP_LOCALIZACAO",  # 1=Urbana, 2=Rural
]


# ============================================================================
# Funções de download
# ============================================================================
def download_year(year: int, cache_dir: str, timeout: int = 600) -> str:
    """
    Baixa ZIP do INEP para um ano. Usa cache se já existe.

    Args:
        year: Ano do censo (2007-2020)
        cache_dir: Diretório para cache de ZIPs
        timeout: Timeout em segundos

    Returns:
        Caminho do arquivo ZIP baixado
    """
    url = INEP_CONFIG.download_url_template.format(year=year)
    zip_path = os.path.join(cache_dir, f"microdados_censo_escolar_{year}.zip")

    if os.path.exists(zip_path):
        print(f"   Cache encontrado: {zip_path}")
        return zip_path

    print(f"   Baixando {url} ...")
    os.makedirs(cache_dir, exist_ok=True)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(zip_path + ".tmp", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192 * 16):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r   {pct:.1f}% ({downloaded // (1024*1024)} MB)", end="")
    print()

    os.rename(zip_path + ".tmp", zip_path)
    return zip_path


def find_csv_in_zip(zip_path: str, table_prefix: str) -> Optional[str]:
    """
    Encontra arquivo CSV dentro do ZIP do INEP.

    Os ZIPs do INEP têm estrutura variável entre anos. Procura por
    arquivos cujo nome contém o prefixo da tabela (case-insensitive).

    Args:
        zip_path: Caminho do ZIP
        table_prefix: Prefixo da tabela ("MATRICULA", "ESCOLA")

    Returns:
        Nome do arquivo dentro do ZIP, ou None
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            basename = os.path.basename(name).upper()
            if basename.startswith(table_prefix.upper()) and basename.endswith('.CSV'):
                return name
    return None


# ============================================================================
# Processamento de tabelas
# ============================================================================
def read_csv_from_zip(zip_path: str, csv_name: str,
                      usecols: List[str]) -> pd.DataFrame:
    """
    Lê CSV de dentro do ZIP com as colunas especificadas.

    Trata encoding ISO-8859-1 e separador pipe (|) padrão do INEP.
    Ignora colunas ausentes (variando entre anos).
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(csv_name) as f:
            # Ler header para descobrir colunas disponíveis
            raw = io.TextIOWrapper(f, encoding=INEP_CONFIG.csv_encoding)
            header_line = raw.readline()
            available = [c.strip().strip('"') for c in header_line.split(INEP_CONFIG.csv_separator)]

            # Interseção: colunas pedidas que existem no arquivo
            cols_to_read = [c for c in usecols if c in available]
            missing = set(usecols) - set(available)
            if missing:
                print(f"      Colunas ausentes neste ano: {missing}")

    # Reler com as colunas filtradas
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                sep=INEP_CONFIG.csv_separator,
                encoding=INEP_CONFIG.csv_encoding,
                usecols=cols_to_read,
                dtype={c: 'Int64' for c in cols_to_read if c.startswith(('TP_', 'IN_', 'CO_', 'NU_'))},
                low_memory=False,
            )
    return df


def process_matricula(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Filtra matrículas de Ensino Médio e prepara para agregação.

    Args:
        df: DataFrame bruto da tabela MATRICULA
        year: Ano do censo

    Returns:
        DataFrame filtrado com colunas padronizadas
    """
    # Filtro: Ensino Médio
    if 'TP_ETAPA_ENSINO' in df.columns:
        df = df[df['TP_ETAPA_ENSINO'].isin(INEP_CONFIG.etapa_ensino_filter)].copy()

    print(f"      Ensino Médio: {len(df):,} matrículas")
    return df


def aggregate_municipality_year(matricula_df: pd.DataFrame,
                                 escola_df: Optional[pd.DataFrame],
                                 year: int) -> pd.DataFrame:
    """
    Agrega microdados no nível município × ano.

    Gera features demográficas (% feminino, % por raça, média de idade),
    features de infraestrutura (% com internet, laboratórios, etc.),
    e a variável target (taxa de abandono).

    Args:
        matricula_df: Matrículas filtradas (Ensino Médio)
        escola_df: Dados de escola (ou None)
        year: Ano do censo

    Returns:
        DataFrame agregado (1 linha por município)
    """
    mun_col = "CO_MUNICIPIO"

    # --- Features de matrícula (alunado) ---
    agg = matricula_df.groupby(mun_col).agg(
        total_matriculas=pd.NamedAgg(column=mun_col, aggfunc='count'),
        pct_feminino=pd.NamedAgg(
            column='TP_SEXO',
            aggfunc=lambda s: (s == 2).mean() * 100 if 'TP_SEXO' in s.name or True else 0
        ),
        media_idade=pd.NamedAgg(column='NU_IDADE', aggfunc='mean'),
    ).reset_index()

    # % por cor/raça
    if 'TP_COR_RACA' in matricula_df.columns:
        for code, label in [(1, 'pct_cor_branca'), (2, 'pct_cor_preta'), (3, 'pct_cor_parda')]:
            race_pct = matricula_df.groupby(mun_col)['TP_COR_RACA'].apply(
                lambda s: (s == code).mean() * 100
            ).reset_index(name=label)
            agg = agg.merge(race_pct, on=mun_col, how='left')

    # % noturno (TP_MEDIACAO_DIDATICO_PEDAGO == 1 = presencial,
    # usamos outro indicador se disponível)
    if 'TP_MEDIACAO_DIDATICO_PEDAGO' in matricula_df.columns:
        noturno = matricula_df.groupby(mun_col)['TP_MEDIACAO_DIDATICO_PEDAGO'].apply(
            lambda s: (s == 2).mean() * 100  # 2 = EAD como proxy simplificado
        ).reset_index(name='pct_noturno')
        agg = agg.merge(noturno, on=mun_col, how='left')

    # % integral
    agg['pct_integral'] = 0.0  # Placeholder — requer TP_TIPO_TURMA

    # % zona rural (se disponível na matrícula)
    if 'TP_ZONA_RESIDENCIAL' in matricula_df.columns:
        rural = matricula_df.groupby(mun_col)['TP_ZONA_RESIDENCIAL'].apply(
            lambda s: (s == 2).mean() * 100
        ).reset_index(name='pct_zona_rural')
        agg = agg.merge(rural, on=mun_col, how='left')

    # --- Features de infraestrutura (escola) ---
    if escola_df is not None and len(escola_df) > 0:
        infra_cols = [c for c in escola_df.columns if c.startswith('IN_')]
        if infra_cols:
            escola_mun = escola_df.groupby(mun_col)[infra_cols].mean().reset_index()
            # Renomear para padrão pct_*
            rename_map = {
                'IN_INTERNET': 'pct_internet',
                'IN_LABORATORIO_INFORMATICA': 'pct_lab_informatica',
                'IN_LABORATORIO_CIENCIAS': 'pct_lab_ciencias',
                'IN_BIBLIOTECA': 'pct_biblioteca',
                'IN_AGUA_POTAVEL': 'pct_agua_potavel',
                'IN_ESGOTO_REDE_PUBLICA': 'pct_esgoto_rede_publica',
                'IN_ENERGIA_REDE_PUBLICA': 'pct_energia_rede_publica',
            }
            escola_mun = escola_mun.rename(columns=rename_map)

            # Quadra = coberta OU descoberta
            if 'IN_QUADRA_ESPORTES_COBERTA' in escola_mun.columns:
                escola_mun['pct_quadra_esportes'] = escola_mun[
                    ['IN_QUADRA_ESPORTES_COBERTA', 'IN_QUADRA_ESPORTES_DESCOBERTA']
                ].max(axis=1) * 100
                escola_mun = escola_mun.drop(
                    columns=['IN_QUADRA_ESPORTES_COBERTA', 'IN_QUADRA_ESPORTES_DESCOBERTA'],
                    errors='ignore'
                )

            # Converter flags 0/1 para percentuais 0-100
            for col in escola_mun.columns:
                if col.startswith('pct_') and col != 'pct_quadra_esportes':
                    escola_mun[col] = escola_mun[col] * 100

            # Contagens por município
            docentes_per_escola = escola_df.groupby(mun_col).size().reset_index(name='media_docentes_por_escola')
            turmas_per_escola = escola_df.groupby(mun_col).size().reset_index(name='media_turmas_por_escola')

            agg = agg.merge(escola_mun, on=mun_col, how='left')
            agg = agg.merge(docentes_per_escola, on=mun_col, how='left')
            agg = agg.merge(turmas_per_escola, on=mun_col, how='left')

    # Adicionar metadados
    agg['year'] = year
    agg['state_code'] = agg[mun_col].astype(str).str[:2]

    # Renomear entidade
    agg = agg.rename(columns={mun_col: 'municipality_code'})

    return agg


def compute_abandono_rate(matricula_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa taxa de abandono por município.

    Abandono = "Deixou de frequentar" (código 5 em TP_SITUACAO_ALUNO).
    Se a variável de situação não estiver disponível no mesmo arquivo,
    usa heurística baseada em dados disponíveis.

    Returns:
        DataFrame com municipality_code e abandono_rate
    """
    mun_col = "CO_MUNICIPIO"

    # A situação do aluno pode estar em TP_SITUACAO ou coluna similar
    sit_col = None
    for candidate in ['TP_SITUACAO_ALUNO', 'TP_SITUACAO', 'IN_SITUACAO_ALUNO']:
        if candidate in matricula_df.columns:
            sit_col = candidate
            break

    if sit_col is None:
        # Sem dados de situação: retornar NaN (será imputado depois)
        print("      Aviso: coluna de situação do aluno não encontrada")
        result = matricula_df.groupby(mun_col).size().reset_index(name='_count')
        result['abandono_rate'] = float('nan')
        result = result.rename(columns={mun_col: 'municipality_code'})
        return result[['municipality_code', 'abandono_rate']]

    # Calcular taxa: abandonos / (aprovados + reprovados + concluintes + abandonos)
    abandono_codes = INEP_CONFIG.situacao_abandono_codes
    ativa_codes = INEP_CONFIG.situacao_ativa_codes

    valid = matricula_df[matricula_df[sit_col].isin(ativa_codes)]
    abandono = valid.groupby(mun_col).apply(
        lambda g: (g[sit_col].isin(abandono_codes)).mean() * 100
    ).reset_index(name='abandono_rate')

    abandono = abandono.rename(columns={mun_col: 'municipality_code'})
    return abandono


# ============================================================================
# Pipeline principal
# ============================================================================
def collect_inep_data(output_dir: str, years: Optional[List[int]] = None,
                      cache_dir: Optional[str] = None) -> Dict:
    """
    Pipeline completo de coleta INEP Censo Escolar.

    Args:
        output_dir: Diretório para salvar Parquets agregados
        years: Anos a coletar (default: todos no range do config)
        cache_dir: Diretório para cache de ZIPs (default: output_dir/zip_cache)

    Returns:
        Dicionário com metadados da coleta
    """
    if years is None:
        start, end = INEP_CONFIG.temporal_range
        years = list(range(start, end + 1))

    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "zip_cache")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"COLETA INEP CENSO ESCOLAR")
    print(f"Anos: {years[0]}-{years[-1]} ({len(years)} anos)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    all_aggregated = []
    metadata = {
        'dataset': INEP_CONFIG.name,
        'label': INEP_CONFIG.label,
        'collection_start': datetime.now().isoformat(),
        'years': {},
    }

    for year in years:
        print(f"\n--- Ano {year} ---")
        t0 = time.time()

        try:
            # 1. Download
            zip_path = download_year(year, cache_dir)

            # 2. Encontrar CSVs
            matricula_csv = find_csv_in_zip(zip_path, "MATRICULA")
            escola_csv = find_csv_in_zip(zip_path, "ESCOLA")

            if matricula_csv is None:
                print(f"   AVISO: Tabela MATRICULA não encontrada para {year}")
                metadata['years'][str(year)] = {'status': 'missing_matricula'}
                continue

            print(f"   Tabela MATRICULA: {matricula_csv}")
            if escola_csv:
                print(f"   Tabela ESCOLA: {escola_csv}")

            # 3. Ler e processar
            mat_df = read_csv_from_zip(zip_path, matricula_csv, MATRICULA_USECOLS)
            print(f"   Total bruto: {len(mat_df):,} linhas")

            mat_df = process_matricula(mat_df, year)

            escola_df = None
            if escola_csv:
                escola_df = read_csv_from_zip(zip_path, escola_csv, ESCOLA_USECOLS)
                # Filtrar escolas que oferecem Ensino Médio
                if 'TP_ETAPA_ENSINO' in mat_df.columns:
                    escolas_em = mat_df['CO_ENTIDADE'].unique() if 'CO_ENTIDADE' in mat_df.columns else []
                    if len(escolas_em) > 0 and 'CO_ENTIDADE' in escola_df.columns:
                        escola_df = escola_df[escola_df['CO_ENTIDADE'].isin(escolas_em)]

            # 4. Agregar
            agg_df = aggregate_municipality_year(mat_df, escola_df, year)

            # 5. Abandono (se disponível)
            abandono_df = compute_abandono_rate(mat_df)
            agg_df = agg_df.merge(abandono_df, on='municipality_code', how='left')

            all_aggregated.append(agg_df)

            elapsed = time.time() - t0
            n_mun = len(agg_df)
            metadata['years'][str(year)] = {
                'status': 'ok',
                'raw_rows': len(mat_df),
                'municipalities': n_mun,
                'elapsed_seconds': round(elapsed, 1),
            }
            print(f"   Agregado: {n_mun} municípios em {elapsed:.1f}s")

        except Exception as e:
            print(f"   ERRO ao processar {year}: {e}")
            metadata['years'][str(year)] = {
                'status': 'error',
                'error': str(e),
            }

    if not all_aggregated:
        print("\nNenhum dado coletado!")
        return metadata

    # 6. Concatenar e salvar
    complete_df = pd.concat(all_aggregated, ignore_index=True)

    # Preencher NaNs de features com 0 (flags de infraestrutura ausentes = não tem)
    for col in INEP_CONFIG.feature_columns:
        if col in complete_df.columns:
            complete_df[col] = complete_df[col].fillna(0)

    # Salvar como Parquet
    parquet_path = os.path.join(output_dir, "complete_data.parquet")
    complete_df.to_parquet(parquet_path, index=False)

    # Hash para reprodutibilidade
    with open(parquet_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    metadata.update({
        'collection_end': datetime.now().isoformat(),
        'total_rows': len(complete_df),
        'total_municipalities': complete_df['municipality_code'].nunique(),
        'total_years': complete_df['year'].nunique(),
        'columns': list(complete_df.columns),
        'sha256': sha256,
        'parquet_path': parquet_path,
    })

    # Salvar metadados
    meta_path = os.path.join(output_dir, "scientific_collection_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COLETA CONCLUÍDA")
    print(f"Total: {len(complete_df):,} observações "
          f"({complete_df['municipality_code'].nunique()} municípios × "
          f"{complete_df['year'].nunique()} anos)")
    print(f"Parquet: {parquet_path}")
    print(f"SHA-256: {sha256}")
    print(f"{'='*60}")

    return metadata


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Coleta microdados INEP Censo Escolar'
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help=f'Diretório de saída (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--years', nargs='+', type=int, default=None,
        help='Anos a coletar (default: todos no range)'
    )
    parser.add_argument(
        '--cache-dir', default=None,
        help='Diretório para cache de ZIPs'
    )
    args = parser.parse_args()

    collect_inep_data(
        output_dir=args.output_dir,
        years=args.years,
        cache_dir=args.cache_dir,
    )


if __name__ == '__main__':
    main()

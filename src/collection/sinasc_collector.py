#!/usr/bin/env python3
"""
SINASC live-birth panel collector — vaginal-delivery share (F2.1).

Downloads the final-vintage DNBR{year}.dbc files from the DATASUS FTP tree
(Sistema de Informações sobre Nascidos Vivos), parses the compressed DBF
microdata, and aggregates births to the municipality-of-residence x year
level in the framework schema (entity_id / entity_name / entity_stratum /
year / features / target_source_rate) for direct consumption by the
processors.

Target: target_source_rate = 100 * n(PARTO==1) / n(PARTO in {1,2}) per
municipality-year (vaginal share of deliveries with a known route). The
framework derives its target as 100 - target_source_rate (cesarean share).

Discipline (mirrors inep_collector.py):
  * Drop, never impute. Municipality-years with a zero valid PARTO
    denominator are removed; feature shares with a zero valid denominator
    are left as NaN for the models' in-fold median fill (P5).
  * Final vintages only: /SINASC/1996_/Dados/DNRES/DNBR{year}.dbc.
    Never /PRELIM/. Per-UF DN{UF}{year}.dbc is a fallback used only when a
    national DNBR file fails.
  * Every .dbc is cached locally and its SHA-256 recorded in the metadata;
    reruns re-download nothing whose cached size and SHA already match.

Source: ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/1996_/Dados/DNRES/
Citation: BRASIL. Ministério da Saúde. Sistema de Informações sobre
          Nascidos Vivos (SINASC). Brasília: DATASUS.

Usage:
    python src/collection/sinasc_collector.py [--years 2001 2002]
        [--output-dir DIR] [--cache-dir DIR] [--inep-panel PARQUET]
"""

import argparse
import ftplib
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "sinasc", "collection", "raw_data")
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "outputs", "sinasc", "dbc_cache")

# ============================================================================
# Source location — final vintages only (never /PRELIM/, never NOV/DNRES)
# ============================================================================
FTP_HOST = "ftp.datasus.gov.br"
FTP_DIR = "/dissemin/publicos/SINASC/1996_/Dados/DNRES"
YEARS = list(range(2001, 2025))

# All 27 federative units, for the per-UF fallback (DN{UF}{year}.dbc).
UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
       "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
       "RS", "SC", "SE", "SP", "TO"]

# IBGE 2-digit state prefix -> UF abbreviation (same strata naming as the
# INEP dataset config: entity_stratum carries the state abbreviation).
UF_BY_PREFIX = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}

# Microdata fields used by this collector (all others are ignored).
NEEDED_FIELDS = [
    "CODMUNRES", "CODPAISRES", "PARTO", "IDADEMAE", "ESCMAE", "CONSULTAS",
    "GESTACAO", "GRAVIDEZ", "QTDFILVIVO", "QTDFILMORT", "SEXO", "PESO",
    "LOCNASC",
]

# Feature shares. Each is numerator/denominator over its own valid domain;
# invalid/ignored codes (9, 99, blanks) never enter a denominator.
FEATURE_COLS = [
    "share_mother_lt20", "share_mother_ge35",
    "share_escmae_low", "share_escmae_high",
    "share_prenatal_7plus", "share_prenatal_none",
    "share_preterm", "share_multiple", "share_firstbirth",
    "share_male", "share_lbw", "share_hospital",
]

PARSE_CHUNK = 250_000  # records per aggregation chunk (keeps memory flat)

# Written verbatim into the metadata so a reader does not have to open this
# file to learn what each share counts. Codes outside a denominator are the
# SINASC "ignorado" codes (9 / 99 / 9999 / 0 for SEXO and PESO) and blanks;
# they are excluded, never recoded. The numeric windows (IDADEMAE 8-65,
# QTDFIL* 0-30) guard against stray values: 2001-2005 carry a few hundred
# IDADEMAE values of 0-7 per year (missing age written as 0), 2001-2002
# carry PESO 0 / 9999 sentinels, and ESCMAE shows out-of-dictionary codes
# 0/6/7 (14,165 zeros in 2003); from 2006 on the windows exclude nothing
# beyond the ignorado codes. SEXO's ignorado code is 0 through 2009 and 9
# from 2010; neither enters the share_male denominator. CODMUNRES is
# 7 digits wide (with IBGE check digit) in 2001-2005 and 6 from 2006.
FEATURE_DEFINITIONS = {
    "target_source_rate": "100 * n(PARTO==1) / n(PARTO in {1,2}); 9/blank excluded",
    "share_mother_lt20": "100 * n(IDADEMAE<20) / n(8<=IDADEMAE<=65); 99/blank excluded",
    "share_mother_ge35": "100 * n(IDADEMAE>=35) / n(8<=IDADEMAE<=65); 99/blank excluded",
    "share_escmae_low": "100 * n(ESCMAE in {1,2}) / n(ESCMAE in {1,2,3,4,5}); 9/blank excluded (0-3 years of schooling)",
    "share_escmae_high": "100 * n(ESCMAE==5) / n(ESCMAE in {1,2,3,4,5}); 9/blank excluded (12+ years)",
    "share_prenatal_7plus": "100 * n(CONSULTAS==4) / n(CONSULTAS in {1,2,3,4}); 9/blank excluded",
    "share_prenatal_none": "100 * n(CONSULTAS==1) / n(CONSULTAS in {1,2,3,4}); 9/blank excluded",
    "share_preterm": "100 * n(GESTACAO in {1,2,3,4}) / n(GESTACAO in {1..6}); 9/blank excluded (<37 weeks)",
    "share_multiple": "100 * n(GRAVIDEZ in {2,3}) / n(GRAVIDEZ in {1,2,3}); 9/blank excluded",
    "share_firstbirth": "100 * n(QTDFILVIVO==0 & QTDFILMORT==0) / n(0<=QTDFILVIVO<=30 & 0<=QTDFILMORT<=30); 99/blank excluded",
    "share_male": "100 * n(SEXO in {1,M}) / n(SEXO in {1,2,M,F}); 0/blank excluded",
    "share_lbw": "100 * n(0<PESO<2500) / n(0<PESO<9999); 0/9999/blank excluded",
    "share_hospital": "100 * n(LOCNASC==1) / n(LOCNASC in {1,2,3,4,5}); 9/blank excluded",
    "births_total": "all records of the municipality of residence before any validity filter",
}
ROW_RULES = {
    "residence_code": ("CODMUNRES stripped to digits; 7-digit codes (2005-style, "
                       "with IBGE check digit) truncated to the 6-digit key"),
    "residence_country": ("CODPAISRES: records with a numeric value other than 1 "
                          "are dropped; blank or absent field (2001-2009) keeps the record"),
    "dropped_codes": ("6-digit codes ending in 0000, 999999, non-6-digit shapes, "
                      "and codes absent from the INEP-derived municipality table"),
    "dropped_rows": "municipality-years with zero valid PARTO denominator",
    "no_imputation": "zero-denominator shares are NaN; nothing is filled at collection",
}


# ============================================================================
# IBGE municipality codes
# ============================================================================
def compute_ibge_check_digit(code6: str) -> str:
    """
    Standard IBGE modulo-10 check digit for the 6-digit municipality code.

    Weights 1,2,1,2,1,2 over the six digits; two-digit products contribute
    the sum of their digits (i.e. product - 9); the check digit completes
    the total to the next multiple of 10.

    Validated against the full INEP panel universe (5,564 codes): 5,555
    reproduce the official 7th digit exactly; the 9 mismatches are the
    documented legacy codes IBGE assigned outside its own algorithm
    (2201919, 2201988, 2202251, 2611533, 3117836, 3152131, 4305871,
    5203939, 5203962) and are handled by the lookup table, which always
    takes precedence over the computed digit.
    """
    total = 0
    for digit, weight in zip(code6, (1, 2, 1, 2, 1, 2)):
        product = int(digit) * weight
        total += product - 9 if product > 9 else product
    return str((10 - total % 10) % 10)


def find_inep_panel() -> Optional[str]:
    """Locate an INEP panel parquet to derive the municipality code table."""
    pattern = os.path.join(
        PROJECT_ROOT, "outputs", "kaggle", "*", "rampart", "panels",
        "azure_results_v7_inep", "collection", "inep_raw", "complete_data.parquet")
    candidates = sorted(glob.glob(pattern))
    return candidates[0] if candidates else None


def load_municipality_table(inep_panel_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Derive the IBGE municipality table from the INEP panel.

    Returns a DataFrame indexed by the 6-digit code with columns
    [entity_id (7-digit), entity_name, entity_stratum], plus a validation
    dict for the check-digit algorithm (computed digit vs official 7th
    digit over the full universe, with a 20-code sample).
    """
    df = pd.read_parquet(inep_panel_path)
    # Tolerate both schema generations (entity_* and country_*).
    renames = {"country_code": "entity_id", "country_name": "entity_name",
               "country_stratum": "entity_stratum"}
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # One row per municipality; take the name from the most recent year.
    df = df.sort_values("year")
    table = (df.groupby("entity_id")
               .agg(entity_name=("entity_name", "last"),
                    entity_stratum=("entity_stratum", "last"))
               .reset_index())
    table["entity_id"] = table["entity_id"].astype(str)
    if not (table["entity_id"].str.len() == 7).all():
        raise ValueError("INEP panel entity_id is not uniformly 7 digits")
    table["code6"] = table["entity_id"].str[:6]
    if table["code6"].duplicated().any():
        raise ValueError("6-digit municipality prefixes are not unique")

    # Check-digit validation over the full universe.
    computed = table["code6"].map(compute_ibge_check_digit)
    match = computed == table["entity_id"].str[6]
    mismatch_codes = sorted(table.loc[~match, "entity_id"].tolist())
    sample = table.sort_values("entity_id").iloc[::278].head(20)
    validation = {
        "codes_tested": int(len(table)),
        "exact_matches": int(match.sum()),
        "mismatches": mismatch_codes,
        "sample_20": [
            {"code6": row.code6,
             "computed_digit": compute_ibge_check_digit(row.code6),
             "official_code7": row.entity_id,
             "match": compute_ibge_check_digit(row.code6) == row.entity_id[6]}
            for row in sample.itertuples()
        ],
    }
    return table.set_index("code6"), validation


# ============================================================================
# Download (FTP with resume + retry; cache keyed by size and SHA-256)
# ============================================================================
def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(cache_dir: str) -> Dict:
    path = os.path.join(cache_dir, "dbc_manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(cache_dir: str, manifest: Dict) -> None:
    path = os.path.join(cache_dir, "dbc_manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)


def _ftp_connect() -> ftplib.FTP:
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(FTP_DIR)
    ftp.voidcmd("TYPE I")
    return ftp


def download_file(filename: str, cache_dir: str, retries: int = 6) -> str:
    """
    Download one .dbc from the final DATASUS tree, resume-safe.

    Skips the transfer when the cached file already matches the recorded
    size/SHA in the manifest (or, lacking a record, the remote size).
    Partial transfers land in <file>.part and are resumed with REST.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, filename)
    part = dest + ".part"
    manifest = _load_manifest(cache_dir)

    if os.path.exists(dest):
        recorded = manifest.get(filename)
        size = os.path.getsize(dest)
        if recorded and recorded.get("size") == size:
            print(f"   Cache: {dest} ({size:,} bytes, sha recorded)")
            return dest

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            ftp = _ftp_connect()
            remote_size = ftp.size(filename)
            if remote_size is None:
                raise IOError(f"FTP SIZE failed for {filename}")

            if os.path.exists(dest) and os.path.getsize(dest) == remote_size:
                ftp.quit()
                break  # complete file already cached; record below

            offset = os.path.getsize(part) if os.path.exists(part) else 0
            if offset > remote_size:
                os.remove(part)
                offset = 0
            mode = "ab" if offset else "wb"
            print(f"   Downloading {filename} "
                  f"({remote_size:,} bytes, offset {offset:,}, attempt {attempt})")
            with open(part, mode) as handle:
                ftp.retrbinary(f"RETR {filename}", handle.write,
                               blocksize=1 << 16, rest=offset or None)
            ftp.quit()
            if os.path.getsize(part) != remote_size:
                raise IOError(
                    f"{filename}: got {os.path.getsize(part):,} bytes, "
                    f"expected {remote_size:,}")
            os.replace(part, dest)
            break
        except Exception as error:  # noqa: BLE001 — network layer, retried
            last_error = error
            print(f"   Retry {attempt}/{retries} for {filename}: {error}")
            time.sleep(min(60, 10 * attempt))
    else:
        raise IOError(f"Download failed for {filename}: {last_error}")

    manifest[filename] = {"size": os.path.getsize(dest),
                          "sha256": _sha256_file(dest)}
    _save_manifest(cache_dir, manifest)
    return dest


def download_year(year: int, cache_dir: str) -> List[str]:
    """
    Fetch the national DNBR file for one year; fall back to the 27 per-UF
    DN{UF}{year}.dbc files only if the national file fails.
    """
    try:
        return [download_file(f"DNBR{year}.dbc", cache_dir)]
    except Exception as error:  # noqa: BLE001
        print(f"   DNBR{year}.dbc failed ({error}); "
              f"falling back to per-UF files")
        return [download_file(f"DN{uf}{year}.dbc", cache_dir) for uf in UFS]


# ============================================================================
# Parse and aggregate (pyreaddbc dbc2dbf -> dbfread, latin-1)
# ============================================================================
def _aggregate_chunk(raw: Dict[str, list], present: Dict[str, bool],
                     stats: Dict) -> pd.DataFrame:
    """
    Turn one chunk of raw microdata records into per-municipality counts.

    Every feature is a (numerator, denominator) pair over its own valid
    domain. Ignored codes (9/99), blanks and out-of-range values are
    excluded from the denominators — never recoded, never imputed.
    """
    df = pd.DataFrame(raw)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # Residence-country filter: keep Brazil (code 1) and records where the
    # field is absent or blank (the field only exists in later layouts).
    if present["CODPAISRES"]:
        pais = pd.to_numeric(df["CODPAISRES"], errors="coerce")
        foreign = pais.notna() & (pais != 1)
        stats["rows_foreign_dropped"] += int(foreign.sum())
        df = df[~foreign]

    # Municipality of residence, normalised to the 6-digit IBGE code.
    code = df["CODMUNRES"].str.replace(r"\D", "", regex=True)
    code = code.where(code.str.len() != 7, code.str[:6])

    parto = pd.to_numeric(df["PARTO"], errors="coerce")
    age = pd.to_numeric(df["IDADEMAE"], errors="coerce")
    esc = pd.to_numeric(df["ESCMAE"], errors="coerce")
    consultas = pd.to_numeric(df["CONSULTAS"], errors="coerce")
    gest = pd.to_numeric(df["GESTACAO"], errors="coerce")
    grav = pd.to_numeric(df["GRAVIDEZ"], errors="coerce")
    filvivo = pd.to_numeric(df["QTDFILVIVO"], errors="coerce")
    filmort = pd.to_numeric(df["QTDFILMORT"], errors="coerce")
    sexo = df["SEXO"].str.upper()
    peso = pd.to_numeric(df["PESO"], errors="coerce")
    locnasc = pd.to_numeric(df["LOCNASC"], errors="coerce")

    # QA gate 7 bookkeeping: CONSULTAS must stay in the categorical coding
    # {1,2,3,4,9}. Raw visit counts (the 1998-style coding) would surface
    # here as out-of-domain values.
    out_of_domain = consultas.notna() & ~consultas.isin([1, 2, 3, 4, 9])
    if out_of_domain.any():
        for value, count in consultas[out_of_domain].value_counts().items():
            key = str(int(value)) if float(value).is_integer() else str(value)
            stats["consultas_out_of_domain"][key] = (
                stats["consultas_out_of_domain"].get(key, 0) + int(count))

    age_valid = age.between(8, 65)          # 99 = ignored
    fil_valid = filvivo.between(0, 30) & filmort.between(0, 30)  # 99 = ignored
    male = sexo.isin(["1", "M"])
    female = sexo.isin(["2", "F"])
    peso_valid = peso.gt(0) & peso.lt(9999)  # 0/9999 = ignored

    counts = pd.DataFrame({
        "code6": code,
        "births_total": 1,
        "num_parto_vaginal": parto.eq(1),
        "den_parto": parto.isin([1, 2]),
        "num_mother_lt20": age_valid & age.lt(20),
        "num_mother_ge35": age_valid & age.ge(35),
        "den_mother_age": age_valid,
        "num_escmae_low": esc.isin([1, 2]),
        "num_escmae_high": esc.eq(5),
        "den_escmae": esc.isin([1, 2, 3, 4, 5]),
        "num_prenatal_7plus": consultas.eq(4),
        "num_prenatal_none": consultas.eq(1),
        "den_prenatal": consultas.isin([1, 2, 3, 4]),
        "num_preterm": gest.isin([1, 2, 3, 4]),
        "den_gestacao": gest.isin([1, 2, 3, 4, 5, 6]),
        "num_multiple": grav.isin([2, 3]),
        "den_gravidez": grav.isin([1, 2, 3]),
        "num_firstbirth": fil_valid & filvivo.eq(0) & filmort.eq(0),
        "den_firstbirth": fil_valid,
        "num_male": male,
        "den_sexo": male | female,
        "num_lbw": peso_valid & peso.lt(2500),
        "den_peso": peso_valid,
        "num_hospital": locnasc.eq(1),
        "den_locnasc": locnasc.isin([1, 2, 3, 4, 5]),
    })
    return counts.groupby("code6").sum()


def parse_year(dbc_paths: List[str], year: int, cache_dir: str,
               dbc_shas: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse one year of SINASC microdata into per-municipality counts.

    The aggregate is cached (keyed on the .dbc SHA-256 list), so reruns
    after a reboot skip both the transfer and the parse.
    """
    from dbfread import DBF
    from pyreaddbc import dbc2dbf

    agg_dir = os.path.join(cache_dir, "agg")
    os.makedirs(agg_dir, exist_ok=True)
    agg_path = os.path.join(agg_dir, f"sinasc_{year}_agg.parquet")
    stats_path = os.path.join(agg_dir, f"sinasc_{year}_stats.json")

    if os.path.exists(agg_path) and os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if stats.get("dbc_sha256") == dbc_shas:
            print(f"   Aggregate cache: {agg_path}")
            stats["aggregate_from_cache"] = True
            stats["aggregate_written_at"] = datetime.fromtimestamp(
                os.path.getmtime(agg_path)).isoformat()
            return pd.read_parquet(agg_path), stats

    stats = {
        "year": year,
        "dbc_files": [os.path.basename(p) for p in dbc_paths],
        "dbc_sha256": dbc_shas,
        "rows_total": 0,
        "rows_foreign_dropped": 0,
        "consultas_out_of_domain": {},
        "fields_missing": [],
    }
    aggregates: List[pd.DataFrame] = []

    for dbc_path in dbc_paths:
        dbf_path = os.path.join(agg_dir, os.path.basename(dbc_path) + ".dbf.tmp")
        try:
            dbc2dbf(dbc_path, dbf_path)
            table = DBF(dbf_path, encoding="latin-1", recfactory=None,
                        char_decode_errors="replace")
            field_names = table.field_names
            missing = [f for f in NEEDED_FIELDS if f not in field_names]
            present = {f: f in field_names for f in NEEDED_FIELDS}
            for f in missing:
                if f not in stats["fields_missing"]:
                    stats["fields_missing"].append(f)
            indices = {f: field_names.index(f)
                       for f in NEEDED_FIELDS if present[f]}

            raw = {f: [] for f in NEEDED_FIELDS}
            n_in_chunk = 0
            for record in table:
                for f in NEEDED_FIELDS:
                    value = record[indices[f]][1] if present[f] else ""
                    raw[f].append("" if value is None else value)
                n_in_chunk += 1
                stats["rows_total"] += 1
                if n_in_chunk >= PARSE_CHUNK:
                    aggregates.append(_aggregate_chunk(raw, present, stats))
                    raw = {f: [] for f in NEEDED_FIELDS}
                    n_in_chunk = 0
            if n_in_chunk:
                aggregates.append(_aggregate_chunk(raw, present, stats))
        finally:
            if os.path.exists(dbf_path):
                os.remove(dbf_path)

    aggregate = (pd.concat(aggregates).groupby(level=0).sum().astype("int64")
                 if aggregates else pd.DataFrame())
    aggregate.to_parquet(agg_path)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    stats["aggregate_from_cache"] = False
    stats["aggregate_written_at"] = datetime.now().isoformat()
    return aggregate, stats


# ============================================================================
# Panel assembly (framework schema)
# ============================================================================
def build_year_rows(aggregate: pd.DataFrame, year: int,
                    muni_table: pd.DataFrame, drop_log: Dict) -> pd.DataFrame:
    """
    Municipality filters + shares for one year.

    Drops (never imputes): ignored/unknown residence codes (xx0000,
    999999, non-6-digit), codes absent from the INEP-derived municipality
    table, and municipality-years with zero valid PARTO denominator.

    The municipality table comes from the INEP panel, so the entity
    universe is INEP's by construction (5,564 of Brazil's 5,570). Real IBGE
    municipalities that INEP lacks (no upper-secondary indicators) are
    dropped here and listed with their births under
    births_not_in_inep_table, so the loss is auditable.
    """
    agg = aggregate.reset_index().rename(columns={"index": "code6"})
    n0 = len(agg)

    valid_shape = agg["code6"].str.fullmatch(r"\d{6}")
    ignored = agg["code6"].str.endswith("0000") | (agg["code6"] == "999999")
    known = agg["code6"].isin(muni_table.index)
    keep = valid_shape & ~ignored & known
    unknown_real = valid_shape & ~ignored & ~known

    drop_log[str(year)] = {
        "codes_seen": int(n0),
        "codes_bad_shape": int((~valid_shape).sum()),
        "codes_ignored_pattern": int((valid_shape & ignored).sum()),
        "codes_not_in_inep_table": int(unknown_real.sum()),
        "births_not_in_inep_table": {
            row.code6: int(row.births_total)
            for row in agg.loc[unknown_real, ["code6", "births_total"]]
                          .sort_values("code6").itertuples()},
        "births_dropped_unknown_residence": int(agg.loc[~keep, "births_total"].sum()),
        "births_kept": int(agg.loc[keep, "births_total"].sum()),
    }
    agg = agg[keep].copy()

    # Target: vaginal share over deliveries with a known route.
    zero_parto = agg["den_parto"] == 0
    drop_log[str(year)]["municipalities_zero_parto_denominator"] = int(zero_parto.sum())
    agg = agg[~zero_parto].copy()
    agg["target_source_rate"] = 100.0 * agg["num_parto_vaginal"] / agg["den_parto"]

    # Feature shares; zero denominators yield NaN (in-fold median fill later).
    pairs = [
        ("share_mother_lt20", "num_mother_lt20", "den_mother_age"),
        ("share_mother_ge35", "num_mother_ge35", "den_mother_age"),
        ("share_escmae_low", "num_escmae_low", "den_escmae"),
        ("share_escmae_high", "num_escmae_high", "den_escmae"),
        ("share_prenatal_7plus", "num_prenatal_7plus", "den_prenatal"),
        ("share_prenatal_none", "num_prenatal_none", "den_prenatal"),
        ("share_preterm", "num_preterm", "den_gestacao"),
        ("share_multiple", "num_multiple", "den_gravidez"),
        ("share_firstbirth", "num_firstbirth", "den_firstbirth"),
        ("share_male", "num_male", "den_sexo"),
        ("share_lbw", "num_lbw", "den_peso"),
        ("share_hospital", "num_hospital", "den_locnasc"),
    ]
    for share, num, den in pairs:
        agg[share] = np.where(agg[den] > 0,
                              100.0 * agg[num] / agg[den].replace(0, 1), np.nan)

    agg["entity_id"] = agg["code6"].map(muni_table["entity_id"])
    agg["entity_name"] = agg["code6"].map(muni_table["entity_name"])
    agg["entity_stratum"] = agg["code6"].str[:2].map(UF_BY_PREFIX)
    agg["year"] = year

    keep_cols = (["entity_id", "entity_name", "entity_stratum", "year"]
                 + FEATURE_COLS + ["target_source_rate", "births_total"])
    return agg[keep_cols]


# ============================================================================
# QA gates
# ============================================================================
def run_qa_gates(panel: pd.DataFrame, year_stats: Dict[int, Dict],
                 muni_table: pd.DataFrame, years: List[int]) -> Dict:
    """
    Deterministic QA receipt over the collected panel (7 gates).

    A failing gate sets kill_triggered; the receipt reports, it never
    patches.
    """
    receipt: Dict = {"gates": {}, "kill_triggered": False}

    def fail(gate: str, reason: str) -> None:
        receipt["gates"][gate]["passed"] = False
        receipt["gates"][gate]["reason"] = reason
        receipt["kill_triggered"] = True

    # Gate 1 — per-year municipality counts and total rows.
    per_year = panel.groupby("year")["entity_id"].nunique().to_dict()
    rows_year = panel.groupby("year").size().to_dict()
    receipt["gates"]["1_per_year_counts"] = {
        "passed": True,
        "total_rows": int(len(panel)),
        "municipalities_per_year": {str(y): int(per_year.get(y, 0)) for y in years},
        "rows_per_year": {str(y): int(rows_year.get(y, 0)) for y in years},
    }
    missing_years = [y for y in years if per_year.get(y, 0) == 0]
    unstable = [y for y in years if not 5400 <= per_year.get(y, 0) <= 5600]
    if missing_years:
        fail("1_per_year_counts", f"years with no data: {missing_years}")
    elif unstable:
        fail("1_per_year_counts",
             f"municipality universe outside [5400, 5600] in {unstable}")

    # Gate 2 — target bounds and the secular cesarean rise.
    target = panel["target_source_rate"]
    by_year = panel.groupby("year")["target_source_rate"]
    dist = {str(y): {"mean": round(float(g.mean()), 4),
                     "p25": round(float(g.quantile(0.25)), 4),
                     "p50": round(float(g.quantile(0.50)), 4),
                     "p75": round(float(g.quantile(0.75)), 4)}
            for y, g in by_year}
    n_out = int(((target < 0) | (target > 100)).sum())
    years_present = sorted(int(y) for y in dist)
    decline = (dist[str(years_present[0])]["mean"]
               - dist[str(years_present[-1])]["mean"]) if years_present else 0.0
    receipt["gates"]["2_target_range"] = {
        "passed": True,
        "min": round(float(target.min()), 4),
        "max": round(float(target.max()), 4),
        "n_outside_0_100": n_out,
        "n_exactly_0": int((target == 0).sum()),
        "n_exactly_100": int((target == 100).sum()),
        "per_year_distribution": dist,
        "vaginal_share_decline_first_to_last": round(float(decline), 4),
    }
    if n_out > 0:
        fail("2_target_range", f"{n_out} rows outside [0, 100]")
    elif (len(years_present) >= 10
          and years_present[-1] - years_present[0] >= 10 and decline <= 5.0):
        # The secular-rise check only means something over a long span.
        fail("2_target_range",
             "no secular cesarean rise: vaginal share did not decline")

    # Gate 3 — cross-sectional lag-1 persistence of the target.
    wide = panel.pivot_table(index="entity_id", columns="year",
                             values="target_source_rate")
    persistence = {}
    for y0, y1 in zip(years[:-1], years[1:]):
        if y0 in wide.columns and y1 in wide.columns:
            pair = wide[[y0, y1]].dropna()
            persistence[f"{y0}-{y1}"] = round(float(pair[y0].corr(pair[y1])), 4)
    values = list(persistence.values())
    receipt["gates"]["3_lag1_persistence"] = {
        "passed": True,
        "per_pair": persistence,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": round(float(np.mean(values)), 4) if values else None,
        "expected_band": [0.85, 0.95],
    }
    if values and min(values) < 0.80:
        fail("3_lag1_persistence",
             f"minimum adjacent-year correlation {min(values)} < 0.80")

    # Gate 4 — entity universe overlap with the INEP panel.
    sinasc_entities = set(panel["entity_id"].unique())
    inep_entities = set(muni_table["entity_id"])
    overlap = len(sinasc_entities & inep_entities)
    receipt["gates"]["4_inep_overlap"] = {
        "passed": overlap >= 5500,
        "sinasc_entities": len(sinasc_entities),
        "inep_entities": len(inep_entities),
        "overlap": overlap,
        "sinasc_only": len(sinasc_entities - inep_entities),
        "inep_only": len(inep_entities - sinasc_entities),
        "note": ("sinasc_only is 0 by construction: the municipality table is "
                 "derived from the INEP panel. Real IBGE codes absent from INEP "
                 "are dropped before this gate and listed per year in "
                 "metadata row_filters[year]['births_not_in_inep_table']. "
                 "inep_only measures municipalities with no SINASC row in any year."),
    }
    if overlap < 5500:
        fail("4_inep_overlap", f"overlap {overlap} < 5500")

    # Gate 5 — NaN census per feature (informational; NaN is the documented
    # zero-denominator outcome, filled in-fold by the models).
    receipt["gates"]["5_nan_census"] = {
        "passed": True,
        "nan_counts": {c: int(panel[c].isna().sum()) for c in FEATURE_COLS},
        "nan_fractions": {c: round(float(panel[c].isna().mean()), 6)
                          for c in FEATURE_COLS},
    }

    # Gate 6 — no duplicated (entity_id, year).
    n_dup = int(panel.duplicated(["entity_id", "year"]).sum())
    receipt["gates"]["6_no_duplicates"] = {"passed": n_dup == 0,
                                           "duplicated_pairs": n_dup}
    if n_dup:
        fail("6_no_duplicates", f"{n_dup} duplicated (entity_id, year) pairs")

    # Gate 7 — CONSULTAS coding must stay categorical ({1,2,3,4,9}).
    coding = {}
    breaches = []
    for year in years:
        stats = year_stats.get(year, {})
        bad = stats.get("consultas_out_of_domain", {})
        n_bad = sum(bad.values())
        frac = n_bad / max(stats.get("rows_total", 1), 1)
        coding[str(year)] = {"out_of_domain_count": n_bad,
                             "out_of_domain_values": bad,
                             "fraction": round(frac, 8)}
        if frac > 0.001:
            breaches.append(year)
    receipt["gates"]["7_consultas_coding"] = {"passed": not breaches,
                                              "per_year": coding}
    if breaches:
        fail("7_consultas_coding",
             f"1998-style CONSULTAS coding detected in {breaches}: "
             f"values outside {{1,2,3,4,9}} above 0.1% of records")

    receipt["all_passed"] = all(g["passed"] for g in receipt["gates"].values())
    return receipt


# ============================================================================
# Main pipeline
# ============================================================================
def collect_sinasc_data(output_dir: str, years: Optional[List[int]] = None,
                        cache_dir: Optional[str] = None,
                        inep_panel: Optional[str] = None) -> Dict:
    """
    Complete collection pipeline for the SINASC vaginal-delivery panel.

    Downloads (cached, resume-safe), parses, aggregates, assembles the
    framework-schema panel, runs the QA gates and writes the artifacts:
    complete_data.parquet, scientific_collection_metadata.json,
    scientific_imputation_log.json (empty by construction), qa_receipt.json.
    """
    if years is None:
        years = YEARS
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    if inep_panel is None:
        inep_panel = find_inep_panel()
    if inep_panel is None or not os.path.exists(inep_panel):
        raise FileNotFoundError(
            "No INEP panel found to derive the municipality table; "
            "pass --inep-panel")
    os.makedirs(output_dir, exist_ok=True)

    print(f"SINASC collection - vaginal-delivery share, "
          f"{years[0]}-{years[-1]} ({len(years)} years)")
    print(f"Output: {output_dir}")
    print(f"Cache:  {cache_dir}")
    print(f"Municipality table from: {inep_panel}")

    muni_table, dv_validation = load_municipality_table(inep_panel)
    print(f"   {len(muni_table)} municipalities; check digit: "
          f"{dv_validation['exact_matches']}/{dv_validation['codes_tested']} "
          f"exact, {len(dv_validation['mismatches'])} known IBGE exceptions")

    metadata: Dict = {
        "dataset": "sinasc",
        "source": ("DATASUS SINASC - Declaracoes de Nascidos Vivos, "
                   "final vintages (DNRES tree, never PRELIM)"),
        "source_url": f"ftp://{FTP_HOST}{FTP_DIR}/",
        "target_definition": ("target_source_rate = 100 * n(PARTO==1) / "
                              "n(PARTO in {1,2}) per municipality-year "
                              "(vaginal share; framework derives "
                              "cesarean share as 100 - source)"),
        "feature_definitions": FEATURE_DEFINITIONS,
        "row_rules": ROW_RULES,
        "collection_start": datetime.now().isoformat(),
        "entity_mapping": {
            "inep_panel_path": os.path.relpath(inep_panel, PROJECT_ROOT),
            "inep_panel_sha256": _sha256_file(inep_panel),
            "municipalities_in_table": int(len(muni_table)),
            "check_digit_validation": dv_validation,
        },
        "years": {},
        "dbc_files": {},
    }

    frames: List[pd.DataFrame] = []
    year_stats: Dict[int, Dict] = {}
    drop_log: Dict = {}
    manifest = _load_manifest(cache_dir)

    for year in years:
        print(f"\n--- {year} ---")
        t0 = time.time()
        dbc_paths = download_year(year, cache_dir)
        manifest = _load_manifest(cache_dir)
        shas = [manifest[os.path.basename(p)]["sha256"] for p in dbc_paths]
        for path, sha in zip(dbc_paths, shas):
            name = os.path.basename(path)
            metadata["dbc_files"][name] = {
                "url": f"ftp://{FTP_HOST}{FTP_DIR}/{name}",
                "size": manifest[name]["size"], "sha256": sha}

        aggregate, stats = parse_year(dbc_paths, year, cache_dir, shas)
        year_stats[year] = stats
        rows = build_year_rows(aggregate, year, muni_table, drop_log)
        frames.append(rows)

        elapsed = time.time() - t0
        print(f"   {stats['rows_total']:,} births -> {len(rows):,} "
              f"municipalities, mean vaginal share "
              f"{rows['target_source_rate'].mean():.1f}% ({elapsed:.1f}s)")
        metadata["years"][str(year)] = {
            "status": "ok",
            "births_parsed": stats["rows_total"],
            "rows_foreign_dropped": stats["rows_foreign_dropped"],
            "fields_missing": stats["fields_missing"],
            "municipalities": int(len(rows)),
            "vaginal_share_mean": round(float(rows["target_source_rate"].mean()), 2),
            # elapsed_s near zero means the per-year aggregate was served from
            # outputs/sinasc/dbc_cache/agg; the parse time is recorded below.
            "aggregate_from_cache": bool(stats.get("aggregate_from_cache", False)),
            "aggregate_written_at": stats.get("aggregate_written_at"),
            "elapsed_s": round(elapsed, 1),
        }

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["entity_id", "year"]).reset_index(drop=True)

    print("\nRunning QA gates...")
    receipt = run_qa_gates(panel, year_stats, muni_table, years)
    for gate, result in receipt["gates"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"   [{status}] {gate}" +
              ("" if result["passed"] else f" -- {result.get('reason')}"))

    # Persist artifacts. The panel is written even on a failing gate so the
    # failure can be audited, but kill_triggered is recorded loudly.
    parquet_path = os.path.join(output_dir, "complete_data.parquet")
    panel.to_parquet(parquet_path, index=False)
    sha256 = _sha256_file(parquet_path)
    receipt["panel_sha256"] = sha256
    receipt["panel_shape"] = [int(panel.shape[0]), int(panel.shape[1])]

    metadata.update({
        "collection_end": datetime.now().isoformat(),
        "total_rows": int(len(panel)),
        "total_municipalities": int(panel["entity_id"].nunique()),
        "total_years": int(panel["year"].nunique()),
        "columns": list(panel.columns),
        "row_filters": drop_log,
        "sha256": sha256,
        "qa_all_passed": receipt["all_passed"],
        "qa_kill_triggered": receipt["kill_triggered"],
    })

    with open(os.path.join(output_dir, "scientific_collection_metadata.json"),
              "w") as f:
        json.dump(metadata, f, indent=2)
    # No imputation happens anywhere in this collector; the log stays empty
    # by construction (drops are documented in metadata['row_filters']).
    with open(os.path.join(output_dir, "scientific_imputation_log.json"),
              "w") as f:
        json.dump({"methodology": "none",
                   "imputation_log": {},
                   "note": ("No imputation at collection time. Zero-denominator "
                            "shares are NaN, left for the models' in-fold "
                            "median fill (P5); invalid rows are dropped and "
                            "counted in scientific_collection_metadata.json")},
                  f, indent=2)
    with open(os.path.join(output_dir, "qa_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)

    print(f"\nCollection complete: {len(panel):,} obs "
          f"({panel['entity_id'].nunique()} municipalities x "
          f"{panel['year'].nunique()} years)")
    print(f"Parquet: {parquet_path}")
    print(f"SHA-256: {sha256}")
    if receipt["kill_triggered"]:
        print("\nKILL CRITERIA TRIGGERED - see qa_receipt.json. "
              "Do not use this panel without resolving the failing gates.")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="SINASC vaginal-delivery panel collection")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--inep-panel", default=None,
                        help="INEP panel parquet used to derive the "
                             "municipality code table")
    args = parser.parse_args()

    collect_sinasc_data(args.output_dir, args.years, args.cache_dir,
                        args.inep_panel)


if __name__ == "__main__":
    main()

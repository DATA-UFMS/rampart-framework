# Guia de Uso

Instruções para executar, verificar e adaptar o framework. Complementa o [README](README.md).

## Preparação do Ambiente

Requisitos: Python 3.10+, 8 GB RAM, 10 GB disco, acesso à internet.

```bash
git clone https://github.com/anonymous/archbench-framework.git
cd archbench-framework
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Execução

**Pipeline completo** (~20 min na primeira execução, ~5 min com cache):

```bash
python pipeline.py
```

Executa: coleta, processamento (3 paradigmas), gate anti-leakage, modelos, benchmark e validação estatística.

**Componentes individuais:**

```bash
# Benchmark com parâmetros customizados
python src/benchmarking/architectural_benchmark.py --repetitions 5 --warmup 1

# Equivalência (gera JSON + LaTeX)
python src/statistical_validation/equivalence_estimation.py --latex

# Testes
pytest tests/

# Validação negativa do gate (cenários S1-S4)
python scripts/validation/leakage_injection.py
```

**Pós-processamento (tabelas LaTeX a partir dos CSVs de benchmark):**

```bash
python src/benchmarking/derive_latency_percentiles.py
python src/benchmarking/derive_throughput_percentiles.py
python src/benchmarking/derive_resource_usage_table.py
python src/statistical_validation/bootstrap_sensitivity.py --latex
```

## Verificação dos Resultados

Artefatos gerados em `outputs/`:

| Artefato | Caminho | Conteúdo |
|----------|---------|----------|
| Folds temporais | `ml_pipeline/architectures/<arch>/prep/temporal_folds_<arch>.json` | Intervalos treino/val/teste, gaps |
| Estatísticas do target | `ml_pipeline/architectures/<arch>/prep/target_statistics.json` | Distribuição, consistência |
| Equivalência | `statistics/equivalence_estimation.json` | Decisão + IC 95% + Wilcoxon |
| Latência | `benchmarks/architectural_benchmark_results.csv` | Tempo por fase e repetição |
| Recursos | `benchmarks/architectural_benchmark_resource_log.jsonl` | CPU/RAM/IO por amostra |
| Scorecard | `statistics/architectural_scorecard.tex` | Painel consolidado |

## Customização

### Parâmetros

Edite `src/core/scientific_config.py`:

- `temporal_gap_years`, `folds_min_train_years`, `folds_step_years`, `embargo_years`
- `sesoi_r2`, `sesoi_mase`, `sesoi_wape` (limiares de equivalência)
- `bootstrap_iters`

### Nova Arquitetura

1. Crie `src/architectures_ml/<nova>/setup.py` com subclasse de `BaseArchitectureML` definindo `PARADIGM_META` e implementando os métodos abstratos. O anti-leakage é herdado.
2. Crie os módulos de processamento, baseline e hierárquico nos caminhos declarados no `PARADIGM_META`. O framework descobre automaticamente via `__init_subclass__`.

### Novo Dataset

Implemente um `DatasetConfig` em `src/datasets/` e um coletor em `src/collection/`. Use o adapter pattern para converter ao schema interno. Exemplo existente: INEP Censo Escolar (`python pipeline.py --dataset inep_censo`).

### Métricas Extras

Adicione módulos em `src/benchmarking/` ou `src/statistical_validation/` seguindo o padrão JSON → LaTeX.

## FAQ

**Quanto tempo demora?** ~20 min na primeira execução (coleta da API). Com cache, ~5 min.

**Precisa de API key?** Não. A World Bank API é aberta.

**Os resultados devem bater exatamente entre execuções?** Sim. Seeds centralizadas e `n_jobs=1` garantem determinismo. Divergências indicam ambiente diferente.

**Funciona no Windows?** O pipeline foi testado em Linux. DuckDB e Polars funcionam no Windows; Dask distributed pode ter limitações.

---

Em caso de dúvidas, abra uma issue no repositório.

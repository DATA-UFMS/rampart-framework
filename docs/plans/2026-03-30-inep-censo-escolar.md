# Plano: Adicionar Dataset INEP Censo Escolar

**Branch**: `feat/inep-censo-escolar`
**Objetivo**: Adicionar segundo dataset (INEP Censo Escolar, município×ano, ~78K obs) ao framework para validar generalidade e encontrar crossover point DW/DL.

## Steps

- [ ] Step 0: DatasetConfig Protocol (`src/core/dataset_config.py`)
- [ ] Step 1: WorldBankDatasetConfig (`src/datasets/worldbank.py`)
- [ ] Step 2: InepCensoDatasetConfig (`src/datasets/inep_censo.py`)
- [ ] Step 3: Refatorar base_architecture.py para usar DatasetConfig
- [ ] Step 4: Refatorar config.py e scientific_config.py
- [ ] Step 5: INEP Raw Data Collector (`src/collection/inep_collector.py`)
- [ ] Step 6: Processador DuckDB para INEP
- [ ] Step 7: Processador Dask para INEP
- [ ] Step 8: Processador Polars para INEP
- [ ] Step 9: Setup ML para INEP (3 arquiteturas)
- [ ] Step 10: Pipeline CLI com --dataset
- [ ] Step 11: Crossover Experiment
- [ ] Step 12: Testes
- [ ] Step 13: Documentação

Ver plano completo no agente de planejamento.

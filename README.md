# dw-vs-dl-dropout-prediction-latam

Framework open-source para benchmarking reprodutivel de arquiteturas de dados, comparando **tres paradigmas** (DuckDB, Dask, Polars), com **verificacao automatica de anti-leakage temporal**.

```mermaid
flowchart LR
    subgraph U["Upstream -- 1x"]
        A["World Bank API"] --> B["Coleta + Imputacao"]
    end

    B --> DW & DL & PL

    DW["DuckDB\nSQL -- schema-on-write"]:::dw
    DL["Dask\ndistributed -- schema-on-read"]:::dl
    PL["Polars\nlazy eval -- Arrow"]:::pl

    DW & DL & PL --> S

    subgraph D["Downstream -- Nx"]
        S["Setup ML"] --> G{{"Anti-Leak\nGate"}}:::gate
        G --> M["Baseline + Hierarquico"]
    end

    M --> V["Bootstrap CI + Effect Sizes"] --> T["LaTeX"]

    classDef dw fill:#1e88e5,stroke:#0d47a1,color:#fff,font-weight:bold
    classDef dl fill:#43a047,stroke:#1b5e20,color:#fff,font-weight:bold
    classDef pl fill:#fb8c00,stroke:#e65100,color:#fff,font-weight:bold
    classDef gate fill:#e53935,stroke:#b71c1c,color:#fff,font-weight:bold
```

## O problema

Leakage temporal e uma das principais causas de resultados irreplicaveis em machine learning aplicado a educacao. Kapoor & Narayanan (2023) auditaram 294 papers e encontraram leakage em uma parcela significativa deles. Em analytics educacional, o cenario e agravado pela escassez de validacao temporal rigorosa e pela ausencia de ferramentas que automatizem essa verificacao.

## O que este repositorio faz

Este framework fornece um **protocolo reutilizavel de benchmarking com verificacao anti-leakage** para pipelines de ML, demonstrado com dados publicos do Banco Mundial (32 paises, 2000-2023) para predicao de evasao escolar. A contribuicao principal e o protocolo, nao o resultado preditivo.

Como caso de uso, o pipeline processa os mesmos dados em tres fluxos de processamento distintos -- **DuckDB** (SQL analitico, schema-on-write), **Dask** (DataFrames distribuidos, schema-on-read) e **Polars** (lazy evaluation, schema-on-read) -- e verifica se o resultado preditivo e estatisticamente equivalente independente do backend. Isso testa se a escolha de paradigma de processamento introduz vies nos resultados de ML, uma pergunta que a literatura de analytics educacional nao aborda sistematicamente.

O pipeline executa coleta, processamento, treinamento e benchmark de ponta a ponta, com um **gate anti-leakage** que interrompe a execucao se qualquer fold violar integridade temporal. Inclui testes de injecao que deliberadamente tentam quebrar o gate para provar que ele funciona.

### Por que tres paradigmas?

A comparacao original (DuckDB vs Dask) cobria os extremos do espectro: SQL analitico in-process versus DataFrames distribuidos. Polars ocupa um nicho intermediario -- lazy evaluation single-machine com otimizacao de query plan, sem overhead de coordenacao distribuida (Dask) e sem linguagem SQL (DuckDB). Isso permite testar se a equivalencia preditiva se mantem nao apenas entre extremos, mas tambem em um paradigma que combina caracteristicas de ambos: schema-on-read como o Data Lake, mas execucao in-process como o Data Warehouse. Com N=3 paradigmas, a generalizacao da tese ("a arquitetura de processamento nao introduz vies nos resultados de ML") e mais robusta do que com N=2.

### Walk-forward temporal

O framework usa validacao walk-forward com gaps de 2 anos entre treino e teste, garantindo que nenhuma informacao futura contamine o modelo. O diagrama abaixo mostra como os 9 folds se distribuem ao longo de 23 anos de dados:

```mermaid
gantt
    title Walk-Forward Temporal (9 folds, gap = 2 anos)
    dateFormat YYYY
    axisFormat %Y
    todayMarker off

    section Fold 1
    Treino 2000-2005      :done, f1t, 2000, 2006
    Gap                   :crit, f1g, 2006, 2008
    Val 2008              :active, f1v, 2008, 2009
    Teste 2009            :f1e, 2009, 2010

    section Fold 2
    Treino 2000-2007      :done, f2t, 2000, 2008
    Gap                   :crit, f2g, 2008, 2010
    Val 2010              :active, f2v, 2010, 2011
    Teste 2011            :f2e, 2011, 2012

    section Fold 3
    Treino 2000-2009      :done, f3t, 2000, 2010
    Gap                   :crit, f3g, 2010, 2012
    Val 2012              :active, f3v, 2012, 2013
    Teste 2013            :f3e, 2013, 2014

    section Fold 5
    Treino 2000-2013      :done, f5t, 2000, 2014
    Gap                   :crit, f5g, 2014, 2016
    Val 2016              :active, f5v, 2016, 2017
    Teste 2017            :f5e, 2017, 2018

    section Fold 9
    Treino 2000-2017      :done, f9t, 2000, 2018
    Gap                   :crit, f9g, 2018, 2020
    Val 2020              :active, f9v, 2020, 2021
    Teste 2021-2023       :f9e, 2021, 2024
```

### Protocolo anti-leakage (P1-P5)

```mermaid
stateDiagram-v2
    direction LR

    state "Dados Temporais" as INPUT
    state fork_state <<fork>>
    state join_state <<join>>

    state "P1: Ordenacao\ntemporal" as P1
    state "P2: Gap\nminimo 2a" as P2
    state "P3: Separacao\nde features" as P3
    state "P4: Selecao no\nescopo do treino" as P4
    state "P5: Scaling/imput.\nso no treino" as P5

    state check <<choice>>

    state "Pipeline ML\n(baseline + hierarquico)" as ML
    state "ValueError!\nExecucao interrompida" as FAIL

    INPUT --> fork_state
    fork_state --> P1
    fork_state --> P2
    fork_state --> P3
    fork_state --> P4
    fork_state --> P5
    P1 --> join_state
    P2 --> join_state
    P3 --> join_state
    P4 --> join_state
    P5 --> join_state
    join_state --> check
    check --> ML : Todas OK
    check --> FAIL : Qualquer violacao
```

### DuckDB vs Dask vs Polars: o que cada um faz

```mermaid
block-beta
    columns 4

    space header["Mesmo dado, mesmo modelo, backends diferentes"] space space

    block:dw:1
        columns 1
        dw_title["Data Warehouse"]
        dw1["DuckDB in-process"]
        dw2["SQL views (zero I/O)"]
        dw3["Schema-on-write"]
        dw4["Buffer pool implicito"]
        dw5["LAG() window functions"]
    end

    block:shared:1
        columns 1
        sh_title["Compartilhado"]
        sh1["World Bank API"]
        sh2["9 walk-forward folds"]
        sh3["Ridge + Random Forest"]
        sh4["Anti-leakage gate"]
        sh5["SESOI + Bootstrap CI"]
    end

    block:dl:1
        columns 1
        dl_title["Data Lake"]
        dl1["Dask distributed"]
        dl2["Parquet materializado"]
        dl3["Schema-on-read"]
        dl4[".persist() explicito"]
        dl5["merge() para lags"]
    end

    block:pl:1
        columns 1
        pl_title["Polars DataFrame"]
        pl1["Polars lazy engine"]
        pl2["Schema-on-read (Parquet)"]
        pl3["In-process, single-machine"]
        pl4["Expressions idiomaticas"]
        pl5["Query plan optimizer"]
    end

    style header fill:transparent,stroke:none,color:#333
    style dw_title fill:#42a5f5,stroke:#1565c0,color:#fff
    style dl_title fill:#66bb6a,stroke:#2e7d32,color:#fff
    style pl_title fill:#ffa726,stroke:#e65100,color:#fff
    style sh_title fill:#ff9800,stroke:#e65100,color:#fff
    style dw fill:#e3f2fd,stroke:#1565c0
    style shared fill:#fff3e0,stroke:#e65100
    style dl fill:#e8f5e9,stroke:#2e7d32
    style pl fill:#ffe0b2,stroke:#e65100
```

### Separacao upstream / downstream

O benchmark separa o pipeline em duas camadas:

- **Upstream** (1x) -- coleta e processamento produzem dados deterministicos identicos em toda execucao. Repeti-los N vezes apenas desperdicaria tempo com chamadas HTTP e I/O sem adicionar informacao estatistica.

- **Downstream** (Nx) -- setup, baseline e hierarchical contem a logica arquitetural que diferencia os paradigmas. Sao repetidos N vezes para derivar intervalos de confianca e effect sizes.

### Garantias de fairness no benchmark

Cada iteracao do benchmark aplica 4 controles:

1. **Ordem randomizada** -- `random.Random(42)` decide a ordem DW/DL/PL a cada iteracao. Elimina vies sistematico de OS page cache.
2. **gc.collect()** -- garbage collection forcado entre cada execucao de arquitetura e entre fases. Evita que objetos residuais beneficiem ou prejudiquem a proxima.
3. **Feature set unificado** -- todas as arquiteturas entram no filtro de colinearidade com o mesmo conjunto base de features.
4. **Amostragem normalizada** -- a matriz de correlacao e computada sobre a mesma populacao de linhas completas em todos os paradigmas.

### Garantias do pipeline

- **Anti-leakage automatico (P1-P5)** em todas as 3 arquiteturas -- ordenacao temporal (P1), gap minimo de 2 anos (P2), separacao de features e deteccao de proxy (P3), escopo temporal da selecao de features (P4), e escopo de preprocessing com scaling/imputacao ajustados exclusivamente no treino (P5). Violacoes de P1-P4 geram `ValueError` e interrompem a execucao; P5 e enforced por contrato e testes unitarios. Cobre as principais categorias de leakage identificadas por Kapoor & Narayanan (2023).
- **HPO sem contaminacao** -- hiperparametros selecionados via grid search no conjunto de validacao; modelo final retreinado no treino completo. Previne leakage por otimizacao no conjunto de teste.
- **Equivalencia estatistica, nao p-hacking** -- comparacao arquitetural via SESOI + IC 95% por bootstrap, com Wilcoxon e Hodges-Lehmann como suporte. Limiares SESOI definidos a priori: R2 = 0.01, MASE = 0.05, WAPE = 0.05 (Lakens et al., 2018).
- **Reprodutibilidade integral** -- seeds centralizadas, `n_jobs=1`, snapshot de ambiente (packages, hardware, git commit) e 80 testes automatizados.
- **Extensivel por design** -- `BaseArchitectureML` (11 metodos abstratos, Template Method) com auto-descoberta de paradigmas via `__init_subclass__`; novas arquiteturas sao registradas automaticamente ao serem importadas, sem editar codigo existente.

### Limitacoes explicitas

Os dados sao macro-educacionais (agregados por pais/ano), nao logs individuais de alunos. O walk-forward com gaps de 2 anos produz n=9 folds, o maximo sem comprometer o anti-leakage temporal. Isso limita o poder do Wilcoxon pareado (~30% para efeitos medios), por isso a decisao primaria usa bootstrap CI e o Wilcoxon e complemento de robustez. Um resultado "inconclusivo" e esperado e reflete a precisao disponivel, nao falha metodologica (Lakens et al., 2018).

## Quickstart

```bash
git clone https://github.com/anonymous/dw-vs-dl-dropout-prediction-latam.git
cd dw-vs-dl-dropout-prediction-latam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pipeline completo: coleta -> validacao -> benchmark -> artefatos LaTeX
python pipeline.py

# Testes (80 testes)
pytest tests/
```

## Estrutura do projeto

```
src/
├── core/                    # Base do framework
│   ├── base_architecture.py # Classe abstrata (Template Method, 11 metodos)
│   ├── paradigm_registry.py # Auto-descoberta de paradigmas via __init_subclass__
│   ├── validation.py        # TemporalValidator + DataIntegrityValidator
│   ├── scientific_config.py # Parametros centralizados (gaps, SESOI, seeds)
│   └── models/baseline.py   # Estrategias de modelos baseline (Ridge, RF)
├── collection/              # Coleta e processamento de dados brutos
│   ├── raw_data_collector.py
│   ├── data_lake/           # Processador Dask (schema-on-read)
│   ├── data_warehouse/      # Processador DuckDB (schema-on-write)
│   └── polars_dataframe/    # Processador Polars (lazy evaluation)
├── architectures_ml/        # Implementacoes por arquitetura
│   ├── data_lake/           # Setup ML + modelos hierarquicos (Ridge, RF)
│   ├── data_warehouse/      # Setup ML + modelos hierarquicos (Ridge, RF)
│   └── polars_dataframe/    # Setup ML + modelos hierarquicos (Ridge, RF)
├── benchmarking/            # Instrumentacao e derivacao de metricas
└── statistical_validation/  # Equivalencia, bootstrap, effect sizes, scorecard
tests/
├── test_unit_core.py        # Testes unitarios (transforms, folds, anti-leakage)
├── test_framework_discovery.py # Testes de auto-descoberta de paradigmas
├── test_dataset_config.py   # Testes de configuracao de datasets
└── test_lag_anti_leak.py    # Testes de integridade temporal
pipeline.py                  # Orquestra tudo
```

### Estrutura de outputs

```
outputs/
├── collection/
│   ├── raw_data/                          # Dados brutos
│   ├── data_lake/                         # Parquet particionado (Dask)
│   ├── data_warehouse/                    # DuckDB + Parquet
│   └── polars_dataframe/                  # Parquet (Polars)
├── ml_pipeline/
│   └── architectures/
│       ├── data_lake/prep/                # Folds, features, modelos DL
│       ├── data_warehouse/prep/           # Folds, features, modelos DW
│       └── polars_dataframe/prep/         # Folds, features, modelos PL
├── benchmarks/
│   ├── architectural_benchmark_results.csv
│   ├── architectural_benchmark_resource_log.jsonl
│   └── architectural_benchmark_summary.json
└── statistics/
    ├── effect_sizes_summary.csv/json
    ├── significance_summary.csv/json
    ├── equivalence_estimation.json/tex
    └── architectural_scorecard.tex
```

## Como adaptar para seu dominio

1. **Nova arquitetura** -- crie uma subclasse de `BaseArchitectureML` em `src/architectures_ml/<novo>/setup.py` com `PARADIGM_META` definido. O framework descobre automaticamente via `__init_subclass__` -- nenhum arquivo existente precisa ser editado.

2. **Novos parametros** -- edite `src/core/scientific_config.py`: gaps temporais, limiares SESOI, embargo, bootstrap iterations.

3. **Novas metricas** -- estenda `src/benchmarking/` ou `src/statistical_validation/` seguindo o padrao de entrada/saida JSON -> LaTeX dos scripts existentes.

4. **Outro dominio** -- ajuste os indicadores no coletor de dados e os limiares SESOI. Os protocolos permanecem os mesmos.

Para detalhes operacionais, veja o [`USAGE_GUIDE.md`](USAGE_GUIDE.md).

---

**Contato**: [Removido para revisao double-blind]

# Diagrama do Pipeline — Data Warehouse vs Data Lake vs Polars DataFrame

Documentação técnica do fluxo completo do framework de benchmarking reprodutível. Mostra os três paradigmas (DuckDB, Dask e Polars) em paralelo, a separação upstream/downstream, e as garantias de fairness implementadas.

## Visão geral

```mermaid
graph TB
    classDef configClass fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    classDef collectionClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef dlClass fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef dwClass fill:#b3e5fc,stroke:#0277bd,stroke-width:2px
    classDef plClass fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    classDef benchClass fill:#ffccbc,stroke:#d84315,stroke-width:2px
    classDef validationClass fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef outputClass fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    classDef fairnessClass fill:#e8eaf6,stroke:#283593,stroke-width:2px

    %% ============================================================
    %% UPSTREAM — executa UMA vez (run_id=0)
    %% ============================================================
    subgraph UPSTREAM ["UPSTREAM — execução única"]
        direction TB

        CONFIG[/"scientific_config.py\n• Seeds centralizadas\n• Gaps temporais (2a)\n• Limiares SESOI\n• Parâmetros de benchmark"/]:::configClass

        SNAPSHOT["Snapshot Científico\nscientific_config_snapshot.json\n• Timestamp + Git commit\n• Ambiente Python + Hardware"]:::configClass

        CONFIG --> SNAPSHOT

        COLLECT["FASE 1 · Coleta\nraw_data_collector.py\n• World Bank API\n• 32 países, 2000-2023\n• Cache local < 24h"]:::collectionClass

        SNAPSHOT --> COLLECT

        RAWDATA[("raw_data/\ncomplete_data.parquet")]:::collectionClass

        COLLECT --> RAWDATA

        subgraph PROC ["FASE 2 · Processamento Arquitetural"]
            direction LR

            subgraph DL_PROC_G ["Data Lake"]
                direction TB
                DL_PROC["processor.py\nDask · schema-on-read\n• Lazy evaluation\n• Parquet particionado"]:::dlClass
                DL_DATA[("data_lake/\nParquet")]:::dlClass
                DL_PROC --> DL_DATA
            end

            subgraph DW_PROC_G ["Data Warehouse"]
                direction TB
                DW_PROC["processor.py\nDuckDB · schema-on-write\n• SIMD vetorizado\n• SQL transacional"]:::dwClass
                DW_DATA[("data_warehouse/\nDuckDB + Parquet")]:::dwClass
                DW_PROC --> DW_DATA
            end

            subgraph PL_PROC_G ["Polars DataFrame"]
                direction TB
                PL_PROC["processor.py\nPolars · lazy evaluation\n• Query plan optimizer\n• In-process engine"]:::plClass
                PL_DATA[("polars_dataframe/\nParquet")]:::plClass
                PL_PROC --> PL_DATA
            end
        end

        RAWDATA --> DL_PROC
        RAWDATA --> DW_PROC
        RAWDATA --> PL_PROC
    end

    %% ============================================================
    %% DOWNSTREAM — repete 30x (+ 2 warmup descartados)
    %% ============================================================
    subgraph DOWNSTREAM ["DOWNSTREAM — 30 repetições + 2 warmup"]
        direction TB

        FAIR["Fairness por iteração\n• Ordem DL/DW randomizada (seed=42)\n• gc.collect() entre fases\n• Feature set unificado"]:::fairnessClass

        subgraph SETUP_G ["FASE 3 · Setup ML"]
            direction LR

            BASE["BaseArchitectureML\n11 métodos abstratos\nTemplate Method\nEnforcement anti-leakage"]:::configClass

            subgraph DL_ML_G ["ML Data Lake"]
                direction TB
                DL_SETUP["setup.py · Dask\n• .persist() após read_parquet\n• Lags via merge()\n• Folds → Parquet"]:::dlClass
                DL_FOLDS[("temporal_folds_data_lake.json\nfeature_selection_data_lake.json")]:::dlClass
                DL_SETUP --> DL_FOLDS
            end

            subgraph DW_ML_G ["ML Data Warehouse"]
                direction TB
                DW_SETUP["setup.py · DuckDB\n• SQL views (zero I/O)\n• LAG() window functions\n• Folds → Views"]:::dwClass
                DW_FOLDS[("temporal_folds_data_warehouse.json\nfeature_selection_data_warehouse.json")]:::dwClass
                DW_SETUP --> DW_FOLDS
            end

            subgraph PL_ML_G ["ML Polars DataFrame"]
                direction TB
                PL_SETUP["setup.py · Polars\n• .lazy() expressions\n• Lags via shift()\n• Folds → Parquet"]:::plClass
                PL_FOLDS[("temporal_folds_polars_dataframe.json\nfeature_selection_polars_dataframe.json")]:::plClass
                PL_SETUP --> PL_FOLDS
            end

            BASE -.-> DL_SETUP
            BASE -.-> DW_SETUP
            BASE -.-> PL_SETUP
        end

        FAIR --> SETUP_G

        GATE{"GATE ANTI-LEAKAGE\nTemporalValidator.enforce_walk_forward()\n• P1: Ordenação temporal\n• P2: Gap mínimo 2 anos\n• P3: Separação de features\n• P4: Seleção no escopo do treino\n• P5: Scaling/imputação só no treino\nraise ValueError se violado"}:::validationClass

        DL_FOLDS --> GATE
        DW_FOLDS --> GATE
        PL_FOLDS --> GATE

        subgraph BASELINE_G ["FASE 4 · Modelos Baseline"]
            direction LR
            DL_BASE["baseline_analysis.py\nData Lake (.persist())\n• Média histórica\n• Tendência linear\n• Naive · Cross-country"]:::dlClass
            DW_BASE["baseline_analysis.py\nData Warehouse (SQL)\n• Média histórica\n• Tendência linear\n• Naive · Cross-country"]:::dwClass
            PL_BASE["baseline_analysis.py\nPolars (.collect())\n• Média histórica\n• Tendência linear\n• Naive · Cross-country"]:::plClass
        end

        GATE --> DL_BASE
        GATE --> DW_BASE
        GATE --> PL_BASE

        subgraph HIER_G ["FASE 5 · Modelos Hierárquicos"]
            direction LR
            DL_HIER["hierarchical_models.py\nData Lake (.persist())\n• Ridge hierárquico\n• Random Forest\n• James-Stein shrinkage"]:::dlClass
            DW_HIER["hierarchical_models.py\nData Warehouse (SQL-first)\n• Ridge hierárquico\n• Random Forest\n• James-Stein shrinkage"]:::dwClass
            PL_HIER["hierarchical_models.py\nPolars (.collect())\n• Ridge hierárquico\n• Random Forest\n• James-Stein shrinkage"]:::plClass
        end

        DL_BASE --> DL_HIER
        DW_BASE --> DW_HIER
        PL_BASE --> PL_HIER
    end

    DL_DATA --> DL_SETUP
    DW_DATA --> DW_SETUP
    PL_DATA --> PL_SETUP
    CONFIG --> BASE

    %% ============================================================
    %% BENCHMARK + VALIDAÇÃO
    %% ============================================================
    BENCH["FASE 6 · Benchmark Arquitetural\narchitectural_benchmark.py\n• perf_counter_ns()\n• psutil (CPU/RAM/I/O)\n• 30 repetições + 2 warmup\n• Ordem randomizada por iteração"]:::benchClass

    DL_HIER --> BENCH
    DW_HIER --> BENCH
    PL_HIER --> BENCH
    CONFIG --> BENCH

    BENCH_OUT[("benchmarks/\n• architectural_benchmark_results.csv\n• architectural_benchmark_resource_log.jsonl\n• architectural_benchmark_summary.json")]:::benchClass

    BENCH --> BENCH_OUT

    subgraph STATS_G ["FASE 7 · Validação Estatística"]
        direction TB
        EFFECT["effect_analysis.py\n• Cohen's dz pareado\n• Interpretação prática"]:::validationClass
        SIGNIF["significance_tests.py\n• Wilcoxon pareado\n• Bootstrap CI 95%\n• Hodges-Lehmann"]:::validationClass
        EQUIV["equivalence_estimation.py\n• TOST por bootstrap\n• SESOI: R²=0.01, MASE/WAPE=0.05\n• Sensibilidade 0.5x–1.5x"]:::validationClass
        SCORE["make_scorecard.py\n• Tabelas LaTeX\n• Métricas comparativas"]:::validationClass

        EFFECT --> SCORE
        SIGNIF --> SCORE
        EQUIV --> SCORE
    end

    BENCH_OUT --> EFFECT
    BENCH_OUT --> SIGNIF
    BENCH_OUT --> EQUIV

    FINAL_OUT[("statistics/\n• effect_sizes_summary.csv/json\n• significance_summary.csv/json\n• equivalence_estimation.json/tex\n• architectural_scorecard.tex")]:::outputClass

    SCORE --> FINAL_OUT

    AUTOGATE["Framework: auto-discovery + gate anti-leakage\n• Folds gerados deterministicamente por arquitetura\n• TemporalValidator.enforce_walk_forward() valida cada fold\n• Divergências interrompem execução via ValueError"]:::validationClass

    DL_FOLDS --> AUTOGATE
    DW_FOLDS --> AUTOGATE
    PL_FOLDS --> AUTOGATE
    AUTOGATE --> FINAL_OUT
```

## Separação upstream / downstream

O benchmark separa o pipeline em duas camadas com justificativas distintas:

```mermaid
flowchart LR
    subgraph U ["Upstream (1x)"]
        direction TB
        C["Coleta\nWorld Bank API"]
        P["Processamento\nDL + DW + PL"]
        C --> P
    end

    subgraph D ["Downstream (30x + 2 warmup)"]
        direction TB
        S["Setup ML"]
        B["Baseline"]
        H["Hierárquico"]
        S --> B --> H
    end

    U --> D

    style U fill:#fff9c4,stroke:#f57f17
    style D fill:#ffccbc,stroke:#d84315
```

**Upstream** — coleta e processamento produzem dados determinísticos idênticos em toda execução. Repetí-los N vezes apenas desperdiça tempo com chamadas HTTP e I/O sem adicionar informação estatística. O cache local (`_cache_is_valid()`) evita chamadas redundantes à API do World Bank quando os dados já existem e têm menos de 24 horas.

**Downstream** — setup, baseline e hierarchical contêm a lógica arquitetural que diferencia DW e DL. São o alvo real do benchmark e por isso são repetidos N vezes para poder derivar intervalos de confiança e effect sizes.

## Garantias de fairness no benchmark

```mermaid
stateDiagram-v2
    direction LR

    state "Início da\niteração i" as START
    state fork_fair <<fork>>
    state join_fair <<join>>

    state "Randomizar ordem\nDL/DW/PL (seed=42)" as RAND
    state "gc.collect()\nentre fases" as GC
    state "Feature set\nunificado" as FEAT
    state "Amostragem\nnormalizada" as SAMP

    state "Medir fase\n(perf_counter_ns)" as MEASURE

    START --> fork_fair
    fork_fair --> RAND
    fork_fair --> GC
    fork_fair --> FEAT
    fork_fair --> SAMP
    RAND --> join_fair
    GC --> join_fair
    FEAT --> join_fair
    SAMP --> join_fair
    join_fair --> MEASURE
```

Cada iteração do benchmark aplica 4 controles de fairness:

**Ordem randomizada** — a cada iteração, `random.Random(42)` decide se DL ou DW executa primeiro. Elimina viés sistemático de OS page cache onde uma arquitetura sempre se beneficiaria do I/O prévio da outra. O seed é fixo para reprodutibilidade.

**gc.collect()** — garbage collection forçado entre cada execução de arquitetura e entre fases. Evita que objetos Python residuais de uma arquitetura fiquem em memória beneficiando ou prejudicando a próxima.

**Feature set unificado** — lag features (`dropout_rate_lag_2/3`) são excluídas do `get_numeric_features()` em ambas as arquiteturas. O DL as criava em `create_target_implementation` (antes do filtro de colinearidade) e o DW só as criava em `prepare_features` (depois), gerando conjuntos iniciais de 23 vs 21 features. Agora ambos entram com o mesmo conjunto base no filtro.

**Amostragem normalizada** — o filtro de colinearidade do DW aplicava `IS NOT NULL` apenas nas primeiras 10 features antes de amostrar. Agora filtra em todas as features candidatas, equivalente ao `.dropna()` que o DL faz. Ambos computam a matriz de correlação sobre a mesma população de linhas completas.

## Protocolo anti-leakage (P1-P5)

```mermaid
flowchart TB
    classDef pass fill:#c8e6c9,stroke:#2e7d32
    classDef fail fill:#ffcdd2,stroke:#c62828
    classDef check fill:#f3e5f5,stroke:#6a1b9a

    INPUT["Dados temporais\n(country, year, features)"]

    P1["P1 · Ordenação temporal\ntrain.year.max < val.year.min\nval.year.max < test.year.min"]:::check
    P2["P2 · Gap mínimo\nval.year.min - train.year.max >= 2"]:::check
    P3["P3 · Separação de features\nProxy detection\nLag features isolados"]:::check
    P4["P4 · Seleção temporal\nCorrelação/colinearidade\nsomente em dados <= train_end"]:::check
    P5["P5 · Preprocessing\nScaling/imputação ajustados\nexclusivamente no treino"]:::check

    OK["Pipeline ML prossegue"]:::pass
    FAIL["ValueError!\nExecução interrompida"]:::fail

    INPUT --> P1 --> P2 --> P3 --> P4 --> P5
    P5 -->|Todas OK| OK
    P1 -->|Violação| FAIL
    P2 -->|Violação| FAIL
    P3 -->|Violação| FAIL
    P4 -->|Violação| FAIL
    P5 -->|Violação| FAIL
```

O gate valida cada fold individualmente antes de liberar para os modelos. Cobre as categorias L1.1-L1.4, L2 e L3.2-L3.3 da taxonomia de Kapoor & Narayanan (2023). P1-P4 geram `ValueError` em runtime; P5 é enforced por contrato e testes unitários. A injeção de leakage (`test_leakage_injection.py`, cenários S1-S4) verifica que o gate detecta violações deliberadas.

## Comparação arquitetural

```mermaid
block-beta
    columns 4

    block:dw:1
        columns 1
        dw_t["Data Warehouse (DuckDB)"]
        dw1["In-process SQL engine"]
        dw2["Schema-on-write (ACID)"]
        dw3["Views para folds (zero I/O)"]
        dw4["LAG() para features temporais"]
        dw5["information_schema para tipos"]
        dw6["CORR() pairwise via SQL"]
        dw7["Buffer pool implícito"]
    end

    block:shared:1
        columns 1
        sh_t["Compartilhado"]
        sh1["World Bank API (cache 24h)"]
        sh2["9 walk-forward folds"]
        sh3["Gap 2 anos, embargo"]
        sh4["Ridge + Random Forest"]
        sh5["Filtro colinearidade (0.8)"]
        sh6["Symmetric log transform"]
        sh7["SESOI + Bootstrap CI"]
    end

    block:pl:1
        columns 1
        pl_t["Polars DataFrame"]
        pl1["In-process lazy engine"]
        pl2["Schema-on-read (Parquet)"]
        pl3["LazyFrame → collect()"]
        pl4["Expressions para lags"]
        pl5["select_dtypes via schema"]
        pl6[".lazy().collect() pipeline"]
        pl7["Query plan optimization"]
    end

    block:dl:1
        columns 1
        dl_t["Data Lake (Dask)"]
        dl1["Distributed scheduler"]
        dl2["Schema-on-read (Parquet)"]
        dl3["Parquet por fold (I/O)"]
        dl4["merge() para lags"]
        dl5["select_dtypes() inference"]
        dl6[".corr() via Pandas sample"]
        dl7[".persist() explícito"]
    end

    style dw_t fill:#42a5f5,stroke:#1565c0,color:#fff
    style sh_t fill:#ff9800,stroke:#e65100,color:#fff
    style pl_t fill:#ffa726,stroke:#e65100,color:#fff
    style dl_t fill:#66bb6a,stroke:#2e7d32,color:#fff
    style dw fill:#e3f2fd,stroke:#1565c0
    style shared fill:#fff3e0,stroke:#e65100
    style pl fill:#fff3e0,stroke:#e65100
    style dl fill:#e8f5e9,stroke:#2e7d32
```

## Resultados de referência (Azure L4as_v4, 32 GB RAM, n=30)

```mermaid
xychart-beta
    title "Latência por fase (segundos)"
    x-axis ["Setup", "Processing", "Baseline", "Hierarchical"]
    y-axis "Tempo (s)" 0 --> 200
    bar [0.41, 0.18, 1.191, 15.25]
    bar [0.054, 0.016, 1.076, 15.38]
    bar [179.01, 0.90, 7.793, 16.96]
```

| Fase | DuckDB (s) | Polars (s) | Dask (s) | Ratio | Cohen's dz |
|------|-----------|-----------|---------|-------|-----------|
| Setup (n=30) | 0.41 ± 0.01 | 0.054 ± 0.001 | 179.01 ± 0.27 | **437x** | — |
| Processing (n=1) | 0.18 | 0.016 | 0.90 | **56x** | — |
| Baseline (n=30) | 1.191 ± 0.009 | 1.076 ± 0.008 | 7.793 ± 0.019 | **7x** | — |
| Hierarchical (n=30) | 15.25 ± 0.03 | 15.38 ± 0.02 | 16.96 ± 0.03 | **1.1x** | — |
| **Total** | **17.031** | **16.526** | **204.663** | **12x** | — |

Ambiente: Azure Standard_L4as_v4 (4 vCPUs, 32 GB RAM, NVMe), Ubuntu 22.04, Python 3.10. IC 95% via t-distribution. Processing executa uma vez (upstream); demais fases repetidas 30 vezes com 2 warmup descartados. Ordem DL/DW/PL randomizada por iteração com `gc.collect()` entre execuções.

## Legenda

### Código de cores

- **Azul claro** — configuração e reprodutibilidade
- **Amarelo** — coleta de dados brutos (upstream)
- **Verde** — pipeline Data Lake (Dask)
- **Azul** — pipeline Data Warehouse (DuckDB)
- **Âmbar** — pipeline Polars DataFrame
- **Laranja** — benchmarking e instrumentação
- **Roxo** — validação estatística e anti-leakage
- **Índigo** — controles de fairness

### Estrutura de outputs

```
outputs/
├── scientific_config_snapshot.json
├── collection/
│   ├── raw_data/                          # Dados brutos do Banco Mundial
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
    ├── significance_summary.csv/json/md
    ├── equivalence_estimation.json/tex
    └── architectural_scorecard.tex
```

### Execução

```bash
# Pipeline completo (7 fases)
python pipeline.py

# Repetições e warmup configurados em scientific_config.py
# (padrão: 30 repetições downstream + 2 warmup descartados)
python pipeline.py
```

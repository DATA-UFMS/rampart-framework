# dw-vs-dl-dropout-prediction-latam

Framework open-source para benchmarking reprodutível de arquiteturas de dados, com **verificação automática de anti-leakage temporal**.

```mermaid
---
config:
  theme: base
  themeVariables:
    primaryColor: "#4a90d9"
    primaryTextColor: "#fff"
    primaryBorderColor: "#2c5f8a"
    secondaryColor: "#f5a623"
    tertiaryColor: "#7ed321"
    lineColor: "#5c6370"
    fontSize: "14px"
---
flowchart LR
    subgraph upstream ["Upstream (1x)"]
        direction LR
        A["World Bank\nAPI"]
        B["Coleta +\nImputação"]
        A --> B
    end

    subgraph fork [" "]
        direction TB
        DL["Data Lake\n Dask "]:::dask
        DW["Data Warehouse\n DuckDB "]:::duck
    end

    subgraph downstream ["Downstream (35x)"]
        direction LR
        S["Setup\nML"]
        G{{"Anti-Leak\nGate"}}:::gate
        M["Baseline +\nHierárquico"]
        S --> G --> M
    end

    subgraph val ["Validação"]
        direction LR
        V["Bootstrap CI\n+ Effect Sizes"]
        T["Tabelas\nLaTeX"]
        V --> T
    end

    B --> DL & DW
    DL & DW --> S
    M --> V

    classDef dask fill:#66bb6a,stroke:#2e7d32,color:#fff
    classDef duck fill:#42a5f5,stroke:#1565c0,color:#fff
    classDef gate fill:#ef5350,stroke:#b71c1c,color:#fff
```

## O problema

Leakage temporal é uma das principais causas de resultados irreplicáveis em machine learning aplicado a educação. Kapoor & Narayanan (2023) auditaram 294 papers e encontraram leakage em uma parcela significativa deles. Em analytics educacional, o cenário é agravado pela escassez de validação temporal rigorosa e pela ausência de ferramentas que automatizem essa verificação.

## O que este repositório faz

Este framework fornece um **protocolo reutilizável de benchmarking com verificação anti-leakage** para pipelines de ML, demonstrado com dados públicos do Banco Mundial (32 países, 2000-2023) para predição de evasão escolar. A contribuição principal é o protocolo, não o resultado preditivo.

Como caso de uso, o pipeline processa os mesmos dados em dois fluxos de processamento distintos — **DuckDB** (SQL analítico, schema-on-write) e **Dask** (DataFrames distribuídos, schema-on-read) — e verifica se o resultado preditivo é estatisticamente equivalente independente do backend. Isso testa se a escolha de paradigma de processamento introduz viés nos resultados de ML, uma pergunta que a literatura de analytics educacional não aborda sistematicamente.

O pipeline executa coleta, processamento, treinamento e benchmark de ponta a ponta, com um **gate anti-leakage** que interrompe a execução se qualquer fold violar integridade temporal. Inclui testes de injeção que deliberadamente tentam quebrar o gate para provar que ele funciona.

### Walk-forward temporal

O framework usa validação walk-forward com gaps de 2 anos entre treino e teste, garantindo que nenhuma informação futura contamine o modelo. O diagrama abaixo mostra como os 9 folds se distribuem ao longo de 23 anos de dados:

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

    state "P1: Ordenação\ntemporal" as P1
    state "P2: Gap\nmínimo 2a" as P2
    state "P3: Separação\nde features" as P3
    state "P4: Seleção no\nescopo do treino" as P4
    state "P5: Scaling/imput.\nsó no treino" as P5

    state check <<choice>>

    state "Pipeline ML\n(baseline + hierárquico)" as ML
    state "ValueError!\nExecução interrompida" as FAIL

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
    check --> FAIL : Qualquer violação
```

### DuckDB vs Dask: o que cada um faz

```mermaid
block-beta
    columns 3

    space header["Mesmo dado, mesmo modelo, backends diferentes"] space

    block:dw:1
        columns 1
        dw_title["Data Warehouse"]
        dw1["DuckDB in-process"]
        dw2["SQL views (zero I/O)"]
        dw3["Schema-on-write"]
        dw4["Buffer pool implícito"]
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
        dl4[".persist() explícito"]
        dl5["merge() para lags"]
    end

    style header fill:transparent,stroke:none,color:#333
    style dw_title fill:#42a5f5,stroke:#1565c0,color:#fff
    style dl_title fill:#66bb6a,stroke:#2e7d32,color:#fff
    style sh_title fill:#ff9800,stroke:#e65100,color:#fff
    style dw fill:#e3f2fd,stroke:#1565c0
    style shared fill:#fff3e0,stroke:#e65100
    style dl fill:#e8f5e9,stroke:#2e7d32
```

### Benchmark (Azure D4s_v3, n=35)

```mermaid
---
config:
  themeVariables:
    xyChart:
      backgroundColor: transparent
---
xychart-beta
    title "Latência por fase: DuckDB vs Dask (segundos, log scale)"
    x-axis ["Setup", "Processing", "Baseline", "Hierarchical"]
    y-axis "Tempo (s)" 0 --> 500
    bar [0.83, 0.32, 3.81, 43.22]
    bar [478.18, 2.50, 21.27, 47.83]
```

> DuckDB (azul) vs Dask (laranja). Setup: **574x**. Baseline: **6x**. Hierarchical: **1.1x**. Total: **11x**.

### Garantias do pipeline

- **Anti-leakage automático (P1-P5)** — ordenação temporal (P1), gap mínimo de 2 anos (P2), separação de features e detecção de proxy (P3), escopo temporal da seleção de features (P4), e escopo de preprocessing com scaling/imputação ajustados exclusivamente no treino (P5). Violações de P1-P4 geram `ValueError` e interrompem a execução; P5 é enforced por contrato e testes unitários. Cobre as categorias L1.1-L1.4, L2 e L3.2 da taxonomia de Kapoor & Narayanan (2023) e as 4 variantes de Semmelrock et al. (2025).
- **HPO sem contaminação** — hiperparâmetros selecionados via grid search no conjunto de validação; modelo final retreinado no treino completo. Previne leakage L3.3.
- **Equivalência estatística, não p-hacking** — comparação arquitetural via SESOI + IC 95% por bootstrap, com Wilcoxon e Hodges-Lehmann como suporte. Limiares SESOI definidos a priori: R² = 0.01 (metade do efeito pequeno de Cohen 1988), MASE = 0.05, WAPE = 0.05 (resolução prática de decisão, Lakens et al. 2018).
- **Reprodutibilidade integral** — seeds centralizadas, `n_jobs=1`, snapshot de ambiente (packages, hardware, git commit) e 51 testes automatizados + 4 cenários de injeção de leakage.
- **Extensível por design** — `BaseArchitectureML` (11 métodos abstratos, Template Method) permite adicionar novas arquiteturas herdando o enforcement anti-leakage automaticamente.

### Limitações explícitas

Os dados são macro-educacionais (agregados por país/ano), não logs individuais de alunos. O walk-forward com gaps de 2 anos produz n=9 folds, o máximo sem comprometer o anti-leakage temporal. Isso limita o poder do Wilcoxon pareado (~30% para efeitos médios), por isso a decisão primária usa bootstrap CI e o Wilcoxon é complemento de robustez. Um resultado "inconclusivo" é esperado e reflete a precisão disponível, não falha metodológica (Lakens et al. 2018). Expomos essas limitações deliberadamente.

## Quickstart

```bash
git clone https://github.com/DATA-UFMS/dw-vs-dl-dropout-prediction-latam.git
cd dw-vs-dl-dropout-prediction-latam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pipeline completo: coleta → validação → benchmark → artefatos LaTeX
python pipeline.py

# Testes (51 unitários + 4 cenários de injeção de leakage)
pytest tests/test_unit_core.py tests/test_lag_anti_leak.py
python tests/test_leakage_injection.py
```

O pipeline gera todos os artefatos em `outputs/` — folds temporais, resultados de benchmark, tabelas LaTeX publication-ready e um snapshot completo do ambiente para replicação.

## Estrutura do projeto

```
src/
├── core/                    # Base do framework
│   ├── base_architecture.py # Classe abstrata (Template Method, 11 métodos)
│   ├── validation.py        # TemporalValidator + DataIntegrityValidator
│   ├── scientific_config.py # Parâmetros centralizados (gaps, SESOI, seeds)
│   └── models/baseline.py   # Estratégias RF, XGBoost, LightGBM
├── collection/              # Coleta e processamento de dados brutos
│   ├── raw_data_collector.py
│   ├── data_lake/           # Processador Dask (schema-on-read)
│   └── data_warehouse/      # Processador DuckDB (schema-on-write)
├── architectures_ml/        # Implementações por arquitetura
│   ├── data_lake/           # Setup ML + modelos hierárquicos (Ridge, RF)
│   └── data_warehouse/      # Setup ML + modelos hierárquicos (Ridge, RF)
├── benchmarking/            # Instrumentação e derivação de métricas
└── statistical_validation/  # TOST, bootstrap, effect sizes, scorecard
tests/
├── test_unit_core.py        # 49 testes unitários
├── test_lag_anti_leak.py    # 2 testes de integridade temporal
└── test_leakage_injection.py # Validação negativa do gate (S1–S4)
pipeline.py                  # Orquestra tudo
```

## Como adaptar para seu domínio

1. **Nova arquitetura** — crie uma subclasse de `BaseArchitectureML` em `src/architectures_ml/`. O anti-leakage é herdado. Registre no `pipeline.py`.

2. **Novos parâmetros** — edite `src/core/scientific_config.py`: gaps temporais, limiares SESOI (`sesoi_r2`, `sesoi_mase`, `sesoi_wape`), embargo, bootstrap iterations.

3. **Novas métricas** — estenda `src/benchmarking/` ou `src/statistical_validation/` seguindo o padrão de entrada/saída JSON → LaTeX dos scripts existentes.

4. **Outro domínio** — ajuste os indicadores no coletor de dados e os limiares SESOI. Os protocolos permanecem os mesmos.

Para detalhes operacionais, veja o [`USAGE_GUIDE.md`](USAGE_GUIDE.md). O fluxo completo do pipeline está em [`docs/pipeline_diagram.md`](docs/pipeline_diagram.md).

---

**Contato**: {eos.xavier, rosa.livia, vanessa.a.borges}@ufms.br — Faculdade de Computação, UFMS.

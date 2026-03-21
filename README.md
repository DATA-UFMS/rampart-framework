# dw-vs-dl-dropout-prediction-latam

Framework open-source para benchmarking reprodutível de arquiteturas de dados, com **verificação automática de anti-leakage temporal**.

## O problema

Leakage temporal é uma das principais causas de resultados irreplicáveis em machine learning aplicado a educação. Kapoor & Narayanan (2023) auditaram 294 papers e encontraram leakage em uma parcela significativa deles. Em analytics educacional, o cenário é agravado pela escassez de validação temporal rigorosa e pela ausência de ferramentas que automatizem essa verificação.

## O que este repositório faz

Este framework compara dois paradigmas de dados — **DuckDB** (schema-on-write) e **Dask** (schema-on-read) — usando dados públicos do Banco Mundial (32 países, 2000–2023) para predição de evasão escolar. Mas o objetivo principal não é declarar um vencedor: é fornecer um **protocolo reutilizável** que qualquer pesquisador pode adaptar para seu domínio.

O pipeline executa coleta, processamento, treinamento e benchmark de ponta a ponta, com um **gate anti-leakage** que interrompe a execução se qualquer fold violar integridade temporal. Inclui também testes de injeção que deliberadamente tentam quebrar o gate para provar que ele funciona.

### Garantias do pipeline

- **Anti-leakage automático** — o pipeline verifica ordenação temporal, gap mínimo (2 anos) e separação de features a cada fold. Violações geram `ValueError` e interrompem a execução. Suporte a embargo configurável (López de Prado, 2018).
- **Equivalência estatística, não p-hacking** — comparação arquitetural via SESOI + IC 95% por bootstrap, com Wilcoxon e Hodges–Lehmann como suporte.
- **Reprodutibilidade integral** — seeds centralizadas, `n_jobs=1`, snapshot de ambiente (packages, hardware, git commit) e 42 testes automatizados + 4 cenários de injeção de leakage.
- **Extensível por design** — `BaseArchitectureML` (11 métodos abstratos, Template Method) permite adicionar novas arquiteturas herdando o enforcement anti-leakage automaticamente.

### Limitações explícitas

Os dados são macro-educacionais (agregados por país/ano), não logs individuais de alunos. Isso limita a generalização dos resultados preditivos, mas não invalida o protocolo — que é o foco do trabalho. Expomos essa limitação deliberadamente.

## Quickstart

```bash
git clone https://github.com/DATA-UFMS/dw-vs-dl-dropout-prediction-latam.git
cd dw-vs-dl-dropout-prediction-latam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pipeline completo: coleta → validação → benchmark → artefatos LaTeX
python pipeline.py

# Testes (42 unitários + 4 cenários de injeção de leakage)
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
├── test_unit_core.py        # 40 testes unitários
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

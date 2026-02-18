# Framework Metodológico para Avaliação de Arquiteturas de Dados em Analytics Educacional

Este repositório materializa um **framework metodológico completo** para avaliar arquiteturas de dados aplicadas a analytics educacional com rigor científico. O trabalho responde a três questões centrais:

1. **RQ1 — Extensibilidade**: Como projetar uma arquitetura modular que minimize retrabalho ao incorporar novos paradigmas de dados?
2. **RQ2 — Recomendação Automática**: Quais mecanismos de instrumentação suportam recomendação automática do paradigma mais eficiente para um dado contexto?
3. **RQ3 — Reprodutibilidade**: Quais práticas de engenharia garantem reprodutibilidade integral dos experimentos?

A demonstração empírica compara dois paradigmas clássicos — schema-on-write (DuckDB) e schema-on-read (Dask) — em dados públicos do Banco Mundial (2000–2023). O objetivo não é declarar vencedores, mas validar o protocolo e exemplificar como transformar resultados em heurísticas transparentes. Como os indicadores são agregados por país/ano, tratamos o estudo como um benchmark metodológico com dados macro-educacionais, útil quando logs individuais não estão acessíveis e expomos explicitamente as limitações decorrentes dessa granularidade.

## 1. Protocolos Metodológicos

| Protocolo | Objetivo | Componentes-chave |
|-----------|----------|--------------------|
| Validação temporal + Anti-leakage | Evitar vazamentos temporais com enforcement automático (raise em violações) | `src/core/validation.py`, `src/core/base_architecture.py`, `pipeline.py` (gate), `tests/test_lag_anti_leak.py` |
| Equivalência prática | Estimar efeitos arquiteturais com SESOI + IC95% | `src/statistical_validation/tost_baseline.py`, `outputs/statistics/equivalence_estimation.*` |
| Benchmark arquitetural | Mensurar latência, throughput e uso de recursos | `src/benchmarking/architectural_benchmark.py`, `outputs/benchmarks/*` |
| Reprodutibilidade integral | Registrar parâmetros, seeds e artefatos auditáveis | `pipeline.py`, `src/core/scientific_config.py`, `src/core/logging_config.py` |

Os parâmetros científicos (gaps, SESOI, número de bootstrap) são centralizados em `src/core/scientific_config.py`, garantindo consistência entre arquiteturas.

## 2. Como Reproduzir a Demonstração

```bash
# 0. Clonar repositório e instalar dependências
git clone https://github.com/DATA-UFMS/dw-vs-dl-dropout-prediction-latam.git
cd dw-vs-dl-dropout-prediction-latam
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Executar pipeline completo (coleta → validação → benchmark → artefatos)
python pipeline.py

# 2. Rodar testes unitários e anti-leakage (42 testes)
pytest tests/test_unit_core.py tests/test_lag_anti_leak.py

# 3. (Re)gerar tabelas LaTeX a partir das saídas do benchmark
python src/benchmarking/derive_latency_percentiles.py
python src/benchmarking/derive_throughput_percentiles.py
python src/benchmarking/derive_resource_usage_table.py
```

O pipeline inclui um **gate anti-leakage** entre o setup ML e o benchmark: se qualquer fold violar a integridade temporal (ordenação, gap mínimo de 2 anos, ou separação de features), a execução é interrompida com `ValueError`. Seeds centralizadas e `n_jobs=1` garantem determinismo entre execuções.

## 3. Checklist de Reprodutibilidade

Verifique os itens abaixo após rodar o pipeline:

- `outputs/ml_pipeline/architectures/<arch>/prep/temporal_folds_<arch>.json` — fronteiras dos folds com gaps anti-leakage verificados (RQ1/RQ2).
- `outputs/statistics/equivalence_estimation.json` — decisões de equivalência, IC95% e estatísticas de suporte (RQ2).
- `outputs/benchmarks/architectural_benchmark_results.csv` e `architectural_benchmark_resource_log.jsonl` — latências, throughput e monitoramento de recursos (RQ2).
- `outputs/scientific_config_snapshot.json` — snapshot completo do ambiente (packages, hardware, git commit, hash do requirements.txt) (RQ3).
- Tabelas LaTeX em `outputs/statistics/` — artefatos publication-ready.

## 4. Artefatos Essenciais

```
outputs/
├── statistics/
│   ├── equivalence_estimation.(json|tex)
│   ├── architectural_latency_percentiles.(json|tex)
│   ├── architectural_throughput_percentiles.(json|tex)
│   ├── architectural_resource_usage.(json|tex)
│   └── architectural_scorecard.tex
├── benchmarks/
│   ├── architectural_benchmark_results.csv
│   └── architectural_benchmark_summary.json
└── ml_pipeline/
    └── architectures/
        ├── data_lake/
        └── data_warehouse/
```

Cada arquivo possui cabeçalho com metadados (timestamp, versão do protocolo, seed). Utilize-os como evidência de replicação ou como base para meta-análises.

## 5. Extender o Framework

1. **Adicionar nova arquitetura**
   Implemente uma subclasse de `BaseArchitectureML` (11 métodos abstratos, padrão Template Method) em `src/architectures_ml/` replicando o padrão `data_lake`/`data_warehouse`. O enforcement anti-leakage é herdado automaticamente da classe base. Registre a nova arquitetura no pipeline principal (`pipeline.py`).

2. **Ajustar parâmetros científicos**
   Atualize `SCIENTIFIC_CONFIG` com novos gaps, SESOI (`sesoi_r2`, `sesoi_wape`, `sesoi_mase`) ou número de iterações de bootstrap. Para inspecionar a sensibilidade das decisões sem reexecutar todo o pipeline, rode `python src/statistical_validation/bootstrap_sensitivity.py --latex`, que gera resumos em `outputs/statistics/bootstrap_sensitivity.*`.

3. **Instrumentar novas métricas**
   Estenda `src/benchmarking/` ou `src/statistical_validation/` mantendo contratos de entrada/saída. Scripts já existentes servem de templates.

4. **Documentar heurísticas**
   Adicione notas ao `tema4_metodologia_analysis.md` ou crie novos memos para contextualizar interpretações, preservando transparência.

## 6. Referências Internas

- `docs/paper_sbc.tex` — Artigo completo (SBC) descrevendo o protocolo e a demonstração.
- `USAGE_GUIDE.md` — Guia operacional detalhado para uso e adaptação do framework.
- `fair_comparison_analysis.md` — Análise crítica de vieses potenciais em comparações arquiteturais.
- `tema4_metodologia_analysis.md` — Registro das decisões de reposicionamento metodológico.

## 7. Referências Bibliográficas de Apoio

Principais trabalhos que embasam o protocolo:
- Hyndman & Koehler (2006), Lakens (2017), Cerqueira et al. (2020) — Métricas e equivalência em séries temporais.
- Roberts et al. (2017) — Boas práticas para validação temporal com blocked designs.
- Kapoor & Narayanan (2023) — Leakage e crise de reprodutibilidade em ML (294 papers auditados).
- Semmelrock et al. (2025) — Barreiras à reprodutibilidade em ML (5 pilares).
- Romero & Ventura (2020), Hellas et al. (2018) — Lacunas metodológicas em analytics educacional.
- Wilkinson et al. (2016) — Princípios FAIR para dados científicos.
- Harby & Zulkernine (2024) — Survey comparativo de arquiteturas lake/warehouse/lakehouse.
- Raasveldt & Mühleisen (2019), Zhang et al. (2023) — Caracterização de paradigmas arquiteturais.

---

**Contato**: {eos.xavier, rosa.livia, vanessa.a.borges}@ufms.br — Faculdade de Computação, UFMS.

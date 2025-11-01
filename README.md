# Framework Metodológico para Avaliação de Arquiteturas de Dados em Analytics Educacional

Este repositório materializa um **framework metodológico completo** para avaliar arquiteturas de dados aplicadas a analytics educacional com rigor científico. O trabalho responde a três questões centrais:

1. **QP1 — Validação Temporal**: Como gerar folds walk-forward com gaps anti-leak auditáveis e reutilizáveis?
2. **QP2 — Equivalência Prática**: Como produzir inferências defensáveis sobre desempenho com amostras pequenas (\(n<15\))?
3. **QP3 — Reprodutibilidade e Benchmarking**: Quais instrumentações são necessárias para reproduzir benchmarks arquiteturais ponta a ponta?

A demonstração empírica compara dois paradigmas clássicos — schema-on-write (DuckDB) e schema-on-read (Dask) — em dados públicos do Banco Mundial (2000–2023). O objetivo não é declarar vencedores, mas validar o protocolo e exemplificar como transformar resultados em heurísticas transparentes. Como os indicadores são agregados por país/ano, tratamos o estudo como um benchmark metodológico com dados macro-educacionais, útil quando logs individuais não estão acessíveis e expomos explicitamente as limitações decorrentes dessa granularidade.

## 1. Protocolos Metodológicos

| Protocolo | Objetivo | Componentes-chave |
|-----------|----------|--------------------|
| Validação temporal | Evitar vazamentos temporais e preservar causalidade | `src/core/scientific_config.py`, `src/architectures_ml/*/setup.py`, `tests/test_lag_anti_leak.py` |
| Equivalência prática | Estimar efeitos arquiteturais com SESOI + IC95% | `src/statistical_validation/tost_baseline.py`, `outputs/statistics/equivalence_estimation.*` |
| Benchmark arquitetural | Mensurar latência, throughput e uso de recursos | `src/benchmarking/architectural_benchmark.py`, `outputs/benchmarks/*` |
| Reprodutibilidade integral | Registrar parâmetros, seeds e artefatos auditáveis | `pipeline.py`, `src/core/logging_config.py`, `outputs/ml_pipeline/` |

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

# 2. Rodar testes de sanidade adicionais (opcional)
pytest tests/test_lag_anti_leak.py

# 3. (Re)gerar tabelas LaTeX a partir das saídas do benchmark
python src/benchmarking/derive_latency_percentiles.py
python src/benchmarking/derive_throughput_percentiles.py
python src/benchmarking/derive_resource_usage_table.py
```

O pipeline imprime seeds e caminhos de saída, garantindo rastreabilidade. Execuções são determinísticas sob o `scientific_config.py` versionado.

## 3. Checklist de Reprodutibilidade

Verifique os itens abaixo após rodar o pipeline:

- `outputs/ml_pipeline/architectures/<arch>/prep/temporal_folds_<arch>.json` — fronteiras dos folds (QP1).
- `outputs/statistics/equivalence_estimation.json` — decisões de equivalência, IC95% e estatísticas de suporte (QP2).
- `outputs/benchmarks/architectural_benchmark_results.csv` e `architectural_benchmark_resource_log.jsonl` — latências, throughput e monitoramento de recursos (QP3).
- `docs/paper_sbc.tex` + tabelas LaTeX em `outputs/statistics/` — artefatos publication-ready.
- `tema4_metodologia_analysis.md` — memo decisório que sumariza heurísticas derivadas do estudo.

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
   ```python
   from core.config import register_architecture
   register_architecture('lakehouse_delta', {...})
   ```
   Implemente uma subclasse de `BaseArchitectureML` em `src/architectures_ml/` replicando o padrão `data_lake`/`data_warehouse`.

2. **Ajustar parâmetros científicos**
   Atualize `SCIENTIFIC_CONFIG` com novos gaps, SESOI (`sesoi_r2`, `sesoi_nrmse`, `sesoi_mase`) ou número de iterações de bootstrap. Para inspecionar a sensibilidade das decisões sem reexecutar todo o pipeline, rode `python src/statistical_validation/bootstrap_sensitivity.py --latex`, que gera resumos em `outputs/statistics/bootstrap_sensitivity.*`.

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
- Hyndman & Athanasopoulos (2021), Lakens (2017), Cerqueira et al. (2020) — Métricas e equivalência em séries temporais.
- Roberts et al. (2016) — Boas práticas para validação temporal.
- Bergner & Kerr (2023), Romero & Ventura (2020) — Lacunas metodológicas em analytics educacional.
- Wilkinson et al. (2016) — Princípios FAIR para dados científicos.
- Dean & Barroso (2013), McSherry et al. (2015), Raasveldt & Mühleisen (2019), Zhang et al. (2023) — Caracterização de paradigmas arquiteturais.

---

**Contato**: {eos.xavier, rosa.livia, vanessa.a.borges}@ufms.br — Faculdade de Computação, UFMS.

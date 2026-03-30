# Guia de Uso do Framework Metodológico

Este guia descreve como executar, verificar e adaptar o framework metodológico de avaliação de arquiteturas de dados em analytics educacional. O material complementa o README principal.

## 1. Visão Geral dos Protocolos

1. **Extensibilidade (O1)** — Arquitetura modular via Template Method com 11 métodos abstratos; novas arquiteturas herdam enforcement anti-leakage automaticamente.
2. **Recomendação automática (O2)** — Benchmark arquitetural com SESOI + IC95% por bootstrap e estatísticas robustas (Wilcoxon, Hodges–Lehmann), gerando recomendação automática de paradigma.
3. **Reprodutibilidade integral (O3)** — Seeds centralizadas, `n_jobs=1`, snapshot completo de ambiente (packages, hardware, git commit) e gate anti-leakage no pipeline.
4. **Validação anti-leakage** — Enforcement automático: `raise ValueError` em violações de ordenação temporal, gap mínimo e separação de features. Gate no pipeline bloqueia execução se integridade temporal falhar. Suporte a embargo configurável (`embargo_years`, default 0) conforme López de Prado (2018).

A demonstração embarcada compara DuckDB (schema-on-write), Dask (schema-on-read) e Polars DataFrame (lazy evaluation); você pode reutilizar os mesmos protocolos para outras arquiteturas.

## 2. Preparação do Ambiente

Requisitos mínimos: Python 3.10+, 8 GB de RAM, 10 GB livres em disco.

```bash
git clone https://github.com/DATA-UFMS/dw-vs-dl-dropout-prediction-latam.git
cd dw-vs-dl-dropout-prediction-latam
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 3. Execução dos Protocolos

1. **Pipeline completo** (recomendado)
   ```bash
   python pipeline.py
   ```
   Executa coleta \u2192 preparação \u2192 **gate anti-leakage** \u2192 benchmark \u2192 análises estatísticas, preservando os parâmetros do `scientific_config.py`. O gate verifica integridade temporal de todos os folds antes de prosseguir ao benchmark.

2. **Componentes individuais** (quando precisar repetir apenas uma etapa)
   ```bash
   # Benchmark arquitetural
   python src/benchmarking/architectural_benchmark.py --repetitions 5 --warmup 1

   # Equivalência prática (gera JSON/LaTeX)
   python src/statistical_validation/equivalence_estimation.py --latex

   # Testes unitários e anti-leakage (73 testes)
   pytest tests/test_unit_core.py tests/test_lag_anti_leak.py

   # Teste de injeção de leakage (validação negativa, cenários S1-S4)
   python tests/test_leakage_injection.py
   ```

3. **Pós-processamento das saídas**
 ```bash
  python src/benchmarking/derive_latency_percentiles.py
  python src/benchmarking/derive_throughput_percentiles.py
  python src/benchmarking/derive_resource_usage_table.py
  python src/statistical_validation/bootstrap_sensitivity.py --latex
  ```
  Esses scripts convertem os CSV/JSON brutos em tabelas LaTeX prontas para publicação.

## 4. Verificação dos Resultados

Após a execução, valide os seguintes artefatos:

- `outputs/ml_pipeline/architectures/<arch>/prep/temporal_folds_<arch>.json` — intervalos treino/validação/teste e gaps \u22652 anos (onde `<arch>` pode ser `data_lake`, `data_warehouse`, ou `polars_dataframe`).
- `outputs/ml_pipeline/architectures/<arch>/prep/target_statistics.json` — estatísticas do target e checagens de consistência.
- `outputs/statistics/equivalence_estimation.json` — decisão (equivalente/superior/inferior) + IC95% + Wilcoxon/Hodges-Lehmann.
- `outputs/benchmarks/architectural_benchmark_results.csv` — latência por fase com identificador de execução.
- `outputs/benchmarks/architectural_benchmark_resource_log.jsonl` — monitoramento de CPU/RAM/IO a cada amostra.
- `outputs/statistics/architectural_scorecard.tex` — painel consolidado para o artigo.

Compare os hashes/timestamps com os registrados nos cabeçalhos para garantir que a execução seja recente e coerente.

## 5. Customização

### 5.1 Ajustar Parâmetros Científicos

Edite `src/core/scientific_config.py` para alterar:

- `temporal_gap_years`, `folds_min_train_years`, `folds_step_years`, `embargo_years` (validação temporal).
- `sesoi_r2`, `sesoi_mase`, `sesoi_wape` (limiares de equivalência por métrica).
- `bootstrap_iters` (número de reamostragens).

Sempre documente alterações em um memo de decisão.

### 5.2 Adicionar Nova Arquitetura

1. Crie `src/architectures_ml/<nova>/setup.py` com uma subclasse de `BaseArchitectureML` que define `PARADIGM_META` (nome, label, módulos) e implementa os 11 métodos abstratos. O enforcement anti-leakage é herdado automaticamente.
2. Crie os módulos de processamento, baseline e hierárquico nos caminhos declarados no `PARADIGM_META`. O framework descobre o novo paradigma automaticamente via `__init_subclass__` — nenhum arquivo existente precisa ser editado.

### 5.3 Instrumentar Métricas Extras

- Extenda `src/benchmarking` para coletar métricas adicionais (p.ex., energia, custo).
- Adicione novos módulos em `src/statistical_validation` mantendo o padrão de entrada/saída JSON/LaTeX.
- Registre scripts personalizados no README.

## 6. Boas Práticas e Sanity Checks

- Execute `pytest tests/test_unit_core.py tests/test_lag_anti_leak.py` (73 testes) e `python tests/test_leakage_injection.py` (validação negativa S1-S4) depois de qualquer alteração em geração de folds ou lógica de validação.
- Compare estatísticas de target e listas de features nos diretórios `outputs/ml_pipeline/architectures/<arch>/prep/` para garantir alinhamento entre as 3 arquiteturas (DuckDB, Dask, Polars DataFrame).
- Para replicações externas, gere um `requirements-lock.txt` atualizado (`pip freeze > requirements-lock.txt`).

## 7. Integração com o Artigo

- Tabelas LaTeX geradas automaticamente (`outputs/statistics/*.tex`) podem ser incluídas diretamente no artigo via `\input{}`.
- Em submissões, inclua a checklist de reprodutibilidade (Seção 4) como material suplementar.

## 8. Perguntas Frequentes

**Os resultados devem bater exatamente?** Sim. Seeds, configuração científica e scripts determinísticos garantem replicabilidade. Divergências indicam ambiente diferente ou alteração não documentada.

**Posso usar apenas parte do framework?** Pode, desde que explique no memo de decisão quais protocolos foram executados e quais foram omitidos.

**Como adapto para outro domínio (ex.: saúde)?** Ajuste os limiares SESOI, redefine indicadores no coletor de dados e atualize a documentação contextual. Os protocolos permanecem idênticos.

---

Em caso de dúvidas, abra uma issue ou contate {eos.xavier, rosa.livia, vanessa.a.borges}@ufms.br.

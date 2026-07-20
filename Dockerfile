# Ambiente de execução fixado, para que uma reprodução não dependa do que está
# instalado na máquina de quem reproduz.
#
# A imagem base é fixada por digest, e não por tag: `python:3.12-slim` aponta para
# imagens diferentes ao longo do tempo, o que é exatamente a variação que este
# arquivo existe para eliminar. Atualizar o digest é uma decisão, não um efeito
# colateral de reconstruir.
#
# Python 3.12 porque é a versão em que os resultados foram produzidos; o pacote
# declara compatibilidade a partir de 3.10, e o CI cobre as duas pontas.
# Digest resolvido do registry em 2026-07-26 para python:3.12-slim.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

# As bibliotecas numéricas dimensionam seus pools de threads no carregamento e não
# podem ser reduzidas depois de dentro do processo. Fixadas aqui e não só no
# pipeline, para que um comando avulso dentro do contêiner tenha o mesmo
# orçamento, e a latência medida não dependa de como o processo foi iniciado.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /rampart

# O lock primeiro, sozinho: a camada de dependências só é invalidada quando as
# dependências mudam, e não a cada alteração de código.
COPY requirements-lock.txt ./
RUN python -m pip install --requirement requirements-lock.txt

# O Dockerfile entra na imagem porque a suíte que roda na construção o
# inspeciona; sem ele oito testes falham e nenhuma imagem é produzida.
COPY pyproject.toml README.md Dockerfile ./
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY pipeline.py ./
RUN python -m pip install --no-deps --editable .

# A suíte roda na construção: uma imagem que não passa nos testes não deveria
# chegar a existir.
RUN python -m pytest tests/ -q

# Sem ENTRYPOINT fixo no pipeline: o dataset é escolha de quem executa, e os dois
# têm custos muito diferentes (World Bank cerca de 1h30, INEP cerca de 29h).
CMD ["bash", "scripts/reproduce.sh", "--help"]

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG INSTALL_LOCAL=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py requirements.txt requirements-local.txt README.md ./
COPY configs ./configs
COPY core ./core
COPY data/probes ./data/probes
COPY models ./models
COPY probes ./probes
COPY scorer ./scorer
COPY scripts ./scripts
COPY translation ./translation
COPY run_bilingual_eval.py run_pilot.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && if [ "$INSTALL_LOCAL" = "true" ]; then python -m pip install -r requirements-local.txt; fi \
    && mkdir -p scorer/models data/eval_outputs/raw data/eval_outputs/scored data/eval_outputs/combined logs \
    && curl -fsSL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
        -o scorer/models/lid.176.ftz

ENTRYPOINT ["gmass"]
CMD ["--help"]

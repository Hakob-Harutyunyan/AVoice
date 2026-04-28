FROM python:3.11-slim

ARG TORCH_VERSION=2.4.1
ARG TORCHAUDIO_VERSION=2.4.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG PREFETCH_AUDIO_TOKENIZER=0
ARG PREFETCH_ASR=0
ARG ASR_MODEL_NAME=openai/whisper-large-v3-turbo

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    OMNIVOICE_MODEL=/app/model \
    OMNIVOICE_HOST=0.0.0.0 \
    OMNIVOICE_PORT=8000 \
    OMNIVOICE_DEVICE=auto \
    OMNIVOICE_DTYPE=auto

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --create-home --gid app app

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app requirements.txt ./
COPY --chown=app:app omnivoice ./omnivoice

RUN pip install --upgrade pip setuptools wheel
RUN pip install --index-url ${TORCH_INDEX_URL} \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION}
RUN pip install -r requirements.txt
RUN pip install --no-deps .

RUN mkdir -p ${HF_HOME} /app/model \
 && chown -R app:app /app ${HF_HOME}

USER app

RUN if [ "${PREFETCH_AUDIO_TOKENIZER}" = "1" ]; then \
      omnivoice-prefetch --model-dir /app/model --copy-audio-tokenizer; \
    fi

RUN if [ "${PREFETCH_ASR}" = "1" ]; then \
      omnivoice-prefetch --prefetch-asr --asr-model "${ASR_MODEL_NAME}"; \
    fi

EXPOSE 8000

# Leave disabled by default; enable if your deployment needs container-level probing.
#HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD \
#  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5).read()"

CMD ["omnivoice-api"]

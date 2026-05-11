# syntax=docker/dockerfile:1
# Transcription app with WhisperX - matches whisperx_transcribe.ipynb
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch
ENV XDG_CACHE_HOME=/cache
ENV WHISPER_CACHE=/cache/whisper

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /cache/huggingface /cache/torch /cache/whisper /tmp/transcribe_uploads

COPY app/requirements.txt app/constraints.txt ./

RUN --mount=type=bind,source=pip-cache,target=/root/.cache/pip \
    python3 -m pip install -c constraints.txt -r requirements.txt

COPY app/ .

ENV HOME=/app
RUN rm -rf /app/.cache 2>/dev/null; ln -sf /cache /app/.cache

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

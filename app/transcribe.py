"""
Transcription service using faster-whisper (lightweight, no PyTorch/pyannote/speechbrain).
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Cached model - loaded once, reused for all transcriptions
_model_cache: dict = {}


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract 16kHz mono WAV from video using ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                audio_path, "-y"
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg_path, "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    audio_path, "-y"
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except Exception:
            return False


def _get_device() -> str:
    return os.environ.get("WHISPER_DEVICE", "cuda")


def preload_model(model_name: str = "large-v2", language: str | None = "ru"):
    """Load model at startup. Uses faster-whisper (ctranslate2)."""
    device = _get_device()
    compute_type = "int8"
    key = (model_name, language)
    if key not in _model_cache:
        try:
            _model_cache[key] = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception:
            if device == "cuda":
                _model_cache[key] = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type=compute_type,
                )
            else:
                raise
    return _model_cache[key]



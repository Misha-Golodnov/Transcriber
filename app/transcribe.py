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


def load_audio(audio_path: str):
    """Load audio as float32 numpy at 16kHz."""
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = data[:, 0]
        if sr != 16000:
            import librosa
            data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=16000)
        return np.array(data, dtype="float32")
    except Exception:
        pass
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        return audio.astype("float32")
    except Exception:
        return None


def transcribe(
    input_path: str,
    language: str = "ru",
    model_name: str = "large-v2",
) -> dict:
    """
    Transcribe audio/video file. Same flow as notebook.
    Returns dict with segments and full text.
    """
    logger.info("[transcribe] start: input_path=%s language=%s", input_path, language)
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    if input_path.suffix.lower() in video_extensions:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        try:
            if not extract_audio(str(input_path), audio_path):
                raise RuntimeError("Failed to extract audio from video")
        except Exception as e:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            raise e
    else:
        audio_path = str(input_path)

    try:
        logger.info("[transcribe] loading model: model_name=%s", model_name)
        model = preload_model(model_name=model_name, language=language)
        logger.info("[transcribe] model loaded, loading audio: %s", audio_path)
        audio = load_audio(audio_path)
        if audio is None:
            raise RuntimeError("Failed to load audio")
        logger.info("[transcribe] audio loaded: %d samples, running inference", len(audio))

        lang = None if language == "auto" else language
        segment_list, info = model.transcribe(audio, language=lang)
        logger.info("[transcribe] inference done")

        segments = [
            {"start": s.start, "end": s.end, "text": (s.text or "").strip()}
            for s in segment_list
        ]
        full_text = "\n".join(s["text"] for s in segments if s["text"])
        detected_lang = info.language if info else language

        return {
            "language": detected_lang or language,
            "segments": segments,
            "full_text": full_text,
        }
    finally:
        if audio_path != str(input_path) and os.path.exists(audio_path):
            os.unlink(audio_path)

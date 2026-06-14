"""Download media from URL or extract audio only (no full video download)."""
import hashlib
import subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests


def extract_audio_from_url(url: str, output_path: Path) -> Path:
    """
    Extract audio from URL directly - no full video download.
    Uses ffmpeg to stream from URL and write only 16kHz mono WAV.
    Falls back to yt-dlp for YouTube and similar sites (downloads audio stream only).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try ffmpeg first (works for direct URLs: .mp4, .mkv, HTTP links, etc.)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", url,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                str(output_path), "-y"
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0 and output_path.exists():
            return output_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: yt-dlp for YouTube, Vimeo, etc. (downloads only audio stream)
    try:
        subprocess.run(
            [
                "yt-dlp", "-x", "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", str(output_path.with_suffix(".%(ext)s")),
                "--no-playlist",
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        # yt-dlp may use different extension
        wav_path = output_path.with_suffix(".wav")
        if wav_path.exists():
            return wav_path
        for p in output_path.parent.glob(output_path.stem + ".*"):
            if p.suffix.lower() in (".wav", ".webm", ".m4a", ".opus"):
                # Resample to 16kHz mono if not already wav
                if p.suffix.lower() != ".wav":
                    subprocess.run(
                        [
                            "ffmpeg", "-i", str(p),
                            "-vn", "-acodec", "pcm_s16le",
                            "-ar", "16000", "-ac", "1",
                            str(wav_path), "-y"
                        ],
                        check=True,
                        capture_output=True,
                        timeout=600,
                    )
                    p.unlink(missing_ok=True)
                    return wav_path
                return p
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Could not extract audio from URL. For direct links, use ffmpeg. "
            f"For YouTube/Vimeo, install yt-dlp: pip install yt-dlp. Error: {e}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp not found. Install with: pip install yt-dlp (needed for YouTube/Vimeo)"
        ) from None

    raise RuntimeError("Could not extract audio from URL")


def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def get_filename_from_url(url: str, default_name: str = "media") -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path = unquote(parsed.path)
    path_parts = [p for p in path.split("/") if p]

    if path_parts:
        original = path_parts[-1]
        if "." in original:
            ext = original.split(".")[-1]
            name = ".".join(original.split(".")[:-1])
        else:
            ext = "mp4"
            name = original
    else:
        ext = "mp4"
        name = default_name

    domain_safe = sanitize_filename(domain.replace(".", "_"))
    name_safe = sanitize_filename(name)
    filename = f"{domain_safe}_{name_safe}.{ext}"

    if len(filename) > 200:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{domain_safe}_{url_hash}.{ext}"

    return filename


def download_from_url(url: str, output_dir: Path, filename: str | None = None) -> Path:
    """Download file from URL to output_dir. Returns path to downloaded file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = filename or get_filename_from_url(url)
    filepath = output_dir / filename

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return filepath

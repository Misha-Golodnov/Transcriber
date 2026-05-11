"""
FastAPI app for audio/video transcription.
"""
import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import ErrorEvent

from download import extract_audio_from_url
from job_queue import JobQueue
from telegram_bot import create_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,
)

UPLOAD_DIR = Path("/tmp/transcribe_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
job_queue = JobQueue(UPLOAD_DIR, telegram_token=TELEGRAM_TOKEN)
logger = logging.getLogger(__name__)


async def _run_telegram_bot(bot: Bot, dp: Dispatcher):
    """Run bot via start_polling (shares event loop with FastAPI)."""
    try:
        await dp.start_polling(bot)
    except asyncio.CancelledError:
        logger.info("[telegram] polling task cancelled")
    finally:
        await bot.session.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    bot_task = None
    bot = None
    if TELEGRAM_TOKEN:
        bot = Bot(
            token=TELEGRAM_TOKEN,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        dp = Dispatcher()

        @dp.errors()
        async def _telegram_error_handler(event: ErrorEvent) -> None:
            logger.exception(
                "[telegram] handler error: %s", event.exception, exc_info=event.exception
            )

        dp.include_router(create_router(job_queue, str(UPLOAD_DIR)))
        bot_task = asyncio.create_task(_run_telegram_bot(bot, dp))
        logger.info("[telegram] bot started (polling)")
    yield
    job_queue.shutdown()
    if bot_task and bot:
        bot_task.cancel()
        try:
            await asyncio.wait_for(bot_task, timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


app = FastAPI(title="Transcription API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/transcribe/url")
async def transcribe_from_url(
    url: str = Form(...),
    language: str = Form("ru"),
):
    """Submit a URL for transcription. Returns job_id to poll for result."""
    logger.info("[api] transcribe_from_url: url=%s", url[:80] + "..." if len(url) > 80 else url)
    filepath = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
    try:
        await asyncio.to_thread(extract_audio_from_url, url, filepath)
        logger.info("[api] extract_audio_from_url done: filepath=%s", filepath)
    except Exception as e:
        if filepath.exists():
            filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    job_id = job_queue.add(
        input_path=str(filepath),
        language=language,
        source="api",
    )
    status = job_queue.get_status(job_id)
    return {"job_id": job_id, "status": "queued", "position": status.get("position", 0)}


@app.post("/api/transcribe/upload")
async def transcribe_from_upload(
    file: UploadFile = File(...),
    language: str = Form("ru"),
):
    """Upload a file for transcription. Returns job_id to poll for result."""
    logger.info("[api] transcribe_from_upload: filename=%s", file.filename)
    job_id = str(uuid.uuid4())
    ext = Path(file.filename or "audio.wav").suffix
    filepath = UPLOAD_DIR / f"{job_id}{ext}"

    try:
        logger.info("[api] transcribe_from_upload: writing file to %s", filepath)
        size = 0
        async with aiofiles.open(filepath, "wb") as f:
            while chunk := await file.read(8192):
                await f.write(chunk)
                size += len(chunk)
        logger.info("[api] transcribe_from_upload: file write complete, size=%d", size)
    except Exception as e:
        if filepath.exists():
            filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("[api] transcribe_from_upload: adding to queue")
    job_queue.add(
        input_path=str(filepath),
        language=language,
        source="api",
        job_id=job_id,
    )
    logger.info("[api] transcribe_from_upload: added, getting status")
    status = job_queue.get_status(job_id)
    return {"job_id": job_id, "status": "queued", "position": status.get("position", 0)}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get transcription job status and queue position."""
    status = job_queue.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Get transcription result (full text and segments)."""
    status = job_queue.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    if status["status"] == "completed":
        return status["result"]
    if status["status"] == "failed":
        raise HTTPException(status_code=500, detail=status.get("error", "Transcription failed"))
    return {"status": status["status"], "position": status.get("position"), "message": "Still processing"}


# Serve static UI
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        async with aiofiles.open(html_path, "r", encoding="utf-8") as f:
            return await f.read()
    return "<h1>Transcription API</h1><p>Use /api/transcribe/upload or /api/transcribe/url</p>"

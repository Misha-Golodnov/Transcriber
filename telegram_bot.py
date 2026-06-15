"""
Telegram bot - forward voice messages for transcription.
Uses aiogram 3.
"""
import uuid
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message


def create_router(queue, upload_dir: str) -> Router:
    """Create router with handlers that use the given queue and upload_dir."""
    router = Router()

    @router.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        await message.answer(
            "🎙️ Send me a voice message and I'll transcribe it.\n\n"
            "I'll reply with your queue position and send the transcription when ready."
        )

    @router.message(F.voice | F.audio)
    async def handle_voice(message: Message, bot: Bot) -> None:
        voice = message.voice or message.audio
        if not voice:
            return

        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)

        job_id = str(uuid.uuid4())
        ext = ".ogg" if message.voice else (
            f".{voice.file_name.rsplit('.', 1)[-1]}"
            if getattr(voice, "file_name", None) and "." in (voice.file_name or "")
            else ".m4a"
        )
        filepath = upload_path / f"{job_id}{ext}"
        await bot.download(voice, destination=filepath)

        queue.add(
            input_path=str(filepath),
            language="ru",
            source="telegram",
            telegram_chat_id=message.chat.id if message.chat else None,
            job_id=job_id,
        )

        pos = queue.get_status(job_id)
        position = pos.get("position", 0)
        if position == 0:
            msg = "🎙️ Processing your voice message..."
        else:
            msg = f"🎙️ Queued at position {position}. I'll send the transcription when ready."
        await message.answer(msg)

    return router

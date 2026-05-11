"""
Centralized transcription queue - single GPU, one job at a time.
Runs transcribe in a separate PROCESS (not thread) - Pyannote/PyTorch
freeze when called from a background thread.
"""
import logging
import threading
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import requests

from transcribe import transcribe

logger = logging.getLogger(__name__)


def _run_transcribe(input_path: str, language: str) -> dict:
    """Wrapper for process pool - must be top-level for pickling."""
    logger.info("[worker] _run_transcribe started: input_path=%s language=%s", input_path, language)
    try:
        result = transcribe(input_path=input_path, language=language)
        logger.info("[worker] _run_transcribe completed: input_path=%s", input_path)
        return result
    except Exception as e:
        logger.exception("[worker] _run_transcribe failed: input_path=%s error=%s", input_path, e)
        raise


class JobQueue:
    def __init__(self, upload_dir, telegram_token: str | None = None):
        self.upload_dir = upload_dir
        self.telegram_token = telegram_token
        self._jobs: dict[str, dict] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()  # RLock: add/get_status call _get_position while holding lock
        logger.info("[queue] initializing ProcessPoolExecutor max_workers=1")
        self._executor = ProcessPoolExecutor(max_workers=1)
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        logger.info("[queue] worker thread started")

    def shutdown(self):
        self._executor.shutdown(wait=False)

    def _get_position(self, job_id: str) -> int:
        """1-based position in queue. 0 = currently processing."""
        with self._lock:
            if self._jobs.get(job_id, {}).get("status") == "processing":
                return 0
            try:
                idx = list(self._queue).index(job_id)
                return idx + 1
            except ValueError:
                return -1

    def add(
        self,
        input_path: str,
        language: str = "ru",
        source: str = "api",
        telegram_chat_id: int | None = None,
        job_id: str | None = None,
    ) -> str:
        job_id = job_id or str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "position": 0,
                "result": None,
                "error": None,
                "input_path": input_path,
                "language": language,
                "source": source,
                "telegram_chat_id": telegram_chat_id,
            }
            self._queue.append(job_id)
            pos = self._get_position(job_id)
            self._jobs[job_id]["position"] = pos
            qlen = len(self._queue)
        logger.info("[queue] job added: job_id=%s input_path=%s queue_len=%d", job_id, input_path, qlen)
        return job_id

    def get_status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            pos = self._get_position(job_id)
            return {
                "job_id": job_id,
                "status": job["status"],
                "position": pos,
                "result": job.get("result"),
                "error": job.get("error"),
            }

    def _send_telegram(self, chat_id: int, text: str):
        if not self.telegram_token:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception:
            pass

    def _run_worker(self):
        logger.info("[queue] worker loop started")
        while True:
            job_id = None
            with self._lock:
                if self._queue:
                    job_id = self._queue.popleft()
                    self._jobs[job_id]["status"] = "processing"
                    self._jobs[job_id]["position"] = 0

            if not job_id:
                import time
                time.sleep(1)
                continue

            job = self._jobs[job_id]
            logger.info("[queue] picked job: job_id=%s input_path=%s submitting to executor", job_id, job["input_path"])
            try:
                # Run in separate process - Pyannote freezes when run from thread
                future = self._executor.submit(
                    _run_transcribe,
                    job["input_path"],
                    job.get("language", "ru"),
                )
                logger.info("[queue] submit done, waiting for result: job_id=%s", job_id)
                result = future.result()
                logger.info("[queue] result received: job_id=%s", job_id)
                with self._lock:
                    self._jobs[job_id]["status"] = "completed"
                    self._jobs[job_id]["result"] = result

                chat_id = job.get("telegram_chat_id")
                if chat_id:
                    text = result.get("full_text", "(empty)")[:4000]
                    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    self._send_telegram(chat_id, f"<b>Transcription:</b>\n\n{text}")

            except Exception as e:
                logger.exception("[queue] job failed: job_id=%s error=%s", job_id, e)
                with self._lock:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = str(e)

                chat_id = job.get("telegram_chat_id")
                if chat_id:
                    self._send_telegram(chat_id, f"❌ Transcription failed: {e}")

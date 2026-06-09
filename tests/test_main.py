import importlib
import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

main_module = importlib.import_module("main")


client = TestClient(main_module.app)


class FakeJobQueue:
    def __init__(self, status=None, added=None):
        self.status = status or {"position": 0}
        self.added = added if added is not None else []

    def add(self, input_path, language, source, job_id=None):
        self.added.append({
            "input_path": input_path,
            "language": language,
            "source": source,
            "job_id": job_id,
        })
        return job_id or "fake-job-id"

    def get_status(self, job_id):
        return self.status


def test_status_returns_404_for_unknown_job():
    response = client.get("/api/status/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_status_returns_job_status(monkeypatch):
    fake_queue = FakeJobQueue(status={"status": "processing", "position": 0})
    monkeypatch.setattr(main_module, "job_queue", fake_queue)

    response = client.get("/api/status/job-123")

    assert response.status_code == 200
    assert response.json() == {"status": "processing", "position": 0}


def test_result_returns_processing_message(monkeypatch):
    fake_queue = FakeJobQueue(status={"status": "queued", "position": 1})
    monkeypatch.setattr(main_module, "job_queue", fake_queue)

    response = client.get("/api/result/job-123")

    assert response.status_code == 200
    assert response.json() == {
        "status": "queued",
        "position": 1,
        "message": "Still processing",
    }


def test_result_returns_completed_result(monkeypatch):
    fake_queue = FakeJobQueue(
        status={"status": "completed", "position": 0, "result": {"full_text": "hello"}}
    )
    monkeypatch.setattr(main_module, "job_queue", fake_queue)

    response = client.get("/api/result/job-123")

    assert response.status_code == 200
    assert response.json() == {"full_text": "hello"}


def test_result_returns_failed_result(monkeypatch):
    fake_queue = FakeJobQueue(status={"status": "failed", "position": 0, "error": "boom"})
    monkeypatch.setattr(main_module, "job_queue", fake_queue)

    response = client.get("/api/result/job-123")

    assert response.status_code == 500
    assert response.json() == {"detail": "boom"}


def test_url_endpoint_returns_job_id(monkeypatch):
    fake_queue = FakeJobQueue(status={"position": 1})
    monkeypatch.setattr(main_module, "job_queue", fake_queue)
    monkeypatch.setattr(main_module, "extract_audio_from_url", lambda url, path: None)

    response = client.post(
        "/api/transcribe/url",
        data={"url": "https://example.com/audio.mp3", "language": "en"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {"job_id": "fake-job-id", "status": "queued", "position": 1}
    assert fake_queue.added


def test_upload_endpoint_returns_job_id(monkeypatch):
    fake_queue = FakeJobQueue(status={"position": 0})
    monkeypatch.setattr(main_module, "job_queue", fake_queue)

    response = client.post(
        "/api/transcribe/upload",
        files={"file": ("sample.wav", io.BytesIO(b"fake-audio"), "audio/wav")},
        data={"language": "ru"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["job_id"]
    assert payload["position"] == 0
    assert fake_queue.added

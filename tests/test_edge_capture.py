from types import SimpleNamespace

import pytest
import requests

from edge.capture import capture_jpeg, send_snapshot


def test_snapshot_uploader_uses_authenticated_jpeg_and_cold_start_timeout(monkeypatch):
    def post(url, **kwargs):
        assert url == "https://api.example/ingest/mess_main"
        assert kwargs["headers"] == {"X-Camera-Key": "secret"}
        assert kwargs["files"]["file"][2] == "image/jpeg"
        assert kwargs["timeout"] == (10, 180)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"status": "ok", "metrics": {"headcount": 4}},
        )

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.delenv("UPLOAD_TIMEOUT_SECONDS", raising=False)
    assert (
        send_snapshot(b"\xff\xd8\xffjpeg", "https://api.example/", "mess_main", "secret")[
            "headcount"
        ]
        == 4
    )


def test_upload_failure_is_not_acknowledged(monkeypatch):
    def fail(*_args, **_kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(requests.Timeout):
        send_snapshot(b"\xff\xd8\xffjpeg", "https://api.example", "mess_main", "secret")
    with pytest.raises(ValueError, match="JPEG"):
        send_snapshot(b"not jpeg", "https://api.example", "mess_main", "secret")


def test_modern_camera_command_is_preferred_and_capture_has_timeout(monkeypatch):
    monkeypatch.setattr("edge.capture.shutil.which", lambda name: f"/usr/bin/{name}")

    def run(command, **kwargs):
        assert command[0] == "/usr/bin/rpicam-still"
        assert kwargs["timeout"] == 30
        assert command[-1] == "-"
        return SimpleNamespace(stdout=b"\xff\xd8\xffjpeg")

    monkeypatch.setattr("edge.capture.subprocess.run", run)
    assert capture_jpeg().startswith(b"\xff\xd8\xff")

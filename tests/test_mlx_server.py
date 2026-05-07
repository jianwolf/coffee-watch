from __future__ import annotations

import sys

from coffee_watch.mlx_server import MLXServerManager


class _DummyProcess:
    returncode = None

    def poll(self):
        return None


def test_vlm_start_process_passes_model(monkeypatch):
    commands: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        return _DummyProcess()

    monkeypatch.setattr("coffee_watch.mlx_server.subprocess.Popen", fake_popen)

    manager = MLXServerManager(
        model="mlx-community/test-model",
        runtime="vlm",
        host="127.0.0.1",
        port=8123,
        log_path=None,
        trust_remote_code=True,
    )

    manager._start_process()

    assert commands == [
        [
            sys.executable,
            "-m",
            "mlx_vlm.server",
            "--model",
            "mlx-community/test-model",
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
            "--trust-remote-code",
        ]
    ]


def test_lm_start_process_passes_trust_remote_code(monkeypatch):
    commands: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        return _DummyProcess()

    monkeypatch.setattr("coffee_watch.mlx_server.subprocess.Popen", fake_popen)

    manager = MLXServerManager(
        model="mlx-community/test-model",
        runtime="lm",
        host="127.0.0.1",
        port=8124,
        log_path=None,
        trust_remote_code=True,
    )

    manager._start_process()

    assert len(commands) == 1
    assert commands[0][0] == sys.executable
    assert "--model" in commands[0]
    assert "mlx-community/test-model" in commands[0]
    assert "--trust-remote-code" in commands[0]

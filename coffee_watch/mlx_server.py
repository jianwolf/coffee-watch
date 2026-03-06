"""MLX server manager for local model inference.

This module manages MLX server subprocess lifecycle:
- Auto-starts the server with the specified model
- Waits for server readiness via health check
- Cleans up on exit

Runtime-specific base URLs:
- `mlx_lm.server` uses `http://127.0.0.1:{port}/v1`
- `mlx_vlm.server` uses `http://127.0.0.1:{port}`
"""

from __future__ import annotations

import atexit
import logging
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


class MLXServerError(Exception):
    """Raised when MLX server fails to start or respond."""


class MLXServerManager:
    """Manage mlx_lm.server or mlx_vlm.server subprocess lifecycle."""

    def __init__(
        self,
        model: str,
        runtime: str = "lm",
        port: int = 8080,
        host: str = "127.0.0.1",
        startup_timeout: int = 900,
        watchdog_interval: int = 30,
        watchdog_failures: int = 3,
        health_timeout: float = 2.0,
        log_path: str | Path | None = "logs/mlx_server.log",
        trust_remote_code: bool = False,
    ):
        self.model = model
        self.runtime = runtime
        self.port = port
        self.host = host
        self.startup_timeout = startup_timeout
        self.watchdog_interval = watchdog_interval
        self.watchdog_failures = watchdog_failures
        self.health_timeout = health_timeout
        self.trust_remote_code = trust_remote_code
        self._process: subprocess.Popen | None = None
        base_path = "/v1" if runtime == "lm" else ""
        self._base_url = f"http://{host}:{port}{base_path}"
        self._log_path = Path(log_path) if log_path else None
        self._log_file: TextIO | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._lock = threading.Lock()
        self._atexit_registered = False
        self._external_server = False

    @property
    def base_url(self) -> str:
        return self._base_url

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                logger.warning("MLX server already running")
                return

            if self._check_health():
                logger.info(
                    "MLX server already running | url=%s (using existing)",
                    self._base_url,
                )
                self._external_server = True
                if not self._atexit_registered:
                    atexit.register(self.stop)
                    self._atexit_registered = True
                return

            self._external_server = False
            self._start_process()

            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True

        try:
            self._wait_for_ready()
        except MLXServerError:
            if self._check_health():
                with self._lock:
                    if self._process and self._process.poll() is not None:
                        self._process = None
                    self._close_log_file()
                logger.warning(
                    "MLX server became ready externally | url=%s (using existing)",
                    self._base_url,
                )
                self._external_server = True
                return
            raise

        if not self._external_server:
            self._start_watchdog()

    def _start_process(self) -> None:
        entry_script = Path(__file__).with_name("mlx_server_entry.py")
        if self.runtime == "lm" and entry_script.exists():
            cmd = [
                sys.executable,
                str(entry_script),
                "--model",
                self.model,
                "--host",
                self.host,
                "--port",
                str(self.port),
            ]
        elif self.runtime == "lm":
            cmd = [
                sys.executable,
                "-m",
                "mlx_lm.server",
                "--model",
                self.model,
                "--host",
                self.host,
                "--port",
                str(self.port),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "mlx_vlm.server",
                "--host",
                self.host,
                "--port",
                str(self.port),
            ]
            if self.trust_remote_code:
                cmd.append("--trust-remote-code")

        logger.info("Starting MLX server | cmd=%s", " ".join(cmd))

        try:
            stdout_target = subprocess.DEVNULL
            if self._log_path:
                stdout_target = self._ensure_log_file() or subprocess.DEVNULL
            self._process = subprocess.Popen(
                cmd,
                stdout=stdout_target,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as exc:
            self._close_log_file()
            package_name = "mlx-vlm" if self.runtime == "vlm" else "mlx-lm"
            raise MLXServerError(
                f"{package_name} not found. Install with: pip install {package_name}"
            ) from exc
        except Exception as exc:
            self._close_log_file()
            raise MLXServerError(
                f"Failed to start MLX server ({type(exc).__name__}): {exc}"
            ) from exc

    def _ensure_log_file(self) -> TextIO | None:
        if not self._log_path:
            return None
        if self._log_file and not self._log_file.closed:
            return self._log_file
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("a", encoding="utf-8")
            return self._log_file
        except Exception as exc:
            logger.warning(
                "Failed to open MLX server log file | path=%s error=%s",
                self._log_path,
                exc,
            )
            self._log_file = None
            return None

    def _wait_for_ready(self, shutdown_on_timeout: bool = True) -> None:
        health_url = self._health_url()
        start_time = time.time()

        logger.info("Waiting for MLX server to be ready (downloading model if needed)...")

        while time.time() - start_time < self.startup_timeout:
            if self._process and self._process.poll() is not None:
                log_hint = f" See {self._log_path} for details." if self._log_path else ""
                crash_hint = self._diagnose_crash()
                raise MLXServerError(
                    f"MLX server exited unexpectedly (code {self._process.returncode})."
                    f"{log_hint}{crash_hint}"
                )

            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=self.health_timeout) as resp:
                    if resp.status == 200:
                        logger.info("MLX server ready | url=%s", self._base_url)
                        return
            except urllib.error.URLError:
                pass
            except Exception:
                pass

            time.sleep(2)

        if shutdown_on_timeout:
            self.stop()
        else:
            self._stop_process()
        manual_cmd = (
            f"mlx_lm.server --model {self.model}"
            if self.runtime == "lm"
            else "mlx_vlm.server"
        )
        raise MLXServerError(
            f"MLX server did not become ready within {self.startup_timeout}s. "
            "This may be due to model download. Try running manually first: "
            f"{manual_cmd}"
        )

    def _check_health(self) -> bool:
        health_url = self._health_url()
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=self.health_timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _health_url(self) -> str:
        if self.runtime == "vlm":
            return f"http://{self.host}:{self.port}/health"
        return f"http://{self.host}:{self.port}/v1/models"

    def _start_watchdog(self) -> None:
        if self.watchdog_interval <= 0 or self.watchdog_failures <= 0:
            return
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="mlx-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        failures = 0
        while not self._watchdog_stop.wait(self.watchdog_interval):
            with self._lock:
                process = self._process
            if process is None:
                continue
            if process.poll() is not None:
                failures += 1
            else:
                if self._check_health():
                    failures = 0
                    continue
                failures += 1

            if failures >= self.watchdog_failures:
                failures = 0
                logger.warning(
                    "MLX server unresponsive; restarting | interval=%ds threshold=%d",
                    self.watchdog_interval,
                    self.watchdog_failures,
                )
                self._restart_from_watchdog()

    def _restart_from_watchdog(self) -> None:
        if self._watchdog_stop.is_set():
            return
        with self._lock:
            if self._watchdog_stop.is_set():
                return
            self._stop_process()
            self._start_process()

        try:
            self._wait_for_ready(shutdown_on_timeout=False)
        except Exception as exc:
            logger.warning("MLX watchdog restart failed | error=%s", exc)

    def stop(self) -> None:
        self._watchdog_stop.set()
        watchdog_thread = self._watchdog_thread
        if watchdog_thread and watchdog_thread.is_alive():
            watchdog_thread.join(timeout=5)
            if watchdog_thread.is_alive():
                logger.warning("MLX watchdog did not stop cleanly")

        with self._lock:
            if self._process is None or self._external_server:
                self._close_log_file()
                return

        logger.info("Stopping MLX server...")

        with self._lock:
            self._stop_process()
            self._close_log_file()
            self._external_server = False

        try:
            atexit.unregister(self.stop)
            self._atexit_registered = False
        except Exception:
            pass

    def _stop_process(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("MLX server did not terminate, killing...")
                self._process.kill()
                self._process.wait()
        except Exception as exc:
            logger.warning("Error stopping MLX server: %s", exc)
        finally:
            self._process = None

    def _close_log_file(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _diagnose_crash(self) -> str:
        if not self._log_path:
            return ""
        try:
            tail = self._read_log_tail(max_lines=80)
        except Exception:
            return ""
        if "NSRangeException" in tail and "index 0 beyond bounds" in tail:
            return (
                " Metal device list was empty. This can happen in headless or sandboxed "
                "shells; try starting `mlx_lm.server` in a regular Terminal session and rerun."
            )
        return ""

    def _read_log_tail(self, max_lines: int = 80) -> str:
        if not self._log_path:
            return ""
        try:
            with self._log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
            return "".join(lines[-max_lines:])
        except Exception:
            return ""

"""Run a training job in a separate process and stream its output.

Training never runs inside the GUI process (DEVELOPMENT.md 3.6): a crashed or
OOM-killed trainer must not take the labelling session with it.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from .jobs import TrainJob

LineFn = Optional[Callable[[str], None]]
DoneFn = Optional[Callable[[int], None]]

METRIC_PREFIX = "@@DSL_METRIC@@ "     # trainers emit one JSON line per epoch


class NoTrainerPython(RuntimeError):
    """No interpreter available to run training in."""


def resolve_python(preferred: str = "") -> str:
    """Pick the interpreter that will run the trainer.

    From source this is simply the running interpreter. A frozen build cannot
    use itself: `sys.executable` is the exe, which does not accept `-m`, and
    the heavy training stack is deliberately not bundled (DEVELOPMENT.md 2.2). So
    a packaged app must be pointed at an outside python that has torch.
    """
    if preferred:
        if not Path(preferred).exists():
            raise NoTrainerPython(
                f"설정된 학습용 파이썬을 찾을 수 없습니다:\n  {preferred}")
        return preferred
    if not getattr(sys, "frozen", False):
        return sys.executable
    raise NoTrainerPython(
        "실행파일 버전에서는 학습에 외부 파이썬이 필요합니다.\n\n"
        "이유: 실행파일은 학습 프로세스를 직접 띄우지 못하고, 용량 때문에 "
        "torch 를 포함하지 않았습니다.\n\n"
        "1) 파이썬 환경을 준비하고 학습 패키지를 설치하세요:\n"
        "     uv pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126\n"
        "     uv pip install ultralytics\n"
        "2) [설정 > 학습용 파이썬 경로] 에 그 python.exe 를 지정하세요.")


class TrainingProcess:
    def __init__(self, job: TrainJob, on_line: LineFn = None,
                 on_metric: Optional[Callable[[dict], None]] = None,
                 on_done: DoneFn = None, python_exe: str = ""):
        self.job = job
        self.on_line = on_line
        self.on_metric = on_metric
        self.on_done = on_done
        # Resolved lazily in start(): raising from __init__ would make the
        # panel unable to even build a job it could then fix the setting for.
        self._python_pref = python_exe
        self.proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.job_path: Optional[Path] = None
        self.returncode: Optional[int] = None

    # ---- control ----------------------------------------------------------
    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("이미 학습이 실행 중입니다.")
        self.proc = None
        self.returncode = None
        self._stop.clear()
        self.python = resolve_python(self._python_pref)   # may raise NoTrainerPython
        self.job.fill_defaults()
        self.job_path = self.job.save()

        from ..core.settings import train_source_root
        src_root = train_source_root()

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # Overwrite rather than setdefault: an inherited PYTHONPATH from the
        # user's shell must not decide which copy of `dsl` the trainer imports.
        env["PYTHONPATH"] = str(src_root)
        # keep the labelling GUI responsive while a trainer saturates the CPU
        env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.proc = subprocess.Popen(
            [self.python, "-u", "-m", "dsl.train.cli", str(self.job_path)],
            cwd=str(src_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env, creationflags=creationflags)

        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            # Keep draining after a stop request: leaving the pipe full can
            # wedge the child (and this thread's wait()) until it is killed.
            if self._stop.is_set():
                continue
            line = raw.rstrip("\n")
            if line.startswith(METRIC_PREFIX):
                if self.on_metric:
                    try:
                        self.on_metric(json.loads(line[len(METRIC_PREFIX):]))
                    except json.JSONDecodeError:
                        pass
                continue
            if self.on_line:
                self.on_line(line)
        self.proc.wait()
        self.returncode = self.proc.returncode
        if self.on_done:
            self.on_done(self.returncode)

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the trainer to stop; kill it if it will not."""
        if self.proc is None or self.proc.poll() is not None:
            return
        self._stop.set()
        try:
            if sys.platform == "win32":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            try:
                self.proc.kill()
            except OSError:
                pass

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

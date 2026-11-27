"""Single-instance process lock helpers."""

from __future__ import annotations

import atexit
import os
import threading
from pathlib import Path

from filelock import FileLock, Timeout

from worker.logger import logger


class ProcessLock:
    """Cross-platform process lock to prevent duplicate worker runs."""

    def __init__(self, name: str, directory: str | os.PathLike[str] = ".") -> None:
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name).strip("._")
        if not safe_name:
            safe_name = "worker"
        self.path = Path(directory) / f".{safe_name}.lock"
        self.fd: int | None = None
        self._release_lock = threading.Lock()
        self._atexit_registered = False
        self._file_lock = FileLock(str(self.path))

    def acquire(self) -> "ProcessLock":
        with self._release_lock:
            if self._file_lock.is_locked:
                # Already acquired by this process/instance.
                return self
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._file_lock.acquire(timeout=0)
            except Timeout as err:
                raise RuntimeError(f"Another instance appears to be running (lock: {self.path})") from err
            except OSError as err:
                raise RuntimeError(f"Failed to acquire process lock {self.path}: {err}") from err

            try:
                fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
                try:
                    os.ftruncate(fd, 0)
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.write(fd, str(os.getpid()).encode("utf-8", errors="ignore"))
                finally:
                    os.close(fd)
            except OSError:
                # Lock acquired; PID write failure should not prevent startup.
                logger.debug("Could not write PID to lock file {}", self.path)

            if not self._atexit_registered:
                atexit.register(self.release)
                self._atexit_registered = True
            logger.debug("Acquired process lock {}", self.path)
            return self

    def release(self) -> None:
        with self._release_lock:
            if not self._file_lock.is_locked:
                if self._atexit_registered:
                    try:
                        atexit.unregister(self.release)
                    except Exception:
                        pass
                    self._atexit_registered = False
                return
            if self._atexit_registered:
                try:
                    atexit.unregister(self.release)
                except Exception:
                    pass
                self._atexit_registered = False
            try:
                self._file_lock.release()
            except Exception:
                logger.debug("Could not release process lock {}", self.path)
        logger.debug("Released process lock {}", self.path)

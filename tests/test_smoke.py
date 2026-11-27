import sys
import tempfile
import types
import unittest
from pathlib import Path


# Stub worker.logger when optional runtime deps are unavailable.
if "worker.logger" not in sys.modules:
    try:
        import worker.logger  # noqa: F401
    except Exception:
        stub = types.ModuleType("worker.logger")

        class _StubLogger:
            def debug(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

            def error(self, *args, **kwargs):
                pass

            def info(self, *args, **kwargs):
                pass

        stub.logger = _StubLogger()
        sys.modules["worker.logger"] = stub


class ResponseJsonTests(unittest.TestCase):
    def test_response_json_returns_default_on_invalid(self):
        from worker.utils.network import response_json

        class BadResponse:
            def json(self):
                raise ValueError("bad json")

        self.assertEqual(response_json(BadResponse(), default={"ok": False}), {"ok": False})

    def test_response_json_returns_parsed(self):
        from worker.utils.network import response_json

        class GoodResponse:
            def json(self):
                return {"reward": 5}

        self.assertEqual(response_json(GoodResponse(), default={}), {"reward": 5})


class ProcessLockTests(unittest.TestCase):
    def test_process_lock_acquire_and_release(self):
        from worker.utils.process_lock import ProcessLock

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "worker.lock"
            lock1 = ProcessLock("worker-test", directory=tmp)
            lock2 = ProcessLock("worker-test", directory=tmp)
            self.assertIs(lock1.acquire(), lock1)
            with self.assertRaises(RuntimeError):
                lock2.acquire()
            lock1.release()
            self.assertIs(lock2.acquire(), lock2)
            lock2.release()


class RequestWithRetryTests(unittest.TestCase):
    def test_returns_response_on_success(self):
        from worker.utils.network import request_with_retry

        class Resp:
            ok = True
            status_code = 200

        class Session:
            def request(self, method, url, **kwargs):
                self.called = (method, url, kwargs)
                return Resp()

        s = Session()
        r = request_with_retry(s, "GET", "https://example.test", timeout=1, retries=0)
        self.assertIsNotNone(r)
        self.assertTrue(r.ok)
        self.assertEqual(s.called[0], "GET")

    def test_retries_then_returns_none(self):
        from unittest import mock

        import requests

        from worker.utils.network import request_with_retry

        class Session:
            def __init__(self):
                self.calls = 0

            def request(self, method, url, **kwargs):
                self.calls += 1
                raise requests.exceptions.ConnectionError("boom")

        s = Session()
        with mock.patch("worker.utils.network.time.sleep"):
            r = request_with_retry(s, "GET", "https://example.test", timeout=1, retries=2, backoff_seconds=0)
        self.assertIsNone(r)
        self.assertEqual(s.calls, 3)


if __name__ == "__main__":
    unittest.main()

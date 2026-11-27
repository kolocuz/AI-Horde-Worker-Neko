import sys
import types
import unittest
from unittest import mock

# Some test environments may not have loguru installed; provide a tiny stub.
try:
    import loguru  # noqa: F401
except Exception:
    class _LoggerStub:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None

            return _noop

        def log(self, *args, **kwargs):
            return None

        def remove(self, *args, **kwargs):
            return None

        def add(self, *args, **kwargs):
            return 0

        def level(self, *args, **kwargs):
            return None

        def catch(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    sys.modules["loguru"] = types.SimpleNamespace(logger=_LoggerStub())

from worker.enums import JobStatus
from worker.jobs.framework import HordeJobFramework


class DummyBridgeData:
    api_key = "0000000000"
    horde_url = "https://example.test"
    suppress_speed_warnings = True


class FrameworkJobTests(unittest.TestCase):
    def setUp(self):
        self.job = HordeJobFramework(mm=None, bd=DummyBridgeData(), pop={"id": "job-1"})
        self.job.current_id = "job-1"
        self.job.submit_endpoint = "/api/v2/generate/submit"

    def test_fault_marks_status_and_starts_submit(self):
        self.job.start_submit_thread = mock.Mock()
        self.job.fault("boom", submit=True)
        self.assertEqual(self.job.status, JobStatus.FAULTED)
        self.job.start_submit_thread.assert_called_once()

    def test_fault_oom_status(self):
        self.job.start_submit_thread = mock.Mock()
        self.job.fault(submit=True, status=JobStatus.OUT_OF_MEMORY)
        self.assertEqual(self.job.status, JobStatus.OUT_OF_MEMORY)
        self.job.start_submit_thread.assert_called_once()

    def test_submit_faulted_uses_fault_payload(self):
        self.job.status = JobStatus.FAULTED

        class Resp:
            ok = True
            status_code = 200
            elapsed = types.SimpleNamespace(total_seconds=lambda: 0.1)

            def json(self):
                return {"reward": 0}

        self.job.session.post = mock.Mock(return_value=Resp())
        self.job.submit_job()

        sent = self.job.session.post.call_args.kwargs["json"]
        self.assertEqual(sent["id"], "job-1")
        self.assertEqual(sent["state"], "faulted")
        self.assertEqual(self.job.status, JobStatus.FAULTED)

    def test_post_submit_exception_does_not_break_done(self):
        self.job.status = JobStatus.FINALIZING
        self.job.prepare_submit_payload = mock.Mock()
        self.job.post_submit_tasks = mock.Mock(side_effect=RuntimeError("hook fail"))

        class Resp:
            ok = True
            status_code = 200
            elapsed = types.SimpleNamespace(total_seconds=lambda: 0.1)

            def json(self):
                return {"reward": 1.5}

        self.job.session.post = mock.Mock(return_value=Resp())
        self.job.submit_job()
        self.assertEqual(self.job.status, JobStatus.DONE)


if __name__ == "__main__":
    unittest.main()

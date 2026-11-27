"""Bridge Stats Tracker"""
import json
import threading
import time
from collections import deque


class BridgeStats:
    """Convenience functions for the stats"""

    stats = {}  # Backward-compatible alias; instance keeps canonical state.

    def __init__(self):
        self.kudos_record = deque()
        self.pop_record = deque()
        # We are called from diverse thread contexts
        self._mutex = threading.Lock()
        self._worker_state = self._default_worker_state()
        self.stats = {}
        BridgeStats.stats = self.stats

    @staticmethod
    def _default_worker_state():
        return {
            "kind": "idle",
            "text": "Waiting for work...",
            "progress": None,
            "transient_until": None,
        }

    def reset(self):
        with self._mutex:
            self.kudos_record = deque()
            self.pop_record = deque()
            self.stats = {}
            BridgeStats.stats = self.stats
            self._worker_state = self._default_worker_state()

    def update_pop_stats(self, node, pop_time):
        with self._mutex:
            self.pop_record.append((node, pop_time, time.time()))

            # only keep pop stats for an hour
            now = time.time()
            too_old = now - 3600
            while self.pop_record and self.pop_record[0][2] < too_old:
                self.pop_record.popleft()

            # Calculate some facts
            recent = now - (60 * 5)
            average_1_hour = (
                sum(poptime for _, poptime, _ in self.pop_record) / len(self.pop_record) if self.pop_record else 0
            )
            data_5_mins = [poptime for _, poptime, when in self.pop_record if when > recent]
            average_5_mins = sum(data_5_mins) / len(data_5_mins) if data_5_mins else 0

            if self.pop_record:
                self.stats["pop_time_avg_5_mins"] = round(average_5_mins, 2)
                self.stats["pop_time_avg_1_hour"] = round(average_1_hour, 2)

    def record_job_completed(self):
        """Record a successfully completed job for session/UI stats."""
        with self._mutex:
            self.stats["jobs_completed"] = self.stats.get("jobs_completed", 0) + 1

    def update_inference_stats(self, model_name, kudos):
        """Updates the stats for a model inference"""
        try:
            kudos = float(kudos)
        except (TypeError, ValueError):
            kudos = 0.0
        with self._mutex:
            if "inference" not in self.stats:
                self.stats["inference"] = {}
            if model_name not in self.stats["inference"]:
                self.stats["inference"][model_name] = {"kudos": 0, "count": 0}
            self.stats["inference"][model_name]["count"] += 1
            self.stats["inference"][model_name]["kudos"] = round(self.stats["inference"][model_name]["kudos"] + kudos)
            stats_for_model = self.stats["inference"][model_name]
            self.stats["inference"][model_name]["avg_kpr"] = round(
                stats_for_model["kudos"] / stats_for_model["count"],
                2,
            )
            self.stats["jobs_completed"] = self.stats.get("jobs_completed", 0) + 1

            # Remember the kudos we got awarded over the last hour
            now = time.time()
            too_old = now - 3600
            self.kudos_record.append((kudos, now))
            while self.kudos_record and self.kudos_record[0][1] < too_old:
                self.kudos_record.popleft()
            oldest = self.kudos_record[0][1] if self.kudos_record else now
            period = now - oldest

            # Calculate the total kudos
            total_kudos = sum(score for score, _ in self.kudos_record)
            # If period is less than an hour, extrapolate
            total_kudos = 0 if period < 10 else total_kudos * (3600 / period)
            jobs_per_hour = 1 if period < 10 else len(self.kudos_record) * (3600 / period)

            if self.kudos_record:
                self.stats["kudos_per_hour"] = round(total_kudos)
                self.stats["jobs_per_hour"] = round(jobs_per_hour)
                self.stats["avg_kudos_per_job"] = round(total_kudos / jobs_per_hour, 1)

    def set_worker_state(self, kind, text, progress=None, transient_ttl=None):
        """Set the current worker status for the TUI.

        kind: idle | working | done | fault
        progress: None for indeterminate, or 0-100
        transient_ttl: seconds to keep done/fault before returning to idle
        """
        transient_until = None
        if transient_ttl:
            transient_until = time.time() + float(transient_ttl)
        with self._mutex:
            self._worker_state = {
                "kind": kind,
                "text": str(text),
                "progress": progress,
                "transient_until": transient_until,
            }

    def get_worker_state(self):
        with self._mutex:
            state = dict(self._worker_state)
            transient_until = state.get("transient_until")
            if transient_until and time.time() >= transient_until:
                self._worker_state = self._default_worker_state()
                state = dict(self._worker_state)
            return state

    def get_stats_snapshot(self):
        """Return a locked snapshot for readers that need consistent cross-field reads."""
        with self._mutex:
            snapshot = dict(self.stats)
            inference = snapshot.get("inference")
            if isinstance(inference, dict):
                snapshot["inference"] = {k: dict(v) if isinstance(v, dict) else v for k, v in inference.items()}
            return snapshot

    def get_pretty_stats(self):
        """Returns a pretty string of the stats"""
        with self._mutex:
            return json.dumps(self.stats, indent=4)


bridge_stats = BridgeStats()

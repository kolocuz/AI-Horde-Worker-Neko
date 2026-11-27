"""Get and process a job from the horde"""
import contextlib
import copy
import json
import sys
import threading
import time

import requests

from worker.enums import JobStatus
from worker.logger import logger


class HordeJobFramework:
    """Get and process a job from the horde"""

    retry_interval = 1
    submit_endpoint = None
    _active_submit_endpoint = None

    def __init__(self, mm, bd, pop):
        self.model_manager = mm
        self.bridge_data = copy.deepcopy(bd)
        self.pop = pop
        self.loop_retry = 0
        self.current_id = None
        self.status = JobStatus.INIT
        self.start_time = time.time()
        self.process_time = time.time()
        self.stale_time = None
        self.submit_dict = {}
        self.headers = {"apikey": self.bridge_data.api_key}
        self.session = requests.Session()
        self._submit_lock = threading.Lock()
        self._submit_started = False

    def close(self):
        with contextlib.suppress(Exception):
            self.session.close()

    def __del__(self):
        self.close()

    def is_finished(self):
        """Check if the job is finished"""
        return self.status not in [JobStatus.WORKING, JobStatus.POLLING, JobStatus.INIT]

    def is_polling(self):
        """Check if the job is polling"""
        return self.status in [JobStatus.POLLING]

    def is_finalizing(self):
        """True if generation has finished even if upload is still remaining"""
        return self.status in [JobStatus.FINALIZING, JobStatus.FINALIZING_FAULTED]

    def is_stale(self):
        """Check if the job is stale"""
        if time.time() - self.start_time > 1200:
            return True
        if not self.stale_time:
            return False
        # Jobs which haven't started yet are not considered stale.
        if self.status != JobStatus.WORKING:
            return False
        return time.time() > self.stale_time

    def is_faulted(self):
        """Check if the job is faulted"""
        return self.status in [JobStatus.FAULTED, JobStatus.FINALIZING_FAULTED, JobStatus.OUT_OF_MEMORY]

    def is_out_of_memory(self):
        """Check if the job ran out of memory"""
        return self.status in [JobStatus.OUT_OF_MEMORY]

    def fault(self, reason=None, submit=True, status=JobStatus.FAULTED):
        """Mark the job as faulted and optionally start submit thread."""
        if reason:
            logger.error(reason)
        self.status = status
        if submit and self.status in [JobStatus.FAULTED, JobStatus.OUT_OF_MEMORY]:
            self.start_submit_thread()

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a job from a pop request
        This method MUST be extended with the specific logic for this worker
        At the end it MUST create a new thread to submit the results to the horde"""
        # Pop new request from the Horde
        if self.pop is None:
            logger.error("No pop data available for job; aborting.")
            self.status = JobStatus.FAULTED
            return
        self.process_time = time.time()
        self.status = JobStatus.WORKING
        # Continue with the specific worker logic from here
        # At the end, you must call self.start_submit_thread()

    def start_submit_thread(self):
        """Starts a thread with submit_job so that we don't wait for the upload to complete
        # Not a daemon, so that it can survive after this class is garbage collected"""
        with self._submit_lock:
            if self._submit_started:
                logger.debug("Submit thread already started for job {}", getattr(self, "current_id", "unknown"))
                return
            self._submit_started = True
        submit_thread = threading.Thread(target=self._submit_job_wrapper, args=())
        submit_thread.start()
        logger.debug("Finished job in threadpool")

    def _submit_job_wrapper(self):
        try:
            self.submit_job()
        except Exception as err:
            logger.error("Submit thread crashed for job {}: {}", getattr(self, "current_id", "unknown"), err)
            self.status = JobStatus.FAULTED
            # Best-effort terminal notification so Horde does not keep waiting.
            try:
                self.prepare_faulted_submit_payload()
                endpoint = self.submit_endpoint
                if endpoint and self.current_id is not None:
                    self.session.post(
                        self.bridge_data.horde_url + endpoint,
                        json=self.submit_dict,
                        headers=self.headers,
                        timeout=20,
                    )
            except Exception as submit_err:
                logger.warning(
                    "Best-effort faulted submit failed for job {}: {}",
                    getattr(self, "current_id", "unknown"),
                    submit_err,
                )
            self.close()

    def submit_job(self, endpoint=None):
        """Submits the job to the server to earn our kudos.
        This method MUST be extended with the specific logic for this worker
        At the end it MUST set the job state to DONE"""
        endpoint = endpoint or self.submit_endpoint or self._active_submit_endpoint
        self._active_submit_endpoint = endpoint
        if not endpoint:
            logger.error("submit_job called without endpoint for {}", self.__class__.__name__)
            self.status = JobStatus.FAULTED
            return
        try:
            if self.status == JobStatus.FAULTED or self.status == JobStatus.OUT_OF_MEMORY:
                self.prepare_faulted_submit_payload()
                self.status = JobStatus.FINALIZING_FAULTED
            else:
                self.status = JobStatus.FINALIZING
                self.prepare_submit_payload()
        except Exception as err:
            logger.error(
                "Failed to prepare submit payload for job {}: {}. Falling back to faulted payload.",
                getattr(self, "current_id", "unknown"),
                err,
            )
            self.submit_dict = {
                "id": self.current_id,
                "state": "faulted",
                "generation": "faulted",
                "seed": -1,
            }
            self.status = JobStatus.FINALIZING_FAULTED
        # Submit back to horde
        max_submit_retries = 20
        while self.is_finalizing():
            if self.loop_retry >= max_submit_retries:
                logger.error(f"Exceeded retry count {self.loop_retry} for job id {self.current_id}. Aborting job!")
                self.status = JobStatus.FAULTED
                break
            self.loop_retry += 1
            try:
                logger.debug(
                    f"posting payload with size of {round(sys.getsizeof(json.dumps(self.submit_dict)) / 1024,1)} kb",
                )
                submit_req = self.session.post(
                    self.bridge_data.horde_url + endpoint,
                    json=self.submit_dict,
                    headers=self.headers,
                    timeout=60,
                )
                logger.debug(f"Upload completed in {submit_req.elapsed.total_seconds()}")
                try:
                    submit = submit_req.json()
                except json.decoder.JSONDecodeError:
                    logger.error(
                        f"Something has gone wrong with {self.bridge_data.horde_url} during submit. "
                        f"Please inform its administrator!  (Retry {self.loop_retry}/{max_submit_retries})",
                    )
                    time.sleep(min(self.retry_interval * self.loop_retry, 10))
                    continue
                if submit_req.status_code == 404:
                    logger.warning("The job we were working on got stale. Aborting!")
                    self.status = JobStatus.FAULTED
                    break
                if not submit_req.ok:
                    submit_message = submit.get("message", "No message") if isinstance(submit, dict) else "No message"
                    if submit_req.status_code == 400:
                        logger.warning(
                            f"During gen submit, server {self.bridge_data.horde_url} "
                            f"responded with status code {submit_req.status_code}: "
                            f"Job took {round(time.time() - self.start_time,1)} seconds since queued "
                            f"and {round(time.time() - self.process_time,1)} since start. "
                            f"{submit_message}. Aborting job!",
                        )
                        self.status = JobStatus.FAULTED
                        break
                    wait_seconds = min(2 * self.loop_retry, 15)
                    logger.warning(
                        f"During gen submit, server {self.bridge_data.horde_url} "
                        f"responded with status code {submit_req.status_code}: "
                        f"{submit_message}. Waiting for {wait_seconds} seconds...  (Retry {self.loop_retry}/{max_submit_retries})",
                    )
                    if isinstance(submit, dict) and "errors" in submit:
                        logger.warning(f"Detailed Request Errors: {submit['errors']}")
                    time.sleep(wait_seconds)
                    continue
                reward = submit.get("reward", 0) if isinstance(submit, dict) else 0
                time_spent_processing = round(time.time() - self.process_time, 1)

                try:
                    reward = float(reward)
                except (TypeError, ValueError):
                    reward = 0.0
                if time_spent_processing > (reward * 3) and not self.bridge_data.suppress_speed_warnings:
                    logger.warning(
                        "This job took longer than average to process."
                        " Please consider lowering your max_power.",
                    )

                logger.info(
                    f"Submitted job with id {self.current_id} and contributed for {reward:.1f}. "
                    f"Job took {round(time.time() - self.start_time,1)} seconds since queued "
                    f"and {time_spent_processing} since start.",
                )

                try:
                    self.post_submit_tasks(submit_req)
                except Exception as err:
                    logger.warning(
                        "post_submit_tasks failed for job {}: {}",
                        getattr(self, "current_id", "unknown"),
                        err,
                    )
                if self.status == JobStatus.FINALIZING_FAULTED:
                    self.status = JobStatus.FAULTED
                else:
                    self.status = JobStatus.DONE
                break
            except (TypeError, ValueError) as err:
                logger.error(
                    "Submit payload serialization/validation failed for job {}: {}",
                    getattr(self, "current_id", "unknown"),
                    err,
                )
                self.prepare_faulted_submit_payload()
                self.status = JobStatus.FINALIZING_FAULTED
                # Try once more with a minimal faulted payload; if that also fails, exit loop.
                if self.loop_retry >= max_submit_retries:
                    self.status = JobStatus.FAULTED
                    break
                time.sleep(min(2 * self.loop_retry, 10))
                continue
            except requests.exceptions.ConnectionError:
                wait_seconds = min(5 * self.loop_retry, 30)
                logger.warning(
                    f"Server {self.bridge_data.horde_url} unavailable during submit. "
                    f"Waiting {wait_seconds} seconds...  (Retry {self.loop_retry}/{max_submit_retries})",
                )
                time.sleep(wait_seconds)
                continue
            except requests.exceptions.ReadTimeout:
                wait_seconds = min(5 * self.loop_retry, 30)
                logger.warning(
                    f"Server {self.bridge_data.horde_url} timed out during submit. "
                    f"Waiting {wait_seconds} seconds...  (Retry {self.loop_retry}/{max_submit_retries})",
                )
                time.sleep(wait_seconds)
                continue
            except requests.exceptions.RequestException as err:
                wait_seconds = min(5 * self.loop_retry, 30)
                logger.warning(
                    f"Network error during submit to {self.bridge_data.horde_url}: {err}. "
                    f"Waiting {wait_seconds} seconds...  (Retry {self.loop_retry}/{max_submit_retries})",
                )
                time.sleep(wait_seconds)
                continue
        if self.is_finalizing():
            logger.error(
                "Submit loop ended without completion for job {}. Marking as faulted.",
                getattr(self, "current_id", "unknown"),
            )
            self.status = JobStatus.FAULTED
        self.close()

    def prepare_submit_payload(self):
        """Should be overriden and prepare a self.submit_dict dictionary with the payload needed
        for this job to be submitted"""
        self.submit_dict = {}

    def prepare_faulted_submit_payload(self):
        """Prepare payload for faulted jobs. Override for endpoints with different schemas."""
        self.submit_dict = {
            "id": self.current_id,
            "state": "faulted",
            "generation": "faulted",
            "seed": -1,
        }

    def post_submit_tasks(self, submit_req):
        """Optional job which will execute only if the submit is successfull"""

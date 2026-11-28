"""Get and process a job from the horde"""
import base64
import time
import traceback
from io import BytesIO

import requests
from hordelib.blip.caption import Caption
from hordelib.clip.interrogate import Interrogator
from hordelib.safety_checker import is_image_nsfw

from worker.consts import (
    KNOWN_CAPTION_MODELS,
    KNOWN_CLIP_INTERROGATORS,
    KNOWN_POST_PROCESSORS,
    KNOWN_UPSCALERS,
)
from worker.enums import JobStatus
from worker.jobs.framework import HordeJobFramework
from worker.logger import logger
from worker.post_process import post_process
from worker.stats import bridge_stats


class InterrogationHordeJob(HordeJobFramework):
    submit_endpoint = "/api/v2/interrogate/submit"
    """Get and process an image interrogation job from the horde"""

    def __init__(self, mm, bd, pop):
        super().__init__(mm, bd, pop)
        self.current_form = self.pop.get("form")
        self.current_id = self.pop.get("id")
        self.current_payload = self.pop.get("payload", {})
        self.image = self.pop.get("image")
        self.r2_upload = self.pop.get("r2_upload", False)
        # We allow a generation a plentiful 10 seconds per form before we consider it stale
        self.result = None

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a Stable Diffusion job from a pop request"""
        logger.debug("Starting job in threadpool for model: {}", self.current_form)
        super().start_job()
        if self.status == JobStatus.FAULTED:
            self.start_submit_thread()
            return
        if self.image is None:
            logger.error("No image provided for interrogation job {}. Aborting.", self.current_id)
            self.fault(submit=True)
            return
        stale_buffer = 10  # seconds
        if self.current_form in KNOWN_UPSCALERS:
            stale_buffer = self.calculate_upscale_chunks() * 5
        self.stale_time = time.time() + stale_buffer + 5
        interrogator = None
        payload_kwargs = {}
        logger.info(f"Starting {self.current_form} alchemy {self.current_id}")
        if self.current_form == "nsfw":
            self.result = is_image_nsfw(self.image)
        elif self.current_form in KNOWN_POST_PROCESSORS:
            try:
                strength = self.current_payload.get("facefixer_strength", 0.5)
                self.image = post_process(self.current_form, self.image, strength=strength)
                self.result = "R2"
            except Exception as err:
                logger.error(
                    "Post-Processor form '{}' encountered an error when working on image . Skipping! {}",
                    self.current_form,
                    err,
                )
                self.fault(submit=True)
                return
        else:
            if self.current_form == "caption":
                blip_model_name = next(
                    (name for name in ("BLIP_Large", "BLIP") if self.model_manager.loaded_models.get(name) is not None),
                    None,
                )
                blip_model = self.model_manager.loaded_models.get(blip_model_name) if blip_model_name else None
                if blip_model is None:
                    logger.error("No caption model loaded (expected one of {}); cannot process caption form. Skipping!", sorted(KNOWN_CAPTION_MODELS))
                    self.fault(submit=True)
                    return
                interrogator = Caption(
                    blip_model,
                )
                payload_kwargs = {
                    "sample": True,
                    "num_beams": self.current_payload.get("num_beams", 7),
                    "min_length": self.current_payload.get("min_length", 20),
                    "max_length": self.current_payload.get(
                        "max_length",
                        self.current_payload.get("min_length", 20) + 30,
                    ),
                    "top_p": self.current_payload.get("top_p", 0.9),
                    "repetition_penalty": self.current_payload.get("repetition_penalty", 1.4),
                }
            if self.current_form == "interrogation":
                clip_model_name = next(
                    (name for name in ("ViT-L/14", "coca_ViT-L-14") if self.model_manager.loaded_models.get(name) is not None),
                    None,
                )
                clip_model = self.model_manager.loaded_models.get(clip_model_name) if clip_model_name else None
                if clip_model is None:
                    logger.error("No interrogation CLIP model loaded (expected one of {}); cannot process interrogation form. Skipping!", sorted(KNOWN_CLIP_INTERROGATORS))
                    self.fault(submit=True)
                    return
                interrogator = Interrogator(
                    clip_model,
                )
                payload_kwargs = {
                    "rank": self.current_payload.get(
                        "rank",
                        True,
                    ),  # TODO: Change after payload onboards rank/similarity
                    "similarity": self.current_payload.get("similarity", False),
                    "top_count": self.current_payload.get("top_count", 5),  # TODO: Add to payload
                }
            if interrogator is None:
                logger.error(
                    "Unknown interrogation form '{}'. Requesting it be skipped.",
                    self.current_form,
                )
                self.fault("Unknown interrogation form", submit=True)
                return
            try:
                self.result = interrogator(self.image, **payload_kwargs)
            except RuntimeError as err:
                logger.error(
                    "Something went wrong when processing request. "
                    "Please check your trace.log file for the full stack trace. "
                    f"Form: {self.current_form}. "
                    f"Payload: {payload_kwargs}."
                    f"URL: {self.pop.get('source_image')}.",
                )
                trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
                logger.trace(trace)
                self.fault(submit=True)
                return
        logger.info(f"Finished alchemy {self.current_id}")
        interrogator = None
        self.start_submit_thread()

    def submit_job(self, endpoint="/api/v2/interrogate/submit"):
        """Submits the job to the server to earn our kudos."""
        super().submit_job(endpoint=endpoint)

    def prepare_submit_payload(self):
        # images, seed, info, stats = txt2img(**self.current_payload)
        self.submit_dict = {"id": self.current_id}
        if self.current_form == "caption":
            self.submit_dict["result"] = {"caption": self.result}
        elif self.current_form == "interrogation":
            self.submit_dict["result"] = {"interrogation": self.result}
        elif self.current_form == "nsfw":
            self.submit_dict["result"] = {"nsfw": self.result}
        elif self.current_form in KNOWN_POST_PROCESSORS:
            buffer = BytesIO()
            # We send as WebP to avoid using all the horde bandwidth
            self.image.save(buffer, format="WebP", quality=95, method=6)
            result_value = base64.b64encode(buffer.getvalue()).decode("utf8")
            if self.r2_upload:
                try:
                    put_response = self.session.put(self.r2_upload, data=buffer.getvalue(), timeout=60)
                    logger.debug("R2 Upload response: {}", put_response)
                    if put_response.ok:
                        result_value = "R2"
                    else:
                        logger.warning(
                            "R2 upload failed with status {}; falling back to inline upload. {}",
                            put_response.status_code,
                            put_response.text[:200],
                        )
                except requests.exceptions.RequestException as err:
                    logger.warning("R2 upload failed: {}; falling back to inline upload.", err)
            self.submit_dict["result"] = {self.current_form: result_value}

    def prepare_faulted_submit_payload(self):
        self.submit_dict = {
            "id": self.current_id,
            "state": "faulted",
        }

    def post_submit_tasks(self, submit_req):
        bridge_stats.record_job_completed()

    def calculate_upscale_chunks(self):
        width, height = self.image.size

        tiles_x = (width + 511) // 512
        tiles_y = (height + 511) // 512
        return tiles_x * tiles_y

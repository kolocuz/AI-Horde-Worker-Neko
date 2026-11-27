import base64
import copy
import json
import time
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError

from worker.consts import BRIDGE_VERSION, KNOWN_INTERROGATORS, KNOWN_POST_PROCESSORS, POST_PROCESSORS_HORDELIB_MODELS, RELEASE
from worker.logger import logger
from worker.stats import bridge_stats
from worker.utils.network import request_with_retry


class JobPopper:
    retry_interval = 1
    pop_timeout = 40
    image_timeout = 10

    def __init__(self, mm, bd):
        self.model_manager = mm
        self.bridge_data = copy.deepcopy(bd)
        self.bridge_agent = f"Horde Worker Neko:{RELEASE}:https://github.com/kolocuz/AI-Horde-Worker-Neko/releases/tag/v{RELEASE}"
        backend_engine = getattr(self.bridge_data, "backend_engine", None)
        if backend_engine and str(backend_engine).lower() != "unknown":
            self.bridge_agent = (
                f"Horde Worker Neko~{backend_engine}:{RELEASE}:https://github.com/kolocuz/AI-Horde-Worker-Neko/releases/tag/v{RELEASE}"
            )
        self.pop = None
        self.headers = {"apikey": self.bridge_data.api_key}
        self.session = requests.Session()
        # This should be set by the extending class
        self.endpoint = None

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def horde_pop(self):
        """Get a job from the horde"""
        try:
            # logger.debug(self.headers)
            pop_req = self.session.post(
                self.bridge_data.horde_url + self.endpoint,
                json=self.pop_payload,
                headers=self.headers,
                timeout=self.pop_timeout,
            )
            node = pop_req.headers.get("horde-node", "unknown")
            logger.debug(f"Job pop took {pop_req.elapsed.total_seconds()} (node: {node})")
            bridge_stats.update_pop_stats(node, pop_req.elapsed.total_seconds())
        except requests.exceptions.ConnectionError:
            logger.warning(f"Server {self.bridge_data.horde_url} unavailable during pop. Waiting 10 seconds...")
            time.sleep(10)
            return None
        except TypeError:
            logger.warning(f"Server {self.bridge_data.horde_url} unavailable during pop. Waiting 2 seconds...")
            time.sleep(2)
            return None
        except requests.exceptions.ReadTimeout:
            logger.warning(f"Server {self.bridge_data.horde_url} timed out during pop. Waiting 2 seconds...")
            time.sleep(2)
            return None
        except requests.exceptions.InvalidHeader:
            logger.warning(
                f"Server {self.bridge_data.horde_url} Something is wrong with the API key you are sending. "
                "Please check your bridgeData api_key variable. Waiting 10 seconds...",
            )
            time.sleep(10)
            return None
        except requests.exceptions.RequestException as err:
            logger.warning(f"Network error during pop from {self.bridge_data.horde_url}: {err}. Waiting 10 seconds...")
            time.sleep(10)
            return None

        try:
            self.pop = pop_req.json()  # I'll use it properly later
        except json.decoder.JSONDecodeError:
            logger.error(
                f"Could not decode response from {self.bridge_data.horde_url} as json. "
                "Please inform its administrator!",
            )
            time.sleep(2)
            return None
        if not pop_req.ok:
            if isinstance(self.pop, dict):
                logger.warning(f"{self.pop.get('message', 'Job pop failed')} ({pop_req.status_code})")
                if "errors" in self.pop:
                    logger.warning(f"Detailed Request Errors: {self.pop['errors']}")
            else:
                logger.warning(f"Job pop failed with status {pop_req.status_code}: {pop_req.text[:200]}")
            time.sleep(2)
            return None
        return [self.pop]

    def report_skipped_info(self):
        job_skipped_info = self.pop.get("skipped") if isinstance(self.pop, dict) else None
        if job_skipped_info and len(job_skipped_info):
            self.skipped_info = f" Skipped Info: {job_skipped_info}."
        else:
            self.skipped_info = ""
        logger.debug(f"Server {self.bridge_data.horde_url} has no valid generations for us to do.{self.skipped_info}")
        time.sleep(self.retry_interval)

    def download_image_data(self, image_url):
        """Returns the image data, not a PIL"""
        img_data = None
        try:
            with self.session.get(image_url, stream=True, timeout=self.image_timeout) as r:
                size = r.headers.get("Content-Length", 0)
                try:
                    size = int(size)
                except (TypeError, ValueError):
                    size = 0
                if size > 5120000:
                    logger.warning(f"Provided image ({image_url}) cannot be larger than 5Mb")
                    return None
                mbs = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        if mbs == 0:
                            img_data = chunk
                        else:
                            img_data += chunk
                        mbs += 1
                        if mbs > 5:
                            logger.warning(f"Provided image ({image_url}) cannot be larger than 5Mb")
                            return None
        except (requests.exceptions.RequestException, OSError) as err:
            logger.warning(f"Could not download source image from {image_url}: {err}")
            return None
        if not img_data:
            logger.warning(f"Could not download source image from R2 {image_url}. Skipping source image.")
        return img_data

    def convert_image_data_to_pil(self, img_data):
        try:
            img = Image.open(BytesIO(img_data))
            if len(img.split()) == 4:
                return img.convert("RGBA")

            return img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            logger.warning(f"Error when creating image: {e}.")
            return None

    def report_faulted_interrogation_form(self, form):
        form_id = form.get("id")
        if not form_id:
            return
        submit_req = request_with_retry(
            self.session,
            "POST",
            self.bridge_data.horde_url + "/api/v2/interrogate/submit",
            json={"id": form_id, "state": "faulted"},
            headers=self.headers,
            timeout=20,
            retries=1,
            backoff_seconds=2,
            log_prefix=f"Faulted interrogation report {form_id}",
        )
        if submit_req is None:
            return
        if not submit_req.ok:
            logger.warning(
                "Failed to report faulted interrogation form {}: {} ({})",
                form_id,
                submit_req.text,
                submit_req.status_code,
            )


class StableDiffusionPopper(JobPopper):
    def __init__(self, mm, bd):
        super().__init__(mm, bd)
        self.endpoint = "/api/v2/generate/pop"
        # logger.debug("Cron: Starting get_loaded_model_names()")
        self.available_models = self.model_manager.get_loaded_models_names()
        # logger.debug("Cron: End get_loaded_model_names()")
        for util_model in (
            list(KNOWN_INTERROGATORS)
            + list(POST_PROCESSORS_HORDELIB_MODELS)
            + [
                "LDSR",
                "safety_checker",
            ]
        ):
            if util_model in self.available_models:
                self.available_models.remove(util_model)
        # logger.debug("Cron: Starting constructing pop payload")
        self.pop_payload = {
            "name": self.bridge_data.worker_name,
            "max_pixels": self.bridge_data.max_pixels,
            "priority_usernames": self.bridge_data.priority_usernames,
            "nsfw": self.bridge_data.nsfw,
            "blacklist": self.bridge_data.blacklist,
            "models": self.available_models,
            "allow_img2img": self.bridge_data.allow_img2img,
            "allow_painting": self.bridge_data.allow_painting,
            "allow_unsafe_ip": self.bridge_data.allow_unsafe_ip,
            "threads": self.bridge_data.max_threads,
            "allow_post_processing": self.bridge_data.allow_post_processing,
            "allow_controlnet": self.bridge_data.allow_controlnet,
            "allow_lora": self.bridge_data.allow_lora if self.model_manager.lora.are_downloads_complete() else False,
            "require_upfront_kudos": self.bridge_data.require_upfront_kudos,
            "bridge_version": BRIDGE_VERSION,
            "bridge_agent": self.bridge_agent,
        }
        # logger.debug("Cron: End constructing pop payload")

    def horde_pop(self):
        # logger.debug("Cron: Starting job pop")
        if not super().horde_pop():
            return None
        if not self.pop.get("id"):
            self.report_skipped_info()
            return None
        # In the stable diffusion popper, the whole return is always a single payload, so we return it as a list
        self.pop["source_image"] = self.download_source(self.pop.get("source_image"))
        self.pop["source_mask"] = self.download_source(self.pop.get("source_mask"))
        source_processing = self.pop.get("source_processing", "txt2img")
        if source_processing in {"img2img", "inpainting", "outpainting"} and not self.pop.get("source_image"):
            logger.warning(
                "Dropping {} job {} because source image could not be downloaded/decoded.",
                source_processing,
                self.pop.get("id"),
            )
            return None
        # logger.debug("Cron: End job pop")
        return [self.pop]

    def download_source(self, source_img):
        if not source_img:
            return None
        if source_img.startswith(("http://", "https://")):
            if img_data := self.download_image_data(source_img):
                img = self.convert_image_data_to_pil(img_data)
                if not img:
                    logger.warning("Non-image data when downloading image! Ignoring")
                return img

            logger.warning(f"Could not download source image from R2 {source_img}. Skipping source image.")
            return None
        try:
            base64_bytes = source_img.encode("utf-8")
            img_bytes = base64.b64decode(base64_bytes)
            img = self.convert_image_data_to_pil(img_bytes)
            if not img:
                logger.warning("Could not decode base64 source image. Skipping source image.")
            return img
        except (ValueError, TypeError) as err:
            logger.warning(f"Could not decode source image with error: '{err}'. Skipping source image.")
            return None


class ScribePopper(JobPopper):
    def __init__(self, mm, bd):
        super().__init__(mm, bd)
        self.endpoint = "/api/v2/generate/text/pop"
        # KAI Only ever offers one single model, so we just add it to the Horde's expected array form.
        self.available_models = [self.bridge_data.model]
        if self.bridge_data.branded_model:
            if not self.bridge_data.username:
                logger.warning("branded_model reqquested but AI Horde username could not be determined.")
            else:
                self.available_models = [f"{self.bridge_data.model}::{self.bridge_data.username}"]
        self.pop_payload = {
            "name": self.bridge_data.worker_name,
            "models": self.available_models,
            "max_length": self.bridge_data.max_length,
            "max_context_length": self.bridge_data.max_context_length,
            "priority_usernames": self.bridge_data.priority_usernames,
            "softprompts": self.bridge_data.softprompts.get(self.bridge_data.model, [])
            if self.bridge_data.openai_api is False
            else [],
            "bridge_agent": self.bridge_agent,
            "threads": self.bridge_data.max_threads,
        }

    def horde_pop(self):
        if not super().horde_pop():
            return None
        if not self.pop.get("id"):
            self.report_skipped_info()
            return None
        return [self.pop]


class InterrogationPopper(JobPopper):
    def __init__(self, mm, bd):
        super().__init__(mm, bd)
        self.endpoint = "/api/v2/interrogate/pop"
        available_forms = []
        self.available_models = self.model_manager.get_loaded_models_names()
        if "BLIP_Large" in self.available_models:
            available_forms.append("caption")
        if "safety_checker" in self.available_models:
            available_forms.append("nsfw")
        if "ViT-L/14" in self.available_models:
            available_forms.append("interrogation")
        if "post-process" in self.bridge_data.forms:
            available_forms.extend(list(KNOWN_POST_PROCESSORS))
        amount = max(self.bridge_data.queue_size, 1)
        self.pop_payload = {
            "name": self.bridge_data.worker_name,
            "forms": available_forms,
            "amount": amount,
            "priority_usernames": self.bridge_data.priority_usernames,
            "threads": self.bridge_data.max_threads,
            "bridge_version": BRIDGE_VERSION,
            "bridge_agent": self.bridge_agent,
            "max_tiles": self.bridge_data.max_power,
        }

    def horde_pop(self):
        if not super().horde_pop():
            return None
        if not self.pop.get("forms"):
            self.report_skipped_info()
            return None
        # In the interrogation popper, the forms key contains an array of payloads to execute
        current_image_url = None
        current_image = None
        non_faulted_forms = []
        for form in self.pop["forms"]:
            source_image = form.get("source_image")
            if not source_image or not isinstance(source_image, str):
                logger.warning("Interrogation form {} missing/invalid source_image; faulting form.", form.get("id"))
                self.report_faulted_interrogation_form(form)
                continue
            if source_image != current_image_url:
                current_image_url = source_image
                current_image = None
                img_data = self.download_image_data(source_image)
                if img_data:
                    current_image = self.convert_image_data_to_pil(img_data)
            if not current_image:
                logger.warning(f"Could not download source image for interrogation: {current_image_url}")
                self.report_faulted_interrogation_form(form)
                continue
            form["image"] = current_image.copy()
            non_faulted_forms.append(form)
        logger.debug(f"Popped {len(non_faulted_forms)} interrogation forms")
        return non_faulted_forms

"""The configuration of the bridge"""
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from hordelib.settings import UserSettings
from PIL import Image

from worker.argparser.stable_diffusion import args
from worker.bridge_data.framework import BridgeDataTemplate
from worker.consts import KNOWN_INTERROGATORS, POST_PROCESSORS_HORDELIB_MODELS
from worker.logger import logger
from worker.utils.network import request_with_retry, response_json


class StableDiffusionBridgeData(BridgeDataTemplate):
    """Configuration object"""

    @staticmethod
    def _safe_load_censor_image(path):
        try:
            return Image.open(path)
        except Exception as err:
            logger.warning(f"Failed to load censor image {path}: {err}. Using placeholder.")
            return Image.new("RGB", (512, 512), (0, 0, 0))

    def __init__(self):
        super().__init__(args)
        self._last_top_n_refresh = 0
        self._last_model_db_refresh = 0
        self._all_model_names = []
        self._top_n_model_names = []
        self.nsfw = os.environ.get("HORDE_NSFW", "true") == "true"
        self.censor_nsfw = os.environ.get("HORDE_CENSOR", "false") == "true"
        self.blacklist = list(filter(lambda a: a, os.environ.get("HORDE_BLACKLIST", "").split(",")))
        self.censorlist = list(filter(lambda a: a, os.environ.get("HORDE_CENSORLIST", "").split(",")))
        self.allow_img2img = os.environ.get("HORDE_IMG2IMG", "true") == "true"
        self.allow_painting = os.environ.get("HORDE_PAINTING", "true") == "true"
        self.allow_post_processing = os.environ.get("ALLOW_POST_PROCESSING", "true") == "true"
        self.allow_controlnet = os.environ.get("ALLOW_CONTROLNET", "false") == "true"
        self.allow_lora = os.environ.get("ALLOW_LORA", "false") == "true"
        raw_model_names = [m.strip() for m in os.environ.get("HORDE_MODELNAMES", "stable_diffusion").split(",") if m.strip()]
        self.model_names = raw_model_names or ["stable_diffusion"]
        self.max_pixels = 64 * 64 * 8 * self.max_power
        self.censor_image_sfw_worker = self._safe_load_censor_image("assets/nsfw_censor_sfw_worker.png")
        self.censor_image_censorlist = self._safe_load_censor_image("assets/nsfw_censor_censorlist.png")
        self.censor_image_sfw_request = self._safe_load_censor_image("assets/nsfw_censor_sfw_request.png")
        self.censor_image_csam = self._safe_load_censor_image("assets/nsfw_censor_csam.png")
        self.models_reloading = False
        self.model = None
        self.always_download = True
        self.dynamic_models = False
        self.number_of_dynamic_models = 0
        self.max_lora_cache_size = int(os.environ.get("HORDE_MAX_LORA_CACHE", "10"))
        self.models_to_skip = os.environ.get("HORDE_SKIPPED_MODELNAMES", "stable_diffusion_inpainting").split(",")
        self.predefined_models = self.model_names.copy()
        self.top_n_refresh_frequency = int(os.environ.get("HORDE_TOP_N_REFRESH", 60 * 60 * 24))
        self.model_database_refresh_frequency = int(os.environ.get("HORDE_MODEL_DB_REFRESH", 0))
        self.ram_to_leave_free = os.environ.get("HORDE_RAM_TO_LEAVE_FREE", "50%")
        self.vram_to_leave_free = os.environ.get("HORDE_VRAM_TO_LEAVE_FREE", "50%")
        self.disable_disk_cache = os.environ.get("HORDE_DISABLE_DISK_CACHE", "false") == "true"
        self.last_lora_check = None
        # Some config file options require us to actually set env vars to pass settings to third party systems
        # Where we load models from
        if not hasattr(self, "cache_home"):
            if not hasattr(self, "nataili_cache_home"):
                self.cache_home = os.environ.get("AIWORKER_CACHE_HOME", "./")
            else:
                self.cache_home = self.nataili_cache_home
        os.environ["AIWORKER_CACHE_HOME"] = self.cache_home
        # Where the temp dir and/or model cache are located
        if hasattr(self, "ray_temp_dir"):
            self.temp_dir = self.ray_temp_dir
        if hasattr(self, "temp_dir"):
            os.environ["AIWORKER_TEMP_DIR"] = self.temp_dir
        else:
            os.environ["AIWORKER_TEMP_DIR"] = "./tmp"

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_url = self.horde_url
        super().reload_data()
        with self.mutex:
            if hasattr(self, "dreamer_name") and not self.args.worker_name:
                self.worker_name = self.dreamer_name
            if not hasattr(self, "models_to_load"):
                self.models_to_load = ["top 2"]
            models_to_load = list(self.models_to_load)

        # Check for magic constants and expand them
        top_n = 0
        all_models = False
        for model in models_to_load[:]:
            # all models
            if match := re.match(r"ALL ((\w*) )?MODELS", model, re.IGNORECASE):
                models_to_load = self.get_all_models(match[2])
                all_models = True
                break  # can't be more
            if match := re.match(r"TOP (\d+)", model, re.IGNORECASE):
                models_to_load.remove(model)
                if int(match[1]) > 0:
                    top_n = int(match[1])
        if top_n:
            models_to_load.extend(self.get_top_n_models(top_n))

        unsupported_models = [model_name for model_name in models_to_load if "SDXL" in model_name]
        for model_name in unsupported_models:
            logger.warning(
                f"SDXL models are not supported in this worker version. Removing {model_name}",
            )
        if unsupported_models:
            models_to_load = [model_name for model_name in models_to_load if "SDXL" not in model_name]

        with self.mutex:
            self.models_to_load = list(models_to_load)
            dynamic_models = bool(self.dynamic_models)
            number_of_dynamic_models = self.number_of_dynamic_models
            allow_post_processing = bool(self.allow_post_processing)

        if dynamic_models and not number_of_dynamic_models:
            logger.warning(
                "Dynamic models are enabled but config option `number_of_dynamic_models` isn't set or is 0. "
                "Disabling dynamic models.",
            )
            dynamic_models = False

        if dynamic_models and (all_models or top_n > number_of_dynamic_models):
            logger.warning(
                f"Dynamic models is configured to load {number_of_dynamic_models} models, but "
                f"{top_n if top_n else 'All Models'} models "
                f"are manually configured to be loaded. Disabling dynamic models.",
            )
            logger.warning(
                "If you want to keep dynamic models enabled, remove the manual model configuration "
                " ('models_to_load') or set `number_of_dynamic_models` to a value higher than the number "
                "of manual models loaded. ",
            )
            dynamic_models = False
            number_of_dynamic_models = 0

        with self.mutex:
            self.dynamic_models = dynamic_models
            self.number_of_dynamic_models = number_of_dynamic_models
            if not self.dynamic_models:
                self.model_names = list(models_to_load)
            else:
                self.predefined_models = list(models_to_load)
                seeded = list(self.predefined_models)
                for required in ("safety_checker", "ViT-L/14"):
                    if required not in seeded:
                        seeded.insert(0, required)
                if allow_post_processing:
                    for pp_model in POST_PROCESSORS_HORDELIB_MODELS:
                        if pp_model not in seeded:
                            seeded.append(pp_model)
                self.model_names = seeded

        with self.mutex:
            if args.max_power:
                self.max_power = args.max_power
            if args.model:
                self.model = [args.model]
            if args.sfw:
                self.nsfw = False
            if args.censor_nsfw:
                self.censor_nsfw = args.censor_nsfw
            if args.blacklist:
                self.blacklist = args.blacklist
            if args.censorlist:
                self.censorlist = args.censorlist
            if args.allow_img2img:
                self.allow_img2img = args.allow_img2img
            if args.allow_painting:
                self.allow_painting = args.allow_painting
            if args.disable_dynamic_models:
                self.dynamic_models = False
            if args.disable_post_processing:
                self.allow_post_processing = False
            if args.disable_controlnet:
                self.allow_controlnet = False
            self.max_power = max(self.max_power, 2)
            self.max_pixels = 64 * 64 * 8 * self.max_power
            if "safety_checker" not in self.model_names:
                self.model_names.insert(0, "safety_checker")
            if "ViT-L/14" not in self.model_names:
                self.model_names.insert(0, "ViT-L/14")
            if self.allow_post_processing:
                for pp_model in POST_PROCESSORS_HORDELIB_MODELS:
                    if pp_model not in self.model_names:
                        self.model_names.append(pp_model)
        if (not self.initialized and not self.models_reloading) or previous_url != self.horde_url:
            logger.init(
                (
                    f"Username '{self.username}'. Server Name '{self.worker_name}'. "
                    f"Horde URL '{self.horde_url}'. Max Pixels {self.max_pixels}. "
                    "Worker Type: Dreamer"
                ),
                status="Joining Horde",
            )
        # Set hordelib settings
        UserSettings.set_ram_to_leave_free_mb(self.ram_to_leave_free)
        UserSettings.set_vram_to_leave_free_mb(self.vram_to_leave_free)
        UserSettings.disable_disk_cache.active = self.disable_disk_cache

    def check_extra_conditions_for_download_choice(self):
        return (self.dynamic_models and self.number_of_dynamic_models) or self.always_download

    def _is_valid_stable_diffusion_model(self, model_name):
        if model_name in ["safety_checker", "LDSR"]:
            return False

        if model_name in POST_PROCESSORS_HORDELIB_MODELS or model_name in KNOWN_INTERROGATORS:
            return False

        return model_name not in self.models_to_skip

    # Get all models directly from the server, not from nataili, as nataili
    # may not be loaded, e.g. in webui.
    def get_all_models(self, style=""):
        # Recognise some magic style constants
        nsfw = None
        if style:
            if style.upper() == "SFW":
                style = ""
                nsfw = False
            elif style.upper() == "NSFW":
                style = ""
                nsfw = True

        with self.mutex:
            refresh_freq = self.model_database_refresh_frequency
            if refresh_freq > 0:
                refresh_freq = max(refresh_freq, 3600)
                self.model_database_refresh_frequency = refresh_freq
            last_refresh = self._last_model_db_refresh
            cached_models = self._all_model_names[:]
            configured_models = list(self.models_to_load) if self.models_to_load else []
            allow_painting = bool(self.allow_painting)

        now = time.monotonic()
        # Should we refresh the model list?
        if refresh_freq > 0 and last_refresh and now - last_refresh < refresh_freq:
            return cached_models

        # If we're never refreshing and have a cache, just use that
        if refresh_freq == 0 and cached_models:
            return cached_models

        logger.info("Refreshing the list of all available models")
        with requests.Session() as session:
            response = request_with_retry(
                session,
                "GET",
                "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/main/stable_diffusion.json",
                timeout=10,
                retries=2,
                backoff_seconds=2,
                log_prefix="Model reference fetch",
            )
        if response is None or not response.ok:
            logger.error("Failed to retrieve the list of all available models")
            return configured_models if configured_models else []

        data = response_json(response, default={})
        if not isinstance(data, dict):
            logger.error("Model reference response is not a mapping; keeping the configured models")
            return configured_models if configured_models else []

        # Get all interesting models
        models = []
        for model in data.values():
            if not isinstance(model, dict):
                continue
            model_name = model.get("name")
            if not model_name or not self._is_valid_stable_diffusion_model(model_name):
                continue  # ignore
            if model.get("type") != "ckpt":
                continue  # ignore
            model_style = model.get("style")
            if style and model_style and style.lower() not in str(model_style).lower():
                continue  # ignore
            if nsfw is False and model.get("nsfw"):
                continue  # ignore
            if nsfw is True and not model.get("nsfw"):
                continue  # ignore
            # Add to the list of models to load
            models.append(model_name)

        models = sorted(set(models))

        # Move the standard SD model to the top of the list
        if "stable_diffusion" in models:
            models.remove("stable_diffusion")
            models.insert(0, "stable_diffusion")

        # Always have usable model something, no matter what went wrong
        if not models:
            models.append("stable_diffusion")

        # Add inpainting model if inpainting is enabled
        if allow_painting:
            models.append("stable_diffusion_inpainting")

        with self.mutex:
            self._all_model_names = models[:]
            self._last_model_db_refresh = time.monotonic()

        return models

    # Get the top n most popular models from the horde server
    def get_top_n_models(self, top_n, period="day"):
        model_list = []

        with self.mutex:
            refresh_freq = self.top_n_refresh_frequency
            if refresh_freq > 0:
                refresh_freq = max(refresh_freq, 3600)
                self.top_n_refresh_frequency = refresh_freq
            last_refresh = self._last_top_n_refresh
            cached_models = list(self._top_n_model_names)
            models_to_skip = set(self.models_to_skip or [])
            horde_url = self.horde_url

        now = time.monotonic()
        if refresh_freq > 0 and last_refresh and now - last_refresh < refresh_freq:
            model_list = cached_models

        # If we're never refreshing and have a cache, just use that
        if refresh_freq == 0 and cached_models:
            model_list = cached_models

        # Update the top n model chart
        if not model_list:
            logger.info("Refreshing the most popular model data")
            with requests.Session() as session:
                req = request_with_retry(
                    session,
                    "GET",
                    f"{horde_url}/api/v2/stats/img/models",
                    timeout=10,
                    retries=1,
                    backoff_seconds=2,
                    log_prefix="Top models stats fetch",
                )
            payload = response_json(req, default={}) if req and req.ok else {}
            models = payload.get(period, {}) if isinstance(payload, dict) else {}
            model_list = sorted(((models[model], model) for model in models), reverse=True)
            model_list = [x for x in model_list if x[1] not in models_to_skip]
            with self.mutex:
                self._top_n_model_names = list(model_list)
                self._last_top_n_refresh = time.monotonic()

        top = [x[1] for x in model_list[:top_n]]

        # Always return something, no matter what went wrong
        if not top:
            top.append("stable_diffusion")
        return top

    @logger.catch(reraise=True)
    def check_models(self, model_manager):
        """Override framework version to handle loras as well"""
        if self.models_reloading:
            return
        super().check_models(model_manager)
        if not self.allow_lora:
            return
        if model_manager.lora is None:  # The lora manager is not loaded yet
            return
        if not model_manager.lora.are_downloads_complete():
            return
        # We only want to check and download new default loras once per day

        if self.last_lora_check and self.last_lora_check > datetime.now(timezone.utc) - timedelta(days=1):
            return
        self.last_lora_check = datetime.now(timezone.utc)
        # This initiates the threads that download the default loras, so it will immediately continue
        model_manager.lora.max_adhoc_disk = self.max_lora_cache_size * 1024
        model_manager.lora.download_default_loras(nsfw=self.nsfw)
        model_manager.lora.wait_for_downloads(900)
        model_manager.lora.wait_for_adhoc_reset(15)
        model_manager.lora.delete_unused_loras(30)

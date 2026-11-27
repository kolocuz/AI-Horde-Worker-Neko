"""The configuration of the bridge"""
import os

from worker.argparser.interrogation import args
from worker.bridge_data.framework import BridgeDataTemplate
from worker.consts import POST_PROCESSORS_HORDELIB_MODELS
from worker.logger import logger


class InterrogationBridgeData(BridgeDataTemplate):
    """Configuration object"""

    def __init__(self):
        super().__init__(args)
        self.forms = [f.strip() for f in os.environ.get("HORDE_INTERROGATION_FORMS", "caption").split(",") if f.strip()]
        self.model_names = []
        # Where we load models from
        if not hasattr(self, "cache_home"):
            if not hasattr(self, "nataili_cache_home"):
                self.cache_home = os.environ.get("AIWORKER_CACHE_HOME", "./")
            else:
                self.cache_home = self.nataili_cache_home

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_url = self.horde_url
        super().reload_data()
        with self.mutex:
            if hasattr(self, "alchemist_name") and not self.args.worker_name:
                self.worker_name = self.alchemist_name
            if args.forms:
                self.forms = args.forms
            # Ensure no duplicates or empty forms
            self.forms = [f for f in list(dict.fromkeys(self.forms)) if f and f.strip()]
        with self.mutex:
            required_models = set()
            if "nsfw" in self.forms:
                required_models.add("safety_checker")
            if "caption" in self.forms:
                required_models.add("BLIP_Large")
            if "interrogation" in self.forms:
                required_models.add("ViT-L/14")
            if "post-process" in self.forms:
                required_models.update(POST_PROCESSORS_HORDELIB_MODELS)

            managed_models = {"safety_checker", "BLIP_Large", "ViT-L/14", *POST_PROCESSORS_HORDELIB_MODELS}
            self.model_names = [model for model in self.model_names if model not in managed_models or model in required_models]

            for required in ("safety_checker", "BLIP_Large", "ViT-L/14"):
                if required in required_models and required not in self.model_names:
                    self.model_names.append(required)
            if "post-process" in self.forms:
                for model in POST_PROCESSORS_HORDELIB_MODELS:
                    if model not in self.model_names:
                        self.model_names.append(model)
            # Deduplicate while preserving order
            self.model_names = list(dict.fromkeys(self.model_names))
        if (not self.initialized and not self.models_reloading) or previous_url != self.horde_url:
            logger.init(
                (
                    f"Username '{self.username}'. Server Name '{self.worker_name}'. "
                    f"Horde URL '{self.horde_url}'. Forms {self.forms}. "
                    "Worker Type: Alchemist"
                ),
                status="Joining Horde",
            )

    def check_extra_conditions_for_download_choice(self):
        """Extend if any condition on the specifics for this bridge_data will force a 'y' result"""
        return True

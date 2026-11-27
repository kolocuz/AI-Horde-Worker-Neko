"""The configuration of the bridge"""
import os

import requests

from worker.argparser.scribe import args
from worker.bridge_data.framework import BridgeDataTemplate
from worker.logger import logger
from worker.utils.network import request_with_retry, response_json


class KoboldAIBridgeData(BridgeDataTemplate):
    """Configuration object"""

    def __init__(self):
        super().__init__(args)
        self.kai_available = False
        self.model = None
        self.kai_url = "http://localhost:5000"
        self.max_length = int(os.environ.get("HORDE_MAX_LENGTH", "80"))
        self.max_context_length = int(os.environ.get("HORDE_MAX_CONTEXT_LENGTH", "1024"))
        self.branded_model = os.environ.get("HORDE_BRANDED_MODEL", "false") == "true"
        self.softprompts = {}
        self.current_softprompt = None
        self.openai_api = os.environ.get("HORDE_BACKEND_OPENAI_API", "false") == "true"

        self.nsfw = os.environ.get("HORDE_NSFW", "true") == "true"
        self.blacklist = list(filter(lambda a: a, os.environ.get("HORDE_BLACKLIST", "").split(",")))

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_url = self.horde_url
        super().reload_data()
        with self.mutex:
            if hasattr(self, "scribe_name") and not self.args.worker_name:
                self.worker_name = self.scribe_name
            if args.kai_url:
                self.kai_url = args.kai_url
            if args.sfw:
                self.nsfw = False
            if args.blacklist:
                self.blacklist = args.blacklist
            if args.openai_api:
                self.openai_api = args.openai_api
        self.validate_kai()
        if self.kai_available and ((not self.initialized and not self.models_reloading) or previous_url != self.horde_url):
            logger.init(
                (
                    f"Username '{self.username}'. Server Name '{self.worker_name}'. "
                    f"Horde URL '{self.horde_url}'. KoboldAI Client URL '{self.kai_url}'"
                    "Worker Type: Scribe"
                ),
                status="Joining Horde",
            )

    @logger.catch(reraise=True)
    def validate_kai(self):
        logger.debug("Retrieving settings from KoboldAI Client...")
        try:
            with requests.Session() as session:
                version_req = request_with_retry(
                    session,
                    "GET",
                    self.kai_url + "/version",
                    retries=1,
                    timeout=10,
                    log_prefix="KoboldAI version check",
                )
                if version_req is not None and version_req.ok:
                    engine = "aphrodite"
                else:
                    engine = "unknown"
                    logger.warning(
                        "Unable to determine OpenAI API compatible backend engine. "
                        "Will report it as unknown to the Horde which will lead to less kudos rewards.",
                    )

                if self.openai_api:
                    req = request_with_retry(
                        session,
                        "GET",
                        self.kai_url + "/v1/models",
                        retries=1,
                        timeout=10,
                        log_prefix="KoboldAI OpenAI model list",
                    )
                    if req is None or not req.ok:
                        raise requests.exceptions.RequestException("Failed to retrieve OpenAI model list")
                    data = response_json(req, default={})
                    self.model = data["data"][0]["id"]
                    logger.debug([self.model, self.backend_engine])
                    engine += "~oai"
                else:
                    req = request_with_retry(
                        session,
                        "GET",
                        self.kai_url + "/api/latest/model",
                        retries=1,
                        timeout=10,
                        log_prefix="KoboldAI model",
                    )
                    if req is None or not req.ok:
                        raise requests.exceptions.RequestException("Failed to retrieve KoboldAI model")
                    data = response_json(req, default={})
                    self.model = data["result"]
                    engine += "~kai"

                # Normalize huggingface and local downloaded model names
                if "/" not in self.model:
                    self.model = self.model.replace("_", "/", 1)

                if not self.openai_api:
                    if self.model not in self.softprompts:
                        req = request_with_retry(
                            session,
                            "GET",
                            self.kai_url + "/api/latest/config/soft_prompts_list",
                            retries=1,
                            timeout=10,
                            log_prefix="KoboldAI softprompts list",
                        )
                        if req is None or not req.ok:
                            raise requests.exceptions.RequestException("Failed to retrieve softprompts list")
                        data = response_json(req, default={})
                        self.softprompts[self.model] = [sp["value"] for sp in data.get("values", [])]
                    req = request_with_retry(
                        session,
                        "GET",
                        self.kai_url + "/api/latest/config/soft_prompt",
                        retries=1,
                        timeout=10,
                        log_prefix="KoboldAI current softprompt",
                    )
                    if req is None or not req.ok:
                        raise requests.exceptions.RequestException("Failed to retrieve current softprompt")
                    data = response_json(req, default={})
                    self.current_softprompt = data.get("value")
            self.backend_engine = engine
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Server {self.kai_url} is up but does not appear to be a KoboldAI server.")
            self.kai_available = False
            return
        except (
            requests.exceptions.RequestException,
            AttributeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            logger.error(f"Server {self.kai_url} is not reachable or returned unexpected response: {exc}")
            self.kai_available = False
            return
        self.kai_available = True

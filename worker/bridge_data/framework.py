"""The configuration of the bridge"""
import copy
import os
import random
import sys
import threading

import requests
import yaml

from worker.consts import BRIDGE_CONFIG_FILE, BRIDGE_VERSION
from worker.logger import logger
from worker.utils.network import request_with_retry, response_json


class BridgeDataTemplate:
    """Configuration object"""

    def __init__(self, args):
        random.seed()
        # I have to pass the args from the extended class, as the framework class doesn't
        # know what kind of polymorphism this worker is using
        self.args = args
        self.mutex = threading.Lock()

        # If there is a YAML config file, load it
        self.load_config()

        self.horde_url = os.environ.get("HORDE_URL", "https://stablehorde.net")
        # Give a cool name to your instance
        self.worker_name = os.environ.get(
            "HORDE_WORKER_NAME",
            f"Automated Instance #{random.randint(-100000000, 100000000)}",
        )
        # The api_key identifies a unique user in the horde
        self.api_key = os.environ.get("HORDE_API_KEY", "0000000000")
        # Put other users whose prompts you want to prioritize.
        # The owner's username is always included so you don't need to add it here,
        # unless you want it to have lower priority than another user
        self.priority_usernames = list(filter(lambda a: a, os.environ.get("HORDE_PRIORITY_USERNAMES", "").split(",")))
        self.max_power = int(os.environ.get("HORDE_MAX_POWER", 8))
        self.max_threads = int(os.environ.get("HORDE_MAX_THREADS", 1))
        self.queue_size = int(os.environ.get("HORDE_QUEUE_SIZE", 0))
        self.allow_unsafe_ip = os.environ.get("HORDE_ALLOW_UNSAFE_IP", "true") == "true"
        self.require_upfront_kudos = os.environ.get("REQUIRE_UPFRONT_KUDOS", "false") == "true"
        self.stats_output_frequency = int(os.environ.get("STATS_OUTPUT_FREQUENCY", 30))
        self.disable_terminal_ui = os.environ.get("DISABLE_TERMINAL_UI", "false") == "true"
        self.initialized = False
        self.username = None
        self.models_reloading = False
        self.max_models_to_download = 10
        self.suppress_speed_warnings = False
        self.backend_engine = os.environ.get("HORDE_BACKEND_ENGINE", "unknown")

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key == "mutex":
                setattr(result, key, threading.Lock())
            elif key == "args":
                setattr(result, key, copy.copy(value))
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result

    def load_config(self):
        # YAML config
        if os.path.exists(BRIDGE_CONFIG_FILE):
            try:
                with open(BRIDGE_CONFIG_FILE, "rt", encoding="utf-8", errors="ignore") as configfile:
                    config = yaml.safe_load(configfile)
            except (OSError, yaml.YAMLError) as err:
                logger.warning("Failed to load config {}: {}. Keeping previous config.", BRIDGE_CONFIG_FILE, err)
                return None
            if not isinstance(config, dict):
                logger.warning("Config file {} does not contain a mapping; ignoring it.", BRIDGE_CONFIG_FILE)
                return None
            # Map the config's values directly into this instance's properties
            protected = {"mutex", "args", "model_database", "SENSITIVE_KEYS"}
            for key, value in config.items():
                if key.startswith("_") or key in protected:
                    continue
                setattr(self, key, value)
            return True  # loaded
        return None

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        with self.mutex:
            previous_api_key = self.api_key
            self.load_config()
            if self.args.api_key:
                self.api_key = self.args.api_key
            if self.args.worker_name:
                self.worker_name = self.args.worker_name
            if self.args.horde_url:
                self.horde_url = self.args.horde_url
            if self.args.priority_usernames:
                self.priority_usernames = self.args.priority_usernames
            if self.args.max_threads is not None:
                self.max_threads = self.args.max_threads
            if self.args.queue_size is not None:
                self.queue_size = self.args.queue_size
            if self.args.allow_unsafe_ip:
                self.allow_unsafe_ip = self.args.allow_unsafe_ip
            if self.args.max_power is not None:
                self.max_power = self.args.max_power
            try:
                self.max_power = max(int(self.max_power), 2)
            except (TypeError, ValueError):
                self.max_power = 8
            for attr in ("max_threads", "queue_size", "stats_output_frequency", "max_models_to_download"):
                try:
                    setattr(self, attr, int(getattr(self, attr)))
                except (TypeError, ValueError):
                    continue
            self.max_threads = max(1, self.max_threads)
            self.queue_size = max(0, self.queue_size)
            self.stats_output_frequency = max(0, self.stats_output_frequency)
            self.max_models_to_download = max(0, self.max_models_to_download)
            need_user_refresh = not self.initialized or previous_api_key != self.api_key
            api_key = self.api_key
            horde_url = self.horde_url

        if need_user_refresh:
            try:
                with requests.Session() as session:
                    user_req = request_with_retry(
                        session,
                        "GET",
                        f"{horde_url}/api/v2/find_user",
                        headers={"apikey": api_key},
                        timeout=10,
                        retries=1,
                        backoff_seconds=2,
                        log_prefix="find_user",
                    )
                user_data = response_json(user_req, default={})
                username = user_data.get("username", "N/A") if user_req and user_req.ok else "N/A"
                if username == "N/A":
                    logger.warning(f"Server {horde_url} error during find_user. Setting username 'N/A'")
            except Exception:
                logger.warning(f"Server {horde_url} error during find_user. Setting username 'N/A'")
                username = "N/A"
            with self.mutex:
                self.username = username

    @logger.catch(reraise=True)
    def check_models(self, model_manager):
        """Check to see if we have the models needed"""
        if self.models_reloading:
            return
        if not self.initialized:
            logger.init("Models", status="Checking")
        models_exist = True
        not_found_models = []
        for model in self.model_names.copy():
            # logger.info(f"Checking: {model}")
            model_info = model_manager.models.get(model, None)
            if not model_info:
                logger.warning(
                    f"Model name requested {model} in bridgeData is unknown to us. "
                    "Please check your configuration. Aborting!",
                )
                with self.mutex:
                    if model in self.model_names:
                        self.model_names.remove(model)
                continue
            if int(model_info.get("min_bridge_version", 0)) > BRIDGE_VERSION:
                logger.warning(
                    f"Model requested {model} in bridgeData is not supported in bridge version {BRIDGE_VERSION}. "
                    "Please upgrade your bridge. Skipping.",
                )
                with self.mutex:
                    if model in self.model_names:
                        self.model_names.remove(model)
                continue
            if model in model_manager.get_loaded_models_names():
                continue
            # TODO: Remove `self.args.skip_md5 or ` after fully deprecating arg.
            skip_md5 = getattr(self.args, "skip_md5", False)
            skip_checksum = getattr(self.args, "skip_checksum", False)
            if not model_manager.validate_model(model, skip_checksum=skip_md5 or skip_checksum):
                logger.debug(f"Model {model} not found or has wrong checksum")
                if (
                    model not in model_manager.get_available_models_by_types()
                    or model_manager.count_available_models_by_types() + len(not_found_models)
                    < self.max_models_to_download
                ):
                    models_exist = False
                    not_found_models.append(model)
                else:
                    logger.debug(f"Downloading Model {model} would exceed max_models_to_download. Skipping")
                    with self.mutex:
                        if model in self.model_names:
                            self.model_names.remove(model)
        if not models_exist:
            if self.args.yes or self.check_extra_conditions_for_download_choice():
                choice = "y"
            elif not sys.stdin.isatty():
                logger.error(
                    "Missing models {} and no interactive terminal available. "
                    "Run with --yes or download models manually.",
                    not_found_models,
                )
                sys.exit(1)
            else:
                choice = input(
                    "You do not appear to have downloaded the models needed yet.\n"
                    "You need at least a main model to proceed. "
                    f"Would you like to download your prespecified models?\n\
                y: Download {not_found_models} (default).\n\
                n: Abort and exit\n\
                all: Download all basic models (This can take a significant amount of time and bandwidth)\n\
                Please select an option: ",
                )
            if choice not in ["y", "Y", "", "yes", "all", "a"]:
                sys.exit(1)

            if choice in ["all", "a"]:
                if not model_manager.download_all():
                    logger.error("Something went wrong when downloading all models.")
            else:
                for model in not_found_models:
                    if not model_manager.download_model(model):
                        logger.error(
                            "Something went wrong when downloading model {}. It may not fit the expected checksum.",
                            model,
                        )
                        with self.mutex:
                            if model in self.model_names:
                                self.model_names.remove(model)
        if not self.initialized:
            logger.init_ok("Models", status="OK")
        if os.path.exists("bridgeData.py") or os.path.exists(BRIDGE_CONFIG_FILE):
            if not self.initialized:
                logger.init_ok("Bridge Config", status="OK")
        elif sys.stdin.isatty() and input(
            "You do not appear to have a bridgeData configuration file. "
            "Would you like to create it from the template now? (y/n)",
        ) in ["y", "Y", "", "yes"]:
            with open("bridgeData_template.yaml", "r", encoding="utf-8") as firstfile, open(
                "bridgeData.yaml",
                "a",
                encoding="utf-8",
            ) as secondfile:
                for line in firstfile:
                    secondfile.write(line)
            logger.message(
                "bridgeData.yaml created. Bridge will exit. "
                "Please edit bridgeData.yaml with your setup and restart the worker",
            )
            sys.exit(2)
        elif not os.path.exists("bridgeData.py") and not os.path.exists(BRIDGE_CONFIG_FILE):
            logger.error(
                "No bridgeData configuration file found and no interactive terminal available. "
                "Create bridgeData.yaml from bridgeData_template.yaml.",
            )
            sys.exit(2)

    def check_extra_conditions_for_download_choice(self):
        """Extend if any condition on the specifics for this bridge_data will force a 'y' result"""
        return False

    def reload_models(self, model_manager):
        """Reloads models - Note this is IN A THREAD"""
        with self.mutex:
            if self.models_reloading:
                return
            self.models_reloading = True
        thread = threading.Thread(target=self._reload_models, args=(model_manager,), daemon=True)
        thread.start()

    @logger.catch(reraise=True)
    def _reload_models(self, model_manager):
        try:
            with self.mutex:
                model_names = self.model_names[:]
            for model in model_manager.get_loaded_models_names():
                if model not in model_names:
                    logger.init(f"{model}", status="Unloading")
                    model_manager.unload_model(model)
            for model in model_names:
                if model not in model_manager.get_loaded_models_names():
                    if model == "safety_checker":
                        success = model_manager.load(model, cpu_only=True)
                    else:
                        success = model_manager.load(model)
                    if not success:
                        logger.init_err(f"{model}", status="Error")
            with self.mutex:
                self.initialized = True
        finally:
            with self.mutex:
                self.models_reloading = False

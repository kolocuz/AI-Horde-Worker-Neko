import os
from pathlib import Path

import yaml

from worker.consts import BRIDGE_CONFIG_FILE


def _load_config():
    config_file_as_path = Path(BRIDGE_CONFIG_FILE)
    if not config_file_as_path.exists():
        return None
    try:
        with open(config_file_as_path, "rt", encoding="utf-8", errors="ignore") as configfile:
            config = yaml.safe_load(configfile)
    except (OSError, yaml.YAMLError):
        return None
    return config if isinstance(config, dict) else None


def set_worker_env_vars_from_config():
    config = _load_config()
    if config is None:
        return False
    cache_home = config.get("cache_home") or config.get("nataili_cache_home")
    temp_dir = config.get("temp_dir") or config.get("ray_temp_dir")
    if cache_home:
        os.environ["AIWORKER_CACHE_HOME"] = str(cache_home)
    if temp_dir:
        os.environ["AIWORKER_TEMP_DIR"] = str(temp_dir)
    return True


def get_models_to_load():
    config = _load_config()
    if config is None:
        return None
    models = config.get("models_to_load", [])
    return models if isinstance(models, list) else []

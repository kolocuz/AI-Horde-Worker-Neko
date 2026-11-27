"""This is the bridge, It connects the horde with the ML processing"""
# isort: off
# We need to import the argparser first, as it sets the necessary Switches
import contextlib

from worker.argparser.interrogation import args
from worker.utils.set_envs import set_worker_env_vars_from_config

set_worker_env_vars_from_config()  # Get `cache_home` from `bridgeconfig.yaml` into the environment variable
import hordelib
from worker.utils.transformers_compat import patch_transformers_modeling_utils

patch_transformers_modeling_utils()

# We need to remove these, to avoid comfyUI trying to use them
hordelib.initialise()
from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.horde import SharedModelManager

# isort: on

from worker.bridge_data.interrogation import InterrogationBridgeData
from worker.logger import logger, quiesce_logger, set_logger_verbosity
from worker.utils.process_lock import ProcessLock
from worker.workers.interrogation import InterrogationWorker

if __name__ == "__main__":
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)

    try:
        ProcessLock("horde-worker-alchemy").acquire()
    except RuntimeError as err:
        logger.error("{}", err)
        raise SystemExit(1)

    bridge_data = None
    try:
        bridge_data = InterrogationBridgeData()
        bridge_data.reload_data()
        SharedModelManager.load_model_managers(
            [
                MODEL_CATEGORY_NAMES.safety_checker,
                MODEL_CATEGORY_NAMES.clip,
                MODEL_CATEGORY_NAMES.blip,
                MODEL_CATEGORY_NAMES.codeformer,
                MODEL_CATEGORY_NAMES.esrgan,
                MODEL_CATEGORY_NAMES.gfpgan,
            ],
        )
        worker = InterrogationWorker(SharedModelManager.manager, bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    except Exception as err:
        logger.opt(exception=err).critical("Fatal error in bridge_alchemy startup/run: {}", err)
    finally:
        if bridge_data is not None:
            with contextlib.suppress(Exception):
                logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")

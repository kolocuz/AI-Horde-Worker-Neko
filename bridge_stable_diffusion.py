"""This is the bridge, It connects the horde with the ML processing"""

# isort: off
# We need to import the argparser first, as it sets the necessry Switches
from worker.argparser.stable_diffusion import args
from worker.utils.set_envs import set_worker_env_vars_from_config

set_worker_env_vars_from_config()  # Get `cache_home` from `bridgeconfig.yaml` into the environment variable

import hordelib
from worker.utils.transformers_compat import patch_transformers_modeling_utils

patch_transformers_modeling_utils()

# We need to remove these, to avoid comfyUI trying to use them
hordelib.initialise()
from hordelib.horde import SharedModelManager
from hordelib.consts import MODEL_CATEGORY_NAMES

# isort: on
from worker.bridge_data.stable_diffusion import StableDiffusionBridgeData
from worker.logger import logger, quiesce_logger, set_logger_verbosity
from worker.utils.process_lock import ProcessLock
from worker.workers.stable_diffusion import StableDiffusionWorker


def main():
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)
    # TODO: Remove check after fully deprecating arg.
    if args.skip_md5:
        logger.warning("DeprecationWarning: `--skip_md5` has been deprecated. Please use `--skip_checksum` instead.")

    try:
        ProcessLock("horde-worker-stable-diffusion").acquire()
    except RuntimeError as err:
        logger.error("{}", err)
        raise SystemExit(1) from err

    bridge_data = None
    try:
        bridge_data = StableDiffusionBridgeData()
        bridge_data.reload_data()

        SharedModelManager.load_model_managers(
            [
                MODEL_CATEGORY_NAMES.blip,
                MODEL_CATEGORY_NAMES.clip,
                MODEL_CATEGORY_NAMES.compvis,
                MODEL_CATEGORY_NAMES.controlnet,
                MODEL_CATEGORY_NAMES.codeformer,
                MODEL_CATEGORY_NAMES.gfpgan,
                MODEL_CATEGORY_NAMES.esrgan,
                MODEL_CATEGORY_NAMES.safety_checker,
                MODEL_CATEGORY_NAMES.lora,
                MODEL_CATEGORY_NAMES.ti,
            ],
        )

        worker = StableDiffusionWorker(SharedModelManager.manager, bridge_data)

        worker.model_manager = SharedModelManager.manager

        if bridge_data.allow_controlnet:
            annotators_preloaded_successfully = SharedModelManager.preloadAnnotators()
            if not annotators_preloaded_successfully:
                logger.warning("Annotators were not preloaded. ControlNet will not be available.")
                bridge_data.allow_controlnet = False

        worker.start()

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    except Exception as err:
        logger.opt(exception=err).critical("Fatal error in stable diffusion bridge")
    finally:
        if bridge_data is not None:
            logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")


if __name__ == "__main__":
    main()

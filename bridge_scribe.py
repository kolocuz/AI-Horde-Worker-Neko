"""This is the bridge, It connects the horde with the ML processing"""
# isort: off
# We need to import the argparser first, as it sets the necessary Switches
import contextlib

from worker.argparser.scribe import args
from worker.utils.set_envs import set_worker_env_vars_from_config

set_worker_env_vars_from_config()  # Get `cache_home` from `bridgeconfig.yaml` into the environment variable

from worker.bridge_data.scribe import KoboldAIBridgeData  # noqa: E402
from worker.logger import logger, quiesce_logger, set_logger_verbosity  # noqa: E402
from worker.utils.process_lock import ProcessLock  # noqa: E402
from worker.workers.scribe import ScribeWorker  # noqa: E402

# isort: on


def main():
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)
    try:
        ProcessLock("horde-worker-scribe").acquire()
    except RuntimeError as err:
        logger.error("{}", err)
        raise SystemExit(1) from err
    bridge_data = None
    try:
        bridge_data = KoboldAIBridgeData()
        bridge_data.reload_data()
        worker = ScribeWorker(bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    except Exception as err:
        logger.opt(exception=err).critical("Fatal error in bridge_scribe startup/run: {}", err)
    finally:
        if bridge_data is not None:
            with contextlib.suppress(Exception):
                logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")


if __name__ == "__main__":
    main()

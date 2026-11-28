"""Post process images"""
import threading

import rembg
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from worker.logger import logger

hordelib = HordeLib()
_REMBG_SESSION = None
_REMBG_LOCK = threading.Lock()


# TODO: move to hordelib or ComfyUI
def strip_background(payload):
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        with _REMBG_LOCK:
            if _REMBG_SESSION is None:
                _REMBG_SESSION = rembg.new_session("u2net")
    with _REMBG_LOCK:
        return rembg.remove(
            payload["source_image"],
            session=_REMBG_SESSION,
            only_mask=False,
            alpha_matting=10,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )


# At the bottom, as we need to define the method first
KNOWN_POST_PROCESSORS = {
    "RealESRGAN_x4plus": hordelib.image_upscale,
    "RealESRGAN_x2plus": hordelib.image_upscale,
    "RealESRGAN_x4plus_anime_6B": hordelib.image_upscale,
    "NMKD_Siax": hordelib.image_upscale,
    "4x_AnimeSharp": hordelib.image_upscale,
    "4x_NMKD_Superscale_SP": hordelib.image_upscale,
    "strip_background": strip_background,
    "GFPGAN": hordelib.image_facefix,
    "CodeFormers": hordelib.image_facefix,
}


def post_process(model, image, strength=1.0):
    """This is the post-processing function,
    it takes the model name, and the image, and returns the post processed image"""
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        strength = 1.0
    if strength != 1.0 and model in {"RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B", "NMKD_Siax", "4x_AnimeSharp", "4x_NMKD_Superscale_SP", "GFPGAN", "CodeFormers"}:
        logger.debug(f"Post processor {model} received strength={strength}, but current hordelib pipeline does not apply it; using processor default.")
    if model not in KNOWN_POST_PROCESSORS:
        logger.warning(f"Post processor {model} is unknown. Returning original image")
        return image

    if model != "strip_background":
        try:
            if not SharedModelManager.manager.is_model_loaded(model):
                logger.init(f"{model}", status="Loading")
                load_result = SharedModelManager.manager.load(model)
                if not load_result:
                    logger.init_err(f"{model}", status="Error")
                    return image
                logger.init_ok(f"{model}", status="Success")
        except Exception as err:
            logger.warning(f"Failed to load post-processor {model}: {err}. Returning original image.")
            return image

    try:
        pprocessor = KNOWN_POST_PROCESSORS[model]
        payload = {
            "model": model,
            "source_image": image,
        }
        return pprocessor(payload)
    except Exception as err:
        logger.warning(f"Post-processor {model} failed: {err}. Returning original image.")
        return image

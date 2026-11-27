"""Utility script to show available models."""

import sys

# isort: off
import hordelib
from worker.utils.transformers_compat import patch_transformers_modeling_utils

patch_transformers_modeling_utils()
hordelib.initialise()

from hordelib.horde import SharedModelManager  # noqa: E402
from hordelib.shared_model_manager import MODEL_CATEGORY_NAMES  # noqa: E402

# isort: on

if __name__ == "__main__":
    # TODO: huggingface_hub or some way to use token instead of username/password
    SharedModelManager.load_model_managers(
        [MODEL_CATEGORY_NAMES.compvis],
    )
    mm = SharedModelManager.manager
    if mm is None or not hasattr(mm, "compvis") or mm.compvis is None:
        print("CompVis model manager is not available.")
        raise SystemExit(1)

    filtered_models = mm.compvis.get_filtered_models(type="ckpt") or {}

    sorted_model_names = sorted(filtered_models.keys())

    ppmodels = ""
    for model_name in sorted_model_names:
        if model_name == "LDSR":
            continue
        ppmodels += model_name
        if filtered_models[model_name].get("description"):
            ppmodels += f" : {filtered_models[model_name].get('description')}"
        ppmodels += "\n"
    print(f"## Known ckpt Models ##\n{ppmodels}")
    try:
        if sys.stdin.isatty():
            input("Press ENTER to continue")
    except EOFError:
        pass

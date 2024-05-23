from pathlib import Path
import folder_paths
from .dit import load_dit

MAX_RESOLUTION=8192

class DitCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "ExtraModels/DiT"
    TITLE = "DitCheckpointLoader"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = load_dit(
            model_path = ckpt_path,
        )
        return out[:3]
    
NODE_CLASS_MAPPINGS = {
    "DitCheckpointLoader":DitCheckpointLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DitCheckpointLoader":"DitCheckpointLoaderSimple",
}


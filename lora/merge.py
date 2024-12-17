import torch
import os
from hydit.config import get_args
from hydit.modules.models import HUNYUAN_DIT_MODELS

from hydit.inference import _to_tuple

args = get_args()

image_size = _to_tuple(args.image_size)
latent_size = (image_size[0] // 8, image_size[1] // 8)

model = HUNYUAN_DIT_MODELS[args.model](
    args,
    input_size=latent_size,
    log_fn=print,
)
model_path = os.path.join(
    args.model_root, "t2i", "model", f"pytorch_model_{args.load_key}.pt"
)
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

print(f"Loading model from {model_path}")
model.load_state_dict(state_dict)

print(f"Loading lora from {args.lora_ckpt}")
model.load_adapter(args.lora_ckpt)
model.merge_and_unload()

torch.save(model.state_dict(), args.output_merge_path)
print(f"Model saved to {args.output_merge_path}")

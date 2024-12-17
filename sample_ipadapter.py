from pathlib import Path

from loguru import logger

from mllm.dialoggen_demo import DialogGen
from hydit.config import get_args
from hydit.inference_ipadapter import End2End

from torchvision import transforms as T
import numpy as np

norm_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

from PIL import Image


def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    gen = End2End(args, models_root_path)

    # Try to enhance prompt
    if args.enhance:
        logger.info("Loading DialogGen model (for prompt enhancement)...")
        enhancer = DialogGen(str(models_root_path / "dialoggen"), args.load_4bit)
        logger.info("DialogGen model loaded.")
    else:
        enhancer = None

    return args, gen, enhancer


if __name__ == "__main__":

    args, gen, enhancer = inferencer()

    if enhancer:
        logger.info("Prompt Enhancement...")
        success, enhanced_prompt = enhancer(args.prompt)
        if not success:
            logger.info("Sorry, the prompt is not compliant, refuse to draw.")
            exit()
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
    else:
        enhanced_prompt = None
   
    # # Run inference
    logger.info("Generating images...")
    height, width = args.image_size

    ref_image = Image.open(args.ref_image_path).convert("RGB")
    i_scale = args.i_scale

    results = gen.predict(
        args.prompt,
        height=height,
        width=width,
        image=ref_image,
        i_scale=i_scale,
        t_scale=1,
        seed=3333,
        enhanced_prompt=enhanced_prompt,
        negative_prompt=args.negative,
        infer_steps=args.infer_steps,
        guidance_scale=3,
        batch_size=args.batch_size,
        src_size_cond=args.size_cond,
    )
    images = results["images"]

    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    # Find the first available index
    all_files = list(save_dir.glob("*.png"))
    if all_files:
        start = max([int(f.stem) for f in all_files]) + 1
    else:
        start = 0

    for idx, pil_img in enumerate(images):
        save_path = save_dir / f"{idx + start}.png"
        pil_img.save(save_path)
        logger.info(f"Save to {save_path}")

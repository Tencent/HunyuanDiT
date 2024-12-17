import random
import torch
from diffusers import HunyuanDiTPipeline
from transformers import T5EncoderModel
import time
from loguru import logger
import gc
import sys

NEGATIVE_PROMPT = ""

TEXT_ENCODER_CONF = {
    "negative_prompt": NEGATIVE_PROMPT,
    "prompt_embeds": None,
    "negative_prompt_embeds": None,
    "prompt_attention_mask": None,
    "negative_prompt_attention_mask": None,
    "max_sequence_length": 256,
    "text_encoder_index": 1,
}


def flush():
    gc.collect()
    torch.cuda.empty_cache()


class End2End(object):
    def __init__(self, model_id="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ========================================================================
        self.default_negative_prompt = NEGATIVE_PROMPT
        logger.info("==================================================")
        logger.info(f"                Model is ready.                  ")
        logger.info("==================================================")

    def load_pipeline(self):
        self.pipeline = HunyuanDiTPipeline.from_pretrained(
            self.model_id,
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch.float16,
        ).to(self.device)

    def get_text_emb(self, prompts):
        with torch.no_grad():
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                load_in_8bit=True,
                device_map="auto",
            )
            encoder_pipeline = HunyuanDiTPipeline.from_pretrained(
                self.model_id,
                text_encoder_2=text_encoder_2,
                transformer=None,
                vae=None,
                torch_dtype=torch.float16,
                device_map="balanced",
            )
            TEXT_ENCODER_CONF["negative_prompt"] = self.default_negative_prompt
            prompt_emb1 = encoder_pipeline.encode_prompt(
                prompts, negative_prompt=self.default_negative_prompt
            )
            prompt_emb2 = encoder_pipeline.encode_prompt(prompts, **TEXT_ENCODER_CONF)
            del text_encoder_2
            del encoder_pipeline
        flush()
        return prompt_emb1, prompt_emb2

    def predict(
        self,
        user_prompt,
        seed=None,
        enhanced_prompt=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6,
        batch_size=1,
    ):
        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if seed is None:
            seed = random.randint(0, 1_000_000)
        if not isinstance(seed, int):
            raise TypeError(f"`seed` must be an integer, but got {type(seed)}")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(user_prompt, str):
            raise TypeError(
                f"`user_prompt` must be a string, but got {type(user_prompt)}"
            )
        user_prompt = user_prompt.strip()
        prompt = user_prompt

        if enhanced_prompt is not None:
            if not isinstance(enhanced_prompt, str):
                raise TypeError(
                    f"`enhanced_prompt` must be a string, but got {type(enhanced_prompt)}"
                )
            enhanced_prompt = enhanced_prompt.strip()
            prompt = enhanced_prompt

        # negative prompt
        if negative_prompt is not None and negative_prompt != "":
            self.default_negative_prompt = negative_prompt
        if not isinstance(self.default_negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )

        # ========================================================================

        logger.debug(
            f"""
                       prompt: {user_prompt}
              enhanced prompt: {enhanced_prompt}
                         seed: {seed}
              negative_prompt: {negative_prompt}
                   batch_size: {batch_size}
               guidance_scale: {guidance_scale}
                  infer_steps: {infer_steps}
        """
        )

        # get text embeding
        flush()
        prompt_emb1, prompt_emb2 = self.get_text_emb(prompt)
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = prompt_emb1
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
        ) = prompt_emb2
        del prompt_emb1
        del prompt_emb2
        # get pipeline
        self.load_pipeline()
        samples = self.pipeline(
            prompt_embeds=prompt_embeds,
            prompt_embeds_2=prompt_embeds_2,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            prompt_attention_mask=prompt_attention_mask,
            prompt_attention_mask_2=prompt_attention_mask_2,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            negative_prompt_attention_mask_2=negative_prompt_attention_mask_2,
            num_images_per_prompt=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=infer_steps,
            generator=generator,
        ).images[0]

        return {
            "images": samples,
            "seed": seed,
        }


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(
            "Usage: python lite/inference.py ${model_id} ${prompt} ${infer_steps} ${guidance_scale}"
        )
        print(
            "model_id: Choose a diffusers repository from the official Hugging Face repository https://huggingface.co/Tencent-Hunyuan, "
            "such as Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers, "
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled, "
            "Tencent-Hunyuan/HunyuanDiT-Diffusers, or Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled."
        )
        print("prompt: the input prompt")
        print("infer_steps: infer_steps")
        print("guidance_scale: guidance_scale")
        sys.exit(1)
    model_id = sys.argv[1]
    prompt = sys.argv[2]
    infer_steps = int(sys.argv[3])
    guidance_scale = int(sys.argv[4])
    gen = End2End(model_id)
    seed = 42
    results = gen.predict(
        prompt,
        seed=seed,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
    )
    results["images"].save("./lite_image.png")

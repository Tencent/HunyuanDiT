import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from polygraphy import cuda

from .engine import Engine


class TRTModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels=4,
        model_name="unet-dyn",
        engine_dir="./unet",
        device_id=0,
        fp16=True,
        image_width=1024,
        image_height=1024,
        text_maxlen=77,
        embedding_dim=768,
        max_batch_size=1,
        plugin_path="./ckpts/trt_model/fmha_plugins/10.1_plugin_cuda11/fMHAPlugin.so",
    ):
        super().__init__()
        # create engine
        self.in_channels = in_channels  # For pipeline compatibility
        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.model_name = model_name
        self.engine_dir = engine_dir
        self.engine = Engine(self.model_name, self.engine_dir)
        self.engine.activate(plugin_path)
        # create cuda stream
        self.stream = torch.cuda.Stream().cuda_stream
        self.latent_width = image_width // 8
        self.latent_height = image_height // 8
        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        device = "cuda:{}".format(device_id)
        self.engine_device = torch.device(device)
        print("[INFO] Create hcf nv controlled unet success")

    @property
    def device(self):
        return self.engine_device

    def __call__(
        self,
        x,
        t_emb,
        context,
        image_meta_size,
        style,
        freqs_cis_img0,
        freqs_cis_img1,
        text_embedding_mask,
        encoder_hidden_states_t5,
        text_embedding_mask_t5,
    ):
        return self.forward(
            x=x,
            t_emb=t_emb,
            context=context,
            image_meta_size=image_meta_size,
            style=style,
            freqs_cis_img0=freqs_cis_img0,
            freqs_cis_img1=freqs_cis_img1,
            text_embedding_mask=text_embedding_mask,
            encoder_hidden_states_t5=encoder_hidden_states_t5,
            text_embedding_mask_t5=text_embedding_mask_t5,
        )

    def get_shared_memory(self):
        return self.engine.get_shared_memory()

    def set_shared_memory(self, shared_memory):
        self.engine.set_shared_memory(shared_memory)

    def forward(
        self,
        x,
        t_emb,
        context,
        image_meta_size,
        style,
        freqs_cis_img0,
        freqs_cis_img1,
        text_embedding_mask,
        encoder_hidden_states_t5,
        text_embedding_mask_t5,
    ):
        x_c = x.half()
        t_emb_c = t_emb.half()
        context_c = context.half()
        if image_meta_size is not None:
            image_meta_size_c = image_meta_size.half().contiguous()
            self.engine.context.set_input_shape(
                "image_meta_size", image_meta_size_c.shape
            )
            self.engine.context.set_tensor_address(
                "image_meta_size", image_meta_size_c.contiguous().data_ptr()
            )
        if style is not None:
            style_c = style.long().contiguous()
            self.engine.context.set_input_shape("style", style_c.shape)
            self.engine.context.set_tensor_address(
                "style", style_c.contiguous().data_ptr()
            )

        freqs_cis_img0_c = freqs_cis_img0.float()
        freqs_cis_img1_c = freqs_cis_img1.float()
        text_embedding_mask_c = text_embedding_mask.long()
        encoder_hidden_states_t5_c = encoder_hidden_states_t5.half()
        text_embedding_mask_t5_c = text_embedding_mask_t5.long()

        self.engine.context.set_input_shape("x", x_c.shape)
        self.engine.context.set_input_shape("t", t_emb_c.shape)
        self.engine.context.set_input_shape("encoder_hidden_states", context_c.shape)
        self.engine.context.set_input_shape(
            "text_embedding_mask", text_embedding_mask_c.shape
        )
        self.engine.context.set_input_shape(
            "encoder_hidden_states_t5", encoder_hidden_states_t5_c.shape
        )
        self.engine.context.set_input_shape(
            "text_embedding_mask_t5", text_embedding_mask_t5_c.shape
        )
        self.engine.context.set_input_shape("cos_cis_img", freqs_cis_img0_c.shape)
        self.engine.context.set_input_shape("sin_cis_img", freqs_cis_img1_c.shape)

        self.engine.context.set_tensor_address("x", x_c.contiguous().data_ptr())
        self.engine.context.set_tensor_address("t", t_emb_c.contiguous().data_ptr())
        self.engine.context.set_tensor_address(
            "encoder_hidden_states", context_c.contiguous().data_ptr()
        )
        self.engine.context.set_tensor_address(
            "text_embedding_mask", text_embedding_mask_c.contiguous().data_ptr()
        )
        self.engine.context.set_tensor_address(
            "encoder_hidden_states_t5",
            encoder_hidden_states_t5_c.contiguous().data_ptr(),
        )
        self.engine.context.set_tensor_address(
            "text_embedding_mask_t5", text_embedding_mask_t5_c.contiguous().data_ptr()
        )
        self.engine.context.set_tensor_address(
            "cos_cis_img", freqs_cis_img0_c.contiguous().data_ptr()
        )
        self.engine.context.set_tensor_address(
            "sin_cis_img", freqs_cis_img1_c.contiguous().data_ptr()
        )

        output = torch.zeros(
            (2 * self.max_batch_size, 8, self.latent_height, self.latent_width),
            dtype=torch.float16,
            device="cuda",
        )
        self.engine.context.set_tensor_address("output", output.contiguous().data_ptr())

        self.engine.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()

        output.resize_(tuple(self.engine.context.get_tensor_shape("output")))
        return output

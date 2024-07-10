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
            plugin_path="./ckpts/trt_model/fmha_plugins/9.2_plugin_cuda11/fMHAPlugin.so",
    ):
        super().__init__()
        # create engine
        self.in_channels = in_channels      # For pipeline compatibility
        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.model_name = model_name
        self.engine_dir = engine_dir
        self.engine = Engine(self.model_name, self.engine_dir)
        self.engine.activate(plugin_path)
        # create cuda stream
        self.stream = cuda.Stream()
        # create inputs buffer
        self.latent_width = image_width // 8
        self.latent_height = image_height // 8
        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        shape_dict = {
            'x': (2 * self.max_batch_size, 4, self.latent_height, self.latent_width),
            't': (2 * self.max_batch_size,),
            'encoder_hidden_states': (2 * self.max_batch_size, self.text_maxlen, self.embedding_dim),
            'text_embedding_mask': (2 * self.max_batch_size, self.text_maxlen),
            'encoder_hidden_states_t5': (2 * self.max_batch_size, 256, 2048),
            'text_embedding_mask_t5': (2 * self.max_batch_size, 256),
            'image_meta_size': (2 * self.max_batch_size, 6),
            'style': (2 * self.max_batch_size,),
            'cos_cis_img': (6400, 88),
            'sin_cis_img': (6400, 88),
            'output': (2 * self.max_batch_size, 8, self.latent_height, self.latent_width),
        }
        device = "cuda:{}".format(device_id)
        self.engine_device = torch.device(device)
        self.engine.allocate_buffers(shape_dict=shape_dict, device=device)

        print("[INFO] Create hcf nv controlled unet success")

    @property
    def device(self):
        return self.engine_device

    def __call__(self, x, t_emb, context, image_meta_size, style, freqs_cis_img0,
                 freqs_cis_img1, text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5):
        return self.forward(x=x, t_emb=t_emb, context=context, image_meta_size=image_meta_size, style=style,
                            freqs_cis_img0=freqs_cis_img0, freqs_cis_img1=freqs_cis_img1,
                            text_embedding_mask=text_embedding_mask, encoder_hidden_states_t5=encoder_hidden_states_t5,
                            text_embedding_mask_t5=text_embedding_mask_t5)

    def get_shared_memory(self):
        return self.engine.get_shared_memory()

    def set_shared_memory(self, shared_memory):
        self.engine.set_shared_memory(shared_memory)

    def forward(self, x, t_emb, context, image_meta_size, style, freqs_cis_img0,
                freqs_cis_img1, text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5):
        x_c = x.half()
        t_emb_c = t_emb.half()
        context_c = context.half()
        image_meta_size_c = image_meta_size.half()
        style_c = style.long()
        freqs_cis_img0_c = freqs_cis_img0.float()
        freqs_cis_img1_c = freqs_cis_img1.float()
        text_embedding_mask_c = text_embedding_mask.long()
        encoder_hidden_states_t5_c = encoder_hidden_states_t5.half()
        text_embedding_mask_t5_c = text_embedding_mask_t5.long()
        dtype = np.float16
        batch_size = x.shape[0] // 2
        if batch_size <= self.max_batch_size:
            sample_inp = cuda.DeviceView(ptr=x_c.reshape(-1).data_ptr(), shape=x_c.shape, dtype=np.float16)
            t_emb_inp = cuda.DeviceView(ptr=t_emb_c.reshape(-1).data_ptr(), shape=t_emb_c.shape, dtype=np.float16)
            embeddings_inp = cuda.DeviceView(ptr=context_c.reshape(-1).data_ptr(), shape=context_c.shape,
                                             dtype=np.float16)
            image_meta_size_inp = cuda.DeviceView(ptr=image_meta_size_c.reshape(-1).data_ptr(),
                                                  shape=image_meta_size_c.shape, dtype=np.float16)
            style_inp = cuda.DeviceView(ptr=style_c.reshape(-1).data_ptr(), shape=style_c.shape, dtype=np.int64)
            freqs_cis_img0_inp = cuda.DeviceView(ptr=freqs_cis_img0_c.reshape(-1).data_ptr(),
                                                 shape=freqs_cis_img0_c.shape, dtype=np.float32)
            freqs_cis_img1_inp = cuda.DeviceView(ptr=freqs_cis_img1_c.reshape(-1).data_ptr(),
                                                 shape=freqs_cis_img1_c.shape, dtype=np.float32)
            text_embedding_mask_inp = cuda.DeviceView(ptr=text_embedding_mask_c.reshape(-1).data_ptr(),
                                                      shape=text_embedding_mask_c.shape, dtype=np.int64)
            encoder_hidden_states_t5_inp = cuda.DeviceView(ptr=encoder_hidden_states_t5_c.reshape(-1).data_ptr(),
                                                           shape=encoder_hidden_states_t5_c.shape, dtype=np.float16)
            text_embedding_mask_t5_inp = cuda.DeviceView(ptr=text_embedding_mask_t5_c.reshape(-1).data_ptr(),
                                                         shape=text_embedding_mask_t5_c.shape, dtype=np.int64)
            feed_dict = {"x": sample_inp,
                         "t": t_emb_inp,
                         "encoder_hidden_states": embeddings_inp,
                         "image_meta_size": image_meta_size_inp,
                         "text_embedding_mask": text_embedding_mask_inp,
                         "encoder_hidden_states_t5": encoder_hidden_states_t5_inp,
                         "text_embedding_mask_t5": text_embedding_mask_t5_inp,
                         "style": style_inp, "cos_cis_img": freqs_cis_img0_inp,
                         "sin_cis_img": freqs_cis_img1_inp}
            latent = self.engine.infer(feed_dict, self.stream)
            return latent['output']
        else:
            raise ValueError(
                "[ERROR] Input batch_size={} execeed max_batch_size={}".format(batch_size, self.max_batch_size))

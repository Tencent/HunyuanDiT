import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
from comfy import model_management
from .supported_dit_models import HunYuan_DiT, HYDiT_Model
from .clip import CLIP
import os
import folder_paths
import torch

sampling_settings = {
	"beta_schedule" : "linear",
	"linear_start"  : 0.00085,
	"linear_end"    : 0.03,
	"timesteps"     : 1000,
}

hydit_conf = {
	"G/2": { # Seems to be the main one
		"unet_config": {
			"depth"       :   40,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1408,
			"mlp_ratio" : 4.3637,
			"input_size": (1024//8, 1024//8),
		},
		"sampling_settings" : sampling_settings,
	},
}

def load_dit(model_path, output_clip=True, output_model=True, output_vae=True):
	state_dict = comfy.utils.load_torch_file(model_path)
	state_dict = state_dict.get("model", state_dict)
	parameters = comfy.utils.calculate_parameters(state_dict)
	unet_dtype = model_management.unet_dtype(model_params=parameters)
	load_device = comfy.model_management.get_torch_device()
	offload_device = comfy.model_management.unet_offload_device()
	clip = None,
	vae = None
	model_patcher = None

	# ignore fp8/etc and use directly for now
	manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)
	root = os.path.join(folder_paths.models_dir, "hunyuan/ckpts/t2i")
	if manual_cast_dtype:
		print(f"DiT: falling back to {manual_cast_dtype}")
		unet_dtype = manual_cast_dtype

	#model_conf["unet_config"]["num_classes"] = state_dict["y_embedder.embedding_table.weight"].shape[0] - 1 # adj. for empty

	if output_model:
		model_conf = HunYuan_DiT(hydit_conf["G/2"])
		model = HYDiT_Model(
			model_conf,
			model_type=comfy.model_base.ModelType.V_PREDICTION,
			device=model_management.get_torch_device()
		)

		from ..hydit.modules.models import HunYuanDiT as HYDiT

		model.diffusion_model = HYDiT(model_conf.dit_conf, **model_conf.unet_config).half().to(load_device)

		model.diffusion_model.load_state_dict(state_dict)
		#model.diffusion_model.dtype = unet_dtype
		model.diffusion_model.eval()
		model.diffusion_model.to(unet_dtype)

		model_patcher = comfy.model_patcher.ModelPatcher(
			model,
			load_device = load_device,
			offload_device = offload_device,
			current_device = "cpu",
		)
		#model_patcher['model_options']['dit'] = 'hunyuan'
	if output_clip:
		clip = CLIP(root)

	if output_vae:
		vae_path = os.path.join(root, 'sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors')
		sd = comfy.utils.load_torch_file(vae_path)
		vae = comfy.sd.VAE(sd=sd)

	return (model_patcher, clip, vae)

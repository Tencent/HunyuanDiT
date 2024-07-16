import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
from comfy import model_management



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
			#"disable_unet_model_creation": True,
		},
		"sampling_settings" : sampling_settings,
	},
}



def load_checkpoint(model_path, version):
	if version == "v1.1":
		from .supported_dit_models_v1_1 import HunYuan_DiT, HYDiT_Model, ModifiedHunYuanDiT
	elif version == "v1.2":
		from .supported_dit_models import HunYuan_DiT, HYDiT_Model, ModifiedHunYuanDiT	
	else:
		pass

	state_dict = comfy.utils.load_torch_file(model_path)


	state_dict = state_dict.get("model", state_dict)
	parameters = comfy.utils.calculate_parameters(state_dict)
	unet_dtype = model_management.unet_dtype(model_params=parameters)
	load_device = comfy.model_management.get_torch_device()
	offload_device = comfy.model_management.unet_offload_device()
	model_patcher = None

	# ignore fp8/etc and use directly for now
	manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)
	if manual_cast_dtype:
		print(f"DiT: falling back to {manual_cast_dtype}")
		unet_dtype = manual_cast_dtype


	model_conf = HunYuan_DiT(hydit_conf["G/2"])
	model = HYDiT_Model(
		model_conf,
		model_type=comfy.model_base.ModelType.V_PREDICTION,
		device=model_management.get_torch_device()
	)


	model.diffusion_model = ModifiedHunYuanDiT(model_conf.dit_conf, **model_conf.unet_config).half().to(load_device)

	model.diffusion_model.load_state_dict(state_dict)
	model.diffusion_model.eval()
	model.diffusion_model.to(unet_dtype)

	model_patcher = comfy.model_patcher.ModelPatcher(
		model,
		load_device = load_device,
		offload_device = offload_device,
		current_device = "cpu",)
	return (model_patcher,)



def load_vae(vae_path):

	sd = comfy.utils.load_torch_file(vae_path)
	vae = comfy.sd.VAE(sd=sd)

	return (vae,)
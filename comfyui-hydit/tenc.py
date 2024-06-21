# This is for loading the CLIP (bert?) + mT5 encoder for HunYuanDiT
import os
import torch
from transformers import AutoTokenizer, modeling_utils
from transformers import T5Config, T5EncoderModel, BertConfig, BertModel

from comfy import model_management
import comfy.model_patcher
import comfy.utils

class mT5Model(torch.nn.Module):
	def __init__(self, textmodel_json_config=None, device="cpu", max_length=256, freeze=True, dtype=None):
		super().__init__()
		self.device = device
		self.dtype = dtype
		self.max_length = max_length
		if textmodel_json_config is None:
			textmodel_json_config = os.path.join(
				os.path.dirname(os.path.realpath(__file__)),
				f"config_mt5.json"
			)
		config = T5Config.from_json_file(textmodel_json_config)
		with modeling_utils.no_init_weights():
			self.transformer = T5EncoderModel(config)
		self.to(dtype)
		if freeze:
			self.freeze()

	def freeze(self):
		self.transformer = self.transformer.eval()
		for param in self.parameters():
			param.requires_grad = False

	def load_sd(self, sd):
		return self.transformer.load_state_dict(sd, strict=False)

	def to(self, *args, **kwargs):
		return self.transformer.to(*args, **kwargs)

class hyCLIPModel(torch.nn.Module):
	def __init__(self, textmodel_json_config=None, device="cpu", max_length=77, freeze=True, dtype=None):
		super().__init__()
		self.device = device
		self.dtype = dtype
		self.max_length = max_length
		if textmodel_json_config is None:
			textmodel_json_config = os.path.join(
				os.path.dirname(os.path.realpath(__file__)),
				f"config_clip.json"
			)
		config = BertConfig.from_json_file(textmodel_json_config)
		with modeling_utils.no_init_weights():
			self.transformer = BertModel(config)
		self.to(dtype)
		if freeze:
			self.freeze()

	def freeze(self):
		self.transformer = self.transformer.eval()
		for param in self.parameters():
			param.requires_grad = False

	def load_sd(self, sd):
		return self.transformer.load_state_dict(sd, strict=False)

	def to(self, *args, **kwargs):
		return self.transformer.to(*args, **kwargs)

class EXM_HyDiT_Tenc_Temp:
	def __init__(self, no_init=False, device="cpu", dtype=None, model_class="mT5", *kwargs):
		if no_init:
			return

		size = 8 if model_class == "mT5" else 2
		if dtype == torch.float32:
			size *= 2
		size *= (1024**3)

		if device == "auto":
			self.load_device = model_management.text_encoder_device()
			self.offload_device = model_management.text_encoder_offload_device()
			self.init_device = "cpu"
		elif device == "cpu":
			size = 0 # doesn't matter
			self.load_device = "cpu"
			self.offload_device = "cpu"
			self.init_device="cpu"
		elif device.startswith("cuda"):
			print("Direct CUDA device override!\nVRAM will not be freed by default.")
			size = 0 # not used
			self.load_device = device
			self.offload_device = device
			self.init_device = device
		else:
			self.load_device = model_management.get_torch_device()
			self.offload_device = "cpu"
			self.init_device="cpu"

		self.dtype = dtype
		self.device = self.load_device
		if model_class == "mT5":
			self.cond_stage_model = mT5Model(
				device         = self.load_device,
				dtype          = self.dtype,
			)
			#tokenizer_args = {"subfolder": "t2i/mt5"} # web
			tokenizer_path = os.path.join( # local
				os.path.dirname(os.path.realpath(__file__)),
				"tokenizer_mt5",
			)
		else:
			self.cond_stage_model = hyCLIPModel(
				device         = self.load_device,
				dtype          = self.dtype,
			)
			#tokenizer_args = {"subfolder": "t2i/tokenizer",} # web
			tokenizer_path = os.path.join( # local
				os.path.dirname(os.path.realpath(__file__)),
				"tokenizer_clip",
			)
		# self.tokenizer = AutoTokenizer.from_pretrained(
			# "Tencent-Hunyuan/HunyuanDiT",
			# **tokenizer_args
		# )
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
		self.patcher = comfy.model_patcher.ModelPatcher(
			self.cond_stage_model,
			load_device    = self.load_device,
			offload_device = self.offload_device,
			current_device = self.load_device,
			size           = size,
		)

	def clone(self):
		n = EXM_HyDiT_Tenc_Temp(no_init=True)
		n.patcher = self.patcher.clone()
		n.cond_stage_model = self.cond_stage_model
		n.tokenizer = self.tokenizer
		return n

	def load_sd(self, sd):
		return self.cond_stage_model.load_sd(sd)

	def get_sd(self):
		return self.cond_stage_model.state_dict()

	def load_model(self):
		if self.load_device != "cpu":
			model_management.load_model_gpu(self.patcher)
		return self.patcher

	def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
		return self.patcher.add_patches(patches, strength_patch, strength_model)

	def get_key_patches(self):
		return self.patcher.get_key_patches()

def load_clip(model_path, **kwargs):
	model = EXM_HyDiT_Tenc_Temp(model_class="clip", **kwargs)
	sd = comfy.utils.load_torch_file(model_path)

	prefix = "bert."
	state_dict = {}
	for key in sd:
		nkey = key
		if key.startswith(prefix):
				nkey = key[len(prefix):]
		state_dict[nkey] = sd[key]

	m, e = model.load_sd(state_dict)
	if len(m) > 0 or len(e) > 0:
		print(f"HYDiT: clip missing {len(m)} keys ({len(e)} extra)")
	return model

def load_t5(model_path, **kwargs):
	model = EXM_HyDiT_Tenc_Temp(model_class="mT5", **kwargs)
	sd = comfy.utils.load_torch_file(model_path)
	m, e = model.load_sd(sd)
	if len(m) > 0 or len(e) > 0:
		print(f"HYDiT: mT5 missing {len(m)} keys ({len(e)} extra)")
	return model

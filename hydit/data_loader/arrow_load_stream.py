import pickle
import random
from pathlib import Path
import ast
import numpy as np
import re
import json
import time
from functools import partial
from PIL import Image

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

from IndexKits.index_kits import ArrowIndexV2, MultiResolutionBucketIndexV2, MultiIndexV2


class TextImageArrowStream(Dataset):
    def __init__(self,
                 args,
                 resolution=512,
                 random_flip=None,
                 enable_CN=True,
                 log_fn=print,
                 index_file=None,
                 multireso=False,
                 batch_size=-1,
                 world_size=1,
                 random_shrink_size_cond=False,
                 merge_src_cond=False,
                 uncond_p=0.0,
                 text_ctx_len=77,
                 tokenizer=None,
                 uncond_p_t5=0.0,
                 text_ctx_len_t5=256,
                 tokenizer_t5=None,
                 ):
        self.args = args
        self.resolution = resolution
        self.log_fn = lambda x: log_fn(f"    {Path(__file__).stem} | " + x)

        self.random_flip = random_flip
        # If true, the Chinese prompt from the `text_zh` column will be taken from the arrow file;
        # otherwise, the English prompt from the `text_en` column will be taken,
        # provided that `text_zh` or `text_en` exists in the arrow file.
        self.enable_CN = enable_CN
        self.index_file = index_file
        self.multireso = multireso
        self.batch_size = batch_size
        self.world_size = world_size
        self.index_manager = self.load_index()

        # clip params
        self.uncond_p = uncond_p
        self.text_ctx_len = text_ctx_len
        self.tokenizer = tokenizer

        # t5 params
        self.uncond_p_t5 = uncond_p_t5
        self.text_ctx_len_t5 = text_ctx_len_t5
        self.tokenizer_t5 = tokenizer_t5

        # size condition
        self.random_shrink_size_cond = random_shrink_size_cond
        self.merge_src_cond = merge_src_cond

        assert isinstance(resolution, int), f"resolution must be an integer, got {resolution}"
        self.flip_norm = T.Compose(
            [
                T.RandomHorizontalFlip() if self.random_flip else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

        # show info
        if self.merge_src_cond:
            self.log_fn("Enable merging src condition: (oriW, oriH) --> ((WH)**0.5, (WH)**0.5)")

        self.log_fn("Enable image_meta_size condition (original_size, target_size, crop_coords)")
        self.log_fn(f"Image_transforms: {self.flip_norm}")

    def load_index(self):
        multireso = self.multireso
        index_file = self.index_file
        batch_size = self.batch_size
        world_size = self.world_size

        if multireso:
            if isinstance(index_file, (list, tuple)):
                if len(index_file) > 1:
                    raise ValueError(f"When enabling multireso, index_file should be a single file, but got {index_file}")
                index_file = index_file[0]
            index_manager = MultiResolutionBucketIndexV2(index_file, batch_size, world_size)
            self.log_fn(f"Using MultiResolutionBucketIndexV2: {len(index_manager):,}")
        else:
            if isinstance(index_file, str):
                index_file = [index_file]
            if len(index_file) == 1:
                index_manager = ArrowIndexV2(index_file[0])
                self.log_fn(f"Using ArrowIndexV2: {len(index_manager):,}")
            else:
                index_manager = MultiIndexV2(index_file)
                self.log_fn(f"Using MultiIndexV2: {len(index_manager):,}")

        return index_manager

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def get_raw_image(self, index, image_key="image"):
        try:
            ret = self.index_manager.get_image(index, image_key)
        except Exception as e:
            self.log_fn(f'get_raw_image | Error: {e}')
            ret = Image.new("RGB", (256, 256), (255, 255, 255))
        return ret

    @staticmethod
    def random_crop_image(image, origin_size, target_size):
        aspect_ratio = float(origin_size[0]) / float(origin_size[1])
        if origin_size[0] < origin_size[1]:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        if new_width > target_size[0]:
            x_start = random.randint(0, new_width - target_size[0])
            y_start = 0
        else:
            x_start = 0
            y_start = random.randint(0, new_height - target_size[1])
        image_crop = image.crop((x_start, y_start, x_start + target_size[0], y_start + target_size[1]))
        crops_coords_top_left = (x_start, y_start)
        return image_crop, crops_coords_top_left

    def get_style(self, index):
        "Here we use a default learned embedder layer for future extension."
        style = 0
        return style

    def get_image_with_hwxy(self, index, image_key="image"):

        image = self.get_raw_image(index, image_key=image_key)
        origin_size = image.size

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)
            image, crops_coords_top_left = self.index_manager.resize_and_crop(
                image, target_size, resample=Image.LANCZOS, crop_type='random')
            image_tensor = self.flip_norm(image)
        else:
            target_size = (self.resolution, self.resolution)
            image_crop, crops_coords_top_left = self.random_crop_image(image, origin_size, target_size)
            image_tensor = self.flip_norm(image_crop)

        if self.random_shrink_size_cond:
            origin_size = (1024 if origin_size[0] < 1024 else origin_size[0],
                           1024 if origin_size[1] < 1024 else origin_size[1])
        if self.merge_src_cond:
            val = (origin_size[0] * origin_size[1]) ** 0.5
            origin_size = (val, val)

        image_meta_size = tuple(origin_size) + tuple(target_size) + tuple(crops_coords_top_left)
        kwargs = {
            'image_meta_size': image_meta_size,
        }

        style = self.get_style(index)
        kwargs['style'] = style

        return image_tensor, kwargs

    def get_text_info_with_encoder(self, description):
        pad_num = 0
        text_inputs = self.tokenizer(
            description,
            padding="max_length",
            max_length=self.text_ctx_len,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids[0]
        attention_mask = text_inputs.attention_mask[0].bool()
        if pad_num > 0:
            attention_mask[1:pad_num + 1] = False
        return description, text_input_ids, attention_mask

    def fill_t5_token_mask(self, fill_tensor, fill_number, setting_length):
        fill_length = setting_length - fill_tensor.shape[1]
        if fill_length > 0:
            fill_tensor = torch.cat((fill_tensor, fill_number * torch.ones(1, fill_length)), dim=1)
        return fill_tensor

    def get_text_info_with_encoder_t5(self, description_t5):
        text_tokens_and_mask = self.tokenizer_t5(
            description_t5,
            max_length=self.text_ctx_len_t5,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        text_input_ids_t5=self.fill_t5_token_mask(text_tokens_and_mask["input_ids"], fill_number=1, setting_length=self.text_ctx_len_t5).long()
        attention_mask_t5=self.fill_t5_token_mask(text_tokens_and_mask["attention_mask"], fill_number=0, setting_length=self.text_ctx_len_t5).bool()
        return description_t5, text_input_ids_t5, attention_mask_t5

    def get_original_text(self, ind):
        text = self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en')
        text = str(text).strip()
        return text

    def get_text(self, ind):
        text =  self.get_original_text(ind)
        if text == '':
            text = '随机生成一张图片'
        return text

    def __getitem__(self, ind):
        # Get text
        if random.random() < self.uncond_p:
            description = ""
        else:
            description = self.get_text(ind)

        # Get text for t5
        if random.random() < self.uncond_p_t5:
            description_t5 = ""
        else:
            description_t5 = self.get_text(ind)

        original_pil_image, kwargs = self.get_image_with_hwxy(ind)

        # Use encoder to embed tokens online
        text, text_embedding, text_embedding_mask = self.get_text_info_with_encoder(description)

        text_t5, text_embedding_t5, text_embedding_mask_t5 = self.get_text_info_with_encoder_t5(description_t5)
        return (
            original_pil_image,
            text_embedding.clone().detach(),
            text_embedding_mask.clone().detach(),
            text_embedding_t5.clone().detach(),
            text_embedding_mask_t5.clone().detach(),
            {k: torch.tensor(np.array(v)).clone().detach() for k, v in kwargs.items()},
        )

    def __len__(self):
        return len(self.index_manager)

#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections import OrderedDict
from copy import copy

import numpy as np
import tensorrt as trt
import torch
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util
import ctypes
from glob import glob
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt_util.TRT_LOGGER = TRT_LOGGER


class Engine():
    def __init__(
            self,
            model_name,
            engine_dir,
            onnx_file=None,
    ):
        self.engine_path = os.path.join(engine_dir, model_name + '.plan')
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

        self.weightNameList = None
        self.refitter = None
        self.onnx_initializers = None
        self.onnx_file = onnx_file
        self.trt_lora_weight = None
        self.trt_lora_weight_mem = None
        self.torch_weight = None

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(self, onnx_path, fp16, input_profile=None, enable_preview=False, sparse_weights=False):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        preview_features = []
        if enable_preview:
            trt_version = [int(i) for i in trt.__version__.split(".")]
            # FASTER_DYNAMIC_SHAPES_0805 should only be used for TRT 8.5.1 or above.
            if trt_version[0] > 8 or \
                    (trt_version[0] == 8 and (trt_version[1] > 5 or (trt_version[1] == 5 and trt_version[2] >= 1))):
                preview_features = [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]

        engine = engine_from_network(network_from_onnx_path(onnx_path), config=CreateConfig(fp16=fp16, profiles=[p],
                                                                                            preview_features=preview_features,
                                                                                            sparse_weights=sparse_weights))
        save_engine(engine, path=self.engine_path)

    def activate(self, plugin_path=""):
        ctypes.cdll.LoadLibrary(plugin_path)
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
        self.context = self.engine.create_execution_context()

    def get_shared_memory(self):
        _, device_memory = cudart.cudaMalloc(self.engine.device_memory_size)
        self.device_memory = device_memory
        return self.device_memory

    def set_shared_memory(self, device_memory_size):
        self.context.device_memory = device_memory_size

    def binding_input(self, name, shape):
        idx = self.engine.get_binding_index(name)
        result = self.context.set_binding_shape(idx, shape)
        return result

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        print("Allocate buffers and bindings inputs:")
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            print("binding: ", binding)
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            nv_dtype = self.engine.get_binding_dtype(binding)
            dtype_map = {trt.DataType.FLOAT: np.float32,
                         trt.DataType.HALF: np.float16,
                         trt.DataType.INT8: np.int8,
                         trt.DataType.INT64: np.int64,
                         trt.DataType.BOOL: bool}
            if hasattr(trt.DataType, 'INT32'):
                dtype_map[trt.DataType.INT32] = np.int32
            dtype = dtype_map[nv_dtype]
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            # Workaround to convert np dtype to torch
            np_type_tensor = np.empty(shape=[], dtype=dtype)
            torch_type_tensor = torch.from_numpy(np_type_tensor)
            tensor = torch.empty(tuple(shape), dtype=torch_type_tensor.dtype).to(device=device)

            print(f"  binding={binding}, shape={shape}, dtype={tensor.dtype}")
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
            self.binding_input(name, buf.shape)
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if not self.engine.binding_is_input(binding):
                shape = self.context.get_binding_shape(idx)
                self.tensors[binding].resize_(tuple(shape))
        return self.tensors

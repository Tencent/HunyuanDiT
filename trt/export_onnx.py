from pathlib import Path

import torch
from loguru import logger

from hydit.config import get_args
from hydit.modules.models import HunYuanDiT, HUNYUAN_DIT_CONFIG

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import polygraphy.backend.onnx.loader
from copy import deepcopy


def _to_tuple(val):
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            val = [val[0], val[0]]
        elif len(val) == 2:
            val = tuple(val)
        else:
            raise ValueError(f"Invalid value: {val}")
    elif isinstance(val, (int, float)):
        val = (val, val)
    else:
        raise ValueError(f"Invalid value: {val}")
    return val


class ExportONNX(object):
    def __init__(self, args, models_root_path):
        self.args = args
        self.model = None
        # Set device and disable gradient
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)

        # Check arguments
        t2i_root_path = Path(models_root_path) / "t2i"
        self.root = t2i_root_path
        logger.info(f"Got text-to-image model root path: {t2i_root_path}")

        # Create folder to save onnx model
        onnx_workdir = Path(self.args.onnx_workdir)
        self.onnx_workdir = onnx_workdir
        self.onnx_export = self.onnx_workdir / "export/model.onnx"
        self.onnx_export.parent.mkdir(parents=True, exist_ok=True)
        self.onnx_modify = self.onnx_workdir / "export_modified/model.onnx"
        self.onnx_modify.parent.mkdir(parents=True, exist_ok=True)
        self.onnx_fmha = self.onnx_workdir / "export_modified_fmha/model.onnx"
        self.onnx_fmha.parent.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        # ========================================================================
        # Create model structure and load the checkpoint
        logger.info(f"Building HunYuan-DiT model...")
        model_config = HUNYUAN_DIT_CONFIG[self.args.model]
        image_size = _to_tuple(self.args.image_size)
        latent_size = (image_size[0] // 8, image_size[1] // 8)

        model_dir = self.root / "model"
        model_path = model_dir / f"pytorch_model_{self.args.load_key}.pt"
        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")

        # Build model structure
        self.model = (
            HunYuanDiT(
                self.args,
                input_size=latent_size,
                **model_config,
                log_fn=logger.info,
            )
            .half()
            .to(self.device)
        )  # Force to use fp16
        # Load model checkpoint
        logger.info(f"Loading torch model {model_path}...")
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Loading torch model finished")
        logger.info("==================================================")
        logger.info(f"                Model is ready.                  ")
        logger.info("==================================================")

    def export(self):
        if self.model is None:
            self.load_model()

        # Construct model inputs
        latent_model_input = torch.randn(2, 4, 128, 128, device=self.device).half()
        t_expand = torch.randint(0, 1000, [2], device=self.device).half()
        prompt_embeds = torch.randn(2, 77, 1024, device=self.device).half()
        attention_mask = torch.randint(0, 2, [2, 77], device=self.device).long()
        prompt_embeds_t5 = torch.randn(2, 256, 2048, device=self.device).half()
        attention_mask_t5 = torch.randint(0, 2, [2, 256], device=self.device).long()
        ims = torch.tensor(
            [[1024, 1024, 1024, 1024, 0, 0], [1024, 1024, 1024, 1024, 0, 0]],
            device=self.device,
        ).half()
        style = torch.tensor([0, 0], device=self.device).long()
        freqs_cis_img = (
            torch.randn(4096, 88),
            torch.randn(4096, 88),
        )

        save_to = self.onnx_export
        logger.info(f"Exporting ONNX model {save_to}...")
        logger.info(f"Exporting ONNX external data {save_to.parent}...")
        # Hydit version 1.2
        if not self.args.use_style_cond:
            model_args = (
                latent_model_input,
                t_expand,
                prompt_embeds,
                attention_mask,
                prompt_embeds_t5,
                attention_mask_t5,
                freqs_cis_img[0],
                freqs_cis_img[1],
            )
            torch.onnx.export(
                self.model,
                model_args,
                str(save_to),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=[
                    "x",
                    "t",
                    "encoder_hidden_states",
                    "text_embedding_mask",
                    "encoder_hidden_states_t5",
                    "text_embedding_mask_t5",
                    "cos_cis_img",
                    "sin_cis_img",
                ],
                output_names=["output"],
                dynamic_axes={
                    "x": {0: "2B", 2: "H", 3: "W"},
                    "t": {0: "2B"},
                    "encoder_hidden_states": {0: "2B"},
                    "text_embedding_mask": {0: "2B"},
                    "encoder_hidden_states_t5": {0: "2B"},
                    "text_embedding_mask_t5": {0: "2B"},
                    "cos_cis_img": {0: "seqlen"},
                    "sin_cis_img": {0: "seqlen"},
                },
            )
        # Hydit version 1.0 or 1.1
        else:
            model_args = (
                latent_model_input,
                t_expand,
                prompt_embeds,
                attention_mask,
                prompt_embeds_t5,
                attention_mask_t5,
                freqs_cis_img[0],
                freqs_cis_img[1],
                ims,
                style,
            )
            torch.onnx.export(
                self.model,
                model_args,
                str(save_to),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=[
                    "x",
                    "t",
                    "encoder_hidden_states",
                    "text_embedding_mask",
                    "encoder_hidden_states_t5",
                    "text_embedding_mask_t5",
                    "cos_cis_img",
                    "sin_cis_img",
                    "image_meta_size",
                    "style",
                ],
                output_names=["output"],
                dynamic_axes={
                    "x": {0: "2B", 2: "H", 3: "W"},
                    "t": {0: "2B"},
                    "encoder_hidden_states": {0: "2B"},
                    "text_embedding_mask": {0: "2B"},
                    "encoder_hidden_states_t5": {0: "2B"},
                    "text_embedding_mask_t5": {0: "2B"},
                    "image_meta_size": {0: "2B"},
                    "style": {0: "2B"},
                    "cos_cis_img": {0: "seqlen"},
                    "sin_cis_img": {0: "seqlen"},
                },
            )
        logger.info("Exporting onnx finished")

    def postprocessing(self):
        load_from = self.onnx_export
        save_to = self.onnx_modify
        logger.info(f"Postprocessing ONNX model {load_from}...")

        onnxModel = onnx.load(str(load_from), load_external_data=False)
        onnx.load_external_data_for_model(onnxModel, str(load_from.parent))
        graph = gs.import_onnx(onnxModel)

        # ADD GAMMA BETA FOR LN
        for node in graph.nodes:
            if node.name == "/final_layer/norm_final/LayerNormalization":
                constantKernel = gs.Constant(
                    "final_layer.norm_final.weight",
                    np.ascontiguousarray(np.ones((1408,), dtype=np.float32)),
                )
                constantBias = gs.Constant(
                    "final_layer.norm_final.bias",
                    np.ascontiguousarray(np.zeros((1408,), dtype=np.float32)),
                )
                node.inputs = [node.inputs[0], constantKernel, constantBias]
            if node.op == "LayerNormalization":
                input_fp32 = gs.Variable(name=node.name + "_input_tensor_fp32")
                cast_fp32_node = gs.Node(
                    op="Cast",
                    name=node.name + "_cast_to_fp32",
                    attrs={"to": onnx.TensorProto.FLOAT},
                    inputs=[node.inputs[0]],
                    outputs=[input_fp32],
                )
                scale = np.array(
                    deepcopy(node.inputs[1].values.tolist()), dtype=np.float32
                )
                bias = np.array(
                    deepcopy(node.inputs[2].values.tolist()), dtype=np.float32
                )
                scale_constant = gs.Constant(
                    node.inputs[1].name + "_fp32",
                    np.ascontiguousarray(scale.reshape(-1)),
                )
                bias_constant = gs.Constant(
                    node.inputs[2].name + "_fp32",
                    np.ascontiguousarray(bias.reshape(-1)),
                )
                node.inputs = [input_fp32, scale_constant, bias_constant]
                out = node.outputs[0]
                output_fp32 = gs.Variable(name=node.name + "_output_tensor_fp32")
                node.outputs = [output_fp32]
                cast_fp16_node = gs.Node(
                    op="Cast",
                    name=node.name + "_cast_to_fp16",
                    attrs={"to": onnx.TensorProto.FLOAT16},
                    inputs=[output_fp32],
                    outputs=[out],
                )
                graph.nodes.append(cast_fp16_node)
                graph.nodes.append(cast_fp32_node)

        graph.cleanup().toposort()
        onnx.save(
            gs.export_onnx(graph.cleanup()),
            str(save_to),
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            location=str(save_to.parent),
        )
        logger.info(f"Postprocessing ONNX model finished: {save_to}")

    def fuse_attn(self):
        load_from = self.onnx_modify
        save_to = self.onnx_fmha
        logger.info(f"FuseAttn ONNX model {load_from}...")

        onnx_graph = polygraphy.backend.onnx.loader.fold_constants(
            onnx.load(str(load_from)),
            allow_onnxruntime_shape_inference=True,
        )
        graph = gs.import_onnx(onnx_graph)

        cnt = 0
        for node in graph.nodes:

            if (
                node.op == "Softmax"
                and node.i().op == "MatMul"
                and node.o().op == "MatMul"
                and node.o().o().op == "Transpose"
            ):

                if "pooler" in node.name:
                    continue

                if "attn1" in node.name:
                    matmul_0 = node.i()
                    transpose = matmul_0.i(1, 0)
                    transpose.attrs["perm"] = [0, 2, 1, 3]
                    k = transpose.outputs[0]
                    q = gs.Variable(
                        "transpose_0_v_{}".format(cnt), np.dtype(np.float16)
                    )
                    transpose_0 = gs.Node(
                        "Transpose",
                        "Transpose_0_{}".format(cnt),
                        attrs={"perm": [0, 2, 1, 3]},
                        inputs=[matmul_0.inputs[0]],
                        outputs=[q],
                    )
                    graph.nodes.append(transpose_0)

                    matmul_1 = node.o()
                    v = gs.Variable(
                        "transpose_1_v_{}".format(cnt), np.dtype(np.float16)
                    )
                    transpose_1 = gs.Node(
                        "Transpose",
                        "Transpose_1_{}".format(cnt),
                        attrs={"perm": [0, 2, 1, 3]},
                        inputs=[matmul_1.inputs[1]],
                        outputs=[v],
                    )
                    graph.nodes.append(transpose_1)

                    output_variable = node.o().o().outputs[0]
                    # fMHA_v = gs.Variable("fMHA_v", np.dtype(np.float16))
                    fMHA = gs.Node(
                        "fMHAPlugin",
                        "fMHAPlugin_1_{}".format(cnt),
                        # attrs={"scale": 1.0},
                        inputs=[q, k, v],
                        outputs=[output_variable],
                    )
                    graph.nodes.append(fMHA)
                    node.o().o().outputs = []
                    cnt = cnt + 1

                elif "attn2" in node.name:
                    matmul_0 = node.i()
                    transpose_q = matmul_0.i()
                    transpose_k = matmul_0.i(1, 0)
                    matmul_1 = node.o()
                    transpose_v = matmul_1.i(1, 0)
                    q = transpose_q.inputs[0]
                    k = transpose_k.inputs[0]
                    v = transpose_v.inputs[0]
                    output_variable = node.o().o().outputs[0]
                    fMHA = gs.Node(
                        "fMHAPlugin",
                        "fMHAPlugin_1_{}".format(cnt),
                        # attrs={"scale": 1.0},
                        inputs=[q, k, v],
                        outputs=[output_variable],
                    )
                    graph.nodes.append(fMHA)
                    node.o().o().outputs = []
                    cnt = cnt + 1
        print("mha count: ", cnt)
        logger.info("mha count: ", cnt)

        onnx.save(
            gs.export_onnx(graph.cleanup()),
            str(save_to),
            save_as_external_data=True,
        )
        logger.info(f"FuseAttn ONNX model finished: {save_to}")


if __name__ == "__main__":
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    exporter = ExportONNX(args, models_root_path)
    exporter.export()
    exporter.postprocessing()
    exporter.fuse_attn()

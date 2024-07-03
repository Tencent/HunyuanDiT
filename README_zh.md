<!-- ## **HunyuanDiT** -->

<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/logo.png"  height=100>
</p>

# Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding

<div align="center">
  <a href="https://github.com/Tencent/HunyuanDiT"><img src="https://img.shields.io/static/v1?label=Hunyuan-DiT Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://dit.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2405.08748"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv:HunYuan-DiT&color=red&logo=arxiv"></a> &ensp;
  <a href="https://arxiv.org/abs/2403.08857"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:DialogGen&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/Tencent-Hunyuan/HunyuanDiT"><img src="https://img.shields.io/static/v1?label=Hunyuan-DiT&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://hunyuan.tencent.com/bot/chat"><img src="https://img.shields.io/static/v1?label=Hunyuan Bot&message=Web&color=green"></a> &ensp;
  <a href="https://huggingface.co/spaces/Tencent-Hunyuan/HunyuanDiT"><img src="https://img.shields.io/static/v1?label=Hunyuan-DiT Demo&message=HuggingFace&color=yellow"></a> &ensp;
</div>

-----

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring Hunyuan-DiT. You can find more visualizations on our [project page](https://dit.hunyuan.tencent.com/).

> [**Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding**](https://arxiv.org/abs/2405.08748) <br>

> [**DialogGen: Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation**](https://arxiv.org/abs/2403.08857) <br>

## 🔥🔥🔥 News!!
* Jun 27, 2024: :art: Hunyuan-Captioner is released, providing fine-grained caption for training data. See [mllm](./mllm) for details.
* Jun 27, 2024: :tada: Support LoRa and ControlNet in diffusers. See [diffusers](./diffusers) for details.
* Jun 27, 2024: :tada: 6GB GPU VRAM Inference scripts are released. See [lite](./lite) for details.
* Jun 19, 2024: :tada: ControlNet is released, supporting canny, pose and depth control. See [training/inference codes](#controlnet) for details.
* Jun 13, 2024: :zap: HYDiT-v1.1 version is released, which mitigates the issue of image oversaturation and alleviates the watermark issue. Please check [HunyuanDiT-v1.1 ](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1) and 
[Distillation-v1.1](https://huggingface.co/Tencent-Hunyuan/Distillation-v1.1) for more details.
* Jun 13, 2024: :truck: The training code is released, offering [full-parameter training](#full-parameter-training) and [LoRA training](#lora).
* Jun 06, 2024: :tada: Hunyuan-DiT is now available in ComfyUI. Please check [ComfyUI](#using-comfyui) for more details.
* Jun 06, 2024: 🚀 We introduce Distillation version for Hunyuan-DiT acceleration, which achieves **50%** acceleration on NVIDIA GPUs. Please check [Distillation](https://huggingface.co/Tencent-Hunyuan/Distillation) for more details.
* Jun 05, 2024: 🤗 Hunyuan-DiT is now available in 🤗 Diffusers! Please check the [example](#using--diffusers) below.
* Jun 04, 2024: :globe_with_meridians: Support Tencent Cloud links to download the pretrained models! Please check the [links](#-download-pretrained-models) below.
* May 22, 2024: 🚀 We introduce TensorRT version for Hunyuan-DiT acceleration, which achieves **47%** acceleration on NVIDIA GPUs. Please check [TensorRT-libs](https://huggingface.co/Tencent-Hunyuan/TensorRT-libs) for instructions.
* May 22, 2024: 💬 We support demo running multi-turn text2image generation now. Please check the [script](#using-gradio) below.

## 🤖 Try it on the web

Welcome to our web-based [**Tencent Hunyuan Bot**](https://hunyuan.tencent.com/bot/chat), where you can explore our innovative products! Just input the suggested prompts below or any other **imaginative prompts containing drawing-related keywords** to activate the Hunyuan text-to-image generation feature.  Unleash your creativity and create any picture you desire, **all for free!**

You can use simple prompts similar to natural language text

> 画一只穿着西装的猪
>
> draw a pig in a suit
>
> 生成一幅画，赛博朋克风，跑车
> 
> generate a painting, cyberpunk style, sports car

or multi-turn language interactions to create the picture. 

> 画一个木制的鸟
>
> draw a wooden bird
>
> 变成玻璃的
>
> turn into glass

## 📑 Open-source Plan

- Hunyuan-DiT (Text-to-Image Model)
  - [x] Inference 
  - [x] Checkpoints 
  - [x] Distillation Version
  - [x] TensorRT Version
  - [x] Training
  - [x] Lora
  - [x] Controlnet (Pose, Canny, Depth)
  - [x] 6GB GPU VRAM Inference 
  - [ ] IP-adapter
  - [ ] Hunyuan-DiT-S checkpoints (0.7B model)
- Mllm
  - Hunyuan-Captioner (Re-caption the raw image-text pairs)
    - [x] Inference
  - [Hunyuan-DialogGen](https://github.com/Centaurusalpha/DialogGen) (Prompt Enhancement Model)
    - [x] Inference
- [X] Web Demo (Gradio) 
- [x] Multi-turn T2I Demo (Gradio)
- [X] Cli Demo 
- [X] ComfyUI
- [X] Diffusers
- [ ] Kohya
- [ ] WebUI


## Contents
- [Hunyuan-DiT](#hunyuan-dit--a-powerful-multi-resolution-diffusion-transformer-with-fine-grained-chinese-understanding)
  - [Abstract](#abstract)
  - [🎉 Hunyuan-DiT Key Features](#-hunyuan-dit-key-features)
    - [Chinese-English Bilingual DiT Architecture](#chinese-english-bilingual-dit-architecture)
    - [Multi-turn Text2Image Generation](#multi-turn-text2image-generation)
  - [📈 Comparisons](#-comparisons)
  - [🎥 Visualization](#-visualization)
  - [📜 Requirements](#-requirements)
  - [🛠 Dependencies and Installation](#%EF%B8%8F-dependencies-and-installation)
  - [🧱 Download Pretrained Models](#-download-pretrained-models)
  - [:truck: Training](#truck-training)
    - [Data Preparation](#data-preparation)
    - [Full Parameter Training](#full-parameter-training)
    - [LoRA](#lora)
  - [🔑 Inference](#-inference)
    - [6GB GPU VRAM Inference](#6gb-gpu-vram-inference)
    - [Using Gradio](#using-gradio)
    - [Using Diffusers](#using--diffusers)
    - [Using Command Line](#using-command-line)
    - [More Configurations](#more-configurations)
    - [Using ComfyUI](#using-comfyui)
  - [:building_construction: Adatper](#building_construction-adapter)
    - [ControlNet](#controlnet)
  - [:art: Hunyuan-Captioner](#art-hunyuan-captioner)
  - [🚀 Acceleration (for Linux)](#-acceleration-for-linux)
  - [🔗 BibTeX](#-bibtex)

## **Abstract**

We present Hunyuan-DiT, a text-to-image diffusion transformer with fine-grained understanding of both English and Chinese. To construct Hunyuan-DiT, we carefully designed the transformer structure, text encoder, and positional encoding. We also build from scratch a whole data pipeline to update and evaluate data for iterative model optimization. For fine-grained language understanding, we train a Multimodal Large Language Model to refine the captions of the images. Finally, Hunyuan-DiT can perform multi-round multi-modal dialogue with users, generating and refining images according to the context.
Through our carefully designed holistic human evaluation protocol with more than 50 professional human evaluators, Hunyuan-DiT sets a new state-of-the-art in Chinese-to-image generation compared with other open-source models.


## 🎉 **Hunyuan-DiT Key Features**
### **Chinese-English Bilingual DiT Architecture**
Hunyuan-DiT is a diffusion model in the latent space, as depicted in figure below. Following the Latent Diffusion Model, we use a pre-trained Variational Autoencoder (VAE) to compress the images into low-dimensional latent spaces and train a diffusion model to learn the data distribution with diffusion models. Our diffusion model is parameterized with a transformer. To encode the text prompts, we leverage a combination of pre-trained bilingual (English and Chinese) CLIP and multilingual T5 encoder.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/framework.png"  height=450>
</p>

### Multi-turn Text2Image Generation
Understanding natural language instructions and performing multi-turn interaction with users are important for a
text-to-image system. It can help build a dynamic and iterative creation process that bring the user’s idea into reality
step by step. In this section, we will detail how we empower Hunyuan-DiT with the ability to perform multi-round
conversations and image generation. We train MLLM to understand the multi-round user dialogue
and output the new text prompt for image generation.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/mllm.png"  height=300>
</p>

## 📈 Comparisons
In order to comprehensively compare the generation capabilities of HunyuanDiT and other models, we constructed a 4-dimensional test set, including Text-Image Consistency, Excluding AI Artifacts, Subject Clarity, Aesthetic. More than 50 professional evaluators performs the evaluation.

<p align="center">
<table> 
<thead> 
<tr> 
    <th rowspan="2">Model</th> <th rowspan="2">Open Source</th> <th>Text-Image Consistency (%)</th> <th>Excluding AI Artifacts (%)</th> <th>Subject Clarity (%)</th> <th rowspan="2">Aesthetics (%)</th> <th rowspan="2">Overall (%)</th> 
</tr> 
</thead> 
<tbody> 
<tr> 
    <td>SDXL</td> <td> ✔ </td> <td>64.3</td> <td>60.6</td> <td>91.1</td> <td>76.3</td> <td>42.7</td> 
</tr> 
<tr> 
    <td>PixArt-α</td> <td> ✔</td> <td>68.3</td> <td>60.9</td> <td>93.2</td> <td>77.5</td> <td>45.5</td> 
</tr> 
<tr> 
    <td>Playground 2.5</td> <td>✔</td> <td>71.9</td> <td>70.8</td> <td>94.9</td> <td>83.3</td> <td>54.3</td> 
</tr> 

<tr> 
    <td>SD 3</td> <td>&#10008</td> <td>77.1</td> <td>69.3</td> <td>94.6</td> <td>82.5</td> <td>56.7</td> 
    
</tr> 
<tr> 
    <td>MidJourney v6</td><td>&#10008</td> <td>73.5</td> <td>80.2</td> <td>93.5</td> <td>87.2</td> <td>63.3</td> 
</tr> 
<tr> 
    <td>DALL-E 3</td><td>&#10008</td> <td>83.9</td> <td>80.3</td> <td>96.5</td> <td>89.4</td> <td>71.0</td> 
</tr> 
<tr style="font-weight: bold; background-color: #f2f2f2;"> 
    <td>Hunyuan-DiT</td><td>✔</td> <td>74.2</td> <td>74.3</td> <td>95.4</td> <td>86.6</td> <td>59.0</td> 
</tr>
</tbody>
</table>
</p>

## 🎥 Visualization

* **Chinese Elements**
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/chinese elements understanding.png"  height=220>
</p>

* **Long Text Input**


<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/long text understanding.png"  height=310>
</p>

* **Multi-turn Text2Image Generation**

https://github.com/Tencent/tencent.github.io/assets/27557933/94b4dcc3-104d-44e1-8bb2-dc55108763d1



---

## 📜 需求

该版本包括了 DialogGen (一种提示增强的模型)和 Hunyuan-DiT (一种文本到图像的模型)。

下表表明了运行模型的要求 (batch size = 1):

|          模型           | --加载-4bit (DialogGen) | GPU最低显存      |       GPU型号       |
|:-----------------------:|:-----------------------:|:---------------:|:---------------:|
| DialogGen + Hunyuan-DiT |            ✘            |       32G       |      A100       |
| DialogGen + Hunyuan-DiT |            ✔            |       22G       |      A100       |
|       Hunyuan-DiT       |            -            |       11G       |      A100       |
|       Hunyuan-DiT       |            -            |       14G       | RTX3090/RTX4090 |

*需要一个支持CUDA的英伟达GPU。
  * 我们在V100和A100的GPUs上进行测试。
  * **最低配置**: GPU最小显存应该达到11GB。
  * **推荐配置**: 我们推荐使用显存为32GB的GPU以获得更好的生成质量。
* 测试使用的操作系统: Linux

## 🛠️ 环境依赖与安装

首先克隆该仓库:
```shell
git clone https://github.com/tencent/HunyuanDiT
cd HunyuanDiT
```

### Linux系统的安装指南

我们提供了一个名为 `environment.yml`的文件来创造一个Conda环境。
Conda的安装说明可以查阅[这里](https://docs.anaconda.com/free/miniconda/index.html).

我们推荐CUDA的版本11.7或12.0+.

```shell
# 1. 创建conda环境
conda env create -f environment.yml

# 2. 激活环境
conda activate HunyuanDiT

# 3. 安装环境依赖
python -m pip install -r requirements.txt

# 4. (可选)安装用于加速的 flash attention v2(需要CUDA11.6或者更高的版本)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

## 🧱 下载预训练模型
要下载模型，首先要安装huggingface-cli。 (详细的说明见[此处](https://huggingface.co/docs/huggingface_hub/guides/cli)。)

```shell
python -m pip install "huggingface_hub[cli]"
```

然后使用以下命令下载模型:

```shell
# Create a directory named 'ckpts' where the model will be saved, fulfilling the prerequisites for running the demo.
mkdir ckpts
# Use the huggingface-cli tool to download the model.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

<details>
<summary>💡使用huggingface-cli的小技巧 (网络下载问题)</summary>

##### 1. 使用HF-Mirror

如果在中国遇到下载速度慢的问题，可以尝试使用镜像来加快下载速度。 例如,

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

##### 2. 恢复下载

`huggingface-cli`支持回复下载。如果下载中端，只需重新运行下载命令就能恢复下载进程。

注意: 如果出现`No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` 错误，可以忽略该错误并重新运行下载命令。

</details>

---

所有的模型都可以免费下载。 若要获取更多有关模型的信息，请访问“Hugging Face”[资源库](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT)。

|       模型        | #参数量 |                                      “Hugging Face”下载链接                                           |                                      腾讯云下载链接                                |
|:------------------:|:-------:|:-------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|        mT5         |  1.6B   |               [mT5](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/mt5)                |               [mT5](https://dit.hunyuan.tencent.com/download/HunyuanDiT/mt5.zip)                |
|        CLIP        |  350M   |        [CLIP](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder)        |        [CLIP](https://dit.hunyuan.tencent.com/download/HunyuanDiT/clip_text_encoder.zip)        |
|      Tokenizer     |  -      |     [Tokenizer](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/tokenizer)              |      [Tokenizer](https://dit.hunyuan.tencent.com/download/HunyuanDiT/tokenizer.zip)             |
|     DialogGen      |  7.0B   |           [DialogGen](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/dialoggen)            |           [DialogGen](https://dit.hunyuan.tencent.com/download/HunyuanDiT/dialoggen.zip)        |
| sdxl-vae-fp16-fix  |   83M   | [sdxl-vae-fp16-fix](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/sdxl-vae-fp16-fix)  | [sdxl-vae-fp16-fix](https://dit.hunyuan.tencent.com/download/HunyuanDiT/sdxl-vae-fp16-fix.zip)  |
|    Hunyuan-DiT-v1.0     |  1.5B   |          [Hunyuan-DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/model)           |          [Hunyuan-DiT-v1.0](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model.zip)           |
|    Hunyuan-DiT-v1.1     |  1.5B   |          [Hunyuan-DiT-v1.1](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1/tree/main/t2i/model)    |          [Hunyuan-DiT-v1.1](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model-v1_1.zip)            |
|    Data demo       |  -      |                                    -                                                                    |      [Data demo](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip)             |

## :truck: 训练

### 数据准备

  请参考下面的命令来准备训练数据。
  
  1. 安装依赖项
  
      我们提供了一个名为“IndexKits”的高效数据管理库，它支持在训练过程中读取数以亿计的数据。更多详细信息见此[文档](./IndexKits/README.md)。
      ```shell
      # 1 安装依赖项
      cd HunyuanDiT
      pip install -e ./IndexKits
     ```
  2. 下载数据 
  
     欢迎随时下载数据，通过[数据演示](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip)。
     ```shell
     # 2 下载数据
     wget -O ./dataset/data_demo.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip
     unzip ./dataset/data_demo.zip -d ./dataset
     mkdir ./dataset/porcelain/arrows ./dataset/porcelain/jsons
     ```
  3. 数据转换
  
     为训练数据创建一个 CSV 文件，其中包含下表列出的字段。
    
     |    字段       | 是否需求  |  描述     |   示例   |
     |:---------------:| :------:  |:----------------:|:-----------:|
     |   `image_path`  | 是  |  图像路径               |     `./dataset/porcelain/images/0.png`        | 
     |   `text_zh`     | 是  |    描述文本              |  青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 | 
     |   `md5`         | 可选  |    图像的信息摘要(md5)  |    `d41d8cd98f00b204e9800998ecf8427e`         | 
     |   `width`       | 可选  |    图像宽度    |     `1024 `       | 
     |   `height`      | 可选  |    图像高度    |    ` 1024 `       | 
     
     > ⚠️ 图像的md5、宽度和高度等可选字段可以省略。如果省略，下面的脚本会自动计算。在处理大规模训练数据时，这一过程可能会比较耗时。
  
    我们使用[Arrow](https://github.com/apache/arrow)格式作为训练数据格式，以提供标准高效的内存数据表示。同时，我们提供了将 CSV 格式转换为 Arrow 格式的转换脚本。.
     ```shell  
     # 3 Data conversion 
     python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows 1
     ```
  
  4. 数据筛选和配置文件创建 
     
      我们通过 YAML 文件配置训练数据。在这些文件中，你可以设置有关训练数据的过滤、复制、重复数据等标准数据处理策略。有关详细信息，请参见[./IndexKits](IndexKits/docs/MakeDataset.md).
  
      有关示例文件，请参阅[文件](./dataset/yamls/porcelain.yaml).。如需完整的参数配置文件，请参阅[文件](./IndexKits/docs/MakeDataset.md)。
  
     
  5. 使用 YAML 文件创建训练数据索引文件。
    
     ```shell
      # 制备单分辨率的数据集
      idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json
   
      # 制备多分辨率的数据集     
      idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json
      ```
   
  "瓷器"数据集的目录结构应为:

  ```shell
   cd ./dataset
  
   porcelain
      ├──images/  (图像文件)
      │  ├──0.png
      │  ├──1.png
      │  ├──......
      ├──csvfile/  (包含配对"文本-图片"的 csv 文件)
      │  ├──image_text.csv
      ├──arrows/  (包含所有必要训练数据的 arrow 文件)
      │  ├──00000.arrow
      │  ├──00001.arrow
      │  ├──......
      ├──jsons/  (在训练期间从arrow文件中读取数据的最终训练数据索引文件)
      │  ├──porcelain.json
      │  ├──porcelain_mt.json
   ```

### 全量训练
 
  要在训练中利用 DeepSpeed，您可以通过调整 '--hostfile' 和 '--master_addr' 等参数，灵活控制单节点/多节点训练。 For more details, see [link](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

  ```shell
  # 单分辨率训练
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain.json
  
  # 多分辨率训练
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain_mt.json --multireso --reso-step 64
  ```

### LoRA



我们提供了 LoRA 的训练和推理脚本, 更多细节见[./lora](./lora/README.md). 

  ```shell
  # 训练"瓷器"相关的LoRA。
  PYTHONPATH=./ sh lora/train_lora.sh --index-file dataset/porcelain/jsons/porcelain.json

  # 使用预训练的 LORA 权重进行推理。
  python sample_t2i.py --prompt "青花瓷风格，一只小狗"  --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt
  ```
 我们为'瓷器'和'玉器'提供两个预训练的 LoRA 权重, 更多细节请参阅[链接](https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA)
  ```shell
  cd HunyuanDiT
  # 使用 huggingface-cli 工具来下载.
  huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora
  
  # 快速使用
  python sample_t2i.py --prompt "青花瓷风格，一只猫在追蝴蝶"  --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain
  ```
 <table>
  <tr>
    <td colspan="4" align="center">训练数据示例</td>
  </tr>
  
  <tr>
    <td align="center"><img src="lora/asset/porcelain/train/0.png" alt="Image 0" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/train/1.png" alt="Image 1" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/train/2.png" alt="Image 2" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/train/3.png" alt="Image 3" width="200"/></td>
  </tr>
  <tr>
    <td align="center">青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 （Porcelain style, a blue bird stands on a blue vase, surrounded by white flowers, with a white background.
）</td>
    <td align="center">青花瓷风格，这是一幅蓝白相间的陶瓷盘子，上面描绘着一只狐狸和它的幼崽在森林中漫步，背景是白色 （Porcelain style, this is a blue and white ceramic plate depicting a fox and its cubs strolling in the forest, with a white background.）</td>
    <td align="center">青花瓷风格，在黑色背景上，一只蓝色的狼站在蓝白相间的盘子上，周围是树木和月亮 （Porcelain style, on a black background, a blue wolf stands on a blue and white plate, surrounded by trees and the moon.）</td>
    <td align="center">青花瓷风格，在蓝色背景上，一只蓝色蝴蝶和白色花朵被放置在中央 （Porcelain style, on a blue background, a blue butterfly and white flowers are placed in the center.）</td>
  </tr>
  <tr>
    <td colspan="4" align="center">推理结果示例</td>
  </tr>
  <tr>
    <td align="center"><img src="lora/asset/porcelain/inference/0.png" alt="Image 4" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/inference/1.png" alt="Image 5" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/inference/2.png" alt="Image 6" width="200"/></td>
    <td align="center"><img src="lora/asset/porcelain/inference/3.png" alt="Image 7" width="200"/></td>
  </tr>
  <tr>
    <td align="center">青花瓷风格，苏州园林 （Porcelain style,  Suzhou Gardens.）</td>
    <td align="center">青花瓷风格，一朵荷花 （Porcelain style,  a lotus flower.）</td>
    <td align="center">青花瓷风格，一只羊（Porcelain style, a sheep.）</td>
    <td align="center">青花瓷风格，一个女孩在雨中跳舞（Porcelain style, a girl dancing in the rain.）</td>
  </tr>
  
</table>


## 🔑 推理

### 6GB GPU VRAM 推理
基于[diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuandit)，现在可以在 6GB GPU VRAM 下运行HunyuanDiT。我们将为您提供快速上手的说明和演示。

> 6GB 版本支持 Nvidia Ampere 架构系列显卡，如 RTX 3070/3080/4080/4090，A100 等。

您唯一需要做的就是安装以下库：

```bash
pip install -U bitsandbytes
pip install git+https://github.com/huggingface/diffusers
pip install torch==2.0.0
```

然后，您就可以在 6GB GPU VRAM 下直接享受 HunyuanDiT 文字转图像功能了！

下面为您提供一个示例。

```bash
cd HunyuanDiT

# 快速使用
model_id=Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled
prompt=一个宇航员在骑马
infer_steps=50
guidance_scale=6
python3 lite/inference.py ${model_id} ${prompt} ${infer_steps} ${guidance_scale}
```

详情见[./lite](lite/README.md)。


### 使用Gradio

在运行以下命令前，请确保已激活 Conda 环境。

```shell
# 默认情况下，我们使用的是中文界面。
python app/hydit_app.py

# 使用 Flash Attention 进行加速。
python app/hydit_app.py --infer-mode fa

# 如果 GPU 内存不足，可以禁用增强模型。
# 该增强功能将不可用，直到您重新启动应用程序时不使用'--no-enhance'。
python app/hydit_app.py --no-enhance

# 使用英文界面。
python app/hydit_app.py --lang en

# 使用 multi-turn T2I 生成交互界面. 
# 如果您的 GPU 显存小于 32GB，请使用"--load-4bit "启用 4 位量化，但这至少需要 22GB 显存。
python app/multiTurnT2I_app.py
```
然后就可以通过 http://0.0.0.0:443 访问演示程序了。需要注意的是，这里的 0.0.0.0 需要与您的服务器 IP X.X.X.X保持一致。

### 使用 🤗 Diffusers

请提前安装 PyTorch 2.0 或更高版本，以满足指定版本“diffusers”库的要求。

安装 🤗 diffusers, 请确保其版本至少为 0.28.1:

```shell
pip install git+https://github.com/huggingface/diffusers.git
```
或者
```shell
pip install diffusers
```

您可以通过以下 Python 脚本使用中文和英文提示来生成图像：
```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

# 您可以使用英文提示，因为 HunyuanDiT 支持英文和中文
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt).images[0]
```
您可以使用我们经过蒸馏的模型更快地生成图像：

```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled", torch_dtype=torch.float16)
pipe.to("cuda")

# 您可以使用英文提示，因为 HunyuanDiT 支持英文和中文
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt, num_inference_steps=25).images[0]
```
更多细节请参见[HunyuanDiT-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled)

**更多功能:** 如需了解 LoRA 和 ControlNet 等其他功能，请参阅 [./diffusers](diffusers)中的“README”文件。

### 使用命令行

我们提供了几条快速启动的命令: 

```shell
# Prompt Enhancement + Text-to-Image. Torch mode
python sample_t2i.py --prompt "渔舟唱晚"

# Only Text-to-Image. Torch mode
python sample_t2i.py --prompt "渔舟唱晚" --no-enhance

# Only Text-to-Image. Flash Attention mode
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"

# Generate an image with other image sizes.
python sample_t2i.py --prompt "渔舟唱晚" --image-size 1280 768

# Prompt Enhancement + Text-to-Image. DialogGen loads with 4-bit quantization, but it may loss performance.
python sample_t2i.py --prompt "渔舟唱晚"  --load-4bit

```

更多提示范例请参见[example_prompts.txt](example_prompts.txt)。

### 更多配置

我们列出了一些更有用的配置，以方便使用:

|    参数     |  默认  |                     描述                    |
|:---------------:|:---------:|:---------------------------------------------------:|
|   `--prompt`    |   None    |        The text prompt for image generation         |
| `--image-size`  | 1024 1024 |           The size of the generated image           |
|    `--seed`     |    42     |        The random seed for generating images        |
| `--infer-steps` |    100    |          The number of steps for sampling           |
|  `--negative`   |     -     |      The negative prompt for image generation       |
| `--infer-mode`  |   torch   |       The inference mode (torch, fa, or trt)        |
|   `--sampler`   |   ddpm    |    The diffusion sampler (ddpm, ddim, or dpmms)     |
| `--no-enhance`  |   False   |        Disable the prompt enhancement model         |
| `--model-root`  |   ckpts   |     The root directory of the model checkpoints     |
|  `--load-key`   |    ema    | Load the student model or EMA model (ema or module) |
|  `--load-4bit`  |   Fasle   |     Load DialogGen model with 4bit quantization     |

### 使用 ComfyUI

我们提供了几条快速启动的命令：

```shell
# Download comfyui code
git clone https://github.com/comfyanonymous/ComfyUI.git

# Install torch, torchvision, torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Install Comfyui essential python package.
cd ComfyUI
pip install -r requirements.txt

# ComfyUI has been successfully installed!

# Download model weight as before or link the existing model folder to ComfyUI.
python -m pip install "huggingface_hub[cli]"
mkdir models/hunyuan
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./models/hunyuan/ckpts

# Move to the ComfyUI custom_nodes folder and copy comfyui-hydit folder from HunyuanDiT Repo.
cd custom_nodes
cp -r ${HunyuanDiT}/comfyui-hydit ./
cd comfyui-hydit

# Install some essential python Package.
pip install -r requirements.txt

# Our tool has been successfully installed!

# Go to ComfyUI main folder
cd ../..
# Run the ComfyUI Lauch command
python main.py --listen --port 80

# Running ComfyUI successfully!
```
更多详情请参见[./comfyui-hydit](comfyui-hydit/README.md)。

## :building_construction: Adapter

### ControlNet

We provide training scripts for ControlNet, detailed in the [./controlnet](./controlnet/README.md). 

  ```shell
  # Training for canny ControlNet.
  PYTHONPATH=./ sh hydit/train_controlnet.sh
  ```
 We offer three types of trained ControlNet weights for `canny` `depth` and `pose`, see details at [links](https://huggingface.co/Tencent-Hunyuan/HYDiT-ControlNet)
  ```shell
  cd HunyuanDiT
  # Use the huggingface-cli tool to download the model.
  # We recommend using distilled weights as the base model for ControlNet inference, as our provided pretrained weights are trained on them.
  huggingface-cli download Tencent-Hunyuan/HYDiT-ControlNet --local-dir ./ckpts/t2i/controlnet
  huggingface-cli download Tencent-Hunyuan/Distillation-v1.1 ./pytorch_model_distill.pt --local-dir ./ckpts/t2i/model
  
  # Quick start
  python3 sample_controlnet.py  --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0
  ```
 
 <table>
  <tr>
    <td colspan="3" align="center">Condition Input</td>
  </tr>
  
   <tr>
    <td align="center">Canny ControlNet </td>
    <td align="center">Depth ControlNet </td>
    <td align="center">Pose ControlNet </td>
  </tr>

  <tr>
    <td align="center">在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围<br>（At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere.） </td>
    <td align="center">在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足<br>（In the dense forest, a black and white panda sits quietly in green trees and red flowers, surrounded by mountains, rivers, and the ocean. The background is the forest in a bright environment.） </td>
    <td align="center">一位亚洲女性，身穿绿色上衣，戴着紫色头巾和紫色围巾，站在黑板前。背景是黑板。照片采用近景、平视和居中构图的方式呈现真实摄影风格<br>（An Asian woman, dressed in a green top, wearing a purple headscarf and a purple scarf, stands in front of a blackboard. The background is the blackboard. The photo is presented in a close-up, eye-level, and centered composition, adopting a realistic photographic style） </td>
  </tr>

  <tr>
    <td align="center"><img src="controlnet/asset/input/canny.jpg" alt="Image 0" width="200"/></td>
    <td align="center"><img src="controlnet/asset/input/depth.jpg" alt="Image 1" width="200"/></td>
    <td align="center"><img src="controlnet/asset/input/pose.jpg" alt="Image 2" width="200"/></td>
    
  </tr>
  
  <tr>
    <td colspan="3" align="center">ControlNet Output</td>
  </tr>

  <tr>
    <td align="center"><img src="controlnet/asset/output/canny.jpg" alt="Image 3" width="200"/></td>
    <td align="center"><img src="controlnet/asset/output/depth.jpg" alt="Image 4" width="200"/></td>
    <td align="center"><img src="controlnet/asset/output/pose.jpg" alt="Image 5" width="200"/></td>
  </tr>
 
</table>

## :art: Hunyuan-Captioner
Hunyuan-Captioner meets the need of text-to-image techniques by maintaining a high degree of image-text consistency. It can generate high-quality image descriptions from a variety of angles, including object description, objects relationships, background information, image style, etc. Our code is based on [LLaVA](https://github.com/haotian-liu/LLaVA) implementation.

### Examples

<td align="center"><img src="./asset/caption_demo.jpg" alt="Image 3" width="1200"/></td>

### Instructions
a. Install dependencies
     
The dependencies and installation are basically the same as the [**base model**](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1).

b. Model download
```shell
# Use the huggingface-cli tool to download the model.
huggingface-cli download Tencent-Hunyuan/HunyuanCaptioner --local-dir ./ckpts/captioner
```

### Inference

Our model supports three different modes including: **directly generating Chinese caption**, **generating Chinese caption based on specific knowledge**, and **directly generating English caption**. The injected information can be either accurate cues or noisy labels (e.g., raw descriptions crawled from the internet). The model is capable of generating reliable and accurate descriptions based on both the inserted information and the image content.

|Mode           | Prompt Template                           |Description                           | 
| ---           | ---                                       | ---                                  |
|caption_zh     | 描述这张图片                               |Caption in Chinese                    | 
|insert_content | 根据提示词“{}”,描述这张图片                 |Caption with inserted knowledge| 
|caption_en     | Please describe the content of this image |Caption in English                    |
|               |                                           |                                      |
 

a. Single picture inference in Chinese

```bash
python mllm/caption_demo.py --mode "caption_zh" --image_file "mllm/images/demo1.png" --model_path "./ckpts/captioner"
```

b. Insert specific knowledge into caption

```bash
python mllm/caption_demo.py --mode "insert_content" --content "宫保鸡丁" --image_file "mllm/images/demo2.png" --model_path "./ckpts/captioner"
```

c. Single picture inference in English

```bash
python mllm/caption_demo.py --mode "caption_en" --image_file "mllm/images/demo3.png" --model_path "./ckpts/captioner"
```

d. Multiple pictures inference in Chinese

```bash
### Convert multiple pictures to csv file. 
python mllm/make_csv.py --img_dir "mllm/images" --input_file "mllm/images/demo.csv"

### Multiple pictures inference
python mllm/caption_demo.py --mode "caption_zh" --input_file "mllm/images/demo.csv" --output_file "mllm/images/demo_res.csv" --model_path "./ckpts/captioner"
```

(Optional) To convert the output csv file to Arrow format, please refer to [Data Preparation #3](#data-preparation) for detailed instructions. 


### Gradio 
To launch a Gradio demo locally, please run the following commands one by one. For more detailed instructions, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA). 
```bash
cd mllm
python -m llava.serve.controller --host 0.0.0.0 --port 10000

python -m llava.serve.gradio_web_server --controller http://0.0.0.0:10000 --model-list-mode reload --port 443

python -m llava.serve.model_worker --host 0.0.0.0 --controller http://0.0.0.0:10000 --port 40000 --worker http://0.0.0.0:40000 --model-path "../ckpts/captioner" --model-name LlavaMistral
```
Then the demo can be accessed through http://0.0.0.0:443. It should be noted that the 0.0.0.0 here needs to be X.X.X.X with your server IP.

## 🚀 Acceleration (for Linux)

- We provide TensorRT version of HunyuanDiT for inference acceleration (faster than flash attention).
See [Tencent-Hunyuan/TensorRT-libs](https://huggingface.co/Tencent-Hunyuan/TensorRT-libs) for more details.

- We provide Distillation version of HunyuanDiT for inference acceleration.
See [Tencent-Hunyuan/Distillation](https://huggingface.co/Tencent-Hunyuan/Distillation) for more details.

## 🔗 BibTeX
If you find [Hunyuan-DiT](https://arxiv.org/abs/2405.08748) or [DialogGen](https://arxiv.org/abs/2403.08857) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{li2024hunyuandit,
      title={Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding}, 
      author={Zhimin Li and Jianwei Zhang and Qin Lin and Jiangfeng Xiong and Yanxin Long and Xinchi Deng and Yingfang Zhang and Xingchao Liu and Minbin Huang and Zedong Xiao and Dayou Chen and Jiajun He and Jiahao Li and Wenyue Li and Chen Zhang and Rongwei Quan and Jianxiang Lu and Jiabin Huang and Xiaoyan Yuan and Xiaoxiao Zheng and Yixuan Li and Jihong Zhang and Chao Zhang and Meng Chen and Jie Liu and Zheng Fang and Weiyan Wang and Jinbao Xue and Yangyu Tao and Jianchen Zhu and Kai Liu and Sihuan Lin and Yifu Sun and Yun Li and Dongdong Wang and Mingtao Chen and Zhichao Hu and Xiao Xiao and Yan Chen and Yuhong Liu and Wei Liu and Di Wang and Yong Yang and Jie Jiang and Qinglin Lu},
      year={2024},
      eprint={2405.08748},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{huang2024dialoggen,
  title={DialogGen: Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation},
  author={Huang, Minbin and Long, Yanxin and Deng, Xinchi and Chu, Ruihang and Xiong, Jiangfeng and Liang, Xiaodan and Cheng, Hong and Lu, Qinglin and Liu, Wei},
  journal={arXiv preprint arXiv:2403.08857},
  year={2024}
}
```

## Start History

<a href="https://star-history.com/#Tencent/HunyuanDiT&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date" />
 </picture>
</a>

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

## 📜 Requirements

This repo consists of DialogGen (a prompt enhancement model) and Hunyuan-DiT (a text-to-image model).

The following table shows the requirements for running the models (batch size = 1):

|          Model          | --load-4bit (DialogGen) | GPU Peak Memory |       GPU       |
|:-----------------------:|:-----------------------:|:---------------:|:---------------:|
| DialogGen + Hunyuan-DiT |            ✘            |       32G       |      A100       |
| DialogGen + Hunyuan-DiT |            ✔            |       22G       |      A100       |
|       Hunyuan-DiT       |            -            |       11G       |      A100       |
|       Hunyuan-DiT       |            -            |       14G       | RTX3090/RTX4090 |

* An NVIDIA GPU with CUDA support is required. 
  * We have tested V100 and A100 GPUs.
  * **Minimum**: The minimum GPU memory required is 11GB.
  * **Recommended**: We recommend using a GPU with 32GB of memory for better generation quality.
* Tested operating system: Linux

## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/tencent/HunyuanDiT
cd HunyuanDiT
```

### Installation Guide for Linux

We provide an `environment.yml` file for setting up a Conda environment.
Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

We recommend CUDA versions 11.7 and 12.0+.

```shell
# 1. Prepare conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate HunyuanDiT

# 3. Install pip dependencies
python -m pip install -r requirements.txt

# 4. (Optional) Install flash attention v2 for acceleration (requires CUDA 11.6 or above)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

## 🧱 Download Pretrained Models
To download the model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Create a directory named 'ckpts' where the model will be saved, fulfilling the prerequisites for running the demo.
mkdir ckpts
# Use the huggingface-cli tool to download the model.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

<details>
<summary>💡Tips for using huggingface-cli (network problem)</summary>

##### 1. Using HF-Mirror

If you encounter slow download speeds in China, you can try a mirror to speed up the download process. For example,

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

##### 2. Resume Download

`huggingface-cli` supports resuming downloads. If the download is interrupted, you can just rerun the download 
command to resume the download process.

Note: If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download 
process, you can ignore the error and rerun the download command.

</details>

---

All models will be automatically downloaded. For more information about the model, visit the Hugging Face repository [here](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).

|       Model        | #Params |                                      Huggingface Download URL                                           |                                      Tencent Cloud Download URL                                 |
|:------------------:|:-------:|:-------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|        mT5         |  1.6B   |               [mT5](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/mt5)                |               [mT5](https://dit.hunyuan.tencent.com/download/HunyuanDiT/mt5.zip)                |
|        CLIP        |  350M   |        [CLIP](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder)        |        [CLIP](https://dit.hunyuan.tencent.com/download/HunyuanDiT/clip_text_encoder.zip)        |
|      Tokenizer     |  -      |     [Tokenizer](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/tokenizer)              |      [Tokenizer](https://dit.hunyuan.tencent.com/download/HunyuanDiT/tokenizer.zip)             |
|     DialogGen      |  7.0B   |           [DialogGen](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/dialoggen)            |           [DialogGen](https://dit.hunyuan.tencent.com/download/HunyuanDiT/dialoggen.zip)        |
| sdxl-vae-fp16-fix  |   83M   | [sdxl-vae-fp16-fix](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/sdxl-vae-fp16-fix)  | [sdxl-vae-fp16-fix](https://dit.hunyuan.tencent.com/download/HunyuanDiT/sdxl-vae-fp16-fix.zip)  |
|    Hunyuan-DiT-v1.0     |  1.5B   |          [Hunyuan-DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/model)           |          [Hunyuan-DiT-v1.0](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model.zip)           |
|    Hunyuan-DiT-v1.1     |  1.5B   |          [Hunyuan-DiT-v1.1](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1/tree/main/t2i/model)    |          [Hunyuan-DiT-v1.1](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model-v1_1.zip)            |
|    Data demo       |  -      |                                    -                                                                    |      [Data demo](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip)             |

## :truck: 训练部分

### 数据准备

  通过下方步骤中的命令来准备训练数据。
  
  1. 安装依赖

      我们提供了一个高效的数据管理库，名为 IndexKits，支持在训练过程中读取数亿条数据，详见 [docs](./IndexKits/README.md)。
      ```shell
      # 1 安装依赖
      cd HunyuanDiT
      pip install -e ./IndexKits
     ```
  2. 数据下载 
  
     下载链接 [data demo](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip).
     ```shell
     # 2 数据下载
     wget -O ./dataset/data_demo.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip
     unzip ./dataset/data_demo.zip -d ./dataset
     mkdir ./dataset/porcelain/arrows ./dataset/porcelain/jsons
     ```
  3. 数据转换 

     创建CSV文件，并包含如下表格所示字段。
    
     |    Fields       | Required  |  Description     |   Example   |
     |:---------------:| :------:  |:----------------:|:-----------:|
     |   `image_path`  | Required  |  image path               |     `./dataset/porcelain/images/0.png`        | 
     |   `text_zh`     | Required  |    text               |  青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 | 
     |   `md5`         | Optional  |    image md5 (Message Digest Algorithm 5)  |    `d41d8cd98f00b204e9800998ecf8427e`         | 
     |   `width`       | Optional  |    image width    |     `1024 `       | 
     |   `height`      | Optional  |    image height   |    ` 1024 `       | 
     
     > ⚠️ 可选择的字段，如md5码、width和height，可以省略。如果省略，下方的脚本将对它们进行自动计算。当处理大规模训练数据时，这个过程可能会耗费时间。
  
     我们使用[Arrow](https://github.com/apache/arrow)库来进行训练数据的格式化，Arrow提供了一种标准且高效的内存数据表示。我们提供了一个转换脚本，用于将CSV文件转换为Arrow格式。
     ```shell  
     # 3 数据转换
     python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows 1
     ```
  
  4. 数据选择和配置文件创建 
     
      我们通过YAML文件来配置训练数据。在这些YAML文件中，您可以设置标准的数据处理策略，用于过滤、复制、去重等与训练数据相关的操作。更多详情请参见 [./IndexKits](IndexKits/docs/MakeDataset.md)。
  
      对于示例文件，请参见 [文档](./dataset/yamls/porcelain.yaml)。对于完整的参数配置文件，请参见 [文档](./IndexKits/docs/MakeDataset.md)。
  
     
  5. 使用YAML完成训练数据索引文件的创建
    
     ```shell
      # 单分辨率数据准备
      idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json
   
      # 多分辨率数据准备
      idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json
      ```
   
  以 `porcelain` 为例，数据集的目录结构为:

  ```shell
   cd ./dataset
  
   porcelain
      ├──images/  (image files)
      │  ├──0.png
      │  ├──1.png
      │  ├──......
      ├──csvfile/  (csv files containing text-image pairs)
      │  ├──image_text.csv
      ├──arrows/  (arrow files containing all necessary training data)
      │  ├──00000.arrow
      │  ├──00001.arrow
      │  ├──......
      ├──jsons/  (final training data index files which read data from arrow files during training)
      │  ├──porcelain.json
      │  ├──porcelain_mt.json
   ```

### 完整参数的训练

  如果您想使用DeepSpeed加速模型的训练，您可以通过调节参数，如 `--hostfile` 和 `--master_addr`，来控制 **单节点** / **多节点** 训练。更多详情请参见 [链接](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)。

  ```shell
  # 单分辨率训练
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain.json
  
  # 多分辨率训练
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain_mt.json --multireso --reso-step 64
  ```

### LoRA


我们为LoRA提供了训练和推理脚本，详见 [./lora](./lora/README.md)。

  ```shell
  # LoRA训练脚本
  PYTHONPATH=./ sh lora/train_lora.sh --index-file dataset/porcelain/jsons/porcelain.json

  # LoRA推理脚本
  python sample_t2i.py --prompt "青花瓷风格，一只小狗"  --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt
  ```
  我们提供 `porcelain` 和 `jade` 两种训练好的LoRA权重，详见 [链接](https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA)。

  ```shell
  cd HunyuanDiT
  # 使用huggingface-cli下载LoRA权重
  huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora
  
  # 测试推理
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

## 🔑 Inference

### 6GB GPU VRAM Inference
Running HunyuanDiT in under 6GB GPU VRAM is available now based on [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuandit). Here we provide instructions and demo for your quick start.

> The 6GB version supports Nvidia Ampere architecture series graphics cards such as RTX 3070/3080/4080/4090, A100, and so on.

The only thing you need do is to install the following library:

```bash
pip install -U bitsandbytes
pip install git+https://github.com/huggingface/diffusers
pip install torch==2.0.0
```

Then you can enjoy your HunyuanDiT text-to-image journey under 6GB GPU VRAM directly!

Here is a demo for you.

```bash
cd HunyuanDiT

# Quick start
model_id=Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled
prompt=一个宇航员在骑马
infer_steps=50
guidance_scale=6
python3 lite/inference.py ${model_id} ${prompt} ${infer_steps} ${guidance_scale}
```

More details can be found in [./lite](lite/README.md).


### Using Gradio

Make sure the conda environment is activated before running the following command.

```shell
# By default, we start a Chinese UI.
python app/hydit_app.py

# Using Flash Attention for acceleration.
python app/hydit_app.py --infer-mode fa

# You can disable the enhancement model if the GPU memory is insufficient.
# The enhancement will be unavailable until you restart the app without the `--no-enhance` flag. 
python app/hydit_app.py --no-enhance

# Start with English UI
python app/hydit_app.py --lang en

# Start a multi-turn T2I generation UI. 
# If your GPU memory is less than 32GB, use '--load-4bit' to enable 4-bit quantization, which requires at least 22GB of memory.
python app/multiTurnT2I_app.py
```
Then the demo can be accessed through http://0.0.0.0:443. It should be noted that the 0.0.0.0 here needs to be X.X.X.X with your server IP.

### Using 🤗 Diffusers

Please install PyTorch version 2.0 or higher in advance to satisfy the requirements of the specified version of the diffusers library.  

Install 🤗 diffusers, ensuring that the version is at least 0.28.1:

```shell
pip install git+https://github.com/huggingface/diffusers.git
```
or
```shell
pip install diffusers
```

You can generate images with both Chinese and English prompts using the following Python script:
```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt).images[0]
```
You can use our distilled model to generate images even faster:

```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled", torch_dtype=torch.float16)
pipe.to("cuda")

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt, num_inference_steps=25).images[0]
```
More details can be found in [HunyuanDiT-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled)

**More functions:** For other functions like LoRA and ControlNet, please have a look at the README of [./diffusers](diffusers).

### Using Command Line

We provide several commands to quick start: 

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

More example prompts can be found in [example_prompts.txt](example_prompts.txt)

### More Configurations

We list some more useful configurations for easy usage:

|    Argument     |  Default  |                     Description                     |
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

### Using ComfyUI

We provide several commands to quick start: 

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
More details can be found in [./comfyui-hydit](comfyui-hydit/README.md)

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

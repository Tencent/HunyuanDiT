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
* Jul 03, 2024: :tada: Kohya-hydit version now available for v1.1 and v1.2 models, with GUI for inference. Official Kohya version is under review. See [kohya](./kohya_ss-hydit) for details.
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
- [X] Kohya
- [ ] WebUI


## Contents
- [Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](#hunyuan-dit--a-powerful-multi-resolution-diffusion-transformer-with-fine-grained-chinese-understanding)
  - [🔥🔥🔥 News!!](#-news)
  - [🤖 Try it on the web](#-try-it-on-the-web)
  - [📑 Open-source Plan](#-open-source-plan)
  - [Contents](#contents)
  - [**Abstract**](#abstract)
  - [🎉 **Hunyuan-DiT Key Features**](#-hunyuan-dit-key-features)
    - [**Chinese-English Bilingual DiT Architecture**](#chinese-english-bilingual-dit-architecture)
    - [Multi-turn Text2Image Generation](#multi-turn-text2image-generation)
  - [📈 Comparisons](#-comparisons)
  - [🎥 Visualization](#-visualization)
  - [📜 Requirements](#-requirements)
  - [🛠️ Dependencies and Installation](#️-dependencies-and-installation)
    - [Installation Guide for Linux](#installation-guide-for-linux)
        - [1. Using HF-Mirror](#1-using-hf-mirror)
        - [2. Resume Download](#2-resume-download)
  - [:truck: Training](#truck-training)
    - [Data Preparation](#data-preparation)
    - [Full-parameter Training](#full-parameter-training)
    - [LoRA](#lora)
  - [🔑 Inference](#-inference)
    - [6GB GPU VRAM Inference](#6gb-gpu-vram-inference)
    - [Using Gradio](#using-gradio)
    - [Using 🤗 Diffusers](#using--diffusers)
    - [Using Command Line](#using-command-line)
    - [More Configurations](#more-configurations)
    - [Using ComfyUI](#using-comfyui)
    - [Using Kohya](#using-kohya)
  - [:building\_construction: Adapter](#building_construction-adapter)
    - [ControlNet](#controlnet)
  - [:art: Hunyuan-Captioner](#art-hunyuan-captioner)
    - [Examples](#examples)
    - [Instructions](#instructions)
    - [Inference](#inference)
    - [Gradio](#gradio)
  - [🚀 Acceleration (for Linux)](#-acceleration-for-linux)
  - [🔗 BibTeX](#-bibtex)
  - [Start History](#start-history)

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

Additionally, you can also use docker to set up the environment.
```shell
# 1. Use the following link to download the docker image tar file.
# For CUDA 12
wget https://dit.hunyuan.tencent.com/download/HunyuanDiT/hunyuan_dit_cu12.tar
# For CUDA 11
wget https://dit.hunyuan.tencent.com/download/HunyuanDiT/hunyuan_dit_cu11.tar

# 2. Import the docker tar file and show the image meta information
# For CUDA 12
docker load -i hunyuan_dit_cu12.tar
# For CUDA 11
docker load -i hunyuan_dit_cu11.tar  

docker image ls

# 3. Run the container based on the image
docker run -dit --gpus all --init --net=host --uts=host --ipc=host --name hunyuandit --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged  docker_image_tag
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

## :truck: 训练

### 数据准备

  参考以下命令来准备训练数据
  
  1. 安装依赖项
  
      我们提供了一个高效的数据管理库，名为 IndexKits，支持在训练期间进行对读取数亿个数据的管理，详细请见[文档](./IndexKits/README.md)
      ```shell
      # 1 安装依赖项
      cd HunyuanDiT
      pip install -e ./IndexKits
     ```
  2. 数据下载
  
     请下载[演示数据](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip)
     ```shell
     # 2 数据下载
     wget -O ./dataset/data_demo.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip
     unzip ./dataset/data_demo.zip -d ./dataset
     mkdir ./dataset/porcelain/arrows ./dataset/porcelain/jsons
     ```
  3. 数据转换
  
     使用下表中列出的字段为训练数据创建 CSV 文件。
    
     |      领域       |    必需   |       介绍        |    示例     |
     |:---------------:| :------:  |:----------------:|:-----------:|
     |   `image_path`  |    必需   |      图片路径     |`./dataset/porcelain/images/0.png`        | 
     |   `text_zh`     | 必需  |    文字描述               |  青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 | 
     |   `md5`         | 可选  |    图片 md5 (讯息摘要5)  |    `d41d8cd98f00b204e9800998ecf8427e`         | 
     |   `width`       | 可选  |    图片宽度    |     `1024 `       | 
     |   `height`      | 可选  |    图片高度   |    ` 1024 `       | 
     
     > ⚠️ 可以省略 MD5、宽度和高度等可选字段。如果省略，下面的脚本将自动计算它们。在处理大规模训练数据时，此过程可能非常耗时。
  
     我们可以利用[Arrow](https://github.com/apache/arrow) 来训练数据的格式，它提供标准高效的内存数据表示。同时提供了一个转换脚本，用于将 CSV 文件转换为Arrow格式。
     ```shell  
     # 3 数据转换
     python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows 1
     ```
  
  4. 数据选择和配置文件创建 
     
      我们通过 YAML 文件配置训练数据。在这些文件中，您可以设置标准数据处理策略，用于筛选、复制、重复数据删除等有关训练数据。有关详细信息，请参见[./IndexKits](IndexKits/docs/MakeDataset.md)
  
      有关示例文件，请参阅[文件](./dataset/yamls/porcelain.yaml) 有关完整参数配置文件，请参阅[文件](./IndexKits/docs/MakeDataset.md)
  
     
  5. 使用 YAML 文件创建训练数据索引文件
    
     ```shell
      # 单分辨率数据准备
      idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json
   
      # 多分辨率数据准备     
      idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json
      ```
   
  数据集 `porcelain` 的目录结构为:

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

### 全参数训练
 
  要在训练中利用 DeepSpeed，您可以通过调整`--hostfile`和`--master_addr` 等参数来灵活地控制**单节点** / **多节点**训练，有关详细信息，请参阅[链接](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)

  ```shell
  # 单分辨率训练
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain.json
  
  # 多分辨率训练
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain_mt.json --multireso --reso-step 64
  ```

### LoRA



我们提供了 LoRA 的训练和推理脚本，详细请见[./lora](./lora/README.md)

  ```shell
  # 训练 porcelain LoRA.
  PYTHONPATH=./ sh lora/train_lora.sh --index-file dataset/porcelain/jsons/porcelain.json

  # 使用 LORA 权重来进行推理.
  python sample_t2i.py --prompt "青花瓷风格，一只小狗"  --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt
  ```
 我们为 `porcelain` 和 `jade` 提供两种类型的训练 LoRA 权重，有关详细信息，请参阅[链接](https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA)
  ```shell
  cd HunyuanDiT
  # 使用 huggingface-cli 工具来下载模型.
  huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora
  
  # 快速开始
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
以[diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuandit)为基础在6GB以下的GPU VRAM中运行HunyuanDiT。我们在此为您的快速入门提供了说明和演示。

> 6GB的版本支持Nvidia Ampere架构系列显卡，如RTX 3070/3080/4080/4090、A100等。

您唯一需要做的就是安装以下库：

```bash
pip install -U bitsandbytes
pip install git+https://github.com/huggingface/diffusers
pip install torch==2.0.0
```

然后，您可以直接在6GB GPU VRAM下享受HunyuanDiT从文本到图像之旅！

这是一个示例。

```bash
cd HunyuanDiT

# 快速开始
model_id=Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled
prompt=一个宇航员在骑马
infer_steps=50
guidance_scale=6
python3 lite/inference.py ${model_id} ${prompt} ${infer_steps} ${guidance_scale}
```

更多详细信息在[./lite](lite/README.md)


### 使用 Gradio

在运行以下命令之前，请确保已激活 conda 环境。

```shell
# 默认情况下, 我们开启中文用户界面
python app/hydit_app.py

# 使用 Flash Attention 机制来加速
python app/hydit_app.py --infer-mode fa

# 如果 GPU 内存不足，可以禁用增强模式
# 增强功能将不可用，直到你在不使用"--no-enhance "标记的情况下重新启动应用程序为止
python app/hydit_app.py --no-enhance

# 开启英文用户界面
python app/hydit_app.py --lang en

# 启动多轮文本到图像生成的用户界面
# 如果 GPU 内存不足 32GB，请使用 '--load-4bit' 来启用 4bits 量化，这至少需要 22GB 内存
python app/multiTurnT2I_app.py
```
然后可以通过 http://0.0.0.0:443 访问示例. 需要注意的是，这里的 0.0.0.0 需要是带有您的服务器 IP的 X.X.X.X。

### 使用 🤗 Diffusers

请提前安装 PyTorch 2.0 或更高版本，以满足指定版本的 diffusers 库的需求。

安装 🤗 diffusers，确保版本至少为 0.28.1:

```shell
pip install git+https://github.com/huggingface/diffusers.git
```
或
```shell
pip install diffusers
```

您可以使用以下 Python 脚本来生成带有中文和英文提示词的图像:
```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

# 您也可以使用英文提示，因为 HunyuanDiT 支持中文和英文
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt).images[0]
```
您可以使用我们的蒸馏模型来更快地生成图像:

```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled", torch_dtype=torch.float16)
pipe.to("cuda")

# 您也可以使用英文提示，因为 HunyuanDiT 支持中文和英文
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt, num_inference_steps=25).images[0]
```
更多细节在[HunyuanDiT-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled)

**更多功能:** 对于 LoRA 和 ControlNet 等其他功能，请查看[./diffusers](diffusers)的README.

### 使用命令行

We provide several commands to quick start: 

```shell
# 提示词增强 + 文本到图像. Torch 模式
python sample_t2i.py --prompt "渔舟唱晚"

# 仅文本到图像. Torch 模式
python sample_t2i.py --prompt "渔舟唱晚" --no-enhance

# 仅文本到图像. Flash Attention 模式
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"

# 生成其他规格大小的图片.
python sample_t2i.py --prompt "渔舟唱晚" --image-size 1280 768

# 提示词增强 + 文本到图像. DialogGen 采用 4bits 量化加载，但这可能会降低性能.
python sample_t2i.py --prompt "渔舟唱晚"  --load-4bit

```

更多示例提示词在[example_prompts.txt](example_prompts.txt)

### 更多配置

为了便于使用，我们列出了一些更有用的配置：

|    参数     |  默认  |                     介绍                     |
|:---------------:|:---------:|:---------------------------------------------------:|
|   `--prompt`    |   None    |        用于生成图像的文本提示         |
| `--image-size`  | 1024 1024 |           生成图像的大小           |
|    `--seed`     |    42     |        用于生成图像的随机种子        |
| `--infer-steps` |    100    |          采样的步数           |
|  `--negative`   |     -     |      图像生成的负面提示       |
| `--infer-mode`  |   torch   |       推理模式 (torch, fa, 或 trt)        |
|   `--sampler`   |   ddpm    |    扩散采样器 (ddpm, ddim, 或 dpmms)     |
| `--no-enhance`  |   False   |        禁用提示词增强模型         |
| `--model-root`  |   ckpts   |     模型检验点的根目录     |
|  `--load-key`   |    ema    | 加载学生模型或 EMA 模型 (ema 或 module) |
|  `--load-4bit`  |   Fasle   |     加载具有 4bits 量化的 DialogGen 模型     |

### 使用 ComfyUI

我们提供了几个命令来快速入门: 

```shell
# 下载 comfyui 代码
git clone https://github.com/comfyanonymous/ComfyUI.git

# 安装 torch, torchvision, torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# 安装 Comfyui 所必需的 python package.
cd ComfyUI
pip install -r requirements.txt

# ComfyUI 已经被成功安装!

# 像之前一样下载模型权重或将现有模型文件夹链接到 ComfyUI.
python -m pip install "huggingface_hub[cli]"
mkdir models/hunyuan
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./models/hunyuan/ckpts

# 跳转至 ComfyUI 的 custom_nodes 文件夹，并将 comfyui-hydit 文件夹从 HunyuanDiT 仓库复制到此
cd custom_nodes
cp -r ${HunyuanDiT}/comfyui-hydit ./
cd comfyui-hydit

# 安装必需的 python 包
pip install -r requirements.txt

# 我们的工具已经被成功安装了!

# 跳转到 ComfyUI 主文件夹下
cd ../..
# Run the ComfyUI Lauch command
python main.py --listen --port 80

# 成功运行 ComfyUI!
```
更多细节在[./comfyui-hydit](comfyui-hydit/README.md)

### 使用 Kohya

我们提供了几个命令来使用 Kohya 快速启动 LoRA 训练和 DreamBooth 训练: 

```shell
# 下载 kohya_ss 图形用户界面
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss/

# 下载 sd-scripts 训练后端, 使用 dev 分支
git clone -b dev https://github.com/kohya-ss/sd-scripts ./sd-scripts

# 将自定义的图形用户界面代码移至 kohya_ss 图形用户界面，并替换同名文件
cp -Rf ${HunyuanDiT}/kohya_ss-hydit/* ./

# 像之前一样下载模型权重或将现有模型文件夹链接到 kohya_ss/models
python -m pip install "huggingface_hub[cli]"
# 如果要下载完整的模型，请使用以下命令
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1 --local-dir ./models/HunyuanDiT-V1.1
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-V1.2 --local-dir ./models/HunyuanDiT-V1.2
# 或者，如果您想下载经过剪枝的 fp16 模型
huggingface-cli download KBlueLeaf/HunYuanDiT-V1.1-fp16-pruned --local-dir ./models/HunyuanDiT-V1.1-fp16-pruned

# 下载模型后，您可能需要修改文件名，确保其符合 kohya 标准格式:
# 重命名 t2i/ 文件夹中的文件名，如下所示:
# HunyuanDiT-V1.2/t2i/
#  - model/                  -> denoiser/
#  - clip_text_encoder/      -> clip/
#  - mt5/                    -> mt5/
#  - sdxl-vae-fp16-fix/      -> vae/
# 此外，您可能需要将 tokenizer/* 移到 clip/ 文件夹中
mv HunyuanDiT-V1.2/t2i/model/ HunyuanDiT-V1.2/t2i/denoiser/
mv HunyuanDiT-V1.2/t2i/clip_text_encoder/ HunyuanDiT-V1.2/t2i/clip/
mv HunyuanDiT-V1.2/t2i/mt5/ HunyuanDiT-V1.2/t2i/mt5/
mv HunyuanDiT-V1.2/t2i/sdxl-vae-fp16-fix/ HunyuanDiT-V1.2/t2i/vae/
mv HunyuanDiT-V1.2/t2i/tokenizer/* HunyuanDiT-V1.2/t2i/clip/ 

# 安装必需的 python 包
conda create -n hydit-kohya python=3.10.12
conda activate hydit-kohya

# 安装必须的包, 请确保已安装 cuda 环境，且 python 版本为 3.10
# 对于 cuda 12:
pip install torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1
# 对于 cuda 11:
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 xformers==0.0.23.post1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# 为了卸载 CPU 以节省 GPU 内存，我们建议按以下步骤安装 Deepspeed:
DS_BUILD_CPU_ADAM=1 pip install deepspeed==0.14.1

# 安装其他的 python 包
pip install -r hunyuan_requirements.txt

# 运行 Kohya_ss 用户界面的启动命令
python kohya_gui.py
```
更多详细信息在[Kohya_ss](kohya_ss-hydit/README.md)的README。

## :building_construction: 适配器

### ControlNet

我们提供了 ControlNet 的训练脚本，详细请见 [./controlnet](./controlnet/README.md)

  ```shell
  # 训练canny ControlNet.
  PYTHONPATH=./ sh hydit/train_controlnet.sh
  ```
 我们为 `canny` ，`depth` 和 `pose` 提供三种类型的训练 ControlNet 权重，详细请见[链接](https://huggingface.co/Tencent-Hunyuan/HYDiT-ControlNet)
  ```shell
  cd HunyuanDiT
  # 使用 huggingface-cli 工具来下载模型
  # 我们建议使用蒸馏权重作为 ControlNet 推理的基础模型，因为我们提供的预训练权重是在它们上训练的
  huggingface-cli download Tencent-Hunyuan/HYDiT-ControlNet --local-dir ./ckpts/t2i/controlnet
  huggingface-cli download Tencent-Hunyuan/Distillation-v1.1 ./pytorch_model_distill.pt --local-dir ./ckpts/t2i/model
  
  # 快速开始
  python3 sample_controlnet.py  --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0
  ```
 
 <table>
  <tr>
    <td colspan="3" align="center">条件输入</td>
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
    <td colspan="3" align="center">ControlNet 输出</td>
  </tr>

  <tr>
    <td align="center"><img src="controlnet/asset/output/canny.jpg" alt="Image 3" width="200"/></td>
    <td align="center"><img src="controlnet/asset/output/depth.jpg" alt="Image 4" width="200"/></td>
    <td align="center"><img src="controlnet/asset/output/pose.jpg" alt="Image 5" width="200"/></td>
  </tr>
 
</table>

## :art: Hunyuan-Captioner
Hunyuan-Captioner通过保持高度的图像-文本一致性来满足文本到图像技术的需求。它可以从多个角度生成高质量的图像描述，包括对象描述、对象关系、背景信息、图像样式等。我们的代码基于[LLaVA](https://github.com/haotian-liu/LLaVA) 实现.

### 示例

<td align="center"><img src="./asset/caption_demo.jpg" alt="Image 3" width="1200"/></td>

### 教程
a. 安装依赖项
     
依赖项和安装流程与[**基础模型**](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1)基本相同.

b. 模型下载
```shell
# 使用 huggingface-cli 工具来下载模型.
huggingface-cli download Tencent-Hunyuan/HunyuanCaptioner --local-dir ./ckpts/captioner
```

### 推理

我们的模型支持三种不同的模式，包括： **直接生成中文字幕**, **基于特定知识生成中文字幕**, 和 **直接生成英文字幕**. 注入的信息可以是准确的提示，也可以是含有噪声的标签（例如，从互联网上抓取的原始描述）。该模型能够根据插入的信息和图像内容生成可靠和准确的描述。

|模式           | 提示词模版                           |介绍                           | 
| ---           | ---                                       | ---                                  |
|caption_zh     | 描述这张图片                               |中文字幕                    | 
|insert_content | 根据提示词“{}”,描述这张图片                 |带有注入知识的字幕| 
|caption_en     | Please describe the content of this image |英文字幕                    |
|               |                                           |                                      |
 

a. 中文单张图片推理

```bash
python mllm/caption_demo.py --mode "caption_zh" --image_file "mllm/images/demo1.png" --model_path "./ckpts/captioner"
```

b. 在字幕中注入特定知识

```bash
python mllm/caption_demo.py --mode "insert_content" --content "宫保鸡丁" --image_file "mllm/images/demo2.png" --model_path "./ckpts/captioner"
```

c. 英文单张图片推理

```bash
python mllm/caption_demo.py --mode "caption_en" --image_file "mllm/images/demo3.png" --model_path "./ckpts/captioner"
```

d. 中文多图片推理

```bash
### 将多张图片转化为 csv 文件. 
python mllm/make_csv.py --img_dir "mllm/images" --input_file "mllm/images/demo.csv"

### 多张图片推理
python mllm/caption_demo.py --mode "caption_zh" --input_file "mllm/images/demo.csv" --output_file "mllm/images/demo_res.csv" --model_path "./ckpts/captioner"
```

(可选) 要将输出 csv 文件转换为 Arrow 格式，请参阅[Data Preparation #3](#data-preparation).


### Gradio 
要在本地启动 Gradio 示例, 请逐个运行以下命令。有关更详细的说明，请参阅[LLaVA](https://github.com/haotian-liu/LLaVA). 
```bash
cd mllm
python -m llava.serve.controller --host 0.0.0.0 --port 10000

python -m llava.serve.gradio_web_server --controller http://0.0.0.0:10000 --model-list-mode reload --port 443

python -m llava.serve.model_worker --host 0.0.0.0 --controller http://0.0.0.0:10000 --port 40000 --worker http://0.0.0.0:40000 --model-path "../ckpts/captioner" --model-name LlavaMistral
```
然后可以通过 http://0.0.0.0:443 访问示例. 需要注意的是，这里的 0.0.0.0 需要是带有您的服务器 IP的 X.X.X.X.

## 🚀 加速（适用于 Linux）

- 我们提供 HunyuanDiT 的 TensorRT 版本用于推理加速（比 flash attention 更快）。详细请见[Tencent-Hunyuan/TensorRT-libs](https://huggingface.co/Tencent-Hunyuan/TensorRT-libs)

- 我们提供 HunyuanDiT 的蒸馏版本，用于推理加速。详细请见 [Tencent-Hunyuan/Distillation](https://huggingface.co/Tencent-Hunyuan/Distillation)

## 🔗 BibTeX
如果您发现[Hunyuan-DiT](https://arxiv.org/abs/2405.08748) 或 [DialogGen](https://arxiv.org/abs/2403.08857)对您的研究和应用有用，请使用此 BibTeX 进行引用:

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

## Star的历史记录

<a href="https://star-history.com/#Tencent/HunyuanDiT&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date" />
 </picture>
</a>

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

## Contents
- [Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](#hunyuan-dit--a-powerful-multi-resolution-diffusion-transformer-with-fine-grained-chinese-understanding)
  - [📜 模型配置需求](#-模型配置需求)
  - [🛠️ 依赖管理和安装指南](#️-依赖管理和安装指南)
    - [Linux环境下的安装指南](#-Linux环境下的安装指南)
  - [🧱 下载预训练模型](#-下载预训练模型)
        - [1. 使用 HF-Mirror 镜像](#1-使用-HF-Mirror-镜像)
        - [2. 恢复下载 ](#2-恢复下载)
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
    - [Using Previous versions](#using-previous-versions)
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

---

## 📜 模型配置需求

这个项目仓库由 DialoGen（一个提示增强模型）和 Hunyuan-DiT（‌一个文生图模型）‌。‌

下面表格展示了运行上述模型所需要的配置需求（批量大小为1）：

|          模型          | 以4位量化的方式加载DialogGen模型 | GPU 峰值内存 |       GPU       |
|:-----------------------:|:-----------------------:|:---------------:|:---------------:|
| DialogGen + Hunyuan-DiT |            ✘            |       32G       |      A100       |
| DialogGen + Hunyuan-DiT |            ✔            |       22G       |      A100       |
|       Hunyuan-DiT       |            -            |       11G       |      A100       |
|       Hunyuan-DiT       |            -            |       14G       | RTX3090/RTX4090 |

* 一个支持 CUDA 的 NVIDIA GPU 上运行。 
  * 我们已经在 V100 和 A100 这两款 GPU 上进行测试。‌
  * **最低配置**: 至少需要 11GB 的 GPU 内存。‌
  * **推荐配置**: 为了获得更好的生成质量，‌我们推荐使用 32GB 内存的 GPU。‌
* 已测试操作系统：‌Linux。‌

## 🛠️ 依赖管理和安装指南

克隆仓库:
```shell
git clone https://github.com/tencent/HunyuanDiT
cd HunyuanDiT
```

### Linux环境下的安装指南

我们提供了一个 `environment.yml`文件，‌用于设置Conda环境。‌
‌Conda的安装说明可在[此处](https://docs.anaconda.com/free/miniconda/index.html)获得.

我们推荐使用CUDA 11.7和12.0+版本。

```shell
# 1. 创建 Conda 环境。
conda env create -f environment.yml

# 2. 激活创建的 Conda 环境。
conda activate HunyuanDiT

# 3. 安装 pip 依赖 。
python -m pip install -r requirements.txt

# 安装 flash attention v2（‌需要 CUDA 11.6 或更高版本）‌用来加速。
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

另外，‌也可以使用 Docker 来配置环境。
```shell 
# 1. 通过以下链接下载 Docker 镜像的压缩文件。
# 对于 CUDA 12 的用户
wget https://dit.hunyuan.tencent.com/download/HunyuanDiT/hunyuan_dit_cu12.tar
# 对于 CUDA 11 的用户
wget https://dit.hunyuan.tencent.com/download/HunyuanDiT/hunyuan_dit_cu11.tar

# 2. 导入 Docker 压缩文件并查看镜像信息。
# 对于 CUDA 12 的用户
docker load -i hunyuan_dit_cu12.tar
# 对于 CUDA 11 的用户
docker load -i hunyuan_dit_cu11.tar  

docker image ls

# 3. 基于当前镜像运行一个新的容器。
docker run -dit --gpus all --init --net=host --uts=host --ipc=host --name hunyuandit --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged  docker_image_tag
```

## 🧱 下载预训练模型
要下载预训练模型，‌首先需要安装 huggingface-cli 工具‌。‌（ 详细的安装指南可以在[这里](https://huggingface.co/docs/huggingface_hub/guides/cli)找到。）‌

```shell
python -m pip install "huggingface_hub[cli]"
```

安装完成后，‌使用以下命令下载模型 ：

```shell
# 创建一个名为 'ckpts' 的目录，‌用于保存模型，‌这是运行演示所必需的步骤。‌
mkdir ckpts
# 使用 huggingface-cli 工具下载模型。‌
# 下载时间可能因网络条件而异，‌从 10 分钟到 1 小时不等。‌
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.2 --local-dir ./ckpts
```

<details>
<summary>💡使用 huggingface-cli 时的网络问题解决方案 </summary>

##### 1. 使用 HF-Mirror 镜像

如果在中国遇到下载速度慢的问题，可以尝试使用镜像来加快下载速度。例如，

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.2 --local-dir ./ckpts
```

##### 2. 恢复下载 

`huggingface-cli` 支持恢复下载。‌如果下载过程中断，‌只需重新运行下载命令即可继续下载。

注意：如果在下载过程中出现类似 `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` 的错误，可以忽略该错误并重新运行下载命令。

</details>

---

所有模型将自动下载。‌如需更多关于模型的信息，‌请访问[这个](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) Hugging Face 仓库。 

|       模型       | 参数量 |                                        从 Huggingface 下载模型的链接                                        |                                  从 Tencent Cloud 下载模型链接                                |
|:-----------------:|:-------:|:------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
|        mT5        |  1.6B   |               [mT5](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/mt5)               |               [mT5](https://dit.hunyuan.tencent.com/download/HunyuanDiT/mt5.zip)               |
|       CLIP        |  350M   |       [CLIP](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder)        |       [CLIP](https://dit.hunyuan.tencent.com/download/HunyuanDiT/clip_text_encoder.zip)        |
|     Tokenizer     |  -      |         [Tokenizer](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/tokenizer)         |         [Tokenizer](https://dit.hunyuan.tencent.com/download/HunyuanDiT/tokenizer.zip)         |
|     DialogGen     |  7.0B   |           [DialogGen](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/dialoggen)           |         [DialogGen](https://dit.hunyuan.tencent.com/download/HunyuanDiT/dialoggen.zip)         |
| sdxl-vae-fp16-fix |   83M   | [sdxl-vae-fp16-fix](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/sdxl-vae-fp16-fix) | [sdxl-vae-fp16-fix](https://dit.hunyuan.tencent.com/download/HunyuanDiT/sdxl-vae-fp16-fix.zip) |
| Hunyuan-DiT-v1.0  |  1.5B   |          [Hunyuan-DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/model)          |       [Hunyuan-DiT-v1.0](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model.zip)        |
| Hunyuan-DiT-v1.1  |  1.5B   |     [Hunyuan-DiT-v1.1](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1/tree/main/t2i/model)     |     [Hunyuan-DiT-v1.1](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model-v1_1.zip)     |
| Hunyuan-DiT-v1.2  |  1.5B   |     [Hunyuan-DiT-v1.2](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2/tree/main/t2i/model)     |     [Hunyuan-DiT-v1.2](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model-v1_2.zip)     |
|     Data demo     |  -      |                                                   -                                                    |         [Data demo](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip)         |

## :truck: Training

### Data Preparation

  Refer to the commands below to prepare the training data. 
  
  1. Install dependencies
  
      We offer an efficient data management library, named IndexKits, supporting the management of reading hundreds of millions of data during training, see more in [docs](./IndexKits/README.md).
      ```shell
      # 1 Install dependencies
      cd HunyuanDiT
      pip install -e ./IndexKits
     ```
  2. Data download 
  
     Feel free to download the [data demo](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip).
     ```shell
     # 2 Data download
     wget -O ./dataset/data_demo.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip
     unzip ./dataset/data_demo.zip -d ./dataset
     mkdir ./dataset/porcelain/arrows ./dataset/porcelain/jsons
     ```
  3. Data conversion 
  
     Create a CSV file for training data with the fields listed in the table below.
    
     |    Fields       | Required  |  Description     |   Example   |
     |:---------------:| :------:  |:----------------:|:-----------:|
     |   `image_path`  | Required  |  image path               |     `./dataset/porcelain/images/0.png`        | 
     |   `text_zh`     | Required  |    text               |  青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 | 
     |   `md5`         | Optional  |    image md5 (Message Digest Algorithm 5)  |    `d41d8cd98f00b204e9800998ecf8427e`         | 
     |   `width`       | Optional  |    image width    |     `1024 `       | 
     |   `height`      | Optional  |    image height   |    ` 1024 `       | 
     
     > ⚠️ Optional fields like MD5, width, and height can be omitted. If omitted, the script below will automatically calculate them. This process can be time-consuming when dealing with large-scale training data.
  
     We utilize [Arrow](https://github.com/apache/arrow) for training data format, offering a standard and efficient in-memory data representation. A conversion script is provided to transform CSV files into Arrow format.
     ```shell  
     # 3 Data conversion 
     python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows 1
     ```
  
  4. Data Selection and Configuration File Creation 
     
      We configure the training data through YAML files. In these files, you can set up standard data processing strategies for filtering, copying, deduplicating, and more regarding the training data. For more details, see [./IndexKits](IndexKits/docs/MakeDataset.md).
  
      For a sample file, please refer to [file](./dataset/yamls/porcelain.yaml). For a full parameter configuration file, see [file](./IndexKits/docs/MakeDataset.md).
  
     
  5. Create training data index file using YAML file.
    
     ```shell
      # Single Resolution Data Preparation
      idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json
   
      # Multi Resolution Data Preparation     
      idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json
      ```
   
  The directory structure for `porcelain` dataset is:

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

### Full-parameter Training
  
  **Requirement:** 
  1. The minimum requriment is a single GPU with at least 20GB memory, but we recommend to use a GPU with about 30 GB memory to avoid host memory offloading. 
  2. Additionally, we encourage users to leverage the multiple GPUs across different nodes to speed up training on large datasets. 
  
  **Notice:**
  1. Personal users can also use the light-weight Kohya to finetune the model with about 16 GB memory. Currently, we are trying to further reduce the memory usage of our industry-level framework for personal users. 
  2. If you have enough GPU memory, please try to remove  `--cpu-offloading` or `--gradient-checkpointing` for less time costs.

  Specifically for distributed training, you have the flexibility to control **single-node** / **multi-node** training by adjusting parameters such as `--hostfile` and `--master_addr`. For more details, see [link](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

  ```shell
  # Single Resolution Training
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain.json
  
  # Multi Resolution Training
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain_mt.json --multireso --reso-step 64
  
  # Training with old version of HunyuanDiT (<= v1.1)
  PYTHONPATH=./ sh hydit/train_v1.1.sh --index-file dataset/porcelain/jsons/porcelain.json
  ```

  After checkpoints are saved, you can use the following command to evaluate the model.
  ```shell
  # Inference
    #   You should replace the 'log_EXP/xxx/checkpoints/final.pt' with your actual path.
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只可爱的哈士奇" --no-enhance --dit-weight log_EXP/xxx/checkpoints/final.pt --load-key module
  
  # Old version of HunyuanDiT (<= v1.1)
  #   You should replace the 'log_EXP/xxx/checkpoints/final.pt' with your actual path.
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只可爱的哈士奇" --model-root ./HunyuanDiT-v1.1 --use-style-cond --size-cond 1024 1024 --beta-end 0.03 --no-enhance --dit-weight log_EXP/xxx/checkpoints/final.pt --load-key module
  ```

### LoRA



We provide training and inference scripts for LoRA, detailed in the [./lora](./lora/README.md). 

  ```shell
  # Training for porcelain LoRA.
  PYTHONPATH=./ sh lora/train_lora.sh --index-file dataset/porcelain/jsons/porcelain.json

  # Inference using trained LORA weights.
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只小狗"  --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt
  ```
 We offer two types of trained LoRA weights for `porcelain` and `jade`, see details at [links](https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA)
  ```shell
  cd HunyuanDiT
  # Use the huggingface-cli tool to download the model.
  huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora
  
  # Quick start
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只猫在追蝴蝶"  --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain
  ```
 <table>
  <tr>
    <td colspan="4" align="center">Examples of training data</td>
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
    <td colspan="4" align="center">Examples of inference results</td>
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
model_id=Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled
prompt=一个宇航员在骑马
infer_steps=50
guidance_scale=6
python3 lite/inference.py ${model_id} ${prompt} ${infer_steps} ${guidance_scale}
```

More details can be found in [./lite](lite/README.md).


### Using Gradio

Make sure the conda environment is activated before running the following command.

```shell
# By default, we start a Chinese UI. Using Flash Attention for acceleration.
python app/hydit_app.py --infer-mode fa

# You can disable the enhancement model if the GPU memory is insufficient.
# The enhancement will be unavailable until you restart the app without the `--no-enhance` flag. 
python app/hydit_app.py --no-enhance --infer-mode fa

# Start with English UI
python app/hydit_app.py --lang en --infer-mode fa

# Start a multi-turn T2I generation UI. 
# If your GPU memory is less than 32GB, use '--load-4bit' to enable 4-bit quantization, which requires at least 22GB of memory.
python app/multiTurnT2I_app.py --infer-mode fa
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

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16)
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

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", torch_dtype=torch.float16)
pipe.to("cuda")

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt, num_inference_steps=25).images[0]
```
More details can be found in [HunyuanDiT-v1.2-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled)

**More functions:** For other functions like LoRA and ControlNet, please have a look at the README of [./diffusers](diffusers).

### Using Command Line

We provide several commands to quick start: 

```shell
# Only Text-to-Image. Flash Attention mode
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --no-enhance

# Generate an image with other image sizes.
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --image-size 1280 768

# Prompt Enhancement + Text-to-Image. DialogGen loads with 4-bit quantization, but it may loss performance.
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"  --load-4bit

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

- Support two workflows: Standard ComfyUI and Diffusers Wrapper, with the former being recommended.
- Support HunyuanDiT-v1.1 and v1.2.
- Support module, lora and clip lora models trained by Kohya.
- Support module, lora models trained by HunyunDiT official training scripts.
- ControlNet is coming soon.

![Workflow](comfyui-hydit/img/workflow_v1.2_lora.png)
More details can be found in [./comfyui-hydit](comfyui-hydit/README.md)

### Using Kohya

We support custom codes for kohya_ss GUI, and sd-scripts training codes for HunyuanDiT.
![dreambooth](kohya_ss-hydit/img/dreambooth.png)
More details can be found in [./kohya_ss-hydit](kohya_ss-hydit/README.md)

### Using Previous versions

* **Hunyuan-DiT <= v1.1**

```shell
# ============================== v1.1 ==============================
# Download the model
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1 --local-dir ./HunyuanDiT-v1.1
# Inference with the model
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --model-root ./HunyuanDiT-v1.1 --use-style-cond --size-cond 1024 1024 --beta-end 0.03

# ============================== v1.0 ==============================
# Download the model
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./HunyuanDiT-v1.0
# Inference with the model
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --model-root ./HunyuanDiT-v1.0 --use-style-cond --size-cond 1024 1024 --beta-end 0.03
```

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
  huggingface-cli download Tencent-Hunyuan/HYDiT-ControlNet-v1.2 --local-dir ./ckpts/t2i/controlnet
  huggingface-cli download Tencent-Hunyuan/Distillation-v1.2 ./pytorch_model_distill.pt --local-dir ./ckpts/t2i/model
  
  # Quick start
  python3 sample_controlnet.py --infer-mode fa --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0
  
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
    <td align="center">在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足。照片采用特写、平视和居中构图的方式，呈现出写实的效果<br>（In the dense forest, a black and white panda sits quietly among the green trees and red flowers, surrounded by mountains and oceans. The background is a daytime forest with ample light. The photo uses a close-up, eye-level, and centered composition to create a realistic effect.） </td>
    <td align="center">在白天的森林中，一位穿着绿色上衣的亚洲女性站在大象旁边。照片采用了中景、平视和居中构图的方式，呈现出写实的效果。这张照片蕴含了人物摄影文化，并展现了宁静的氛围<br>（In the daytime forest, an Asian woman wearing a green shirt stands beside an elephant. The photo uses a medium shot, eye-level, and centered composition to create a realistic effect. This picture embodies the character photography culture and conveys a serene atmosphere.） </td>
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
     
The dependencies and installation are basically the same as the [**base model**](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2).

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

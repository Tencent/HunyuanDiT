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

此仓库包含了我们探索 Hunyuan-DiT 的论文所需的 PyTorch 模型定义、‌预训练权重以及推理/ 采样代码。‌您可以在我们的[项目页面](https://dit.hunyuan.tencent.com/)上找到更多可视化内容。‌

> [**Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding**](https://arxiv.org/abs/2405.08748) <br>

> [**DialogGen: Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation**](https://arxiv.org/abs/2403.08857) <br>

## Contents
- [Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](#hunyuan-dit--a-powerful-multi-resolution-diffusion-transformer-with-fine-grained-chinese-understanding)
  - [📜 模型配置需求](#-模型配置需求)
  - [🛠️ 依赖管理和安装指南](#️-依赖管理和安装指南)
    - [Linux 环境下的安装指南](#linux-环境下的安装指南)
  - [🧱 下载预训练模型](#-下载预训练模型)
        - [1. 使用 HF-Mirror 镜像](#1-使用-hf-mirror-镜像)
        - [2. 恢复下载 ](#2-恢复下载)
  - [:truck:训练](#truck-训练)
    - [数据准备](#数据准备)
    - [全参数训练](#全参数训练)
    - [LoRA](#lora)
  - [🔑 推理](#-推理)
    - [6GB GPU VRAM 推理](#6gb-gpu-vram-推理)
    - [使用 Gradio](#使用-gradio)
    - [使用 🤗 Diffusers](#使用--diffusers)
    - [使用命令行](#使用命令行)
    - [更多配置选项‌](#更多配置选项)
    - [使用 ComfyUI](#使用-comfyui)
    - [使用 Kohya](#使用-kohya)
    - [使用早期版本](#使用早期版本)
  - [:building\_construction: 适配器](#building_construction-适配器)
    - [ControlNet](#controlnet)
  - [:art: Hunyuan-Captioner](#art-hunyuan-captioner)
    - [示例](#示例)
    - [使用说明](#使用说明)
    - [推理](#推理)
    - [Gradio](#gradio)
  - [🚀 加速 (适用于 Linux)](#-加速-适用于-linux)
  - [🔗 BibTeX](#-bibtex)
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

### Linux 环境下的安装指南

我们提供了一个 `environment.yml`文件，‌用于设置 Conda 环境。‌
‌Conda 的安装说明可在[此处](https://docs.anaconda.com/free/miniconda/index.html)获得.

我们推荐使用 CUDA 11.7 和 12.0+ 版本。

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

所有模型将自动下载。‌如需更多关于模型的信息，‌请访问[这个 Hugging Face ](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) 仓库。 

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

## :truck: 训练

### 数据准备

  请参考以下命令来准备训练数据。
  
  1. 安装依赖
  
      我们提供了一个高效的数据管理库，‌名为 IndexKits，‌它支持在训练过程中读取数亿条数据。‌更多信息，‌请参见 [docs](./IndexKits/README.md)。
      ```shell
      # 1 安装依赖
      cd HunyuanDiT
      pip install -e ./IndexKits
     ```
  2. 下载数据
  
     您可以自由下载[数据示例](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip)。
     ```shell
     # 2 下载数据
     wget -O ./dataset/data_demo.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip
     unzip ./dataset/data_demo.zip -d ./dataset
     mkdir ./dataset/porcelain/arrows ./dataset/porcelain/jsons
     ```
  3. 数据转换 
  
     请根据下表所列字段，‌创建一个用于训练数据的 CSV 文件。‌
    
     |    字段名称      |   是否是必选项   |    描述     |   示例   |
     |:---------------:| :------------------:  |:-----------:|:-----------:|
     |   `image_path`  |    必选     |  图像路径               |     `./dataset/porcelain/images/0.png`        | 
     |   `text_zh`     |    必项     |    文本描述              |  青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 | 
     |   `md5`         |    可选     |    图像MD5 (Message Digest Algorithm 5)  |    `d41d8cd98f00b204e9800998ecf8427e`         | 
     |   `width`       |    可选     |    图像宽度    |     `1024 `       | 
     |   `height`      |    可选     |    图像高度   |    ` 1024 `       | 
     
     > ⚠️ 注意：‌MD5、‌宽度和高度等可选字段可以省略。‌如果省略，‌下面的脚本将自动计算它们。‌但在处理大规模训练数据时，‌这个过程可能会很耗时。‌
  
     我们采用 [Arrow](https://github.com/apache/arrow) 作为训练数据的格式，‌它提供了一种标准和高效的内存数据表示方法。‌为了方便用户，‌我们提供了一个转换脚本，‌可以将 CSV 文件转换为 Arrow 格式。
     ```shell  
     # 3 数据转换
     python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows 1
     ```
  
  4. 数据选择和配置文件创建 
     
      我们通过YAML文件来配置训练数据。‌在这些文件中，‌您可以设置关于训练数据的标准数据处理策略，‌如过滤、‌复制、‌去重等。‌更多详细信息，‌请参见 [./IndexKits](IndexKits/docs/MakeDataset.md)。
  
      请参阅[示例文件](./dataset/yamls/porcelain.yaml)。如果您需要查看完整的参数配置文件，‌请参阅[文件](./IndexKits/docs/MakeDataset.md)。
  
     
  5. 使用YAML文件生成训练数据索引文件。‌
    
     ```shell
      # 单分辨率数据准备
      idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json
   
      # 多分辨率数据准备    
      idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json
      ```
   
  `porcelain` 数据集的目录结构如下:

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
  
  **配置需求：** 
  1. 至少需要一块 20GB 内存的 GPU ，‌但我们更推荐使用约 30GB 内存的 GPU 进行训练以避免主机内存分流。 
  2. 此外，‌我们也鼓励用户利用不同节点上的多块 GPU‌ 来加速大数据集的训练。
  
  **注意事项:**
  1. 个人用户也可以使用轻量级的 Kohya 进行模型微调 ，需要大约 16GB 的内存。‌目前我们正致力于优化工业级框架进一步降低内存使用量，‌以更好地适应个人用户的需求。
  2. 如果GPU内存足够，‌请尝试移除  `--cpu-offloading` 或 `--gradient-checkpointing` 以减少时间成本。

 对于分布式训练，‌您可以通过调整 `--hostfile` 和 `--master_addr` 等参数来灵活地控制使用 **单节点** 或者 **多节点** 进行训练. 如需更多详情，‌请参阅[链接](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)。

  ```shell
  # 单分辨率训练
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain.json
  
  # 多分辨率训练
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain_mt.json --multireso --reso-step 64
  
  # 使用旧版本的HunyuanDiT（<= v1.1）训练 
  PYTHONPATH=./ sh hydit/train_v1.1.sh --index-file dataset/porcelain/jsons/porcelain.json
  ```

  保存检查点后，‌您可以使用以下命令来评估模型。‌
  ```shell
  # 推理
    #  你需要将'log_EXP/xxx/checkpoints/final.pt'替换为你的实际路径。‌
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只可爱的哈士奇" --no-enhance --dit-weight log_EXP/xxx/checkpoints/final.pt --load-key module
  
  # 旧版本的HunyuanDiT（<= v1.1）
  #   您应该将 'log_EXP/xxx/checkpoints/final.pt'  替换为您实际的路径。 
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只可爱的哈士奇" --model-root ./HunyuanDiT-v1.1 --use-style-cond --size-cond 1024 1024 --beta-end 0.03 --no-enhance --dit-weight log_EXP/xxx/checkpoints/final.pt --load-key module
  ```

### LoRA



我们提供了LoRA的训练和推理脚本，详细信息请参阅 [./lora](./lora/README.md). 

  ```shell
  # 针对瓷器LoRA的训练。
  PYTHONPATH=./ sh lora/train_lora.sh --index-file dataset/porcelain/jsons/porcelain.json

  # 使用训练好的LoRA权重进行推理。 
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只小狗"  --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt
  ```
 我们为 `瓷器` 和 `玉器` 提供了两种类型的训练好的LoRA权重，详情请访问[链接](https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA)
  ```shell
  cd HunyuanDiT
  # 使用 huggingface-cli 工具下载模型。
  huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora
  
  # 快速启动。
  python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只猫在追蝴蝶"  --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain
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
现在，基于 [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuandit)，可以在不到 6GB 的 GPU VRAM 中运行 HunyuanDiT。这里我们为您提供快速开始的指导和演示。

> 6GB 版本支持 Nvidia Ampere 架构系列显卡，如 RTX 3070/ 3080/ 4080/ 4090、A100 等。

您唯一需要做的就是安装以下库：

```bash
pip install -U bitsandbytes
pip install git+https://github.com/huggingface/diffusers
pip install torch==2.0.0
```

安装完成后，‌您就可以直接在 6GB GPU VRAM 下享受 HunyuanDiT 的文生图旅程了！‌

这里有一个演示供您参考。

```bash
cd HunyuanDiT

# 快速开始
model_id=Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled
prompt=一个宇航员在骑马
infer_steps=50
guidance_scale=6
python3 lite/inference.py ${model_id} ${prompt} ${infer_steps} ${guidance_scale}
```

更多详细信息请参阅 [./lite](lite/README.md)。


### 使用 Gradio

在运行以下命令之前，请确保已激活 conda 环境。

```shell
# 默认启动一个中文界面。使用 Flash Attention 进行加速。 
python app/hydit_app.py --infer-mode fa

# 如果 GPU 内存不足，您可以禁用增强模型。 
# 该增强功能将不可用，直到您在不带 `--no-enhance` 标志的情况下重新启动应用程序，增强功能才会重新启用。 
python app/hydit_app.py --no-enhance --infer-mode fa

# 启动英文用户界面
python app/hydit_app.py --lang en --infer-mode fa

# 启动多轮 T2I 生成界面。 
# 如果您的 GPU 内存小于 32GB，请使用 '--load-4bit' 启用 4 位量化，这至少需要 22GB 的内存。
python app/multiTurnT2I_app.py --infer-mode fa
```
然后可以通过 http://0.0.0.0:443 访问演示。需要注意的是，这里的 0.0.0.0 需要替换为您的服务器IP地址。

### 使用 🤗 Diffusers

请提前安装 PyTorch 版本 2.0 或更高版本，以满足指定版本的 diffusers 库的要求。

安装 🤗 diffusers，确保版本至少为 0.28.1：

```shell
pip install git+https://github.com/huggingface/diffusers.git
```
或者
```shell
pip install diffusers
```

您可以使用以下 Python 脚本生成包含中英文提示的图像：
```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

# 您也可以使用英文提示，因为 HunyuanDiT 支持中英文 
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt).images[0]
```
您可以使用我们的蒸馏模型来更快地生成图像：

```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", torch_dtype=torch.float16)
pipe.to("cuda")

# 您也可以使用英文提示，因为 HunyuanDiT 支持中英文 
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt, num_inference_steps=25).images[0]
```
更多详情可以参阅 [HunyuanDiT-v1.2-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled)。

**更多功能：** 有关其他功能，如 LoRA 和 ControlNet，请查看 [./diffusers](diffusers) 的 README。

### 使用命令行

我们提供了几个命令来快速启动： 

```shell
# 仅文本到图像。Flash Attention 加速模式。 
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --no-enhance

# 生成不同尺寸的图像。
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --image-size 1280 768

# 提示增强 + 文本到图像。DialogGen 以 4 位量化加载，但可能会损失性能。
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"  --load-4bit

```

更多示例提示可以在 [example_prompts.txt](example_prompts.txt) 中找到。

### 更多配置选项

我们列出了更多有用的配置选项以方便使用：

|    参数         |    默认值  |                         描述                        |
|:---------------:|:---------:|:---------------------------------------------------:|
|   `--prompt`    |   None    |                   用于图像生成的文本提示              |
| `--image-size`  | 1024 1024 |                     生成图像的尺寸                   |
|    `--seed`     |    42     |                    图像生成的随机种子                |
| `--infer-steps` |    100    |                       采样步数                      |
|  `--negative`   |     -     |                   图像生成的负面提示                 |
| `--infer-mode`  |   torch   |                 推理模式（‌torch, fa, 或 trt）‌        |
|   `--sampler`   |   ddpm    |              扩散采样器（‌ddpm, ddim, 或 dpmms）‌      |
| `--no-enhance`  |   False   |                     禁用提示增强模型                 |
| `--model-root`  |   ckpts   |                  模型检查点的根目录                  |
|  `--load-key`   |    ema    |        加载学生模型或者 EMA 模型(ema 或 module)      |
|  `--load-4bit`  |   Fasle   |             加载使用4位量化的DialogGen模型           |

### 使用 ComfyUI

- 支持两种工作流程：‌标准ComfyUI和Diffusers Wrapper，‌推荐使用前者。‌
- 支持 HunyuanDiT-v1.1 和 v1.2 版本。‌
- 支持 Kohya 训练的 module、‌lora 和 clip lora 模型。‌
- 支持 HunyunDiT 官方训练脚本训练的 module、‌lora 模型。‌
- 即将支持 ControlNet。‌

![Workflow](comfyui-hydit/img/workflow_v1.2_lora.png)
更多详情，‌请参阅 [./comfyui-hydit](comfyui-hydit/README.md)。

### 使用 Kohya

我们支持为 kohya_ss GUI 自定义的代码，以及用于 HunyuanDiT 的 sd-scripts 训练代码。
![dreambooth](kohya_ss-hydit/img/dreambooth.png)
更多详情请参阅 [./kohya_ss-hydit](kohya_ss-hydit/README.md)

### 使用早期版本

* **Hunyuan-DiT <= v1.1**

```shell
# ============================== v1.1 ==============================
# 下载模型 
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1 --local-dir ./HunyuanDiT-v1.1
# 使用模型进行推理 
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --model-root ./HunyuanDiT-v1.1 --use-style-cond --size-cond 1024 1024 --beta-end 0.03

# ============================== v1.0 ==============================
# 下载模型 
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./HunyuanDiT-v1.0
# 使用模型进行推理 
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --model-root ./HunyuanDiT-v1.0 --use-style-cond --size-cond 1024 1024 --beta-end 0.03
```

## :building_construction: 适配器

### ControlNet

提供了ControlNet的训练脚本，详情见 [./controlnet](./controlnet/README.md)。

  ```shell
  # 训练 canny ControlNet.
  PYTHONPATH=./ sh hydit/train_controlnet.sh
  ```
 我们为`canny` `depth` 和 `pose`三种类型提供了训练好的 ControlNet 权重，详细信息请访问[链接](https://huggingface.co/Tencent-Hunyuan/HYDiT-ControlNet。
  ```shell
  cd HunyuanDiT
  # 使用huggingface-cli工具下载模型。
  # 我们建议使用蒸馏权重作为ControlNet推理的基础模型，‌因为我们提供的预训练权重是在这些蒸馏权重上训练得到的。‌
  huggingface-cli download Tencent-Hunyuan/HYDiT-ControlNet-v1.2 --local-dir ./ckpts/t2i/controlnet
  huggingface-cli download Tencent-Hunyuan/Distillation-v1.2 ./pytorch_model_distill.pt --local-dir ./ckpts/t2i/model
  
  # 快速开始
  python3 sample_controlnet.py --infer-mode fa --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0
  
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
    <td align="center">在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足。照片采用特写、平视和居中构图的方式，呈现出写实的效果<br>（In the dense forest, a black and white panda sits quietly among the green trees and red flowers, surrounded by mountains and oceans. The background is a daytime forest with ample light. The photo uses a close-up, eye-level, and centered composition to create a realistic effect.） </td>
    <td align="center">在白天的森林中，一位穿着绿色上衣的亚洲女性站在大象旁边。照片采用了中景、平视和居中构图的方式，呈现出写实的效果。这张照片蕴含了人物摄影文化，并展现了宁静的氛围<br>（In the daytime forest, an Asian woman wearing a green shirt stands beside an elephant. The photo uses a medium shot, eye-level, and centered composition to create a realistic effect. This picture embodies the character photography culture and conveys a serene atmosphere.） </td>
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
Hunyuan-Captioner 满足了文生图技术的需求，‌保持了高度的图文一致性。‌它能从物体描述、‌物体关系、‌背景信息、‌图像风格等多个角度生成高质量的图像描述。‌我们的代码基于 [LLaVA](https://github.com/haotian-liu/LLaVA) 实现。

### 示例

<td align="center"><img src="./asset/caption_demo.jpg" alt="Image 3" width="1200"/></td>

### 使用说明
a. 安装依赖项
     
依赖项和安装方法与[**基础模型**](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2)基本相同。‌ 

b. 模型下载
```shell
# 使用huggingface-cli工具下载模型。‌
huggingface-cli download Tencent-Hunyuan/HunyuanCaptioner --local-dir ./ckpts/captioner
```

### 推理

我们的模型支持三种不同的模式，包括：**直接生成中文标题**、**基于特定知识生成中文标题**和**直接生成英文标题**。输入的信息可以是准确的线索或噪声标签（例如，从互联网上抓取的原始描述）。该模型能够根据插入的信息和图像内容生成可靠且准确的描述。

|模式           | 提示模板                                   |描述                           | 
| ---           | ---                                       | ---                           |
|caption_zh     | 描述这张图片                               |中文标题                        | 
|insert_content | 根据提示词“{}”,描述这张图片                 |基于特定知识生成的标题            | 
|caption_en     | Please describe the content of this image |英文标题                        |
|               |                                           |                                |
 

a. 单张图片进行中文推理

```bash
python mllm/caption_demo.py --mode "caption_zh" --image_file "mllm/images/demo1.png" --model_path "./ckpts/captioner"
```

b. 标题插入特定知识进行推理

```bash
python mllm/caption_demo.py --mode "insert_content" --content "宫保鸡丁" --image_file "mllm/images/demo2.png" --model_path "./ckpts/captioner"
```

c. 单张图片进行英文推理

```bash
python mllm/caption_demo.py --mode "caption_en" --image_file "mllm/images/demo3.png" --model_path "./ckpts/captioner"
```

d. 多张图片进行中文推理

```bash
### 将多张图片转换为 csv 文件。‌ 
python mllm/make_csv.py --img_dir "mllm/images" --input_file "mllm/images/demo.csv"

### 多张图片推理
python mllm/caption_demo.py --mode "caption_zh" --input_file "mllm/images/demo.csv" --output_file "mllm/images/demo_res.csv" --model_path "./ckpts/captioner"
```

(可选)将输出的 csv 文件转换为 Arrow 格式，‌具体使用说明请参考‌ [数据准备 #3](#数据准备) 。 


### Gradio 
要在本地启动一个 Gradio 演示，请依次运行以下命令。有关更详细的说明，请参阅 [LLaVA](https://github.com/haotian-liu/LLaVA). 
```bash
cd mllm
python -m llava.serve.controller --host 0.0.0.0 --port 10000

python -m llava.serve.gradio_web_server --controller http://0.0.0.0:10000 --model-list-mode reload --port 443

python -m llava.serve.model_worker --host 0.0.0.0 --controller http://0.0.0.0:10000 --port 40000 --worker http://0.0.0.0:40000 --model-path "../ckpts/captioner" --model-name LlavaMistral
```
然后可以通过 http://0.0.0.0:443 访问演示。需要注意的是，这里的 0.0.0.0 需要替换为您的服务器 IP 地址。

## 🚀 加速 (适用于 Linux)

- 我们提供了 HunyuanDiT 的 TensorRT 版本，用于推理加速（比 flash attention 更快）。
更多细节请参阅 [Tencent-Hunyuan/TensorRT-libs](https://huggingface.co/Tencent-Hunyuan/TensorRT-libs) 。

- 我们也提供了蒸馏版本的 HunyuanDiT 用于推理加速。 
更多细节请参阅 [Tencent-Hunyuan/Distillation](https://huggingface.co/Tencent-Hunyuan/Distillation) 。

## 🔗 BibTeX
如果你觉得 [Hunyuan-DiT](https://arxiv.org/abs/2405.08748) 或者  [DialogGen](https://arxiv.org/abs/2403.08857) 对你的研究和应用有帮助，‌请使用以下BibTeX进行引用。‌

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

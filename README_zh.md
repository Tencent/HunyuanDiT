## 📜 配置需求

本仓库包含 DialogGen（提示增强模型） 和 Hunyuan-DiT（文生图模型）。

下表为运行模型所需的配置 (batch size = 1):

|          模型           | --load-4bit (DialogGen) | GPU 显存需求 |       GPU       |
| :---------------------: | :---------------------: | :----------: | :-------------: |
| DialogGen + Hunyuan-DiT |            ✘            |     32G      |      A100       |
| DialogGen + Hunyuan-DiT |            ✔            |     22G      |      A100       |
|       Hunyuan-DiT       |            -            |     11G      |      A100       |
|       Hunyuan-DiT       |            -            |     14G      | RTX3090/RTX4090 |

* 需要支持 CUDA 的 NVIDA GPU。
  * 我们已经测试了 V100 和 A100 GPU。
  * **最低配置**: 至少需要 11GB 显存。
  * **推荐配置**: 为了获得更好的生成质量，我们建议您使用具有 32GB 显存的 GPU。
* 已测试的操作系统：Linux

## 🛠️ 依赖项与安装

首先，克隆本仓库：

```shell
git clone https://github.com/tencent/HunyuanDiT
cd HunyuanDiT
```

### 在 Linux 上的安装指南

我们提供了 `environment.yml` 文件用于配置 Conda 环境。
Conda 的安装说明可以在[这里](https://docs.anaconda.com/free/miniconda/index.html)找到。

我们推荐使用 CUDA 11.7 和 12.0 及以上的版本。

```shell
# 1. 准备 Conda 环境
conda env create -f environment.yml

# 2. 激活环境
conda activate HunyuanDiT

# 3. 安装 pip 依赖项
python -m pip install -r requirements.txt

# 4.（可选）安装 flash attention v2 以加速模型（需要CUDA 11.6或更高版本）
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

## 🧱 下载预训练模型
要下载模型，首先需要安装 huggingface-cli。（详细说明见[此处](https://huggingface.co/docs/huggingface_hub/guides/cli)）

```shell
python -m pip install "huggingface_hub[cli]"
```

然后使用下面的命令安装模型：

```shell
# 创建一个名为 'ckpts' 的文件夹用于储存模型，以满足运行该 demo 的先决条件
mkdir ckpts
# 使用 huggingface-cli 工具下载模型。
# 根据您的网络状况，下载时间可能从十分钟到一小时不等
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

<details>
<summary>💡使用 huggingface-cli 的技巧 (关于网络问题)</summary>

##### 1. 使用 HF-Mirror

如果您在中国遇到下载速度慢的情况，可以尝试使用镜像来加快下载速度，例如，

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

##### 2. 断点续传

`huggingface-cli` 支持断点续传。如果下载被中断，您只需重新运行下载命令，即可恢复下载进程。

注意：如果在下载过程中出现类似 `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` 的错误，您可以忽略此错误并重新运行下载命令。

</details>

---

所有的模型都能够自动下载。有关模型的更多信息，请访问 [Hugging Face](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) 仓库。

|       模型        | 参数数量 |                                      Huggingface 下载链接                                      |                               腾讯云下载链接                               |
|:------------------:|:-------:|:-------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|        mT5         |  1.6B   |               [mT5](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/mt5)                |               [mT5](https://dit.hunyuan.tencent.com/download/HunyuanDiT/mt5.zip)                |
|        CLIP        |  350M   |        [CLIP](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder)        |        [CLIP](https://dit.hunyuan.tencent.com/download/HunyuanDiT/clip_text_encoder.zip)        |
|      Tokenizer     |  -      |     [Tokenizer](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/tokenizer)              |      [Tokenizer](https://dit.hunyuan.tencent.com/download/HunyuanDiT/tokenizer.zip)             |
|     DialogGen      |  7.0B   |           [DialogGen](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/dialoggen)            |           [DialogGen](https://dit.hunyuan.tencent.com/download/HunyuanDiT/dialoggen.zip)        |
| sdxl-vae-fp16-fix  |   83M   | [sdxl-vae-fp16-fix](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/sdxl-vae-fp16-fix)  | [sdxl-vae-fp16-fix](https://dit.hunyuan.tencent.com/download/HunyuanDiT/sdxl-vae-fp16-fix.zip)  |
|    Hunyuan-DiT-v1.0     |  1.5B   |          [Hunyuan-DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/model)           |          [Hunyuan-DiT-v1.0](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model.zip)           |
|    Hunyuan-DiT-v1.1     |  1.5B   |          [Hunyuan-DiT-v1.1](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1/tree/main/t2i/model)    |          [Hunyuan-DiT-v1.1](https://dit.hunyuan.tencent.com/download/HunyuanDiT/model-v1_1.zip)            |
|    Data demo       |  -      |                                    -                                                                    |      [Data demo](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip)             |
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/logo.png"  height=100>
</p>

# 混元DiT(Hunyuan-DiT)：一个高性能的多分辨率的Diffusion Transformers(DiT)模型，并具备精细的中文理解能力  

<p align="center">
  <a href="./README.md">English</a> |
  <span>简体中文</span>
</p>

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

本仓库包含我们探索 Hunyuan-DiT 的论文的 PyTorch 模型定义、预训练权重和推理/采样代码。更多可视化内容请访问我们的[项目页面](https://dit.hunyuan.tencent.com/)。

> [**Hunyuan-DiT：一个高性能的多分辨率的Diffusion Transformers(DiT)模型，并具备精细的中文理解能力**](https://arxiv.org/abs/2405.08748) <br>

> [**DialogGen：多模态交互对话系统，用于多轮文本生成图像**](https://arxiv.org/abs/2403.08857)<br>


## 🔥🔥🔥 最新动态！！

* 2024年5月22日：🚀 我们推出了 Hunyuan-DiT 的 TensorRT 版本，加速了 NVIDIA GPU 上的推理速度，达到了**47%**的加速效果。请查看 [TensorRT-libs](https://huggingface.co/Tencent-Hunyuan/TensorRT-libs) 获取使用说明。
* 2024年5月22日：💬 我们现在支持多轮文本生成图像的演示运行。请查看下面的[脚本](#using-gradio)。

## 🤖 在网页上试用

欢迎访问我们网页版的[**腾讯混元Bot**](https://hunyuan.tencent.com/bot/chat)，在这里您可以探索我们的创新产品！只需输入下列建议的提示词或任何其他**包含绘画相关关键词的创意提示词**，即可激活混元文本生成图像功能。释放您的创造力，创建任何您想要的图片，**全部免费！**

您可以使用类似自然语言文本的简单提示词

> 画一只穿着西装的猪
>
> 生成一幅画，赛博朋克风，跑车

或通过多轮对话交互来创建图片。

> 画一个木制的鸟
>
> 变成玻璃的

## 📑 开源计划

- 混元-DiT（文本生成图像模型）
  - [x] 推理(Inference)
  - [x] 检查点(Checkpoints)
  - [ ] 蒸馏版本(Distillation Version)（即将推出 ⏩️）
  - [x] TensorRT 版本(TensorRT Version)（即将推出 ⏩️）
  - [ ] 训练(Training)（稍后推出 ⏩️）
- [DialogGen](https://github.com/Centaurusalpha/DialogGen)（提示词增强模型）
  - [x] 推理(Inference)
- [X] 网页版文生图样例(Web Demo) (基于Gradio)
- [x] 网页版多轮对话交互文生图样例(Multi-turn T2I Demo) (基于Gradio)
- [X] 命令行版文生图样例(Cli Demo)

## 目录
- [混元-DiT](#混元-dit--一个高性能的多分辨率的Diffusion Transformers(DiT)模型，并具备精细的中文理解能力)
  - [摘要](#摘要)
  - [🎉 混元-DiT 主要特点](#-混元-dit-主要特点)
    - [中英双语 DiT 架构](#中英双语-dit-架构)
    - [多轮文本生成图像](#多轮文本生成图像)
  - [📈 对比](#-对比)
  - [🎥 生成图像示例](#-生成图像示例)
  - [📜 要求](#-要求)
  - [🛠 依赖和安装](#-依赖和安装)
  - [🧱 下载预训练模型](#-下载预训练模型)
  - [🔑 推理](#-推理)
    - [使用 Gradio](#使用-gradio)
    - [使用命令行](#使用命令行)
    - [使用 ComfyUI](#使用-comfyUI)
    - [更多配置](#更多配置)
  - [🚀 加速（适用于 Linux）](#-加速适用于-linux)
  - [🔗 BibTeX](#-bibtex)

## 摘要

我们呈现了 混元-DiT，一个高性能的多分辨率的Diffusion Transformers(DiT)模型，并具备精细的中文理解能力。为了构建 混元-DiT，我们精心设计了变压器(transformer)结构、文本编码器(text encoder)和位置编码(positional encoding)。我们还从头开始构建了一个完整的数据管道，用来更新和评估数据，从而进行模型迭代优化。为了实现精细的语言理解，我们训练了一个多模态大语言模型(DialogGen)来优化图像的描述。最终，混元-DiT 能够与用户进行多轮多模态对话，根据上下文进行生成和优化图像。
通过我们精心设计的整体人类评估方案，并由50多位专业评估人员进行评估，混元-DiT在中文图像生成方面超越了其他开源模型，达到了新的技术水平。

## 🎉 **混元-DiT 主要特点**
### **中英双语 DiT 架构**
混元-DiT 是一个在潜在空间中的扩散模型，如下图所示。遵循潜在扩散模型的思路，我们使用预训练的变分自编码器（VAE）将图像压缩到低维潜在空间，并训练一个扩散模型来学习数据分布。我们的扩散模型采用了变压器(transformer)参数化。为了对文本提示词进行编码，我们使用了预训练的双语（英语和中文）CLIP和多语言T5编码器。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/framework.png"  height=450>
</p>

### 多轮文本生成图像
理解自然语言指令并与用户进行多轮交互，对于文本生成图像系统来说非常重要。它可以帮助构建一个动态的、迭代的创作过程，逐步将用户的想法变为现实。
在本节中，我们将详细介绍如何赋予 混元-DiT 执行多轮对话和图像生成的能力。我们训练了多模态大语言模型（MLLM）以理解多轮用户对话并输出新的文本提示用于图像生成。
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/mllm.png"  height=300>
</p>

## 📈 对比
为了全面比较 混元-DiT 和其他模型的生成能力，我们构建了一个四维测试集，包括文本与图像一致性、排除AI痕迹、主体清晰度和美学。并邀请了超过50名专业评估员进行了评估。

<p align="center">
<table> 
<thead> 
<tr> 
    <th rowspan="2">模型</th> <th rowspan="2">是否开源</th> <th>文本与图像一致性 (%)</th> <th>排除AI痕迹 (%)</th> <th>主体清晰度 (%)</th> <th rowspan="2">美学 (%)</th> <th rowspan="2">总体 (%)</th> 
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

## 🎥 生成图像示例

为了更好地了解混元-DiT生成图像的细节和风格，我们提供了一些生成图像示例。

* **中国元素**
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/chinese elements understanding.png"  height=220>
</p>

* **长文本输入**


<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/long text understanding.png"  height=310>
</p>

* **多轮对话文生图**

https://github.com/Tencent/tencent.github.io/assets/27557933/94b4dcc3-104d-44e1-8bb2-dc55108763d1



---

## 📜 要求

本仓库包括DialogGen（提示词增强模型）和混元-DiT（文生图模型）。

以下表格显示了运行模型所需的要求（batch size = 1）：

|          Model          | --load-4bit (DialogGen) | GPU Peak Memory |       GPU       |
|:-----------------------:|:-----------------------:|:---------------:|:---------------:|
| DialogGen + Hunyuan-DiT |            ✘            |       32G       |      A100       |
| DialogGen + Hunyuan-DiT |            ✔            |       22G       |      A100       |
|       Hunyuan-DiT       |            -            |       11G       |      A100       |
|       Hunyuan-DiT       |            -            |       14G       | RTX3090/RTX4090 |

* 需要支持CUDA的NVIDIA GPU。
  * 我们已经测试了V100和A100 GPU。
  * **最低要求**：所需的最低GPU内存为11GB。
  * **推荐配置**：我们建议使用具有32GB内存的GPU以获得更好的生成质量。
* 测试操作系统：Linux

## 🛠 依赖和安装

首先克隆本仓库
```shell
git clone https://github.com/tencent/HunyuanDiT
cd HunyuanDiT
```

### Linux安装指南

我们提供了一个 `environment.yml` 文件来配置Conda环境。
Conda的安装说明阅读[Conda安装说明](https://docs.anaconda.com/free/miniconda/index.html)。

```shell
# 1. 准备Conda环境
conda env create -f environment.yml

# 2. 激活环境
conda activate HunyuanDiT

# 3. 安装pip依赖
python -m pip install -r requirements.txt

# 4. （可选）安装flash attention v2加速（需要CUDA 11.6或更高版本）
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

## 🧱 下载预训练模型
下载本模型之前，请首先安装huggingface-cli（详细说明可查看[huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli)）。

```shell
python -m pip install "huggingface_hub[cli]"
```

然后使用以下命令下载模型：

```shell
# 创建一个名为'ckpts'的目录，将模型保存到该目录。
mkdir ckpts
# 使用huggingface-cli工具下载模型。
# 下载时间根据网络条件可能需要10分钟到1小时不等。
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

<details>
<summary>💡 使用huggingface-cli 的小提示（例如网络问题）</summary>

##### 1. 使用HF-Mirror

如果在国内遇到下载速度缓慢的问题，可以尝试使用HF镜像加快下载过程。例如，

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

##### 2. 恢复下载

`huggingface-cli` 支持恢复下载。如果下载中断，只需重新运行下载命令即可恢复下载过程。

注意：如果在下载过程中出现类似于 `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` 的错误，则可以忽略该错误并重新运行下载命令。

</details>

---

所有模型将会自动下载。有关模型的更多信息，请访问[Hugging Face代码库](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT)。

|       模型        | 参数规格 |                                              下载地址                                               |
|:------------------:|:-------:|:-------------------------------------------------------------------------------------------------------:|
|        mT5         |  1.6B   |               [mT5](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/mt5)                |
|        CLIP        |  350M   |        [CLIP](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder)        |
|     DialogGen      |  7.0B   |           [DialogGen](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/dialoggen)            |
| sdxl-vae-fp16-fix  |   83M   | [sdxl-vae-fp16-fix](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/sdxl-vae-fp16-fix)  |
|    Hunyuan-DiT     |  1.5B   |          [Hunyuan-DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/model)           |


## 🔑 推理
### 使用 Gradio
我们提供了一个基于 Gradio 的 Web 界面，用于快速运行推理。请运行以下命令以启动 Gradio 服务。

```shell
# 默认情况下，我们启动一个中文用户界面。
python app/hydit_app.py

# 使用 Flash Attention 进行加速。
python app/hydit_app.py --infer-mode fa

# 如果 GPU 内存不足，您可以禁用提示词增强模型（DialogGen）。
# 直到您不使用`--no-enhance` 标志来重新启动应用程序之前，提示词增强模型（DialogGen）将不可用。
python app/hydit_app.py --no-enhance

# 以英文用户界面启动
python app/hydit_app.py --lang en

# 启动多轮文本图像生成用户界面。
# 如果您的 GPU 内存少于 32GB，请使用 '--load-4bit' 启用 4 位量化，这需要至少 22GB 的内存。
python app/multiTurnT2I_app.py
```
然后可以通过 http://0.0.0.0:443 访问演示。

### 使用命令行

您也可以使用命令行工具运行推理，我们提供了几个命令来快速启动：

```shell
# 使用提示词增强模型 + 文生图模型
python sample_t2i.py --prompt "渔舟唱晚"

# 仅使用文生图模型
python sample_t2i.py --prompt "渔舟唱晚" --no-enhance

# 仅使用文生图模型并用Flash Attention 进行加速
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"

# 使用指定图像尺寸生成图像
python sample_t2i.py --prompt "渔舟唱晚" --image-size 1280 768

# 使用提示词增强模型 + 文生图模型。提示词增强模型以4位量化方式加载，可能会降低效果
python sample_t2i.py --prompt "渔舟唱晚"  --load-4bit

```

### 使用 ComfyUI

 混元-DiT的ComfyUI: [HunyuanDiT-ComfyUI](https://github.com/city96/ComfyUI_ExtraModels)

更多提示词示例可以在[example_prompts.txt](example_prompts.txt)查看。

### 更多配置

我们列出了一些常用的配置参数，以便更简单的上手使用：

|    参数名称     |  默认值  |                     描述                     |
|:---------------:|:---------:|:---------------------------------------------------:|
|   `--prompt`    |   None    |        用于生成图像的文本提示语              |
| `--image-size`  | 1024 1024 |           生成图像的像素大小                |
|    `--seed`     |    42     |        用于生成图像的随机种子                |
| `--infer-steps` |    100    |          采样步数               |
|  `--negative`   |     -     |      用于生成图像的负向提示语       |
| `--infer-mode`  |   torch   |       推理模式（torch、fa 或 trt）        |
|   `--sampler`   |   ddpm    |    扩散采样器（ddpm、ddim 或 dpmms）     |
| `--no-enhance`  |   False   |        禁用提示词增强模型         |
| `--model-root`  |   ckpts   |     模型检查点的根目录     |
|  `--load-key`   |    ema    | 加载module模型或 ema 模型（ema 或 module） |
|  `--load-4bit`  |   Fasle   |     使用 4 位量化加载 DialogGen 模型     |

## 🚀 加速（适用于 Linux）

我们提供了混元-DiT的TensorRT版本，用于推理加速（比Flash Attention更快）。
更多详情请查看[Tencent-Hunyuan/TensorRT-libs](https://huggingface.co/Tencent-Hunyuan/TensorRT-libs)

## 🔗 BibTeX

如果您发现[Hunyuan-DiT](https://arxiv.org/abs/2405.08748)或[DialogGen](https://arxiv.org/abs/2403.08857)对您的研究和应用有帮助，请使用以下BibTeX引用：：

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

## github获赞里程碑

<a href="https://star-history.com/#Tencent/HunyuanDiT&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanDiT&type=Date" />
 </picture>
</a>
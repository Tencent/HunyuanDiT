

## 🔑 Inference

### 在Gradio上使用

确保在运行以下命令之前激活conda环境。


```shell
# 默认启动中文的UI界面
python app/hydit_app.py

# 使用Flash Attention加速
python app/hydit_app.py --infer-mode fa

# 如果GPU内存不足，您可以禁用增强模型。
# The enhancement will be unavailable until you restart the app without the `--no-enhance` flag. 

#在重新启动应用程序并且不带 `--no-enhance` 之前，增强功能将不可用。
python app/hydit_app.py --no-enhance

# 启动英文的UI界面
python app/hydit_app.py --lang en

# Start a multi-turn T2I generation UI. 启动多轮文本生成图像生成界面 
# 如果你的 GPU 内存少于 32GB，使用 `--load-4bit` 以启用 4 位量化，这至少需要 22GB 的内存。
python app/multiTurnT2I_app.py
```
示例程序可以通过访问 http://0.0.0.0:443获取 。需要注意的是，这里的 0.0.0.0 需要替换为你的服务器IP地址。

### 使用🤗 Diffusers

请预先安装 PyTorch 2.0 或更高版本，以满足 diffusers 库指定版本的要求。



安装 🤗 diffusers，确保版本至少为 0.28.1：
```shell
pip install git+https://github.com/huggingface/diffusers.git
```
或
```shell
pip install diffusers
```
您可以使用以下 Python 脚本通过中文和英文提示生成图像：

```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

# 你也可以使用英文提示，HunyuanDiT支持中英文提示
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt).images[0]
```

您可以使用我们的蒸馏模型来更快地生成图像：
```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled", torch_dtype=torch.float16)
pipe.to("cuda")

# 你也可以使用英文提示，因为HunyuanDiT支持中英文提示
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt, num_inference_steps=25).images[0]
```
更多详细信息可以查阅：[HunyuanDiT-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled)

### 使用命令行

我们提供了几种命令以快速开始

```shell
# 提示增强 + 文本生成图像。Torch 模式。
python sample_t2i.py --prompt "渔舟唱晚"

# 仅文本生成图像。Torch 模式。
python sample_t2i.py --prompt "渔舟唱晚" --no-enhance

# 仅文本生成图像。Flash Attention 模式。
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"

# 生成其他尺寸的图像。
python sample_t2i.py --prompt "渔舟唱晚" --image-size 1280 768

# 提示增强 + 文本生成图像。使用 4 位量化加载 DialogGen，但可能会降低性能。
python sample_t2i.py --prompt "渔舟唱晚"  --load-4bit

```

更多prompts样例可以查阅 [example_prompts.txt](example_prompts.txt)

### 更多配置选项
我们列出了一些更常用的配置选项方便用户使用
|    Argument     |  Default  |                     Description                     |
|:---------------:|:---------:|:---------------------------------------------------:|
|   `--prompt`    |   None    |        用于图像生成的文本提示      |
| `--image-size`  | 1024 1024 |           生成图像的大小       |
|    `--seed`     |    42     |        生成图像的随机种子       |
| `--infer-steps` |    100    |         扩散步数        |
|  `--negative`   |     -     |      用于图像生成的负面提示      |
| `--infer-mode`  |   torch   |       推理模式（torch、fa 或 trt）      |
|   `--sampler`   |   ddpm    |    扩散采样器（ddpm、ddim 或 dpmm）   |
| `--no-enhance`  |   False   |     禁用提示增强模型       |
| `--model-root`  |   ckpts   |    模型检查点的根目录   |
|  `--load-key`   |    ema    | 加载学生模型或 EMA 模型（ema 或 module） |
|  `--load-4bit`  |   Fasle   |     使用 4 位量化加载 DialogGen 模型    |

### 使用ComfyUI

我们提供了几种命令以快速开始
We provide several commands to quick start: 

```shell
# 下载 ComfyUI 代码
git clone https://github.com/comfyanonymous/ComfyUI.git

# 安装 torch、torchvision、torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# 安装 ComfyUI 所需的 Python 包
cd ComfyUI
pip install -r requirements.txt

# ComfyUI 已成功安装！

# 下载模型权重或将现有模型文件夹链接到 ComfyUI
python -m pip install "huggingface_hub[cli]"
mkdir models/hunyuan
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./models/hunyuan/ckpts

# 进入 ComfyUI custom_nodes 文件夹并从 HunyuanDiT 仓库中复制 comfyui-hydit 文件夹
cd custom_nodes
cp -r ${HunyuanDiT}/comfyui-hydit ./
cd comfyui-hydit

# 安装一些必要的 Python 包
pip install -r requirements.txt

# 我们的工具已成功安装！

# 进入 ComfyUI 主文件夹
cd ../..
# 运行 ComfyUI 启动命令
python main.py --listen --port 80

# ComfyUI 成功运行！
```
更多详细信息可以查阅： [ComfyUI README](comfyui-hydit/README.md)


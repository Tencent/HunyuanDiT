## 📜 要求
本项目包括DialogGen(一个提示增强模型)和Hunyuan-DiT(文生图模型)。

下表展示了运行本模型时的环境要求(batch size=1)：

|        模型                | 是否加载4bit量化(DialogGen) |   最大GPU显存   |      可支持的GPU      |
|:------------------------:|:---------------------:|:-----------:|:-----------------:|
| DialogGen + Hunyuan-DiT  |       &#x2717;        |     32G     |       A100        |
| DialogGen + Hunyuan-DiT  |       &#x2713;        |     22G     |       A100        |
|       Hunyuan-DiT        |           -           |     11G     |       A100        |
|       Hunyuan-DiT        |           -           |     14G     |  RTX3090/RTX4090  |

*  需要使用支持CUDA的英伟达GPU：
   * 本项目已经测试能够在V100和A100显卡上运行。
   * **最小GPU显存**：GPU最小显存至少为11GB。
   * **推荐**：我们推荐使用32GB显存的显卡，以获得更好的生成质量。
*  测试采用的操作系统：Linux

## 🛠️ 依赖和安装

首先，克隆本项目：
```bash
git clone https://github.com/tencent/HunyuanDiT
cd HunyuanDiT
```


我们提供了一个 `environment.yml`文件用于创建Conda环境。
Conda的安装指引可以参考如下链接： [here](https://docs.anaconda.com/free/miniconda/index.html).


```bash
# 1. 准备conda环境
conda env create -f environment.yml

# 2. 激活环境
conda activate HunyuanDiT

# 3. 安装pip依赖
python -m pip install -r requirements.txt

# 4. (可选的) 安装 flash attention v2 用于加速(要求CUDA 11.6或以上版本)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

我们推荐使用 CUDA versions 11.7 和 12.0+ 版本。



## 🧱 下载预训练模型
为了下载模型，首先请安装huggingface-cli。(指引细节可以参考如下链接：[here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

然后采用如下命令下载模型：

```shell
# 创建一个名为'ckpts'的文件夹，该文件夹下保存模型权重，是运行该demo的先行条件
mkdir ckpts
# 采用 huggingface-cli工具下载模型
# 下载时间可能为10分钟到1小时，取决于你的网络条件。
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```


<details>
<summary>💡使用huggingface-cli的小技巧 (网络问题)</summary>

##### 1. 使用 HF 镜像

如果在中国境内的下载速度较慢，你可以使用镜像加速下载过程，例如
```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```

##### 2. 重新下载

`huggingface-cli` 支持重新下载。如果下载过程被中断，你只需要重新运行下载命令，恢复下载进程。

注意：如果在下载过程中发生类似`No such file or directory: 'ckpts/.huggingface/.gitignore.lock'`的错误，你可以忽略这个错误，
并重新执行以下命令： `huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts`

</details>

---

所有的模型将会自动下载。如果想要了解更多关于模型的信息，请查阅Hugging Face的项目：[here](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT)。

|       模型       |  参数量   |                                             下载链接                                              |
|:------------------:|:------:|:-------------------------------------------------------------------------------------------------------:|
|        mT5         |  1.6B  |               [mT5](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/mt5)                |
|        CLIP        |  350M  |        [CLIP](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder)        |
|     DialogGen      |  7.0B  |           [DialogGen](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/dialoggen)            |
| sdxl-vae-fp16-fix  |  83M   | [sdxl-vae-fp16-fix](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/sdxl-vae-fp16-fix)  |
|    Hunyuan-DiT     |  1.5B  |          [Hunyuan-DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/model)           |





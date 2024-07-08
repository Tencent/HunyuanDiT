## :truck: 训练

### 数据准备

  参考以下命令来准备训练数据
  
  1. 安装依赖项
  
      我们提供了一个高效的数据管理库，名为 IndexKits，支持在训练期间进行对读取数亿个数据的管理，详细请见文档[文档](./IndexKits/README.md).
      ```shell
      # 1 安装依赖项
      cd HunyuanDiT
      pip install -e ./IndexKits
     ```
  2. 数据下载
  
     请下载 [演示数据](https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip).
     ```shell
     # 2 数据下载
     wget -O ./dataset/data_demo.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/data_demo.zip
     unzip ./dataset/data_demo.zip -d ./dataset
     mkdir ./dataset/porcelain/arrows ./dataset/porcelain/jsons
     ```
  3. 数据转换
  
     使用下表中列出的字段为训练数据创建 CSV 文件.
    
     |      领域       |    必需   |       介绍        |    示例     |
     |:---------------:| :------:  |:----------------:|:-----------:|
     |   `image_path`  |    必需   |      图片路径     |`./dataset/porcelain/images/0.png`        | 
     |   `text_zh`     | 必需  |    文字描述               |  青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 | 
     |   `md5`         | 可选  |    图片 md5 (讯息摘要5)  |    `d41d8cd98f00b204e9800998ecf8427e`         | 
     |   `width`       | 可选  |    图片宽度    |     `1024 `       | 
     |   `height`      | 可选  |    图片高度   |    ` 1024 `       | 
     
     > ⚠️ 可以省略 MD5、宽度和高度等可选字段。如果省略，下面的脚本将自动计算它们。在处理大规模训练数据时，此过程可能非常耗时.
  
     我们可以利用[Arrow](https://github.com/apache/arrow) 来训练数据的格式，它提供标准高效的内存数据表示。同时提供了一个转换脚本，用于将 CSV 文件转换为Arrow格式.
     ```shell  
     # 3 数据转换
     python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows 1
     ```
  
  4. 数据选择和配置文件创建 
     
      我们通过 YAML 文件配置训练数据。在这些文件中，您可以设置标准数据处理策略，用于筛选、复制、重复数据删除等有关训练数据。有关详细信息，请参见[./IndexKits](IndexKits/docs/MakeDataset.md).
  
      有关示例文件，请参阅[文件](./dataset/yamls/porcelain.yaml). 有关完整参数配置文件，请参阅[文件](./IndexKits/docs/MakeDataset.md).
  
     
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

### Full-parameter Training
 
  To leverage DeepSpeed in training, you have the flexibility to control **single-node** / **multi-node** training by adjusting parameters such as `--hostfile` and `--master_addr`. For more details, see [link](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

  ```shell
  # Single Resolution Training
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain.json
  
  # Multi Resolution Training
  PYTHONPATH=./ sh hydit/train.sh --index-file dataset/porcelain/jsons/porcelain_mt.json --multireso --reso-step 64
  ```

### LoRA



We provide training and inference scripts for LoRA, detailed in the [./lora](./lora/README.md). 

  ```shell
  # Training for porcelain LoRA.
  PYTHONPATH=./ sh lora/train_lora.sh --index-file dataset/porcelain/jsons/porcelain.json

  # Inference using trained LORA weights.
  python sample_t2i.py --prompt "青花瓷风格，一只小狗"  --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt
  ```
 We offer two types of trained LoRA weights for `porcelain` and `jade`, see details at [links](https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA)
  ```shell
  cd HunyuanDiT
  # Use the huggingface-cli tool to download the model.
  huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora
  
  # Quick start
  python sample_t2i.py --prompt "青花瓷风格，一只猫在追蝴蝶"  --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain
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

### Using Kohya

We provide several commands to quick start LoRA Training and DreamBooth Training with Kohya: 

```shell
# Download kohya_ss GUI
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss/

# Download sd-scripts training backend, use dev branch
git clone -b dev https://github.com/kohya-ss/sd-scripts ./sd-scripts

# Move the costom GUI codes to the kohya_ss GUI, and replace files with the same name
cp -Rf ${HunyuanDiT}/kohya_ss-hydit/* ./

# Download model weights as before or link the existing model folder to kohya_ss/models.
python -m pip install "huggingface_hub[cli]"
# If you want to download the full model, use the following command
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1 --local-dir ./models/HunyuanDiT-V1.1
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-V1.2 --local-dir ./models/HunyuanDiT-V1.2
# Or, if you want to download the fp16 pruned model
huggingface-cli download KBlueLeaf/HunYuanDiT-V1.1-fp16-pruned --local-dir ./models/HunyuanDiT-V1.1-fp16-pruned

# After the model is downloaded, you may need to modify the file name an make sure it follows the kohya standard format:
# rename the file name in t2i/ folder as shown below:
# HunyuanDiT-V1.2/t2i/
#  - model/                  -> denoiser/
#  - clip_text_encoder/      -> clip/
#  - mt5/                    -> mt5/
#  - sdxl-vae-fp16-fix/      -> vae/
# Also you may need to move tokenizer/* into clip/ folder
mv HunyuanDiT-V1.2/t2i/model/ HunyuanDiT-V1.2/t2i/denoiser/
mv HunyuanDiT-V1.2/t2i/clip_text_encoder/ HunyuanDiT-V1.2/t2i/clip/
mv HunyuanDiT-V1.2/t2i/mt5/ HunyuanDiT-V1.2/t2i/mt5/
mv HunyuanDiT-V1.2/t2i/sdxl-vae-fp16-fix/ HunyuanDiT-V1.2/t2i/vae/
mv HunyuanDiT-V1.2/t2i/tokenizer/* HunyuanDiT-V1.2/t2i/clip/ 

# Install some essential python Package, 
conda create -n hydit-kohya python=3.10.12
conda activate hydit-kohya

# Install some essential packages, please make sure cuda environment is installed and python version is 3.10
# For cuda 12:
pip install torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1
# For cuda 11:
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 xformers==0.0.23.post1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# For cpu offloading to save GPU memory, we recommend to install Deepspeed as follows:
DS_BUILD_CPU_ADAM=1 pip install deepspeed==0.14.1

# Install other python package
pip install -r hunyuan_requirements.txt

# Run the Kohya_ss UI launch command
python kohya_gui.py
```
More details can be found in [Kohya_ss README](kohya_ss-hydit/README.md)

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

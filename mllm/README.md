# Hunyuan-MLLM
We provide two multimodal large language models, Hunyuan-Captioner and Dialogen. The former provides fine-grained text descriptions for training data, while the latter enhances the user's prompt input during inference and supports multi-turn text-to-image generation. 

## Contents
- [Hunyuan-Captioner](https://github.com/Tencent/HunyuanDiT/tree/main/mllm#hunyuan-captioner)
  - [Instructions](https://github.com/Tencent/HunyuanDiT/tree/main/mllm#instructions)
  - [Examples](https://github.com/Tencent/HunyuanDiT/tree/main/mllm#examples)
  - [Inference](https://github.com/Tencent/HunyuanDiT/tree/main/mllm#inference)
  - [Gradio](https://github.com/Tencent/HunyuanDiT/tree/main/mllm#gradio)
- [DialogGen](https://github.com/Tencent/HunyuanDiT/tree/main/mllm#dialoggen)
  - [Inference](https://github.com/Tencent/HunyuanDiT/tree/main/mllm#inference-1)
  - [Gradio](https://github.com/Tencent/HunyuanDiT/tree/main/mllm#gradio-1)
- [Acceleration](https://github.com/Tencent/HunyuanDiT/tree/main/mllm/trtllm#acceleration)

## Hunyuan-Captioner
Hunyuan-Captioner meets the need of text-to-image techniques by maintaining a high degree of image-text consistency. It can generate high-quality image descriptions from a variety of angles, including object description, objects relationships, background information, image style, etc. Our code is based on [LLaVA](https://github.com/haotian-liu/LLaVA) implementation.

### Examples

<td align="center"><img src="../asset/caption_demo.jpg" alt="Image 3" width="1200"/></td>
 

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
To launch a Gradio demo locally, please execute the following commands sequentially. Ensure each command is running in the background. For more detailed instructions, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA). 
```bash
cd mllm
python -m llava.serve.controller --host 0.0.0.0 --port 10000
python -m llava.serve.gradio_web_server --controller http://0.0.0.0:10000 --model-list-mode reload --port 443
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://0.0.0.0:10000 --port 40000 --worker http://0.0.0.0:40000 --model-path "../ckpts/captioner" --model-name LlavaMistral
```
Then the demo can be accessed through http://0.0.0.0:443. It should be noted that the 0.0.0.0 here needs to be X.X.X.X with your server IP.

 


## Hunyuan-DialogGen
We additionally provide inference commands for [DialogGen](https://github.com/Centaurusalpha/DialogGen). 
### Inference
```bash
cd HunyuanDiT
python mllm/dialoggen_demo.py --prompt "画一只小猫"
```

### Gradio
```bash
# Start a multi-turn T2I generation UI. 
# If your GPU memory is less than 32GB, use '--load-4bit' to enable 4-bit quantization, which requires at least 22GB of memory.
python app/multiTurnT2I_app.py

```

## Acceleration (for Linux)
We provide TensorRT-LLM (precision: int8 weight-only) version of Hunyuan-Captioner for inference acceleration(for Linux). See: [Acceleration](https://github.com/Tencent/HunyuanDiT/tree/main/mllm/trtllm#acceleration)
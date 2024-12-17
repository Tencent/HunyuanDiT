# Hunyuan-MLLM-TRTLLM
We provide TensorRT-LLM (precision: int8 weight-only) version of Hunyuan-Captioner for inference acceleration(for Linux). 



## Hunyuan-Captioner-TRTLLM


### Instructions
a. Retrieve and launch the docker container
For a list of the supported hardware see the [**Frameworks Support Matrix**](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

It should be noted that Nvidia’s official documentation does not list the support list for consumer-grade graphics cards. Our tests show that 4090 and 3080 graphics cards are supported, but for other consumer-grade graphics cards, we cannot guarantee whether you will encounter performance problems or some bugs, please experiment by yourself.


```bash
docker pull nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3
docker run --rm --ipc=host --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3
```

b. Download Torch model
```shell
huggingface-cli download Tencent-Hunyuan/HunyuanCaptioner --local-dir ../ckpts/captioner
```
b. Build TRTLLM engine 
```shell
sh build_trtllm.sh
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
python3  run_llava.py  --max_new_tokens 512 --hf_model_dir  ./llava-v1.6-mistral-7b-hf-merged/   --visual_engine_dir visual_engines/   --llm_engine_dir trt_engines/llava/int8/1-gpu --mode caption_zh --image_file ../images/demo1.png
```

b. Insert specific knowledge into caption

```bash
python3  run_llava.py  --max_new_tokens 512 --hf_model_dir  ./llava-v1.6-mistral-7b-hf-merged/   --visual_engine_dir visual_engines/   --llm_engine_dir trt_engines/llava/int8/1-gpu --mode insert_content --image_file ../images/demo2.png --content 宫保鸡丁
```

c. Single picture inference in English

```bash
python3  run_llava.py  --max_new_tokens 512 --hf_model_dir  ./llava-v1.6-mistral-7b-hf-merged/   --visual_engine_dir visual_engines/   --llm_engine_dir trt_engines/llava/int8/1-gpu --mode caption_en --image_file ../images/demo1.png
```

d. Multiple pictures inference in Chinese

```bash
### Convert multiple pictures to csv file. 
python3 ../make_csv.py --img_dir ../images --input_file ../images/demo.csv

### Multiple pictures inference
python3  run_llava.py  --max_new_tokens 512 --hf_model_dir  ./llava-v1.6-mistral-7b-hf-merged/  --visual_engine_dir visual_engines/   --llm_engine_dir trt_engines/llava/int8/1-gpu --mode caption_zh --input_file ../images/demo.csv --output_file ../images/demo_res.csv

```

### Benchmark

|Hardware           | GPU Memory Usage (GB)                           |TRTLLM Inference Duration(s)                   | 
| ---           | ---                                       | ---                                  |
|A100    |       8.9                        |    0.73         | 
|4090 |           8.7     |0.73|  
|3080    |           8.7        |1.16

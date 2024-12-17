## Using HunyuanDiT IP-Adapter


### Instructions

 The dependencies and installation are basically the same as the base model, and we use the module weights for training.
 Download the model using the following commands:

```bash
cd HunyuanDiT
# Use the huggingface-cli tool to download the model.
# We recommend using module weights as the base model for IP-Adapter inference, as our provided pretrained weights are trained on them.
huggingface-cli download Tencent-Hunyuan/IP-Adapter ipa.pt --local-dir ./ckpts/t2i/model
huggingface-cli download Tencent-Hunyuan/IP-Adapter clip_img_encoder.pt  --local-dir ./ckpts/t2i/model/clip_img_encoder

# Quick start
python3 sample_ipadapter.py  --infer-mode fa --ref-image-path ipadapter/asset/input/tiger.png --i-scale 1.0 --prompt 一只老虎在海洋中游泳，背景是海洋。构图方式是居中构图，呈现了动漫风格和文化，营造了平静的氛围。 --infer-steps 100 --is-ipa True --load-key distill
```

Examples of ref input and IP-Adapter results are as follows:
<table>
  <tr>
    <td colspan="3" align="center">Ref Input</td>
  </tr>
  
q

  

  <tr>
    <td align="center"><img src="asset/input/tiger.png" alt="Image 0" width="200"/></td>
    <td align="center"><img src="asset/input/beauty.png" alt="Image 1" width="200"/></td>
    <td align="center"><img src="asset/input/xunyicao.png" alt="Image 2" width="200"/></td>
    
  </tr>
  
  <tr>
    <td colspan="3" align="center">IP-Adapter Output</td>
  </tr>

  <tr>
    <td align="center">一只老虎在奔跑。<br>（A tiger running.） </td>
    <td align="center">一个卡通美女，抱着一只小猪。<br>（A cartoon beauty holding a little pig.） </td>
    <td align="center">一片紫色薰衣草地。<br>（A purple lavender field.） </td>
  </tr>

  <tr>
    <td align="center"><img src="asset/output/tiger_run.png" alt="Image 3" width="200"/></td>
    <td align="center"><img src="asset/output/beauty_pig.png" alt="Image 4" width="200"/></td>
    <td align="center"><img src="asset/output/xunyicao_res.png" alt="Image 5" width="200"/></td>
  </tr>

  <tr>
    <td align="center">一只老虎在看书。<br>（A tiger is reading a book.） </td>
    <td align="center">一个卡通美女，穿着绿色衣服。<br>（A cartoon beauty wearing green clothes.） </td>
    <td align="center">一片紫色薰衣草地，有一只可爱的小狗。<br>（A purple lavender field with a cute puppy.） </td>
  </tr>

  <tr>
    <td align="center"><img src="asset/output/tiger_book.png" alt="Image 3" width="200"/></td>
    <td align="center"><img src="asset/output/beauty_green_cloth.png" alt="Image 4" width="200"/></td>
    <td align="center"><img src="asset/output/xunyicao_dog.png" alt="Image 5" width="200"/></td>
  </tr>

  <tr>
    <td align="center">一只老虎在咆哮。<br>（A tiger is roaring.） </td>
    <td align="center">一个卡通美女，戴着墨镜。<br>（A cartoon beauty wearing sunglasses.） </td>
    <td align="center">水墨风格,一片紫色薰衣草地。<br>（Ink style. A purple lavender field.） </td>
  </tr>
  <tr>
    <td align="center"><img src="asset/output/tiger_roar.png" alt="Image 3" width="200"/></td>
    <td align="center"><img src="asset/output/beauty_glass.png" alt="Image 4" width="200"/></td>
    <td align="center"><img src="asset/output/xunyicao_style.png" alt="Image 5" width="200"/></td>
  </tr>
 
  
</table>


### Training

We provide base model weights for IP-Adapter training, you can use `module` weights for IP-Adapter training.

Here is an example, we load the `module` weights into the main model and conduct IP-Adapter training. 

If apply multiple resolution training, you need to add the `--multireso` and `--reso-step 64` parameter. 

```bash
task_flag="IP_Adapter"                                # the task flag is used to identify folders.                         # checkpoint root for resume
index_file=path/to/your/index_file
results_dir=./log_EXP                                        # save root for results
batch_size=1                                                 # training batch size
image_size=1024                                              # training image resolution
grad_accu_steps=1                                            # gradient accumulation
warmup_num_steps=0                                           # warm-up steps
lr=0.0001                                                    # learning rate
ckpt_every=10                                         # create a ckpt every a few steps.
ckpt_latest_every=10000                                    # create a ckpt named `latest.pt` every a few steps.
ckpt_every_n_epoch=2                                         # create a ckpt every a few epochs.
epochs=8                                                     # total training epochs

PYTHONPATH=. \
sh $(dirname "$0")/run_g_ipadapter.sh \
    --task-flag ${task_flag} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --multireso \
    --reso-step 64 \
    --uncond-p 0.22 \
    --uncond-p-t5 0.22\
    --uncond-p-img 0.05\
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --extra-fp16 \
    --results-dir ${results_dir} \
    --resume\
    --resume-module-root ckpts/t2i/model/pytorch_model_module.pt \
    --epochs ${epochs} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --ckpt-every-n-epoch ${ckpt_every_n_epoch} \
    --log-every 10 \
    --deepspeed \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    --no-strict \
    --training-parts ipadapter \
    --is-ipa True \
    --resume-ipa True \
    --resume-ipa-root ckpts/t2i/model/ipa.pt  \
    "$@"

```

Recommended parameter settings

|     Parameter     |  Description  |          Recommended Parameter Value                               | Note|
|:---------------:|:---------:|:---------------------------------------------------:|:--:|
|   `--batch-size` |    Training batch size    |        1        | Depends on GPU memory|
|   `--grad-accu-steps` |    Size of gradient accumulation    |       2        | - |
|   `--lr` |    Learning rate  |        0.0001        | - |
|   `--training-parts` |  be trained parameters when training IP-Adapter  |        ipadapter        | - |
|   `--is-ipa` |  training IP-Adapter or not  |        True       | - |
|   `--resume-ipa-root` |  resume ipa model or not when training  |        ipa model path       | - |


### Inference
Use the following command line for inference.

a. Use the parameter float i-scale to specify the weight of IP-Adapter reference image. The bigger parameter indicates more relativity to reference image.
```bash
python3 sample_ipadapter.py  --infer-mode fa --ref-image-path ipadapter/input/beach.png --i-scale 1.0 --prompt 一只老虎在海洋中游泳，背景是海洋。构图方式是居中构图，呈现了动漫风格和文化，营造了平静的氛围。 --infer-steps 100 --is-ipa True --load-key module
```


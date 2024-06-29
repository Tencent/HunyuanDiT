model='DiT-g/2'                                         # model type
task_flag="lora_porcelain_ema_rank64"                   # task flag 
resume=./ckpts/t2i/model/                               # resume checkpoint 
index_file=dataset/porcelain/jsons/porcelain.json       # the selected data indices
results_dir=./log_EXP                                   # save root for results
batch_size=1                                            # training batch size
image_size=1024                                         # training image resolution
grad_accu_steps=2                                       # gradient accumulation steps
warmup_num_steps=0                                      # warm-up steps
lr=0.0001                                               # learning rate
ckpt_every=100                                          # create a ckpt every a few steps.
ckpt_latest_every=2000                                  # create a ckpt named `latest.pt` every a few steps.
rank=64                                                 # rank of lora
max_training_steps=2000                                 # Maximum training iteration steps

PYTHONPATH=./ deepspeed hydit/train_deepspeed.py \
    --task-flag ${task_flag} \
    --model ${model} \
    --training-parts lora \
    --rank ${rank} \
    --resume-split \
    --resume ${resume} \
    --ema-to-module \
    --lr ${lr} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.03 \
    --predict-type v_prediction \
    --uncond-p 0.44 \
    --uncond-p-t5 0.44 \
    --index-file ${index_file} \
    --random-flip \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --ema-dtype fp32 \
    --results-dir ${results_dir} \
    --ckpt-every ${ckpt_every} \
    --max-training-steps ${max_training_steps}\
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 10 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --qk-norm \
    --rope-img base512 \
    --rope-real \
    --use-style-cond \
    --size-cond 1024 1024 \
    "$@"

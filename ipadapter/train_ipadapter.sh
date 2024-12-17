export CUDA_VISIBLE_DEVICES=0,1,2,3
task_flag="IP_Adapter"                                # the task flag is used to identify folders.                         # checkpoint root for resume
index_file=dataset/porcelain/jsons/porcelain_mt.json 
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
sh ./hydit/run_g_ipadapter.sh \
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
    --resume-module-root ./ckpts/t2i/model/pytorch_model_distill.pt \
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
    --resume-ipa-root ./ckpts/t2i/model/ipa.pt  \
    "$@"

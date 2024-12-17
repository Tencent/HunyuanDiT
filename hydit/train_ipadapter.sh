export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1



task_flag="dit_g2_full_1024p"                                # the task flag is used to identify folders.                         # checkpoint root for resume
resume=ckpts/t2i/model
results_dir=./log_EXP                                        # save root for results
batch_size=1                                                 # training batch size
image_size=1024                                              # training image resolution
grad_accu_steps=1                                            # gradient accumulation
warmup_num_steps=0                                           # warm-up steps
lr=0.0001                                                    # learning rate
ckpt_every=100                                           # create a ckpt every a few steps.
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
    --index-file  \
    your data path \
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
    --resume \
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
    --resume-ipa Ture \
    --resume-ipa-root ckpts/t2i/model/ipa.pt  \
    "$@"

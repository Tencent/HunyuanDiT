model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
deepspeed hydit/train_deepspeed_controlnet.py ${params}  "$@"
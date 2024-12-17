sh env.sh
git clone https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
python3 rename_key.py --huggingface_repo_dir ./llava-v1.6-mistral-7b-hf/ --thirdparty_repo_dir ../ckpts/captioner/ --merged_repo_dir ./llava-v1.6-mistral-7b-hf-merged/
python3 convert_checkpoint.py --model_dir ./llava-v1.6-mistral-7b-hf-merged/ --output_dir tmp/trt_models/llava/int8/1-gpu --dtype float16  --use_weight_only --weight_only_precision int8

trtllm-build --checkpoint_dir tmp/trt_models/llava/int8/1-gpu  --output_dir trt_engines/llava/int8/1-gpu --gemm_plugin float16  --max_batch_size 1 --max_input_len 2048 --max_output_len 512 --max_multimodal_len 576

python3 build_visual_engine.py --model_path ./llava-v1.6-mistral-7b-hf-merged/ --model_type llava_next


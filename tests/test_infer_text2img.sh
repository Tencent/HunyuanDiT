#!/bin/bash

task_name="infer_text2img_flash_attn"
log_file="${task_name}.log"
python sample_t2i.py --infer-mode fa --infer-steps 30 --prompt "青花瓷风格，一只可爱的哈士奇" --no-enhance --load-key distill > "$log_file" 2>&1
exit_status=$?
if [ $exit_status -eq 0 ]; then
    echo -e "\033[0;32m$task_name Passed\033[0m"
else
    echo -e "\033[0;31m$task_name Failed\033[0m"
fi

###
task_name="infer_text2img_raw_attn"
log_file="${task_name}.log"
python sample_t2i.py --infer-mode torch --infer-steps 30 --prompt "青花瓷风格，一只可爱的哈士奇" --no-enhance --load-key distill > "$log_file" 2>&1
exit_status=$?
if [ $exit_status -eq 0 ]; then
    echo -e "\033[0;32m$task_name Passed\033[0m"
else
    echo -e "\033[0;31m$task_name Failed\033[0m"
fi

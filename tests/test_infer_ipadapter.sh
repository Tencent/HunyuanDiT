#!/bin/bash

task_name="infer_ipadapter.sh"
log_file="${task_name}.log"
python3 sample_ipadapter.py  --infer-mode torch --ref-image-path ipadapter/asset/input/tiger.png --i-scale 1.0 --prompt 一只老虎在海洋中游泳，背景是海洋。构图方式是居中构图，呈现了动漫风格和文化，营造了平静的氛围。 --infer-steps 30 --is-ipa True --load-key distill > "$log_file" 2>&1
exit_status=$?
if [ $exit_status -eq 0 ]; then
    echo -e "\033[0;32m$task_name Passed\033[0m"
else
    echo -e "\033[0;31m$task_name Failed\033[0m"
fi

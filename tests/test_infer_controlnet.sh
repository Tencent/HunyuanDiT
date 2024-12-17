#!/bin/bash

task_name="infer_controlnet_canny"
log_file="${task_name}.log"
python sample_controlnet.py --infer-mode torch --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0 > "$log_file" 2>&1
exit_status=$?
if [ $exit_status -eq 0 ]; then
    echo -e "\033[0;32m$task_name Passed\033[0m"
else
    echo -e "\033[0;31m$task_name Failed\033[0m"
fi

###
task_name="infer_controlnet_depth"
log_file="${task_name}.log"
python sample_controlnet.py --infer-mode torch --no-enhance --load-key distill --infer-steps 50 --control-type depth --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/depth.jpg --control-weight 1.0 > "$log_file" 2>&1
exit_status=$?
if [ $exit_status -eq 0 ]; then
    echo -e "\033[0;32m$task_name Passed\033[0m"
else
    echo -e "\033[0;31m$task_name Failed\033[0m"
fi

###
task_name="infer_controlnet_pose"
log_file="${task_name}.log"
python sample_controlnet.py --infer-mode torch --no-enhance --load-key distill --infer-steps 50 --control-type pose --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/pose.jpg --control-weight 1.0 > "$log_file" 2>&1
exit_status=$?
if [ $exit_status -eq 0 ]; then
    echo -e "\033[0;32m$task_name Passed\033[0m"
else
    echo -e "\033[0;31m$task_name Failed\033[0m"
fi

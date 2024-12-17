#!/bin/bash

test_base=./tests # 指定测试目录
export CUDA_VISIBLE_DEVICES=3 # 指定GPU

for file in $(find "$test_base" -maxdepth 1 -name 'test_*.sh'); do
    # 去掉路径前的 './' 以获得文件名
    filename=$(basename "$file")
    echo "################################"
    echo "Running tests in $filename..."

    bash "$file"
    echo "################################"

done
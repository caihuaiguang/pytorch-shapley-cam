#!/bin/bash

# 导出CUDA设备
export CUDA_VISIBLE_DEVICES=4

# 设置PYTHONPATH
export PYTHONPATH=/home/caihuaiguang/DSG/pytorch-shapley-cam:$PYTHONPATH
source /media/caihuaiguang/miniconda3/etc/profile.d/conda.sh
conda activate cords
# 运行Python脚本
python usage_examples/ADCC_imagenet.py 

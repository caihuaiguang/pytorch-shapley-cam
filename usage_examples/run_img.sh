#!/bin/bash

# 导出CUDA设备
export CUDA_VISIBLE_DEVICES=3

# 设置PYTHONPATH
export PYTHONPATH=/home/caihuaiguang/DSG/pytorch-shapley-cam:$PYTHONPATH
source /media/caihuaiguang/miniconda3/etc/profile.d/conda.sh
conda activate cords
# 运行Python脚本
# python usage_examples/ADCC_imagenet.py --model resnet50 --cam-method gradcamplusplus --batch-size 128 --output-file output.txt
# python usage_examples/ADCC_imagenet.py --model resnet50 --cam-method gradcamelementwise  --batch-size 128 --output-file output.txt
# python usage_examples/ADCC_imagenet.py --model resnet50 --cam-method scorecam --batch-size 12 --output-file output.txt
# python usage_examples/ADCC_imagenet.py --model resnet50 --cam-method gradcam  --batch-size 128 --output-file output.txt
# python usage_examples/ADCC_imagenet.py --model resnet50 --cam-method hirescam  --batch-size 128 --output-file output.txt


python usage_examples/ADCC_imagenet.py --model swint --cam-method gradcam  --batch-size 64 --output-file output.txt

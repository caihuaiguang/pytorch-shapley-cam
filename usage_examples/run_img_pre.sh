#!/bin/bash

# 导出CUDA设备
export CUDA_VISIBLE_DEVICES=3

# 设置PYTHONPATH
export PYTHONPATH=/home/caihuaiguang/DSG/pytorch-shapley-cam:$PYTHONPATH
source /media/caihuaiguang/miniconda3/etc/profile.d/conda.sh
conda activate cords

# 模型和CAM方法的数组
# "vit" 
# models=("resnet18")
models=("resnext50" "resnet50" "resnet101" "resnet18" "swint_t" "swint_s" "swint_b" "vgg16" "efficientnetb0" "mobilenetv2")
cam_methods=("shapleycam" "gradcam" "gradcamelementwise" "hirescam" "gradcamplusplus"  "xgradcam" "layercam" "randomcam")

# cam_methods=("shapleycam")
# cam_methods=("scorecam")

# ablationcam and scorecam are too time-comsuming

# 批次大小设置
declare -A batch_sizes
batch_sizes=( ["resnet50"]=128 ["resnext50"]=128 ["resnet18"]=128 ["resnet101"]=64 ["swint_t"]=128 ["swint_s"]=64 ["swint_b"]=48 ["vit"]=64 ["vgg16"]=64  ["efficientnetb0"]=128 ["mobilenetv2"]=128)

# 遍历所有模型和CAM方法
for model in "${models[@]}"; do
  for cam_method in "${cam_methods[@]}"; do
    python usage_examples/ADCC_imagenet_pre.py --model "$model" --cam-method "$cam_method" --batch-size "${batch_sizes[$model]}" --output-file output_ADCC_pre_test.txt
    # python usage_examples/ADCC_imagenet.py --model "$model" --cam-method "$cam_method" --batch-size 32 --output-file output_ADCC_softmax_test.txt
  done
done

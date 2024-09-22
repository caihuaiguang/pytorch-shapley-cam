#!/bin/bash

# 导出CUDA设备
export CUDA_VISIBLE_DEVICES=2

# 设置PYTHONPATH
export PYTHONPATH=/home/caihuaiguang/DSG/pytorch-shapley-cam:$PYTHONPATH
source /media/caihuaiguang/miniconda3/etc/profile.d/conda.sh
conda activate cords

# 模型和CAM方法的数组
# "vit" 
# models=("resnext50" "resnet50" "resnet101" "resnet152" "resnet18"  "vgg16" "efficientnetb0" "mobilenetv2" "vit")

models=("swint_t"  "swint_s" "swint_b" "swint_l" )
cam_methods=("gradcamelementwise" "shapleycam" "shapleycam_mean" "shapleycam_hires" "gradcam"   "hirescam" "gradcamplusplus"  "xgradcam" "layercam" "randomcam")
# cam_methods=("shapleycam"  "gradcamelementwise" "shapleycam_mean"  "gradcam" "gradcamplusplus" "layercam" "randomcam")
# cam_methods=("shapleycam_x"  "xgradcam"  "shapleycam_hires"  "hirescam" )

# cam_methods=( "shapleycam_mean"  "gradcam"  "gradcamplusplus" "layercam" "gradcamelementwise")
# cam_methods=( "scorecam" "shapleycam" ) # 72h,before less than 1 hour

# ablationcam and scorecam are too time-comsuming

# 批次大小设置
declare -A batch_sizes
batch_sizes=( ["resnet50"]=128 ["resnext50"]=128 ["resnet18"]=128  ["resnet101"]=64 ["resnet152"]=64 ["swint_t"]=128 ["swint_s"]=64 ["swint_b"]=48 ["swint_l"]=32 ["vit"]=64 ["vgg16"]=64  ["efficientnetb0"]=128 ["mobilenetv2"]=128)

# 遍历所有模型和CAM方法
for model in "${models[@]}"; do
  for cam_method in "${cam_methods[@]}"; do
    python usage_examples/ADCC_imagenet_conv_gist.py --model "$model" --cam-method "$cam_method" --batch-size "${batch_sizes[$model]}"  --output-file gist2_conv.txt
    # python usage_examples/ADCC_imagenet.py --model "$model" --cam-method "$cam_method" --batch-size 48 --output-file output_ADCC_softmax_test.txt
  done
done

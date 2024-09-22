import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

# 设置设备为GPU或CPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# 标准的ImageNet图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载ImageNet验证集
val_dataset = datasets.ImageFolder('/media/caihuaiguang/data/ILSVRC2012_img_val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 加载更大的预训练的DeiT模型, acc: 81.74%
# model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

# resnet 18, acc: 69.76%
# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# resnet 50, acc: 76.13%
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# resnet 101, acc: 77.38%
# model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

# resnet 152, acc:78.32%
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

# resnext 50, acc: 77.62%
# model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)


# swinT_t, acc: 80.91% similar size to resnet50
# model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

# swinT_s, acc: 83.05% similar size to resnet101
# model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)

# swinT_b, acc: 84.71%
# model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

# swinT_l, acc: 85.83%
# model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)


# vgg16, acc: 71.59%
# model = models.vgg16(pretrained=True)

# mobilenet_v2, 71.88%
# model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)


# efficientnetb0, 77.69%
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)



model = model.to(device)
model.eval()

# 初始化统计变量
correct = 0
total = 0

# 迭代验证集进行预测
with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Processing images", unit="batch"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # 模型推理
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        
        # 统计准确数量
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算并打印准确率
accuracy = 100 * correct / total
print(f'Accuracy on the ImageNet validation set: {accuracy:.2f}%')

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

# 设置设备为GPU或CPU
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# 标准的ImageNet图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载ImageNet验证集
val_dataset = datasets.ImageFolder('/media/caihuaiguang/data/ILSVRC2012_img_val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 加载更大的预训练的DeiT模型, acc: 81.74%
# model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

# resnet 50, acc: 76.13%
# model = models.resnet50(pretrained=True)

# vgg16, acc: 71.59%
# model = models.vgg16(pretrained=True)

# swinT, acc: 84.71%
# model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)


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

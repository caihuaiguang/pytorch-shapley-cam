import timm

# 指定模型名称 (Swin Transformer Large 版本)
model_name = 'swin_large_patch4_window7_224'

# 加载模型（会自动下载到本地缓存目录）
model = timm.create_model(model_name, pretrained=True)

print(f"{model_name} 已下载并缓存到本地！")

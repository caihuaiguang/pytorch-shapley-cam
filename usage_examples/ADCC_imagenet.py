import warnings
warnings.filterwarnings('ignore')
from torchvision import models, datasets, transforms
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, GradCAMElementWise, EigenGradCAM,  XGradCAM, LayerCAM, AblationCAM, RandomCAM, ShapleyCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.metrics.ADCC import ADCC
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # Import tqdm for the progress bar

# Transform for the ILSVRC2012 (ImageNet) dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ILSVRC2012 (ImageNet) validation dataset
val_dataset = datasets.ImageFolder('/media/caihuaiguang/data/ILSVRC2012_img_val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# val_subset = Subset(val_dataset, range(100))
# val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True).to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Define the Grad-CAM and ADCC
target_layers = [model.layer4]
cam = GradCAMElementWise(model=model, target_layers=target_layers)
adcc_metric = ADCC()

# Initialize ADCC sum and counter
adcc_sum = 0.0
avgdrop_sum = 0.0
coh_sum = 0.0
com_sum = 0.0
num_images = 0

# Iterate over the validation dataset
for imgs, labels in tqdm(val_loader, desc="Processing images", unit="batch"):
    # Move the images to the appropriate device
    imgs = imgs.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create input tensor and target
    input_tensor = imgs
    outputs = model(input_tensor)
    target_categories = labels
    targets = [ClassifierOutputSoftmaxTarget(category) for category in target_categories]
    
    # Compute Grad-CAM
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

    # Calculate ADCC and other metrics for the current batch
    adcc_value, avgdrop_value, coh_value, com_value = adcc_metric(input_tensor, grayscale_cams, targets, model, cam)
    
    # Accumulate the metrics
    batch_size = imgs.size(0)
    adcc_sum += adcc_value.sum() 
    avgdrop_sum += avgdrop_value.sum() 
    coh_sum += coh_value.sum() 
    com_sum += com_value.sum()
    num_images += batch_size

# Calculate the average of all metrics over the entire validation dataset
average_adcc = adcc_sum / num_images
average_avgdrop = avgdrop_sum / num_images
average_coh = coh_sum / num_images
average_com = com_sum / num_images

# Print the results
print(f"Average ADCC over the ILSVRC2012 validation dataset: {average_adcc}")
print(f"Average AvgDrop over the ILSVRC2012 validation dataset: {average_avgdrop}")
print(f"Average Coherency over the ILSVRC2012 validation dataset: {average_coh}")
print(f"Average Complexity over the ILSVRC2012 validation dataset: {average_com}")

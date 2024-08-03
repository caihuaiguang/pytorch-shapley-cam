import warnings
warnings.filterwarnings('ignore')
from torchvision import models, datasets, transforms
import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, GradCAMElementWise, EigenGradCAM,  XGradCAM, LayerCAM, AblationCAM, RandomCAM, ShapleyCAM, HiResCAM, ScoreCAM

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.metrics.ADCC import ADCC
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True).to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Define the Grad-CAM and ADCC
target_layers = [model.layer4]
cam = GradCAM(model=model, target_layers=target_layers)
adcc_metric = ADCC()

# Initialize ADCC sum and counter
adcc_sum = 0.0
num_images = 0

# Iterate over the test dataset
for img, label in tqdm(test_loader, desc="Processing images", unit="image"):
    # Move the image to the appropriate device
    img = img.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create input tensor and target
    input_tensor = img
    pre_label = model(input_tensor).argmax()
    # targets = [ClassifierOutputTarget(label.item())]
    targets = None

    # Compute Grad-CAM
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

    # Calculate ADCC for the current image
    adcc_value = adcc_metric(input_tensor, grayscale_cams, targets, model, cam)
    adcc_sum += adcc_value
    num_images += 1

    # print(f"Processed {num_images} images, Current ADCC: {adcc_value}")

# Calculate the average ADCC over the entire test dataset
average_adcc = adcc_sum / num_images
print(f"Average ADCC over the CIFAR-10 test dataset: {average_adcc}")

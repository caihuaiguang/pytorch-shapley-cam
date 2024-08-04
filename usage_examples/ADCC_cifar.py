import warnings
warnings.filterwarnings('ignore')
from torchvision import models, datasets, transforms
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, GradCAMElementWise, EigenGradCAM,  XGradCAM, LayerCAM, AblationCAM, RandomCAM, ShapleyCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.ADCC import ADCC
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.CIFAR10(root='/media/caihuaiguang/data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # Adjust batch size as needed

# Load the pre-trained ResNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Define the Grad-CAM and ADCC
target_layers = [model.layer4]
cam = ScoreCAM(model=model, target_layers=target_layers)  # You can switch to other CAM methods
adcc_metric = ADCC()

# Initialize sums and counters for metrics
adcc_sum = 0.0
avgdrop_sum = 0.0
coh_sum = 0.0
com_sum = 0.0
num_images = 0

# Iterate over the test dataset
for imgs, labels in tqdm(test_loader, desc="Processing images", unit="batch"):
    # Move the images and labels to the appropriate device
    imgs = imgs.to(device)
    labels = labels.to(device)

    # Create input tensor and target
    input_tensor = imgs
    outputs = model(input_tensor)
    target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
    targets = [ClassifierOutputSoftmaxTarget(category) for category in target_categories]

    # Compute Grad-CAM
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

    # Calculate ADCC and other metrics for the current batch
    batch_adcc_values, avgdrop_values, coh_values, com_values = adcc_metric(input_tensor, grayscale_cams, targets, model, cam)
    
    # Accumulate the metrics
    batch_size = len(batch_adcc_values)
    adcc_sum += batch_adcc_values.sum()
    avgdrop_sum += avgdrop_values.sum()
    coh_sum += coh_values.sum()
    com_sum += com_values.sum()
    num_images += batch_size

# Calculate the average of all metrics over the entire test dataset
average_adcc = adcc_sum / num_images
average_avgdrop = avgdrop_sum / num_images
average_coh = coh_sum / num_images
average_com = com_sum / num_images

# Print the results
print(f"Average ADCC over the CIFAR-10 test dataset: {average_adcc}")
print(f"Average AvgDrop over the CIFAR-10 test dataset: {average_avgdrop}")
print(f"Average Coherency over the CIFAR-10 test dataset: {average_coh}")
print(f"Average Complexity over the CIFAR-10 test dataset: {average_com}")

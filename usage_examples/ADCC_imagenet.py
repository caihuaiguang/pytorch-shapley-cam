import warnings
warnings.filterwarnings('ignore')
from torchvision import models, datasets, transforms
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, GradCAMElementWise, EigenGradCAM, XGradCAM, LayerCAM, AblationCAM, RandomCAM, ShapleyCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.metrics.ADCC import ADCC
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import timm


def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def swint_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture (e.g., resnet50, vgg16)')
    parser.add_argument('--cam-method', type=str, default='gradcamplusplus', help='CAM method (e.g., gradcam, gradcam++, scorecam)')
    parser.add_argument('--batch-size', type=int, default=96, help='Batch size for DataLoader')
    parser.add_argument('--output-file', type=str, default='output.txt', help='Output file to append results')
    return parser.parse_args()

def load_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'vit':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    elif model_name == 'swint':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model

def select_cam_method(method_name, model, model_name):
    methods = {
        "gradcam": GradCAM,
        "gradcamplusplus": GradCAMPlusPlus,
        "gradcamelementwise": GradCAMElementWise,
        "scorecam": ScoreCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenGradCAM,
        "layercam": LayerCAM,
        "ablationcam": AblationCAM,
        "shapleycam": ShapleyCAM,
        "hirescam": HiResCAM
    }
    if method_name not in methods:
        raise ValueError(f"CAM method {method_name} not supported")
    target_layers = None
    if model_name == 'resnet50':
        target_layers = [model.layer4[-1]]
    elif model_name == 'vgg16':
        target_layers = [model.features[-1]]
    elif model_name == 'vit':
        target_layers = [model.blocks[-1].norm1]
    elif model_name == 'swint':
        target_layers = [model.layers[-1].blocks[-1].norm2]
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    reshape_transform = None
    if model_name == 'vit':
        reshape_transform = vit_reshape_transform
    elif model_name == 'swint':
        reshape_transform = swint_reshape_transform
    return methods[method_name](model=model, target_layers=target_layers, reshape_transform = reshape_transform)

if __name__ == "__main__":
    args = get_args()

    # Load the model
    model = load_model(args.model)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()


    # Select the CAM method
    cam = select_cam_method(args.cam_method, model, args.model)

    # Transform for the ILSVRC2012 (ImageNet) dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the ILSVRC2012 (ImageNet) validation dataset
    val_dataset = datasets.ImageFolder('/media/caihuaiguang/data/ILSVRC2012_img_val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the ADCC
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
        target_categories = labels
        targets = [ClassifierOutputSoftmaxTarget(category) for category in target_categories]

        # Compute CAMs
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

    # Prepare the output string
    output_str = (f"Model: {args.model}, CAM Method: {args.cam_method}, Batch Size: {args.batch_size}\n"
                  f"Average ADCC: {average_adcc}\n"
                  f"Average AvgDrop: {average_avgdrop}\n"
                  f"Average Coherency: {average_coh}\n"
                  f"Average Complexity: {average_com}\n\n")

    # Print the results to the console
    print(output_str)

    # Write the results to the output file
    with open(args.output_file, 'a') as f:
        f.write(output_str)

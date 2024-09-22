import warnings
warnings.filterwarnings('ignore')
from torchvision import models, datasets, transforms
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, GradCAMElementWise, EigenGradCAM, XGradCAM, LayerCAM, AblationCAM, RandomCAM, ShapleyCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.metrics.ADCC import ADCC
from torch.utils.data import DataLoader, Subset, random_split
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
    ## norm 2
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)

    ## norm1
    
    # result = tensor.transpose(2, 3).transpose(1, 2)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture (e.g., resnet50, vgg16)')
    parser.add_argument('--cam-method', type=str, default='gradcamplusplus', help='CAM method (e.g., gradcam, gradcam++, scorecam)')
    parser.add_argument('--batch-size', type=int, default=96, help='Batch size for DataLoader')
    parser.add_argument('--output-file', type=str, default='output.txt', help='Output file to append results')
    return parser.parse_args()

def load_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'vit':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    elif model_name == 'swint_t':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    elif model_name == 'swint_s':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    elif model_name == 'swint_b':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
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
        "hirescam": HiResCAM,
        "randomcam": RandomCAM
    }
    if method_name not in methods:
        raise ValueError(f"CAM method {method_name} not supported")
    target_layers = None
    if model_name in ['resnet50', 'resnet101', 'resnext50']:
        # target_layers = [model.layer4[-1].conv3]
        target_layers = [model.layer4[-1].relu]
    elif model_name == 'resnet18':
        # target_layers = [model.layer4[-1].conv2]
        target_layers = [model.layer4[-1].relu]
    elif model_name == 'vgg16':
        # target_layers = [model.features[28]]
        target_layers = [model.features[-1]]
    elif model_name == 'vit':
        target_layers = [model.blocks[-1].norm1]
    elif model_name in ['swint_t', 'swint_s', 'swint_b']:
        # target_layers = [model.layers[-1].blocks[-1].norm1]
        target_layers = [model.layers[-1].blocks[-1].norm2]
    elif model_name == 'mobilenetv2':
        # target_layers = [model.features[-1][0]]
        target_layers = [model.features[-1][2]]
    elif model_name == "efficientnetb0":
        # target_layers = [model.features[-1][0]]
        target_layers = [model.features[-1][2]]

    else:
        raise ValueError(f"Model {model_name} not supported")
    
    reshape_transform = None
    if model_name == 'vit':
        reshape_transform = vit_reshape_transform
    elif model_name in ['swint_t', 'swint_s', 'swint_b']:
        reshape_transform = swint_reshape_transform
    return methods[method_name](model=model, target_layers=target_layers, reshape_transform = reshape_transform)


if __name__ == "__main__":
    args = get_args()

    # Load the model
    model = load_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Select the CAM method
    cam = select_cam_method(args.cam_method, model, args.model)

    # Transform for the ILSVRC2012 (ImageNet) dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the ILSVRC2012 (ImageNet) validation dataset
    val_dataset = datasets.ImageFolder('/media/caihuaiguang/data/ILSVRC2012_img_val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # # Calculate the size of the subset (1/100th of the total dataset)
    # subset_size = len(val_dataset) // 100
    # subset_dataset, _ = random_split(val_dataset, [subset_size, len(val_dataset) - subset_size])
    # val_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the ADCC
    adcc_metric = ADCC()

    # Initialize metrics sum and counter
    adcc_sum, avgdrop_sum, coh_sum, com_sum, num_images = 0.0, 0.0, 0.0, 0.0, 0

    # Iterate over the validation dataset
    for imgs, labels in tqdm(val_loader, desc="Processing images", unit="batch"):
        # Move the images and labels to the appropriate device
        imgs = imgs.to(device)
        labels = labels.to(device)

        # # Get model predictions
        # with torch.no_grad():  # Disables gradient calculation to save memory
        #     outputs = model(imgs)
        #     predicted_categories = torch.argmax(outputs, dim=-1)  # Use torch.argmax for tensors

        # # Find correctly predicted samples
        # correct_indices = (predicted_categories == labels).nonzero(as_tuple=True)[0]

        # # Indexing tensors using correct indices
        # # Update imgs and target_categories with correctly predicted samples
        # input_tensor = imgs[correct_indices]
        # target_categories = labels[correct_indices]
        
        input_tensor = imgs
        target_categories = labels

        targets = [ClassifierOutputTarget(category) for category in target_categories]

        # Compute CAMs
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        metric_targets = [ClassifierOutputSoftmaxTarget(category) for category in target_categories]
        
        # Calculate ADCC and other metrics for the current batch
        adcc_value, avgdrop_value, coh_value, com_value = adcc_metric(input_tensor, grayscale_cams, targets, metric_targets, model, cam)
        
        # Accumulate the metrics
        batch_size = imgs.size(0)
        adcc_sum += adcc_value.sum()
        avgdrop_sum += avgdrop_value.sum()
        coh_sum += coh_value.sum()
        com_sum += com_value.sum()
        num_images += batch_size

    # Calculate the average of all metrics over the entire validation dataset
    average_avgdrop = avgdrop_sum / num_images
    average_coh = coh_sum / num_images
    average_com = com_sum / num_images
    # average_adcc = 3 / (1 / average_coh + 1 / (1 - average_com) + 1 / (1 - average_avgdrop))
    average_adcc = adcc_sum / num_images

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
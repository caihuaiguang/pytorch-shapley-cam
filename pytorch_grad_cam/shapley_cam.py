from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM

from pytorch_grad_cam.activations_and_gradients_tensor import ActivationsAndGradientstensor
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

class ShapleyCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            ShapleyCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

        self.activations_and_grads = ActivationsAndGradientstensor(self.model, target_layers, reshape_transform)
    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            # outputs_copy = outputs.clone()
            # target_categories = np.argmax(outputs_copy.cpu().data.numpy(), axis=-1)
            # targets = [ClassifierOutputTarget(category) for category in target_categories]
            
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            # loss.backward(retain_graph=True, create_graph = True)
            torch.autograd.grad(loss, input_tensor,  retain_graph = True, create_graph = True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        activations: List[Tensor]  # type: ignore[assignment]
        grads: List[Tensor]  # type: ignore[assignment]
        # Use the math kernels for scaled dot product attention
        # hvp = torch.autograd.grad(
        #     outputs=self.activations_and_grads.original_gradients[0],
        #     inputs=self.activations_and_grads.original_activations[0],
        #     grad_outputs=self.activations_and_grads.original_activations[0],
        #     retain_graph=False
        # )[0]
        self.activations_and_grads.release()
        hvp = torch.autograd.grad(
            outputs=grads,
            inputs=activations,
            grad_outputs=activations,
            retain_graph=False
        )[0]
        if self.activations_and_grads.reshape_transform is not None:
            print("not None")
            hvp = self.activations_and_grads.reshape_transform(hvp)
            activations = self.activations_and_grads.reshape_transform(activations)
            grads = self.activations_and_grads.reshape_transform(grads)
        # hvp = grads * grads * activations
        print(torch.norm(hvp))
        weight = (grads - 0.5*hvp).cpu().detach().numpy()
            
        # weight = (grads).cpu().detach().numpy()
        activations = activations.cpu().detach().numpy()
        # weight = (grads).cpu().detach().numpy()

        is_flatten = True
        # is_flatten = False

        # 2D image
        if len(weight.shape) == 4:
            if is_flatten is True:
                weight = self.compute_normalized_product(weight*activations, weight)
                # weight = np.mean(weight, axis=(2, 3))
            return weight, activations, is_flatten
        
        # 3D image
        elif len(weight.shape) == 5:
            if is_flatten is True:
                weight = self.compute_normalized_product(weight*activations, weight)
                weight = np.mean(weight, axis=(2, 3, 4))
            return weight, activations, is_flatten
        
        else:
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")


    def minmax_normalize(self, weighted_activations, eps=1e-7):
        # Get the dimensions
        _, N, H, W = weighted_activations.shape
        
        # Initialize the normalized activations array with the same shape
        normalized_activations = np.zeros_like(weighted_activations)
        
        # Iterate over each N dimension
        for i in range(N):
            # Get the H*W layer
            layer = weighted_activations[0, i, :, :]
            
            # Calculate min and max values
            layer_min = np.min(layer)
            layer_max = np.max(layer)
            
            # Apply Min-Max normalization
            normalized_activations[0, i, :, :] = (layer - layer_min) / (layer_max - layer_min + eps)
        
        return normalized_activations

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights, activations, is_flatten = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)

        # 2D conv
        if len(activations.shape) == 4:
            if is_flatten is True:
                weighted_activations = weights[:, :, None, None] * activations
            else:
                weighted_activations = weights * activations
            # weighted_activations = self.minmax_normalize(weighted_activations)

        # 3D conv
        elif len(activations.shape) == 5:
            if is_flatten is True:
                weighted_activations = weights[:, :, None, None, None] * activations
            else:
                weighted_activations = weights * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")
        # weighted_activations = np.maximum(weighted_activations, 0)
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = [a for a in self.activations_and_grads.original_activations]
        grads_list = [g for g in self.activations_and_grads.original_gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

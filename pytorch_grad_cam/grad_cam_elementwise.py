import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class GradCAMElementWise(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None):
        super(
            GradCAMElementWise,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        
        weight = grads
        k = 3
        w_mean = np.mean(weight, axis=(2, 3), keepdims=True)
        w_std = np.std(weight, axis=(2, 3), keepdims=True)
        scale_factor = np.minimum(1, np.abs(w_mean)/(k*(w_std+1e-9)))
        weight = w_mean + scale_factor * (weight-w_mean)
        grads = weight
        elementwise_activations = np.maximum(grads * activations, 0)

        if eigen_smooth:
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
        return cam

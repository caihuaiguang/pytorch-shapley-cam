# https://github.com/aimagelab/ADCC/

from scipy.linalg import norm
from scipy import stats as STS
import torch
import torch.nn.functional as FF
from pytorch_grad_cam.metrics.cam_mult_image import multiply_tensor_with_cam
import numpy as np
from typing import List, Callable
import cv2

def complexity(saliency_map):
    return abs(saliency_map).sum()/(saliency_map.shape[-1]*saliency_map.shape[-2])

def coherency(saliency_map, explanation_map, attr_method, targets):

    saliency_map_B=attr_method(explanation_map, targets)

    A, B = saliency_map.detach(), saliency_map_B.detach()

    '''
    # Pearson correlation coefficient
    # '''
    Asq, Bsq = A.view(1, -1).squeeze(0).cpu(), B.view(1, -1).squeeze(0).cpu()

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if torch.tensor(Asq).isnan().any() or torch.tensor(Bsq).isnan().any():
        y = 0.
    else:
        y, _ = STS.pearsonr(Asq, Bsq)
        y = (y + 1) / 2

    return y,A,B



class ADCC:
    def __init__(self):
        self.perturbation = multiply_tensor_with_cam

    def __call__(self, input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module,
                 cam_method,
                 return_visualization=False):

        with torch.no_grad():
            outputs = model(input_tensor)
            scores = [target(output).cpu().numpy()
                      for target, output in zip(targets, outputs)]
            scores = np.float32(scores)

        perturbated_tensors = []
        cam = cams[0]
        tensor = self.perturbation(input_tensor[0, ...].cpu(),
                                   torch.from_numpy(cam))
        tensor = tensor.to(input_tensor.device)
        perturbated_tensors.append(tensor.unsqueeze(0))
        perturbated_tensors = torch.cat(perturbated_tensors)

        with torch.no_grad():
            outputs_after_imputation = model(perturbated_tensors)
        scores_after_imputation = [
            target(output).cpu().numpy() for target, output in zip(
                targets, outputs_after_imputation)]
        scores_after_imputation = np.float32(scores_after_imputation)

        result = scores_after_imputation - scores
        avgdrop = max(0., scores - scores_after_imputation) / scores
        com = complexity(cam)
        coh,_,_ = coherency(cam, perturbated_tensors, cam_method, targets)

        adcc = 3 / (1 / coh + 1 / (1 - com) + 1 / (1 - avgdrop))

        if return_visualization:
            return adcc, perturbated_tensors
        else:
            return adcc
        
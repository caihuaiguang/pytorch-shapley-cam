from scipy.linalg import norm
from scipy import stats as STS
import torch
import torch.nn.functional as FF
from pytorch_grad_cam.metrics.cam_mult_image import multiply_tensor_with_cam
import numpy as np
from typing import List, Callable
import cv2


def complexity(saliency_map):
    return abs(saliency_map).sum(axis=(1, 2)) / (saliency_map.shape[-1] * saliency_map.shape[-2])


def coherency(A, explanation_map, attr_method, targets):
    B = attr_method(torch.tensor(explanation_map), targets)

    Asq = A.reshape((A.shape[0], -1))
    Bsq = B.reshape((B.shape[0], -1))

    y = np.zeros(A.shape[0])

    for i in range(Asq.shape[0]):
        if np.isnan(Asq[i]).any() or np.isnan(Bsq[i]).any() or np.std(Bsq[i]) == 0 or np.std(Asq[i]) == 0:
            y[i] = 0.0
        else:
            y[i], _ = STS.pearsonr(Asq[i], Bsq[i])
            y[i] = (y[i] + 1) / 2

    return y, A, B


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
            scores = np.float32([target(output).cpu().numpy() for target, output in zip(targets, outputs)])

        perturbated_tensors = []
        for i in range(cams.shape[0]):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(cam))
            tensor = tensor.to(input_tensor.device)
            perturbated_tensors.append(tensor.unsqueeze(0))
        perturbated_tensors = torch.cat(perturbated_tensors)

        with torch.no_grad():
            outputs_after_imputation = model(perturbated_tensors)
        scores_after_imputation = np.float32(
            [target(output).cpu().numpy() for target, output in zip(targets, outputs_after_imputation)])

        avgdrop = np.maximum(0., scores - scores_after_imputation) / scores
        com = complexity(cams)
        coh, _, _ = coherency(cams, perturbated_tensors, cam_method, targets)
        if np.all(coh == 0.0):
            adcc = np.zeros(coh.shape[0])
        else:
            adcc = 3 / (1 / coh + 1 / (1 - com) + 1 / (1 - avgdrop))

        if return_visualization:
            return adcc, perturbated_tensors
        else:
            return adcc, avgdrop, coh, com

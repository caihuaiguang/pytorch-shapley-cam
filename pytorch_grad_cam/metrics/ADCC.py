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
        # Check for NaN and inf values and handle them
        if np.any(np.isnan(Bsq[i])) or np.any(np.isnan(Asq[i])) or np.any(np.isinf(Bsq[i])) or np.any(np.isinf(Asq[i])):
            print(f"Warning: Array contains NaN or inf values at index {i}")
            y[i] = 0.0  # Or some other default value, as appropriate
        elif np.std(Bsq[i]) == 0 or np.std(Asq[i]) == 0:
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
                 metric_targets: List[Callable],
                 model: torch.nn.Module,
                 cam_method,
                 return_visualization=False):
        with torch.no_grad():
            outputs = model(input_tensor)
            scores = np.float32([metric_target(output).cpu().numpy() for metric_target, output in zip(metric_targets, outputs)])

        added_tensors = []
        deleted_tensors = []
        for i in range(cams.shape[0]):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(cam))
            tensor = tensor.to(input_tensor.device)
            added_tensors.append(tensor.unsqueeze(0))
            tensor_delete = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(1-cam))
            tensor_delete = tensor_delete.to(input_tensor.device)
            deleted_tensors.append(tensor_delete.unsqueeze(0))
        added_tensors = torch.cat(added_tensors)
        deleted_tensors = torch.cat(deleted_tensors)

        with torch.no_grad():
            outputs_after_added = model(added_tensors)
            scores_after_added = np.float32([metric_target(output).cpu().numpy() for metric_target, output in zip(metric_targets, outputs_after_added)])
            outputs_after_deleted = model(deleted_tensors)
            scores_after_deleted = np.float32([metric_target(output).cpu().numpy() for metric_target, output in zip(metric_targets, outputs_after_deleted)])

        drop = np.maximum(0., scores - scores_after_added) / scores
        inc = scores_after_added > scores
        dropindeletion = np.maximum(0., scores - scores_after_deleted)/scores
        com = complexity(cams)
        coh, _, _ = coherency(cams, added_tensors, cam_method, targets)

        adcc = 3 / (1 / coh + 1 / (1 - com) + 1 / (1 - drop))

        if return_visualization:
            return adcc, drop, coh, com, inc, dropindeletion, added_tensors, deleted_tensors
        else:
            return adcc, drop, coh, com, inc, dropindeletion

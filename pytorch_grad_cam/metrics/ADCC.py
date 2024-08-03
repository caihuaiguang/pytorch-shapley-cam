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

def coherency(saliency_map, explanation_map, arch, attr_method, out):
    if torch.cuda.is_available():
        explanation_map = explanation_map.cuda()
        arch = arch.cuda()

    class_idx = out.max(1)[1].item()
    saliency_map_B=attr_method(image=explanation_map, model=arch, classidx=class_idx)

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
def average_drop(image, explanation_map, arch, out, class_idx=None):

    with torch.no_grad():
        out_on_exp = FF.softmax(arch(explanation_map), dim=1)

    confidence_on_inp = out.max(1)[0].item()

    if class_idx is None:
        class_idx = out.max(1)[1].item()

    confidence_on_exp = out_on_exp[:,class_idx][0].item()

    return max(0.,confidence_on_inp-confidence_on_exp)/confidence_on_inp



class ADCC:
    def __init__(self):
        self.perturbation = multiply_tensor_with_cam

    def __call__(self, input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module,
                 cam_method,
                 return_visualization=False,
                 return_diff=True):

        if return_diff:
            with torch.no_grad():
                outputs = model(input_tensor)
                scores = [target(output).cpu().numpy()
                          for target, output in zip(targets, outputs)]
                scores = np.float32(scores)

        batch_size = input_tensor.size(0)
        perturbated_tensors = []
        for i in range(batch_size):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(),
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

        if return_diff:
            result = scores_after_imputation - scores
        else:
            result = scores_after_imputation

        if return_visualization:
            return result, perturbated_tensors
        else:
            return result
        
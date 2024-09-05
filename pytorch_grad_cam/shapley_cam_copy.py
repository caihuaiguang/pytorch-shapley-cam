from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import cv2

from pytorch_grad_cam.activations_and_gradients_tensor import ActivationsAndGradientstensor
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

class ShapleyCAM_copy(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            ShapleyCAM_copy,
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
            self.score = [target(output).cpu().detach().numpy() for target, output in zip(targets, outputs)]
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


    def svd_denoising(self, matrix, k=1):
        """
        使用 SVD 进行去噪。
        
        参数:
        - matrix: 输入的 7x7 矩阵
        - k: 保留的奇异值的数量

        返回:
        - denoised_matrix: 去噪后的矩阵
        """
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        S[k:] = 0  # 将奇异值中小于 k 的置为 0
        S = np.diag(S)
        denoised_matrix = np.dot(U, np.dot(S, Vt))
        return denoised_matrix


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
        # self.activations_and_grads.release()
        # print(grads)
        # print(activations)

        hvp = torch.autograd.grad(
            outputs=grads,
            inputs=activations,
            grad_outputs=activations,
            retain_graph=False
        )[0]
        if self.activations_and_grads.reshape_transform is not None:
            # print("not None")
            hvp = self.activations_and_grads.reshape_transform(hvp)
            activations = self.activations_and_grads.reshape_transform(activations)
            grads = self.activations_and_grads.reshape_transform(grads)
        # print(torch.norm(hvp))
        weight = (grads - 0.5*hvp).cpu().detach().numpy()
        # weight = (grads).cpu().detach().numpy()
        # weight_flatten = torch.sum(weight * activations, dim=(2, 3))
        # weight = weight_flatten
        # print(we torch.softmax(weight, dim=-1)
        # weight = grads.cpu().detach().numpy()
        activations = activations.cpu().detach().numpy()
        grads = grads.cpu().detach().numpy()
        # weight = np.exp(weight*activations/activations.shape[3])
        # weight /= np.sum(weight, axis=1, keepdims=True)

        # 对 A 的每ight)
        # weight -= weight.mean()
        # weight =个 7x7 矩阵进行 SVD 去噪
        # denoised_weight = np.empty_like(weight)
        # for i in range(weight.shape[1]):
        #     denoised_weight[0, i] = self.svd_denoising(weight[0, i], 1)
        # weight = denoised_weight
        # weight_1 = np.mean(grads, axis=(2, 3),keepdims=True)
        # weight_2 = grads
        # weight = 0.9*weight_1+0.1*weight_2
        # 创建 3x3 的平滑滤波器（卷积核）
        # kernel = np.ones((3, 3)) / 9
        # # 设置高斯滤波器的标准差 (sigma)
        # # sigma = 1.0

        # # # 对 weight 的最后两维进行高斯滤波
        # # smoothed_weight = np.empty_like(weight)
        # # for i in range(weight.shape[1]):
        # #     smoothed_weight[0, i] = gaussian_filter(weight[0, i], sigma=sigma)

        # # 对 weight 的最后两维进行卷积操作
        # # 使用 'same' 模式保持卷积后输出的大小与输入相同
        # smoothed_weight = np.empty_like(weight)
        # for i in range(weight.shape[1]):
        #     smoothed_weight[0, i] = convolve2d(weight[0, i], kernel, mode='same', boundary='symm')
        # weight = smoothed_weight
        # weight = get_2d_projection(weight)
        # smoothed_weight = np.empty_like(weight)

        # # 对 weight 的最后两维进行中值滤波
        # for i in range(weight.shape[1]):
        #     # OpenCV 的 medianBlur 需要输入是单通道的 2D 矩阵，因此需要从 weight 中提取
        #     # 注意 kernel size 需要为奇数，这里使用 3x3
        #     smoothed_weight[0, i] = cv2.medianBlur(weight[0, i].astype(np.float32), ksize=3)
        # weight = smoothed_weight

        # weight = weight * activations
        # weight = weight[:, :, None, None] * grads
        # is_flatten = True
        is_flatten = False

        # k = 3
        # w_mean = np.mean(weight, axis=(2, 3), keepdims=True)
        # # weight = (weight - w_mean)/20 + w_mean 
        # w_std = np.std(weight, axis=(2, 3), keepdims=True)
        # # weight = (weight-w_mean)/(w_std+1e-9)
        # # # weight = np.clip(weight, -k, k)
        # # # weight = weight * w_std + w_mean
        # # weight = weight*np.abs(w_mean)/k + w_mean
        # # weight = np.clip(weight, -k*w_std+w_mean, k*w_std+w_mean)
        # scale_factor = np.minimum(1, np.abs(w_mean)/(k*(w_std+1e-9)))
        # weight = w_mean + scale_factor * (weight-w_mean)


        # # print(w_std/w_mean)
        # if (np.mean(w_std)>1e-6):
        #     scale_factor = np.minimum(1, 1/3*np.abs(w_mean)/w_std)
        #     weight = w_mean + scale_factor * (weight-w_mean)
            # 使用 tanh 函数对权重进行平滑处理
            # scaled_weight = np.tanh((weight-w_mean)/w_std)
            # scaled_weight = np.clip((weight-w_mean)/w_std, -1, 1)
            # scaled_weight = 0.1 * (weight-w_mean)/w_std
            # 恢复原始均值并缩放 tanh 输出

        # weight = np.ones_like(weight)
        # weight_flatten = np.sum(grads*activations, axis=(2, 3)) 
        # weight_flatten = np.exp(weight_flatten)
        # weight_flatten /= weight_flatten.sum()
        # weight_flatten = np.broadcast_to(weight_flatten[:, :, None, None], grads.shape)
        # weight = weight * weight_flatten
        
        # weight = np.mean(weight, axis=1, keepdims=True)
        # weight = np.broadcast_to(weight, grads.shape)
        # activations = np.mean(activations, axis=1, keepdims=True)
        # activations = np.broadcast_to(activations, grads.shape)
        

        # # 假设 weight 和 activations 是形状为 (1, n, 7, 7) 的 numpy 数组
        # mask = (weight * activations > 0)
        # # shapley_value = np.maximum(0, weight*activations)
        # # weight_flatten = self.compute_normalized_product(weight*activations*mask, activations*mask)[:, :, None, None]
        # weight_flatten = np.sum(weight * mask, axis=(2, 3), keepdims=True) / np.sum(mask, axis=(2, 3), keepdims=True)
        # weight_flatten = np.broadcast_to(weight_flatten, weight.shape)
        
        # weight[mask] = weight_flatten[mask]
        # weight = -np.ones_like(grads)
        # # 分别创建 activation 正负的掩码
        # pos_activation_mask = (activations > 0) & mask  # activation > 0 且 mask 为 True 的位置
        # neg_activation_mask = (activations < 0) & mask  # activation < 0 且 mask 为 True 的位置

        # # 计算正 activation 对应的 weight 的均值
        # pos_weight_mean = np.sum(weight * pos_activation_mask, axis=(2, 3), keepdims=True) / np.sum(pos_activation_mask, axis=(2, 3), keepdims=True)

        # # 计算负 activation 对应的 weight 的均值
        # neg_weight_mean = np.sum(weight * neg_activation_mask, axis=(2, 3), keepdims=True) / np.sum(neg_activation_mask, axis=(2, 3), keepdims=True)


        # # 广播 pos_weight_mean 和 neg_weight_mean 到相同的形状
        # pos_weight_mean_broadcast = np.broadcast_to(pos_weight_mean, weight.shape)
        # neg_weight_mean_broadcast = np.broadcast_to(neg_weight_mean, weight.shape)

        # # 使用布尔索引进行替换
        # weight[pos_activation_mask] = pos_weight_mean_broadcast[pos_activation_mask]
        # weight[neg_activation_mask] = neg_weight_mean_broadcast[neg_activation_mask]


        # 2D image
        if len(activations.shape) == 4:
            if is_flatten is True:
                # mask = (weight > 0)
                # mask = (activations > 0)
                # shapley_value = weight*activations
                # l_a = np.maximum(activations, 0)
                # shapley_value = weight*l_a
                # weight = np.maximum(0,weight)
                # weight_min = np.min(weight, axis=1, keepdims=True)
                # activations_min = np.min(activations, axis=1, keepdims=True)
                # weight -= weight_min
                # l_a = activations - activations_min
                # shapley_value = weight * l_a
                # shapley_value -= shapley_value.mean()
                # shapley_value += self.score[0]/shapley_value.size
                # weight = self.compute_normalized_product(shapley_value, l_a)
                # weight = self.compute_normalized_product(weight*activations*mask, activations*mask)
                # weight = np.sum(grads*activations, axis=(2, 3))/(1e-14+np.sum(activations, axis=(2, 3)))
                # weight = np.sum(grads*activations, axis=(2, 3))
                # weight = np.maximum(0, weight)
                # weight /= weight.sum()
                # shapley_value = np.maximum(0, np.sum(weight*activations, axis=1))
                # shapley_value = np.maximum(0, weight*activations)
                # # shapley_value = np.mean(weight*activations, axis=1)
                # shapley_value = np.broadcast_to(shapley_value, weight.shape)
                # mask = (weight*activations > 0)
                # print(shapley_value.shape)
                # weight = self.compute_normalized_product(shapley_value, activations)
                # print("grads",np.mean(weight ** 2, axis=(2,3)))
                # weight = self.compute_normalized_product(weight*activations, activations)
                # weight = np.sum(weight*mask, axis=(2, 3))/np.sum(mask, axis=(2, 3))
                # weight = np.mean(weight*mask, axis=(2, 3))
                # weight = np.sum(weight, axis=(2, 3))
                # weight -= weight.mean()
                # weight = np.exp(weight)
                # weight /= weight.sum()
                k = 1
                weight = np.mean(weight, axis=(2, 3))
            return weight, activations, is_flatten
        
        # 3D image
        elif len(activations.shape) == 5:
            if is_flatten is True:
                # weight = self.compute_normalized_product(weight*activations, activations)
                weight = np.mean(weight, axis=(2, 3, 4))
            return weight, activations, is_flatten
        
        else:
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")


    def minmax_normalize(self, weighted_activations, eps=1e-14):
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

        # 3D conv
        elif len(activations.shape) == 5:
            if is_flatten is True:
                weighted_activations = weights[:, :, None, None, None] * activations
            else:
                weighted_activations = weights * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")
        # weights_min = weights.min(axis=(2, 3), keepdims=True)
        # weights_max = weights.max(axis=(2, 3), keepdims=True)
        # activations_min = activations.min(axis=(2, 3), keepdims=True)
        # activations_max = activations.max(axis=(2, 3), keepdims=True)

        # # Min-Max 归一化
        # weights_norm_flatten = (weights - weights_min) / (weights_max - weights_min)
        # activations_norm_flatten = (activations - activations_min) / (activations_max - activations_min)

        # # 归一化后矩阵相乘
        # result_flatten_norm = weights_norm_flatten * activations_norm_flatten

        # # 反归一化
        # weighted_activations = result_flatten_norm * (activations_max - activations_min) + activations_min


        weighted_activations = np.maximum(weighted_activations, 0)
        # weighted_activations = self.minmax_normalize(weighted_activations)
        
        # weight_flatten = np.sum(weights * activations, axis=(2, 3))
        # weight_flatten = np.exp(weight_flatten)
        # weight_flatten /= weight_flatten.sum()
        # weighted_activations = weight_flatten[:, :, None, None] * weighted_activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
            # cam = weighted_activations[:,0,:,:]
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
            # print(self.score[0])
            # cam_mean = np.mean(cam)
            # cam -= cam_mean
            # cam += self.score[0]/cam.size
            
            cam = np.maximum(cam, 0)
            
            # cam = np.sqrt(cam)
            # cam = np.tanh(2 * cam)
            # cam = np.maximum(cam, 0)
            # cam = np.where(cam > cam_mean, cam, 0)

            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

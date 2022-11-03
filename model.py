# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "SRResNet", "Discriminator",
    "srresnet_x4", "discriminator", "contextual_bilateral_loss_for_rgb", "contextual_bilateral_loss_for_vgg",
]


class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            num_rcb: int,
            upscale_factor: int
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input image size must equal 96
        assert x.shape[2] == 96 and x.shape[3] == 96, "Image shape must equal 96x96"

        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out


def _extract_image_patches(x: Tensor, patch_size: int, stride: int) -> Tensor:
    """Implementation TensorFlow extract_image_patches() version"""
    batch_size, channels, height, width = x.shape
    patches = (
        x.unfold(2, patch_size, stride)
        .unfold(3, patch_size, stride)
        .reshape(batch_size, channels, -1, patch_size, patch_size)
        .permute(0, 1, 3, 4, 2)
    )

    return patches


def _compute_cosine_distance(x: Tensor, y: Tensor) -> Tensor:
    batch_size, channels, _, _ = x.size()

    # mean shifting by channel-wise mean of `y`.
    y_mean = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mean
    y_centered = y - y_mean

    # L2 normalization
    x_normalized = F_torch.normalize(x_centered, p=2, dim=1)
    y_normalized = F_torch.normalize(y_centered, p=2, dim=1)

    # Channel-wise vectorization
    x_normalized = x_normalized.reshape(batch_size, channels, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(batch_size, channels, -1)  # (N, C, H*W)

    # cosine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

    # convert to distance
    distance = 1 - cosine_sim

    return distance


def _compute_mae_distance(x: torch.Tensor, y: torch.Tensor) -> Tensor:
    batch_size, channels, height, width = x.size()

    x_vec = x.view(batch_size, channels, -1)
    y_vec = y.view(batch_size, channels, -1)

    distance = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    distance = distance.sum(dim=1).abs()
    distance = distance.transpose(1, 2).reshape(batch_size, height * width, height * width)
    distance = distance.clamp(min=0.)

    return distance


def _compute_mse_distance(x: Tensor, y: Tensor) -> Tensor:
    batch_size, channels, height, width = x.size()

    x_vec = x.view(batch_size, channels, -1)
    y_vec = y.view(batch_size, channels, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)
    batch_size, HW = y_s.shape

    A = y_vec.transpose(1, 2) @ x_vec  # N x(HW) x (HW)
    distance = y_s.unsqueeze(dim=2) - 2 * A + x_s.unsqueeze(dim=1)
    distance = distance.transpose(1, 2).reshape(batch_size, height * width, height * width)
    distance = distance.clamp(min=0.)

    return distance


def _compute_relative_distance(distance: Tensor) -> Tensor:
    dist_min, _ = torch.min(distance, dim=2, keepdim=True)
    relative_distance = distance / (dist_min + 1e-5)

    return relative_distance


def _compute_contextual(relative_distance: Tensor, bandwidth: float) -> Tensor:
    # This code easy OOM
    # w = torch.exp((1 - relative_distance) / bandwidth)  # Eq(3)
    # contextual = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)

    # This code is safe
    contextual = F_torch.softmax((1 - relative_distance) / bandwidth, dim=2)

    return contextual


def _compute_meshgrid(shape: Tensor) -> Tensor:
    batch_size, _, height, width = shape

    rows = torch.arange(0, height, dtype=torch.float32) / (height + 1)
    cols = torch.arange(0, width, dtype=torch.float32) / (width + 1)

    feature_grid = torch.meshgrid(rows, cols, indexing="ij")
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(batch_size)], dim=0)

    return feature_grid


def _compute_contextual_bilateral_loss(x: Tensor, y: Tensor, weight_spatial: float, bandwidth: float) -> Tensor:
    device = x.device
    # Calculate two image spatial loss
    grid = _compute_meshgrid(x.shape).to(device)
    distance = _compute_mse_distance(grid, grid)
    relative_distance = _compute_relative_distance(distance)
    contextual_spatial = _compute_contextual(relative_distance, bandwidth)

    # Calculate feature loss
    # Calculate two image distance
    distance = _compute_cosine_distance(x, y)
    # Compute relative distance
    relative_distance = _compute_relative_distance(distance)
    # Calculate two image contextual loss
    contextual_feature = _compute_contextual(relative_distance, bandwidth)

    # Combine loss
    cx_combine = (1. - weight_spatial) * contextual_feature + weight_spatial * contextual_spatial
    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
    cx = k_max_NC.mean(dim=1)
    loss = torch.mean(-torch.log(cx + 1e-5))

    return loss


class _ContextualBilateralLoss(nn.Module):
    """Creates a criterion that measures the contextual loss"""

    def __init__(
            self,
            feature_model_extractor_nodes: list = None,
            feature_model_normalize_mean: list = None,
            feature_model_normalize_std: list = None,
            patch_size: int = 5,
            stride: int = 1,
            weight_spatial: float = 0.1,
            bandwidth: float = 1.0,
            loss_type: str = "rgb",
    ) -> None:
        super(_ContextualBilateralLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.weight_spatial = weight_spatial
        self.bandwidth = bandwidth
        self.loss_type = loss_type

        if loss_type == "vgg":
            # Get the name of the specified feature extraction node
            self.feature_model_extractor_nodes = feature_model_extractor_nodes
            # Load the VGG19 model trained on the ImageNet dataset.
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
            self.feature_extractor = create_feature_extractor(model, feature_model_extractor_nodes)
            # set to validation mode
            self.feature_extractor.eval()

            # The preprocessing method of the input data.
            # This is the VGG model preprocessing method of the ImageNet dataset.
            self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

            # Freeze model parameters.
            for model_parameters in self.feature_extractor.parameters():
                model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> list[Tensor] or list[Tensor, Tensor, Tensor]:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"

        device = sr_tensor.device

        losses = []
        if self.loss_type == "vgg":
            # Standardized operations
            sr_tensor = self.normalize(sr_tensor)
            gt_tensor = self.normalize(gt_tensor)

            # VGG19 conv3_4 feature extraction
            sr_feature = self.feature_extractor(sr_tensor)
            gt_feature = self.feature_extractor(gt_tensor)

            for i in range(len(self.feature_model_extractor_nodes)):
                losses.append(_compute_contextual_bilateral_loss(sr_feature[self.feature_model_extractor_nodes[i]],
                                                                 gt_feature[self.feature_model_extractor_nodes[i]],
                                                                 self.weight_spatial,
                                                                 self.bandwidth))

        elif self.loss_type == "rgb":
            sr_patches = _extract_image_patches(sr_tensor, self.patch_size, self.stride)
            gt_patches = _extract_image_patches(gt_tensor, self.patch_size, self.stride)
            _, _, h_patch, w_patch, n_patches = sr_patches.shape

            sr_tensor = sr_patches.reshape(sr_tensor.shape[0], -1, h_patch, w_patch)
            gt_tensor = gt_patches.reshape(sr_tensor.shape[0], -1, h_patch, w_patch)

            losses.append(_compute_contextual_bilateral_loss(sr_tensor,
                                                             gt_tensor,
                                                             self.weight_spatial,
                                                             self.bandwidth))

        else:
            raise ValueError("`loss_type` only supported `vgg` or `rgb`")

        losses = torch.Tensor([losses]).to(device=device)

        return losses


def srresnet_x4(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale_factor=4, **kwargs)

    return model


def discriminator() -> Discriminator:
    model = Discriminator()

    return model


def contextual_bilateral_loss_for_rgb(**kwargs) -> _ContextualBilateralLoss:
    contextual_loss = _ContextualBilateralLoss(loss_type="rgb", **kwargs)

    return contextual_loss


def contextual_bilateral_loss_for_vgg(**kwargs) -> _ContextualBilateralLoss:
    contextual_loss = _ContextualBilateralLoss(loss_type="vgg", **kwargs)

    return contextual_loss

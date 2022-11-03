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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
d_arch_name = "discriminator"
g_arch_name = "srresnet_x4"
# Model arch config
in_channels = 3
out_channels = 3
channels = 64
num_rcb = 16
# Test upscale factor
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "SRGAN_CoBi_x4-DIV2K"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/SR_RAW/SRGAN_CoBi/train/GT"
    train_lr_images_dir = f"./data/SR_RAW/SRGAN_CoBi/train/LRbicx{upscale_factor}"

    test_gt_images_dir = f"./data/SR_RAW/SRGAN_CoBi/test/GT"
    test_lr_images_dir = f"./data/SR_RAW/SRGAN_CoBi/test/LRbicx{upscale_factor}"

    gt_image_size = 96
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_d_model_weights_path = f""
    pretrained_g_model_weights_path = f"./results/SRResNet_CoBi_x4-DIV2K/g_last.pth.tar"

    # Incremental training and migration training
    resume_d_model_weights_path = f""
    resume_g_model_weights_path = f""

    # Total num epochs (200,000 iters)
    epochs = 303

    # Loss function weight
    cobi_vgg_weight = [0.5, 0.5, 0.5]
    cobi_rgb_weight = [0.8]
    adversarial_weight = [1e-3]

    # Contextual loss parameter configuration
    feature_model_extractor_nodes = ["features.2", "features.7", "features.12"]
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]
    patch_size = 10  # upscale factor = 4, upscale factor = 2, use 5, upscale factor = 8, use 15,
    stride = 1
    weight_spatial = 0.5
    bandwidth = 0.1

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # Dynamically adjust the learning rate policy [100,000 | 200,000]
    lr_scheduler_step_size = epochs // 2
    lr_scheduler_gamma = 0.1

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 1

if mode == "test":
    # Test data address
    gt_dir = f"./data/SR_RAW/SRGAN_CoBi/test/GT"
    sr_dir = f"./results/test/{exp_name}"
    lr_dir = f"./data/SR_RAW/SRGAN_CoBi/test/LRbicx{upscale_factor}"

    g_model_weights_path = f"./samples/SRGAN_CoBi_x4-DIV2K/g_epoch_5.pth.tar"

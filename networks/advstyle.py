
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import kornia.enhance as diff_transforms

from PIL import Image


class OpsMagnitude(torch.nn.Module):
    """
    OpsMagnitude is a neural network module designed for estimating magnitudes of operations (n_ops) in an input image.
    It consists of a convolutional backbone followed by a linear block for magnitude estimation.

    Args:
        n_ops (int): Number of operations to estimate.

    Attributes:
        backbone (torch.nn.Sequential): Convolutional backbone for feature extraction.
        linear_block (torch.nn.Sequential): Linear block for magnitude estimation.

    Methods:
        forward(x): Forward pass through the OpsMagnitude network.

    Example:
        # Instantiate OpsMagnitude with 10 operations to estimate
        ops_magnitude = OpsMagnituGitHub README Templatede(n_ops=10)

        # Forward pass with an input tensor
        input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor
        output_magnitudes = ops_magnitude(input_tensor)
    """
    def __init__(self, n_ops) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1)
        )

        self.linear_block = torch.nn.Sequential(
            torch.nn.Linear(128, 128, bias=False),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, n_ops, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the OpsMagnitude network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor containing estimated magnitudes for each operation.
        """
        # Forward through the backbone
        x = self.backbone(x)

        # Squeeze spatial dimensions and pass through the linear block
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_block(x + 0.1*torch.randn_like(x))
        return x


class DiffAugmentationCNN(torch.nn.Module):
    """
    DiffAugmentationCNN is a neural network module that performs differential augmentation on input images.
    It utilizes a magnitude predictor to estimate magnitudes for different augmentation operations and applies
    these operations to the input images.

    Attributes:
        ops (list): List of tuples, each containing an augmentation function and its corresponding magnitude range.
        magnitude_predictor (OpsMagnitude): Magnitude predictor for estimating augmentation magnitudes.

    Methods:
        forward(img): Forward pass through the DiffAugmentationCNN network.

    Example:
        # Instantiate DiffAugmentationCNN
        diff_augmentation_cnn = DiffAugmentationCNN()

        # Forward pass with an input image tensor
        input_image = torch.randn(1, 3, 64, 64)  # Example input image tensor
        augmented_image = diff_augmentation_cnn(input_image)
    """
    def __init__(self) -> None:
        super().__init__()

        # List of augmentation operations with their magnitude ranges
        self.ops = [
            (diff_transforms.adjust_gamma, (0.6, 1.4)),
            (diff_transforms.adjust_hue, (-math.pi/12, math.pi/12)),
            (diff_transforms.adjust_saturation, (0.6, 1.4)),
            (diff_transforms.sharpness, (0.1, 0.9)),
            (lambda x, v: diff_transforms.adjust_brightness(x, v, clip_output=True), (-0.4, 0.4)),
            (lambda x, v: diff_transforms.adjust_contrast(x, v, clip_output=True), (0.4, 1.6)),
        ]

        # Magnitude predictor for estimating augmentation magnitudes
        self.magnitude_predictor = OpsMagnitude(len(self.ops))

    def forward(self, img):
        """
        Forward pass through the DiffAugmentationCNN network.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Augmented image tensor after applying differential augmentation.
        """
        # Predict augmentation magnitudes
        mag = self.magnitude_predictor(img) # batch x n_ops

        # Initialize output image
        out_img = img

        # Apply differential augmentation based on predicted magnitudes
        for iop, op in enumerate(self.ops):
            op_fn, (min_v, max_v) = op
            v = min_v + (max_v - min_v) * mag[:, iop]
            out_img = op_fn(out_img, v)

        return out_img
    

class LearnableDataAugmentationDINO(nn.Module):
    """
    LearnableDataAugmentationDINO is a neural network module for applying learnable data augmentation in DINO.

    Attributes:
        global_crops_scale (float): Scale factor for the global crops.
        local_crops_scale (float): Scale factor for the local crops.
        local_crops_number (int): Number of local crops.
        image_size (int): Size of the input images.
        embed_dim (int): Dimension of the embedding space.

    Methods:
        forward(imgs): Forward pass through the LearnableDataAugmentationDINO network.

    Example:
        # Instantiate LearnableDataAugmentationDINO
        learnable_augmentation = LearnableDataAugmentationDINO(
            global_crops_scale=0.08, local_crops_scale=0.15, local_crops_number=8, image_size=224, embed_dim=256
        )

        # Forward pass with a batch of input images
        input_images = torch.randn(8, 3, 224, 224)  # Example batch of input images
        augmented_crops = learnable_augmentation(input_images)
    """
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size, embed_dim) -> None:
        super().__init__()

        # Transformation for the first global crop
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        # Random augmentation for first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.Resampling.BILINEAR),
            flip_and_color_jitter,
        ])

        # Learnable augmentation for the second global crop
        self.global_transfo2 = DiffAugmentationCNN()

        # Transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(int(0.43*image_size), scale=local_crops_scale, interpolation=Image.Resampling.BILINEAR),
            flip_and_color_jitter,
        ])

    def forward(self, imgs):
        """
        Forward pass through the LearnableDataAugmentationDINO network.

        Args:
            imgs (torch.Tensor): Batch of input images.

        Returns:
            list: List of augmented crops including global and local crops.
        """
        crops = []
        crops.append(self.global_transfo1(imgs))
        crops.append(self.global_transfo2(imgs))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(imgs))
        
        return crops

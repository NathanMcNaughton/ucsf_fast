import os
import yaml
import torch
import torch.nn as nn


def load_config():
    if os.path.exists('config_local.yaml'):
        with open('config_local.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print('Using config_local.yaml')
    else:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print('Using config.yaml')
    return config


class DICELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DICELoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Compute the negative DICE score between predicted and target segmentation masks.

        Args:
            pred (torch.Tensor): Predicted segmentation mask of shape (batch_size, num_classes, height, width).
            target (torch.Tensor): Target segmentation mask of shape (batch_size, num_classes, height, width).

        Returns:
            torch.Tensor: Negative DICE score averaged across all classes and batch instances.
        """
        num = target.shape[0]
        pred = pred.reshape(num, -1)
        target = target.reshape(num, -1)

        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)
        dice = (2. * intersection) / (union + 1e-8)
        dice = dice.sum() / num

        dice_loss = 1 - dice

        return dice_loss


class DICELossV0(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DICELoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Compute the negative DICE score between predicted and target segmentation masks.

        Args:
            pred (torch.Tensor): Predicted segmentation mask of shape (batch_size, num_classes, height, width).
            target (torch.Tensor): Target segmentation mask of shape (batch_size, num_classes, height, width).

        Returns:
            torch.Tensor: Negative DICE score averaged across all classes and batch instances.
        """
        # Flatten the tensors along the batch and spatial dimensions
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        # Compute the intersection and union for each class
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        # Compute the DICE score for each class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average the DICE score across all classes and batch instances
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss
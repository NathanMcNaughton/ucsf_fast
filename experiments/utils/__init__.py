import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from common.models.small.unet_toy import ToyUNet
from common.models.large.pretrained_backbone_unet import get_pretrained_backbone_unet


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


def visualize_segmentation_overlay(image_tensor, mask_tensor_true, mask_tensor_pred, alpha=1.0, save_path=None):
    """
    Visualizes the segmentation overlay on top of the original image.

    Parameters:
        image_tensor (torch.Tensor): The original image tensor of shape [1, H, W].
        mask_tensor_true (torch.Tensor): The true mask tensor of shape [1, H, W], with 1s for the object.
        mask_tensor_pred (torch.Tensor, optional): The predicted mask tensor of shape [1, H, W], with 1s for the object.
        alpha (float): Opacity level of the mask overlay. Default is 0.3.
        save_path (str, optional): Path to save the visualization. If None, the image is not saved.
    """
    mask_tensor_pred = torch.sigmoid(mask_tensor_pred)

    # Ensure the orignal image and the masks are numpy arrays
    image = image_tensor.detach().squeeze().cpu().numpy()
    mask_true = mask_tensor_true.detach().squeeze().cpu().numpy() * alpha
    mask_pred = mask_tensor_pred.detach().squeeze().cpu().numpy() * alpha

    # Normalize the image for display
    image_normalized = (image - image.min()) / (image.max() - image.min())

    # Rescale original image pixels into the interval [0, 1 - p]
    image_rescaled_true = image_normalized * (1 - mask_true)
    image_rescaled_pred = image_normalized * (1 - mask_pred)

    # Create an RGB version of the image
    image_rgb_true = np.stack([image_rescaled_true]*3, axis=-1)
    image_rgb_pred = np.stack([image_rescaled_pred] * 3, axis=-1)

    # Add the mask values to the red channel
    image_rgb_true[..., 0] += mask_true
    image_rgb_pred[..., 0] +=  mask_pred

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Original image
    axes[0].imshow(image_normalized, cmap='gray', interpolation='nearest')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # True mask overlay
    axes[1].imshow(image_rgb_true, interpolation='nearest')
    axes[1].set_title('True Segmentation Overlay')
    axes[1].axis('off')

    # Predicted mask overlay
    axes[2].imshow(image_rgb_pred, interpolation='nearest')
    axes[2].set_title('Predicted Segmentation Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + '.png', bbox_inches='tight')
        plt.savefig(save_path + '.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def visualize_fixed_set(model, images, masks, epoch, batch, save_dir):
    """
    Visualizes the segmentation mask overlay on the original image for a fixed set of images and masks.
    Parameters:
        model: The trained model.
        images: The fixed set of images.
        masks: The fixed set of masks.
        epoch: The current epoch.
        batch: The current batch.
        save_dir: The directory to save the visualizations.
    """
    model.eval()
    with torch.no_grad():
        pred_masks = model(images)
    for i in range(len(images)):
        save_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{batch}_image_{i}')
        visualize_segmentation_overlay(image_tensor=images[i], mask_tensor_true=masks[i],
                                       mask_tensor_pred=pred_masks[i], save_path=save_path)
    model.train()


def pad_to_divisible_by_32(images, pad_value=0.0):
    # Calculate padding to make image height and width divisible by 32. This is required by the UNet model.
    _, _, height, width = images.shape
    pad_height = (32 - height % 32) % 32
    pad_width = (32 - width % 32) % 32

    padding_top = pad_height // 2
    padding_bottom = pad_height - padding_top
    padding_left = pad_width // 2
    padding_right = pad_width - padding_left

    # Pad images
    padded_images = nn.functional.pad(images, (padding_left, padding_right, padding_top, padding_bottom),
                                      mode='constant', value=pad_value)
    return padded_images


def get_model(model_name, in_channels=1, n_classes=1):
    """
    Get the model by name.
    Parameters:
        model_name: The name of the model.
        n_channels: The number of input channels.
        n_classes: The number of output classes.
    Returns:
        model: The model.
    """
    if model_name == 'toy_unet':
        model = ToyUNet(n_channels=in_channels, n_classes=n_classes)
    elif model_name == 'pretrained_unet':
        model = get_pretrained_backbone_unet(backbone_name='resnet18', in_channels=in_channels, n_classes=n_classes)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    return model


def get_optimizer(optimizer_name, model, learning_rate):
    """
    Get the optimizer by name.
    Parameters:
        optimizer_name: The name of the optimizer.
        model: The model.
        learning_rate: The learning rate.
    Returns:
        optimizer: The optimizer.
    """
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'Invalid optimizer name: {optimizer_name}')
    return optimizer


def test_all(loader, model, loss_fn):
    """
    Test the model on the entire dataset
    Parameters:
        loader: The dataloader for the test set.
        model: The trained model.
        loss_fn: The loss function.
    """
    num_batches = len(loader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (images, masks, free_fluid_labels) in enumerate(loader):
            # Pad the images and masks to make them divisible by 32
            images = pad_to_divisible_by_32(images, pad_value=-1.0)
            masks = pad_to_divisible_by_32(masks, pad_value=0.0)

            images, masks = images.cuda(), masks.cuda()
            pred = model(images)
            test_loss += loss_fn(pred, masks).item()
    test_loss /= num_batches

    return test_loss

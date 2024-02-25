import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from common.datasets import FASTDataset
from common.models.small.unet_toy import ToyUNet


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
    plt.show()


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
        save_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{batch}_image_{i}.png')
        visualize_segmentation_overlay(image_tensor=images[i], mask_tensor_true=masks[i],
                                       mask_tensor_pred=pred_masks[i], save_path=save_path)
    model.train()


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
        for batch, (images, masks) in enumerate(loader):
            images, masks = images.cuda(), masks.cuda()
            pred = model(images)
            test_loss += loss_fn(pred, masks).item()
    test_loss /= num_batches

    return test_loss


def main():
    learning_rate = 1e-3
    loss_fn = 'BCEWithLogitsLoss'
    loss_weight = 1  # 63.29 = 1/0.0158
    optimizer = 'SGD'

    n_total = 384
    n_train = 2
    n_test = n_total - n_train

    batch_size = 10
    n_epochs = 100
    n_batches = int((n_train / batch_size) * n_epochs)
    k = 10  # Visualize the model predictions on the fixed set every k epochs
    print(f'Total number of train batches: {n_batches}')

    if torch.cuda.is_available():
        print(f'CUDA available. Using GPU.')
    else:
        print('CUDA is not available. Try again with a GPU. Exiting...')
        exit(1)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Might change this to get the directory from the config file
    data_dir = '/scratch/users/austin.zane/ucsf_fast/data/pilot_labeling/AnnotationData/MorisonPouchMasks_1-23'
    # fig_dir = '/scratch/users/austin.zane/ucsf_fast/figures/segmentation_overlay/02_21_2024'
    logging_dir = '/scratch/users/austin.zane/ucsf_fast/logging'

    # WandB setup
    os.environ['WANDB_API_KEY'] = config['wandb_api_key']
    proj_name = '02_21_2024_ucsf_fast'
    wandb.init(project=proj_name, entity=config['wandb_entity'])

    wandb.config = {
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_train": n_train,
        "loss_fn": "BCEWithLogitsLoss",
        "loss_weight": "None",
        "optimizer": "SGD"
    }

    project_dir = os.path.join(logging_dir, proj_name)
    if not os.path.exists(project_dir): os.makedirs(project_dir)

    wandb_run_name = wandb.run.name
    run_dir = os.path.join(project_dir, wandb_run_name)
    if not os.path.exists(run_dir): os.makedirs(run_dir)

    fig_dir = os.path.join(run_dir, 'segmentation_overlays')
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)

    # Create the datasets and dataloaders
    img_dataset = FASTDataset(data_dir)

    dataset_train, dataset_test = torch.utils.data.random_split(img_dataset, [n_train, n_test])

    loader_train = DataLoader(dataset_train, batch_size=min(n_train, batch_size), shuffle=True, num_workers=2)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    # Select a fixed set of exactly three test images and masks for visualization during training
    fixed_images, fixed_masks = next(iter(loader_train))
    fixed_images, fixed_masks = fixed_images[:2].cuda(), fixed_masks[:2].cuda()

    model = ToyUNet(n_channels=1, n_classes=1)
    model.cuda()
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weight))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler =

    print('Training...')
    for epoch in range(n_epochs):
        model.train()

        for batch, (images, masks) in enumerate(loader_train):
            model.train()

            images, masks = images.cuda(), masks.cuda()

            # Compute prediction error
            pred = model(images)
            loss = criterion(pred, masks)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        test_loss = test_all(loader_test, model, criterion)
        print(f'[Epoch {epoch+1} of {n_epochs}] \t  Training loss: {loss.item()}. \t Test loss: {test_loss}.')
        wandb.log({'train_loss': loss.item(), 'test_loss': test_loss})

        if (epoch+1) % k == 0:
            visualize_fixed_set(model, fixed_images, fixed_masks, epoch, batch, fig_dir)

    print('Saving model...')

    print('Model saved. Exiting...')

    wandb.finish()


if __name__ == '__main__':
    main()

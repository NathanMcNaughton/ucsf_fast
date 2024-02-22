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


def visualize_segmentation_overlay(image_tensor, mask_tensor_true, mask_tensor_pred=None, alpha=0.4, save_path=None):
    """
    Visualizes the predicted and true segmentation mask overlays on the original image.

    Parameters:
        image_tensor (torch.Tensor): The original image tensor of shape [1, H, W].
        mask_tensor_true (torch.Tensor): The true mask tensor of shape [1, H, W], with 1s for the object.
        mask_tensor_pred (torch.Tensor, optional): The predicted mask tensor of shape [1, H, W], with 1s for the object.
        alpha (float): Opacity level of the mask overlay. Default is 0.3.
        save_path (str, optional): Path to save the visualization. If None, the image is not saved.
    """
    # Convert the image and mask tensors to numpy arrays
    image = image_tensor.detach().squeeze().cpu().numpy()  # Remove channel dimension and convert to numpy
    mask_true = mask_tensor_true.detach().squeeze().cpu().numpy()

    # Normalize the image for display
    image_normalized = (image - image.min()) / (image.max() - image.min())

    # Create an RGB version of the image
    image_rgb = np.stack([image_normalized]*3, axis=-1)

    # Create a mask overlay
    mask_overlay_true = np.zeros_like(image_rgb)
    mask_overlay_true[mask_true > 0.5] = [1, 0, 0]  # Set the mask to red

    # Overlay the mask on the image with the specified opacity
    overlayed_image_true = np.where(mask_overlay_true, (1-alpha)*image_rgb + alpha*mask_overlay_true, image_rgb)

    if mask_tensor_pred is not None:
        mask_pred = mask_tensor_pred.detach().squeeze().cpu().numpy()
        mask_pred_rescaled = (mask_pred - mask_pred.min()) / (mask_pred.max() - mask_pred.min())

        mask_overlay_pred = np.zeros_like(image_rgb)
        #print(f'RGB mask overlay dim: {mask_overlay_pred.shape}')
        #print(f'Predicted mask dim: {mask_pred_rescaled.shape}')
        #print(f'Predicted mask range: {mask_pred_rescaled.min()} to {mask_pred_rescaled.max()}')
        # mask_overlay_pred[mask_pred > 0.5] = [0, 1, 0]
        mask_overlay_pred[:, :, 0] = mask_pred_rescaled
        overlayed_image_pred = np.where(mask_overlay_pred, (1-alpha)*image_rgb + alpha*mask_overlay_pred, image_rgb)

    # Display the overlayed image
    plt.figure(figsize=(10, 10))

    # True mask
    plt.subplot(1, 2, 1)
    plt.imshow(overlayed_image_true, cmap='gray')
    plt.title('True mask')
    plt.axis('off')

    # Predicted mask
    plt.subplot(1, 2, 2)
    if mask_tensor_pred is not None:
        plt.imshow(overlayed_image_pred, cmap='gray')
        plt.title('Predicted mask')
    else:
        plt.imshow(image_rgb, cmap='gray')
        plt.title('Original image')
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        #print(f'Visualization saved to {save_path}')

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

    wandb_run_name = wandb.run.name
    project_dir = os.path.join(logging_dir, proj_name)

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    run_dir = os.path.join(project_dir, wandb_run_name)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    fig_dir = os.path.join(run_dir, 'segmentation_overlays')

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Create the datasets and dataloaders
    img_dataset = FASTDataset(data_dir)

    n_total = len(img_dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    batch_size = 10
    # batches_per_epoch = n_train // batch_size
    n_epochs = 5
    n_batches = int((n_train / batch_size) * n_epochs)
    k = 30  # Evaluate the model on the test set every k batches
    print(f'Total number of train batches: {n_batches}')

    dataset_train, dataset_test = torch.utils.data.random_split(img_dataset, [n_train, n_test])

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    # Select a fixed set of exactly three test images and masks for visualization during training
    fixed_images, fixed_masks = next(iter(loader_test))
    fixed_images, fixed_masks = fixed_images[:3].cuda(), fixed_masks[:3].cuda()

    model = ToyUNet(n_channels=1, n_classes=1)
    model.cuda()
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
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

        visualize_fixed_set(model, fixed_images, fixed_masks, epoch, batch, fig_dir)

    print('Saving model...')

    print('Model saved. Exiting...')

    wandb.finish()



if __name__ == '__main__':
    main()

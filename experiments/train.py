import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import numpy as np
import matplotlib.pyplot as plt

from common.datasets import FASTDataset
from common.models.small.unet_toy import ToyUNet


def visualize_segmentation_overlay(image_tensor, mask_tensor_true, mask_tensor_pred=None, alpha=0.3, save_path=None):
    """
    Visualizes the segmentation mask overlay on the original image.

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

    if mask_tensor_pred:
        mask_pred = mask_tensor_pred.detach().squeeze().cpu().numpy()
        mask_overlay_pred = np.zeros_like(image_rgb)
        mask_overlay_pred[mask_pred > 0.5] = [0, 1, 0]
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
    if mask_tensor_pred:
        plt.imshow(overlayed_image_pred, cmap='gray')
        plt.title('Predicted mask')
    else:
        plt.imshow(image_rgb, cmap='gray')
        plt.title('Original image')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Visualization saved to {save_path}')

    plt.show()


def test_all(loader, model, loss_fn):
    """
    Test the model on the entire dataset
    :param loader:
    :param model:
    :param loss_fn:
    :return:
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


    # wandb.init(project='02_15_2024_ucsf_fast', entity='john-doe')

    # Might change this to get the directory from the config file
    data_dir = '/scratch/users/austin.zane/ucsf_fast/data/pilot_labeling/AnnotationData/MorisonPouchMasks_1-23'

    # Create the datasets and dataloaders
    img_dataset = FASTDataset(data_dir)

    n_total = len(img_dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    batch_size = 10
    num_epochs = 5

    dataset_train, dataset_test = torch.utils.data.random_split(img_dataset, [n_train, n_test])

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    model = ToyUNet(n_channels=1, n_classes=1)
    model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # scheduler =

    print('Training...')
    for epoch in range(num_epochs):
        model.train()

        for batch, (images, masks) in enumerate(loader_train):
            images, masks = images.cuda(), masks.cuda()

            # Compute prediction error
            pred = model(images)
            loss = criterion(pred, masks)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                print(f'Epoch {epoch} of {num_epochs}, batch {batch} of {len(loader_train)}')
                print(f'Training loss: {loss.item()}')

                test_loss = test_all(loader_test, model, criterion)

                print(f'Test loss: {test_loss}')

    print('Saving model...')

    print('Model saved. Exiting...')



if __name__ == '__main__':
    main()

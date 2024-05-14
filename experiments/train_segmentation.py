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

import sys
# root_dir = '/accounts/campus/austin.zane/ucsf_fast'
# sys.path.append(root_dir)
# os.chdir(root_dir)

from experiments.utils import load_config, DICELoss
from common.datasets import FASTSegmentationDataset
from common.models.small.unet_toy import ToyUNet
from common.models.large.pretrained_backbone_unet import get_pretrained_backbone_unet

from common.logging import VanillaLogger

"""
# In the future, we will pass these as command line arguments
    model_name = 'pretrained_unet'
    learning_rate = 3e-4
    loss_fn = 'BCEWithLogitsLoss'
    # loss_fn = 'DICELoss'
    loss_weight = 10  # 63.29 = 1/0.0158
    optimizer = 'adam'

    prop_train = 0.75
    n_total = 500
    n_train = prop_train * n_total
    n_test = n_total - n_train

    batch_size = 10
    n_epochs = 10
    k = 1  # Visualize the model predictions on the fixed set every k epochs

    args = {
        'model': model_name,
        'learning_rate': learning_rate,
        'loss_fn': loss_fn,
        'loss_weight': loss_weight,
        'optimizer': optimizer,
        'n_train': n_train,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'k': k
    }

    n_batches = int((n_train / batch_size) * n_epochs)
    print(f'Total number of train batches: {n_batches}')
"""



parser = argparse.ArgumentParser(description='Training UNet')
parser.add_argument('--proj', default='unet_test', type=str, help='project name')
parser.add_argument('--n_total', default=2000, type=int, help='num. of total images')
parser.add_argument('--n_train', default=350, type=int, help='num. of training images')
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--k', default=1, type=int, help="log every k batches")


parser.add_argument('--model_name', metavar='ARCH', default='pretrained_unet')
# parser.add_argument('--pretrained', type=str, default=None, help='local path to pretrained model state dict (optional)')
# parser.add_argument('--width', default=None, type=int, help="architecture width parameter (optional)")
parser.add_argument('--loss_fn', default='BCEWithLogitsLoss', choices=['BCEWithLogitsLoss', 'DICELoss'],
                    type=str)
parser.add_argument('--loss_weight', default=10.0, type=float,
                    help='positive class weight for BCEWithLogitsLoss')

parser.add_argument('--optimizer', default="adam", type=str)
parser.add_argument('--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--scheduler', default="cosine", type=str, help='lr scheduler')

parser.add_argument('--n_epochs', default=2, type=int)
# for keeping the same LR sched across different samp sizes.
# parser.add_argument('--nbatches', default=None, type=int, help='Total num. batches to train for. If specified, overrides EPOCHS.')
# parser.add_argument('--batches_per_lr_step', default=390, type=int)

parser.add_argument('--momentum', default=0.0, type=float, help='momentum (0 or 0.9)')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')

parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
# parser.add_argument('--half', default=False, action='store_true', help='training with half precision')
# parser.add_argument('--fast', default=False, action='store_true', help='do not log more frequently in early stages')
parser.add_argument('--earlystop', default=False, action='store_true', help='stop when train loss < 0.01')

args = parser.parse_args()


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


def main():
    """
    # In the future, we will pass these as command line arguments
    model_name = 'pretrained_unet'
    learning_rate = 3e-4
    loss_fn = 'BCEWithLogitsLoss'
    # loss_fn = 'DICELoss'
    loss_weight = 10  # 63.29 = 1/0.0158
    optimizer = 'adam'

    prop_train = 0.75
    n_total = 500
    n_train = prop_train * n_total
    n_test = n_total - n_train

    batch_size = 10
    n_epochs = 10
    k = 1  # Visualize the model predictions on the fixed set every k epochs

    args = {
        'model': model_name,
        'learning_rate': learning_rate,
        'loss_fn': loss_fn,
        'loss_weight': loss_weight,
        'optimizer': optimizer,
        'n_train': n_train,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'k': k
    }
    """
    n_batches = int((args.n_train / args.batch_size) * args.n_epochs)
    print(f'Total number of train batches: {n_batches}')

    vis_images = [
        ('1.2.840.114340.3.8251050064157.3.20190421.114851.4961.6_137.jpg', -1),
        ('1.2.840.114340.3.48100021226225.3.20190816.202201.6283.4_39.jpg', -1),
        ('1.2.840.114340.3.8251050064157.3.20180802.150631.1897.6_152.jpg', -1),
        ('1.2.840.114340.3.8251050064157.3.20180425.145653.384.6_0.jpg', -1),
        ('1.2.840.114340.3.48100016190144.3.20200705.200229.5904.4_0.jpg', -1),
        ('1.2.840.114340.3.8251017179172.3.20141207.22903.11967.6_5.jpg', 1),
        ('1.2.840.114340.3.8251017179172.3.20150128.164707.13073.6_79.jpg', 1),
        ('1.2.840.114340.3.8251017179172.3.20140531.234055.7585.6_41.jpg', 1),
        ('1.2.840.114340.3.48100016190144.3.20201005.183539.6271.4_0.jpg', 1),
        ('1.2.840.114340.3.8251017179172.3.20150413.183009.15018.6_53.jpg', 1)
    ]

    vis_images = [img[0] for img in vis_images]

    # Temporary list of randomly-drawn vis images for grant proposal. Will remove later.
    # vis_images = [vis_images[0], vis_images[-1]]

    if torch.cuda.is_available():
        print(f'CUDA available. Using GPU.')
    else:
        print('CUDA is not available. Try again with a GPU. Exiting...')
        exit(1)

    config = load_config()

    # Change this to get the directory from the config file
    data_dir = config['data_dir']
    logging_dir = config['logging_dir']

    # WandB setup
    os.environ['WANDB_API_KEY'] = config['wandb_api_key']
    wandb.init(project=args.proj, entity=config['wandb_entity'])
    wandb.config.update(args)

    project_dir = os.path.join(logging_dir, args.proj)
    if not os.path.exists(project_dir): os.makedirs(project_dir)

    run_dir = os.path.join(project_dir, f"{wandb.run.name}-{wandb.run.id}")
    if not os.path.exists(run_dir): os.makedirs(run_dir)

    fig_dir = os.path.join(run_dir, 'segmentation_overlays')
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)

    # Create the datasets and dataloaders
    img_dataset = FASTSegmentationDataset(data_dir, excluded_files=vis_images)
    # img_dataset = FASTSegmentationDataset(data_dir)

    n_total = min(len(img_dataset), args.n_total)
    # n_train = int(prop_train * n_total)
    n_test = n_total - args.n_train

    dataset_train, dataset_test = torch.utils.data.random_split(img_dataset, [args.n_train, n_test])

    loader_train = DataLoader(dataset_train, batch_size=min(args.n_train, args.batch_size), shuffle=True, num_workers=1)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # Create a separate dataset for the visualization images
    vis_dataset = FASTSegmentationDataset(data_dir, included_files=vis_images)
    loader_vis = DataLoader(vis_dataset, batch_size=len(vis_dataset), shuffle=False, num_workers=1)

    # Get the fixed set of images and masks for visualization
    fixed_images, fixed_masks, fixed_free_fluid_labels = next(iter(loader_vis))

    # Pad the images and masks to make them divisible by 32
    fixed_images = pad_to_divisible_by_32(fixed_images, pad_value=-1.0)
    fixed_masks = pad_to_divisible_by_32(fixed_masks, pad_value=0.0)

    fixed_images, fixed_masks = fixed_images.cuda(), fixed_masks.cuda()

    print(f'You specified {len(vis_images)} visualization images. You ended up with {len(fixed_images)} images.')

    model = get_model(model_name=args.model_name, in_channels=1, n_classes=1)
    model.cuda()
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    # init logging
    logger = VanillaLogger(args, wandb, log_root=logging_dir, hash=True)

    if args.loss_fn == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.loss_weight))
    elif args.loss_fn == 'DICELoss':
        criterion = DICELoss()
    else:
        raise ValueError(f'Invalid loss function: {args.loss_fn}. Only BCEWithLogitsLoss and DICELoss are supported.')
    test_loss = None

    optimizer = get_optimizer(optimizer_name=args.optimizer, model=model, learning_rate=args.learning_rate)
    # scheduler =

    print('Training...')
    for epoch in range(args.n_epochs):
        model.train()

        for batch, (images, masks, free_fluid_labels) in enumerate(loader_train):
            # torch.save(
            #     {'images': images, 'masks': masks, 'free_fluid_labels': free_fluid_labels},
            #     '/scratch/users/austin.zane/ucsf_fast/data/labeled_fast_morison/debugging/training_loop_unpadded_data_check.pth'
            # ) ############ DELETE THIS LINE ############

            model.train()

            # Pad the images and masks to make them divisible by 32
            images = pad_to_divisible_by_32(images, pad_value=-1.0)
            masks = pad_to_divisible_by_32(masks, pad_value=0.0)

            # torch.save(
            #     {'images': images, 'masks': masks, 'free_fluid_labels': free_fluid_labels},
            #     '/scratch/users/austin.zane/ucsf_fast/data/labeled_fast_morison/debugging/training_loop_data_check.pth'
            # ) ############ DELETE THIS LINE ############

            # break ############ DELETE THIS LINE ############

            images, masks = images.cuda(), masks.cuda()

            # Compute prediction error
            pred = model(images)
            loss = criterion(pred, masks)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # break ############ DELETE THIS LINE ############

        test_loss = test_all(loader_test, model, criterion)
        print(f'[Epoch {epoch+1} of {args.n_epochs}] \t  Training loss: {loss.item()}. \t Test loss: {test_loss}.')
        # wandb.log({'train_loss': loss.item(), 'test_loss': test_loss})

        if (epoch+1) % args.k == 0:
            visualize_fixed_set(model, fixed_images, fixed_masks, epoch, batch, fig_dir)

            d = {'epoch': epoch,
                 'lr': args.learning_rate,
                 'n': args.n_train,
                 'train_loss': loss.item(),
                 'test_loss': test_loss}

            logger.log_scalars(d)
            logger.flush()

    ## Final logging
    print('Saving model...')
    logger.save_model(model)

    summary = {}
    summary.update({f'Final Test {args.k}': test_loss})
    summary.update({f'Final Train {args.k}': loss.item()})

    logger.log_summary(summary)
    logger.flush()

    print('Exiting...')

    # wandb.finish()


if __name__ == '__main__':
    main()

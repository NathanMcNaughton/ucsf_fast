import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

from sklearn.metrics import f1_score

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
from experiments.utils import get_optimizer, get_model, test_all, evaluate_and_save_outputs
from experiments.utils import visualize_segmentation_overlay, visualize_fixed_set, pad_to_divisible_by_32

from common.logging import VanillaLogger
from common import print_error


parser = argparse.ArgumentParser(description='Training Segmentation Model')
parser.add_argument('--proj', default='unet_test', type=str, help='project name')
parser.add_argument('--n_total', default=2000, type=int, help='num. of total images')
parser.add_argument('--n_train', default=350, type=int, help='num. of training images')
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--k', default=1, type=int, help="log every k epochs")


parser.add_argument('--model_name', metavar='ARCH', default='pretrained_unet')
# parser.add_argument('--pretrained', type=str, default=None, help='local path to pretrained model state dict (optional)')
# parser.add_argument('--width', default=None, type=int, help="architecture width parameter (optional)")
parser.add_argument('--loss_fn', default='BCEWithLogitsLoss', choices=['BCEWithLogitsLoss', 'DICELoss'],
                    type=str)
parser.add_argument('--loss_weight', default=10.0, type=float,
                    help='positive class weight for BCEWithLogitsLoss')

parser.add_argument('--optimizer', default="adam", type=str)
parser.add_argument('--learning_rate', default=0.0003, type=float, help='initial learning rate')
parser.add_argument('--scheduler', default="cosine", type=str, help='lr scheduler')

parser.add_argument('--n_epochs', default=2, type=int)
# for keeping the same LR sched across different samp sizes.
# parser.add_argument('--nbatches', default=None, type=int, help='Total num. batches to train for. If specified, overrides EPOCHS.')
# parser.add_argument('--batches_per_lr_step', default=390, type=int)

parser.add_argument('--momentum', default=0.0, type=float, help='momentum (0 or 0.9)')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')

parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
# parser.add_argument('--half', default=False, action='store_true', help='training with half precision')
# parser.add_argument('--fast', default=False, action='store_true', help='do not log more frequently in early stages')
parser.add_argument('--earlystop', default=False, action='store_true', help='stop when train loss < 0.01')
parser.add_argument('--model_saving', default=False, action='store_true', help='Save model after training')
parser.add_argument('--output_saving', default=False, action='store_true', help='Save outputs after training')


args = parser.parse_args()



def main():
    config = load_config()

    vis_images = config['VIS_IMAGES']

    if torch.cuda.is_available():
        print(f'CUDA available. Using GPU.')
    else:
        print('CUDA is not available. Try again with a GPU. Exiting...')
        print_error()
        exit(1)

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

    output_dir = os.path.join(config['output_dir'], args.proj, f"{wandb.run.name}-{wandb.run.id}")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Create the datasets and dataloaders
    dataset_train = FASTSegmentationDataset(data_dir, included_studies=config['TRAIN_STUDIES'])
    dataset_val = FASTSegmentationDataset(data_dir, included_studies=config['VAL_STUDIES'])
    dataset_test = FASTSegmentationDataset(data_dir, included_studies=config['TEST_STUDIES'])

    # n_total = min(len(img_dataset), args.n_total)
    # n_train = int(prop_train * n_total)
    # n_test = n_total - args.n_train

    # dataset_train, dataset_test = torch.utils.data.random_split(img_dataset, [args.n_train, n_test])

    loader_train = DataLoader(dataset_train, batch_size=min(len(dataset_train), args.batch_size), shuffle=True, num_workers=args.workers)
    loader_val = DataLoader(dataset_val, batch_size=min(len(dataset_val), args.batch_size), shuffle=True, num_workers=1)
    loader_test = DataLoader(dataset_test, batch_size=min(len(dataset_test), args.batch_size), shuffle=True, num_workers=1)

    # Create a separate dataset for the visualization images
    vis_dataset = FASTSegmentationDataset(data_dir, included_files=vis_images)
    loader_vis = DataLoader(vis_dataset, batch_size=len(vis_dataset), shuffle=False, num_workers=1)

    # Get the fixed set of images and masks for visualization
    fixed_images, fixed_masks, fixed_free_fluid_labels, _ = next(iter(loader_vis))

    # Pad the images and masks to make them divisible by 32
    fixed_images = pad_to_divisible_by_32(fixed_images, pad_value=-1.0)
    fixed_masks = pad_to_divisible_by_32(fixed_masks, pad_value=0.0)

    fixed_images, fixed_masks = fixed_images.cuda(), fixed_masks.cuda()
    print(f'Number of training images: {len(dataset_train)}')
    print(f'Number of validation images: {len(dataset_val)}')
    print(f'Number of visualization images: {len(vis_dataset)}')

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
        print_error()
        raise ValueError(f'Invalid loss function: {args.loss_fn}. Only BCEWithLogitsLoss and DICELoss are supported.')

    test_loss = None

    optimizer = get_optimizer(optimizer_name=args.optimizer, model=model, learning_rate=args.learning_rate)
    # scheduler =

    print('Training...')
    for epoch in range(args.n_epochs):
        model.train()

        for batch, (images, masks, free_fluid_labels, _) in enumerate(loader_train):
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
            pred = torch.sigmoid(pred)
            loss = criterion(pred, masks)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        # break ############ DELETE THIS LINE ############


        if (epoch+1) % args.k == 0:
            test_loss = test_all(loader_val, model, criterion)
            print(
                f'[Epoch {epoch + 1} of {args.n_epochs}] \t  Training loss: {loss.item()}. \t Test loss: {test_loss}.')
            # wandb.log({'train_loss': loss.item(), 'test_loss': test_loss})

            visualize_fixed_set(model, fixed_images, fixed_masks, epoch, batch, fig_dir, binarize=False, cutoff=0.5)

            d = {'epoch': epoch,
                 'lr': args.learning_rate,
                 'n': len(dataset_train),
                 'train_loss': loss.item(),
                 'test_loss': test_loss}

            logger.log_scalars(d)
            logger.flush()

    ## Final logging
    if args.model_saving:
        print('Saving model...')
        logger.save_model(model)

    summary = {}
    summary.update({f'Final Test {args.k}': test_loss})
    summary.update({f'Final Train {args.k}': loss.item()})

    logger.log_summary(summary)
    logger.flush()

    if args.output_saving:
        print('Evaluating on training set and saving outputs...')
        train_output_dir = os.path.join(output_dir, 'train')
        evaluate_and_save_outputs(model, loader_train, train_output_dir, device='cuda')

        print('Evaluating on test set and saving outputs...')
        test_output_dir = os.path.join(output_dir, 'test')
        evaluate_and_save_outputs(model, loader_test, test_output_dir, device='cuda')

    print('Exiting...')

    # wandb.finish()



if __name__ == '__main__':
    main()

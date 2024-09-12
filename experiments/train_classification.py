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

from experiments.utils import load_config
from common.datasets import FASTClassificationDataset, FASTSegmentationDataset
from experiments.utils import get_optimizer, get_model, test_all_classification, evaluate_and_save_class_outputs

from common.logging import VanillaLogger
from common import print_error


parser = argparse.ArgumentParser(description='Training Segmentation Model')
parser.add_argument('--proj', default='unet_test', type=str, help='project name')
parser.add_argument('--seg_proj', default='unet_test_05_18_24', type=str,
                    help='segmentation project name')
parser.add_argument('--seg_run', default='scarlet-water-11-2l4vc76j', type=str,
                    help='segmentation run name')
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--k', default=1, type=int, help="log every k epochs")
parser.add_argument('--cutoff', default=0.5, type=float, help='Threshold for binarizing output')

parser.add_argument('--model_name', default='preresnet18', type=str, help='model architecture')
# parser.add_argument('--pretrained', type=str, default=None, help='local path to pretrained model state dict (optional)')
# parser.add_argument('--width', default=None, type=int, help="architecture width parameter (optional)")
parser.add_argument('--loss_fn', default='BCEWithLogitsLoss', choices=['BCEWithLogitsLoss', 'mse'],
                    type=str)
parser.add_argument('--loss_weight', default=1.0, type=float,
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
parser.add_argument('--earlystop', default=False, action='store_true',
                    help='stop when train loss < 0.01')
parser.add_argument('--model_saving', default=False, action='store_true',
                    help='Save model after training')
parser.add_argument('--output_saving', default=True, action='store_true', help='Save outputs after training')


args = parser.parse_args()



def main():
    config = load_config()

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

    output_dir = os.path.join(config['output_dir'], args.proj, f"{wandb.run.name}-{wandb.run.id}")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    seg_dir = os.path.join(config['output_dir'], args.seg_proj, args.seg_run)

    class_output_dir = os.path.join(config['class_output_dir'], args.proj, f"{wandb.run.name}-{wandb.run.id}")
    if not os.path.exists(class_output_dir): os.makedirs(class_output_dir)

    dataset_train = FASTClassificationDataset(data_dir, pred_mask_dir=seg_dir, split='train')
    dataset_val = FASTClassificationDataset(data_dir, pred_mask_dir=seg_dir, split='val')
    # dataset_test = FASTClassificationDataset(data_dir, split='test')

    loader_train = DataLoader(dataset_train, batch_size=min(len(dataset_train), args.batch_size), shuffle=True, num_workers=args.workers)
    loader_val = DataLoader(dataset_val, batch_size=min(len(dataset_val), args.batch_size), shuffle=True, num_workers=1)
    # loader_test = DataLoader(dataset_test, batch_size=min(len(dataset_test), args.batch_size), shuffle=True, num_workers=1)

    print(f'Number of training images: {len(dataset_train)}')
    print(f'Number of validation images: {len(dataset_val)}')

    model = get_model(model_name=args.model_name, in_channels=1, n_classes=1)
    model.cuda()
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    # init logging
    logger = VanillaLogger(args, wandb, log_root=logging_dir, hash=True)

    if args.loss_fn == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.loss_weight))
    else:
        print_error()
        raise ValueError(f'Invalid loss function: {args.loss_fn}. Only BCEWithLogitsLoss is supported at the moment.')

    test_loss = None

    optimizer = get_optimizer(optimizer_name=args.optimizer, model=model, learning_rate=args.learning_rate)
    # scheduler =

    print('Training...')
    for epoch in range(args.n_epochs):
        model.train()

        for batch, (images, labels) in enumerate(loader_train):
            model.train()

            labels = labels.unsqueeze(1).float()

            # Save the raw images and labels for visualization elsewhere
            if batch == 0 and epoch == 0:
                raw_output_dir = os.path.join(class_output_dir, 'raw')
                if not os.path.exists(raw_output_dir): os.makedirs(raw_output_dir)

                labels_copy = labels.clone().detach().cpu().numpy()
                np.save(os.path.join(raw_output_dir, 'labels.npy'), labels_copy)

                for i in range(len(images)):
                    img = images[i].squeeze().cpu().numpy()
                    np.save(os.path.join(raw_output_dir, f'img_{i}.npy'), img)

            images, labels = images.cuda(), labels.cuda()

            pred = model(images)
            loss = criterion(pred, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch+1) % args.k == 0:
            test_loss, test_acc, ratio_pos_pred = test_all_classification(loader_val, model, criterion)
            print(
                f'[Epoch {epoch + 1} of {args.n_epochs}] \t  Training loss: {loss.item()}. \t Val loss: {test_loss}.')

            d = {'epoch': epoch,
                 'lr': args.learning_rate,
                 'n': len(dataset_train),
                 'train_loss': loss.item(),
                 'test_loss': test_loss,
                 'test_acc': test_acc,
                 'ratio_pos_pred': ratio_pos_pred,
                 }

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
        train_output_dir = os.path.join(class_output_dir, 'train')
        evaluate_and_save_class_outputs(model, loader_train, train_output_dir,
                                      cutoff=args.cutoff, device='cuda')

        print('Evaluating on val set and saving outputs...')
        val_output_dir = os.path.join(class_output_dir, 'val')
        evaluate_and_save_class_outputs(model, loader_val, val_output_dir,
                                      cutoff=args.cutoff, device='cuda')


    print('Exiting...')


if __name__ == '__main__':
    main()

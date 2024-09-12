import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Optional, List


class TransformingTensorDataset(Dataset):
    """
    TensorDataset with support of torchvision transforms.
    """
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        y = self.Y[index]
        
        return x, y
    
    def __len__(self):
        return len(self.X)
    

def censor_image(image, mask):
    # Ensure the image and mask have the same shape
    assert image.shape == mask.shape, "Image and mask must have the same shape"

    # Ensure that mask tensor is binary
    assert torch.all((mask == 0) | (mask == 1)), "Mask tensor must be binary"

    # Create a tensor filled with -1, having the same shape as the image
    censor_value = torch.full_like(image, fill_value=-1)

    # Use the mask to select the pixels to be censored
    censored_image = torch.where(mask == 0, censor_value, image)

    return censored_image


class FASTSegmentationDataset(Dataset):
    """
        A PyTorch dataset for image segmentation tasks, specifically designed for the FAST
        (Free-breathing Abdominal Segmentation Technique) dataset.

        This dataset loads images and their corresponding segmentation masks from a specified directory. It supports
        data splitting (e.g., train, validation, test) and optional filtering of image files based on a provided list
        of included files.

        The dataset assumes that the images and masks are stored in separate subdirectories within the specified data
        directory. The image files are expected to have a '.pt' extension, while the mask files should have
        a '_Mask.pt' suffix.

        Using additional transforms is not recommended with the current version.

        Args:
            data_dir (str): The root directory containing the image and mask subdirectories.
            transform (Optional[transforms.Compose], optional): A composition of image transformations to be applied to
            both the images and masks. Defaults to None.
            split (str, optional): The data split to use (e.g., 'train', 'val', 'test'). Defaults to 'train'.
            included_files (Optional[List[str]], optional): A list of image file names to include in the dataset. If
            provided, only the specified files will be included. Defaults to None.

        Raises:
            RuntimeError: If there is an error opening an image or mask file.

        Returns:
            tuple: A tuple containing the loaded image and its corresponding segmentation mask.
        """
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, split: str = 'train',
                 included_files: Optional[List[str]] = None):
        # data_dir = '/scratch/users/austin.zane/ucsf_fast/data/labeled_fast_morison'

        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.image_files = os.listdir(os.path.join(self.data_dir, 'images'))

        if included_files:
            self.image_files = [f for f in self.image_files if f.replace('.pt', '.jpg') in included_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        mask_name = image_name.replace('.pt', '_Mask.pt')

        image_path = os.path.join(self.data_dir, 'images', image_name)
        mask_path = os.path.join(self.data_dir, 'masks', mask_name)

        try:
            image = torch.load(image_path)
            mask = torch.load(mask_path)
        except IOError as e:
            raise RuntimeError(f'Error opening image or mask: {e}')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, image_name


class FASTClassificationDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', pred_mask_dir: str = None,
                 use_mask: bool = True, use_pred_mask: bool = True):
        self.data_dir = data_dir
        self.split_data_dir = os.path.join(data_dir, split)
        self.pred_mask_dir = os.path.join(pred_mask_dir, split)
        self.use_mask = use_mask
        self.use_pred_mask = use_pred_mask

        self.image_files = os.listdir(os.path.join(self.split_data_dir, 'images'))
        # csv_path = os.path.join(data_dir, 'free_fluid_labels.csv')
        # self.image_metadata = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.split_data_dir, 'images', image_name)

        csv_path = os.path.join(self.data_dir, 'free_fluid_labels.csv')
        image_metadata = pd.read_csv(csv_path)

        try:
            image = torch.load(image_path, map_location=torch.device('cpu'))
        except IOError as e:
            raise RuntimeError(f'Error opening image: {e}')

        image_name_jpg = image_name.replace('.pt', '.jpg')
        metadata = image_metadata[image_metadata['filename'] == image_name_jpg]
        free_fluid_label = metadata['free_fluid_label'].values[0]

        if not self.use_mask:
            return image, free_fluid_label

        if self.use_pred_mask:
            mask_name = image_name.replace('.pt', '_Mask_Pred.pt')
            mask_path = os.path.join(self.pred_mask_dir, mask_name)
        else:
            mask_name = image_name.replace('.pt', '_Mask.pt')
            mask_path = os.path.join(self.split_data_dir, 'masks', mask_name)

        try:
            mask = torch.load(mask_path, map_location=torch.device('cpu'))
        except IOError as e:
            raise RuntimeError(f'Error opening mask: {e}')

        censored_image = censor_image(image, mask)

        free_fluid_label = 0 if free_fluid_label == -1 else 1

        return censored_image, free_fluid_label




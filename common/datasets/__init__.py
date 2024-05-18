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
    
    
# class FASTDataset(Dataset):
#     """
#     This dataset is designed to work with the pilot labeling done in February 2024.
#     The images and the segmentation labels are saved as PNG images in the same
#     directory. There are 384 such pairs.
#
#     This class defaults to resizing the images when they have irregular dimensions.
#     If the option is turned off and this class is used with torch.utils.data.DataLoader,
#     there will likely be errors when PyTorch tries to stack the images in a batch.
#
#     Attributes:
#         data_dir (str): Directory with image files.
#         resize (bool): Flag to resize images to a standard size if they don't match.
#         transform (torchvision.transforms): Transformations to be applied to the images and the masks.
#     """
#     DEFAULT_IMAGE_SIZE = (960, 720)
#
#     def __init__(self, data_dir: str, resize: bool = True, transform = None, included_files = None, excluded_files=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.resize = resize
#         # List all files in directory and filter out segmentation label images
#         if included_files:
#             self.image_files = [f for f in included_files if not '_Morison' in f and f.endswith('.png') and f in os.listdir(data_dir)]
#         else:
#             self.image_files = [f for f in os.listdir(data_dir) if not '_Morison' in f and f.endswith('.png') and f not in excluded_files]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, index):
#         img_name = self.image_files[index]
#         label_name = img_name.rsplit('.', 1)[0] + '_Morison.png'
#
#         img_path = os.path.join(self.data_dir, img_name)
#         label_path = os.path.join(self.data_dir, label_name)
#
#         try:
#             image = Image.open(img_path).convert('L')
#             label = Image.open(label_path).convert('L')
#         except IOError as e:
#             raise RuntimeError(f'Error opening image: {e}')
#
#         # Check for images that are not default size
#         if self.resize and image.size != self.DEFAULT_IMAGE_SIZE:
#
#             # For images, use bilinear interpolation
#             img_resize_transform = transforms.Resize(
#                 size=(self.DEFAULT_IMAGE_SIZE[1], self.DEFAULT_IMAGE_SIZE[0]),
#                 interpolation=InterpolationMode.BILINEAR
#             )
#             # For labels, use nearest neighbor interpolation to avoid introducing non-binary values
#             label_resize_transform = transforms.Resize(
#                 size=(self.DEFAULT_IMAGE_SIZE[1], self.DEFAULT_IMAGE_SIZE[0]),
#                 interpolation=InterpolationMode.NEAREST
#             )
#             image = img_resize_transform(image)
#             label = label_resize_transform(label)
#
#         # Define transformations for the images. Namely, convert to tensor and normalize.
#         img_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5], std=[0.5])
#         ])
#         # Define transformations for the labels. Namely, convert to tensor.
#         label_transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
#
#         image = img_transform(image)
#         label = label_transform(label)
#
#         if self.transform:
#             image = self.transform(image)
#             label = self.transform(label)
#
#         return image, label


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
            raise RuntimeError(f'Error opening image: {e}')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, image_name


# class FASTClassificationDataset(Dataset):



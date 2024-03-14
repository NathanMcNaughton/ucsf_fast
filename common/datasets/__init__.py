import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode


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
    
    
class FASTDataset(Dataset):
    """
    This dataset is designed to work with the pilot labeling done in February 2024.
    The images and the segmentation labels are saved as PNG images in the same 
    directory. There are 384 such pairs.
    
    This class defaults to resizing the images when they have irregular dimensions.
    If the option is turned off and this class is used with torch.utils.data.DataLoader,
    there will likely be errors when PyTorch tries to stack the images in a batch.
    
    Attributes:
        data_dir (str): Directory with image files.
        resize (bool): Flag to resize images to a standard size if they don't match.
        transform (torchvision.transforms): Transformations to be applied to the images and the masks.
    """
    DEFAULT_IMAGE_SIZE = (960, 720)
    
    def __init__(self, data_dir: str, resize: bool = True, transform = None, included_files = None, excluded_files=None):
        self.data_dir = data_dir
        self.transform = transform
        self.resize = resize
        # List all files in directory and filter out segmentation label images
        if included_files:
            self.image_files = [f for f in included_files if not '_Morison' in f and f.endswith('.png') and f in os.listdir(data_dir)]
        else:
            self.image_files = [f for f in os.listdir(data_dir) if not '_Morison' in f and f.endswith('.png') and f not in excluded_files]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_name = self.image_files[index]
        label_name = img_name.rsplit('.', 1)[0] + '_Morison.png'
        
        img_path = os.path.join(self.data_dir, img_name)
        label_path = os.path.join(self.data_dir, label_name)
        
        try:
            image = Image.open(img_path).convert('L')
            label = Image.open(label_path).convert('L')
        except IOError as e:
            raise RuntimeError(f'Error opening image: {e}')

        # Check for images that are not default size
        if self.resize and image.size != self.DEFAULT_IMAGE_SIZE:

            # For images, use bilinear interpolation
            img_resize_transform = transforms.Resize(
                size=(self.DEFAULT_IMAGE_SIZE[1], self.DEFAULT_IMAGE_SIZE[0]),
                interpolation=InterpolationMode.BILINEAR
            )
            # For labels, use nearest neighbor interpolation to avoid introducing non-binary values
            label_resize_transform = transforms.Resize(
                size=(self.DEFAULT_IMAGE_SIZE[1], self.DEFAULT_IMAGE_SIZE[0]),
                interpolation=InterpolationMode.NEAREST
            )
            image = img_resize_transform(image)
            label = label_resize_transform(label)

        # Define transformations for the images. Namely, convert to tensor and normalize.
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # Define transformations for the labels. Namely, convert to tensor.
        label_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = img_transform(image)
        label = label_transform(label)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label
    
    
    
    
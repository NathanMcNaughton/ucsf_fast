import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
import numpy as np
from torchvision.transforms import transforms


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
    directory. There are around 350 such pairs.
    
    This class defaults to resizing the images when they have irregular dimensions.
    If the option is turned off and this class is used with torch.utils.data.DataLoader,
    there will likely be errors when PyTorch tries to stack the images in a batch.
    
    Attributes:
        data_dir (str): Directory with image files.
        resize (bool): Flag to resize images to a standard size if they don't match.
        transform (transforms.]): Transformations to be applied to the images.
    """
    DEFAULT_IMAGE_SIZE = (960, 720)
    
    def __init__(self, data_dir: str, resize: bool = True, transform = None):
        self.data_dir = data_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.resize = resize
        # List all files in directory and filter out segmentation label images
        self.image_files = [f for f in os.listdir(data_dir) if not '_Morison' in f and f.endswith('.png')]
        
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
        
        if self.resize and image.size != self.DEFAULT_IMAGE_SIZE:
            resize_transform = transforms.Resize((self.DEFAULT_IMAGE_SIZE[1], 
                                                  self.DEFAULT_IMAGE_SIZE[0]))
            image = resize_transform(image)
            label = resize_transform(label)
        
        image = self.transform(image)
        label = self.transform(label)
        
        return image, label
    
    
    
    
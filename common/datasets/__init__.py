import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
    """
    def __init__(self, data_dir, transform=None):
        self.directory = directory
        self.transform = transform
        # List all files in directory and filter out segmentation label images
        self.image_files = [f for f in os.listdir(directory) if not '_Morison' in f]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_name = self.image_files[idx]
        
        # Construct a segmentation label name based on the image name
        label_name = img_name.rsplit('.', 1)[0] + '_Morrison.png'
        
        img_path = os.path.join(self.directory, img_name)
        label_path = os.path.join(self.directory, label, name)
        
        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label
    
    
    
    
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
    """
    def __init__(self, data_dir, resize=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.resize = resize
        # List all files in directory and filter out segmentation label images
        self.image_files = [f for f in os.listdir(data_dir) if (not '_Morison' in f and not '.db' in f and not '.json' in f)]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_name = self.image_files[index]
        
        # Construct a segmentation label name based on the image name
        label_name = img_name.rsplit('.', 1)[0] + '_Morison.png'
        
        img_path = os.path.join(self.data_dir, img_name)
        label_path = os.path.join(self.data_dir, label_name)
        
        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')
        
        if self.resize and image._size != (960, 720):
            resize_transform = transforms.Resize((720, 960))
            image = resize_transform(image)
            label = resize_transform(label)
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        else:
            tensor_transform = transforms.ToTensor()
            image = tensor_transform(image)
            label = tensor_transform(label)
        
        return image, label
    
    
    
    
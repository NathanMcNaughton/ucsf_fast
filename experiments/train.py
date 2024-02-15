from torchvision import transforms
from torch.utils.data import DataLoader

from common.datasets import FASTDataset




def main():
    # Might change this to get the directory from the config file
    data_dir = '/scratch/users/austin.zane/ucsf_fast/data/pilot_labeling/AnnotationData/MorisonPouchMasks_1-23'

    # Transform that converts PIL image to tensor and normalizes it
    pil2tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create the dataset and dataloader
    # Only one dataset for now. Will add test and train later.

    img_dataset = FASTDataset(data_dir, transform=pil2tensor)
    dataloader = DataLoader(img_dataset, batch_size=4, shuffle=True, num_workers=4)

    for img, mask in dataloader:
        print(img.shape, mask.shape)
        break



if __name__ == '__main__':
    main()

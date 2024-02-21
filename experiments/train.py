import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from common.datasets import FASTDataset
from common.models.small.unet_toy import ToyUNet



def test_all(loader, model, criterion):
    """
    Test the model on the entire dataset
    :param loader:
    :param model:
    :param criterion:
    :return:
    """

    return 0


def main():
    print(f'Torch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('CUDA is not available. Using CPU')


    # wandb.init(project='02_15_2024_ucsf_fast', entity='john-doe')

    # Might change this to get the directory from the config file
    data_dir = '/scratch/users/austin.zane/ucsf_fast/data/pilot_labeling/AnnotationData/MorisonPouchMasks_1-23'

    # Transform that converts PIL image to tensor and normalizes it
    pil2tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Consider clipping off 0.95th and 99.5th percentiles
    ])

    # Create the datasets and dataloaders

    img_dataset = FASTDataset(data_dir, transform=pil2tensor)

    n_total = len(img_dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    batch_size = 4

    dataset_train, dataset_test = torch.utils.data.random_split(img_dataset, [n_train, n_test])

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    model = ToyUNet(n_channels=1, n_classes=2)
    model.cuda()
    # criterion =
    # optimizer =
    # scheduler =

    for img, mask in loader_train:
        print(img.shape, mask.shape)




if __name__ == '__main__':
    main()

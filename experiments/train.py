import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from common.datasets import FASTDataset



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
    # wandb.init(project='02_15_2024_ucsf_fast', entity='john-doe')

    # Might change this to get the directory from the config file
    data_dir = '/scratch/users/austin.zane/ucsf_fast/data/pilot_labeling/AnnotationData/MorisonPouchMasks_1-23'

    # Transform that converts PIL image to tensor and normalizes it
    pil2tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create the datasets and dataloaders

    img_dataset = FASTDataset(data_dir, transform=pil2tensor)

    n_total = len(img_dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    dataset_train, dataset_test = torch.utils.data.random_split(img_dataset, [n_train, n_test])

    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)
    loader_test = DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=4)

    # model =
    # criterion =
    # optimizer =
    # scheduler =

    for img, mask in loader_train:
        print(img.shape, mask.shape)




if __name__ == '__main__':
    main()

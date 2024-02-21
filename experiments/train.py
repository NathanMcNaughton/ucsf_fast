import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from common.datasets import FASTDataset
from common.models.small.unet_toy import ToyUNet


def test_all(loader, model, loss_fn):
    """
    Test the model on the entire dataset
    :param loader:
    :param model:
    :param loss_fn:
    :return:
    """
    num_batches = len(loader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (images, masks) in enumerate(loader):
            images, masks = images.cuda(), masks.cuda()
            pred = model(images)
            test_loss += loss_fn(pred, masks).item()
    test_loss /= num_batches

    return test_loss


def main():
    if torch.cuda.is_available():
        print(f'CUDA available. Using GPU.')
    else:
        print('CUDA is not available. Try again with a GPU. Exiting...')
        exit(1)


    # wandb.init(project='02_15_2024_ucsf_fast', entity='john-doe')

    # Might change this to get the directory from the config file
    data_dir = '/scratch/users/austin.zane/ucsf_fast/data/pilot_labeling/AnnotationData/MorisonPouchMasks_1-23'

    # Create the datasets and dataloaders
    img_dataset = FASTDataset(data_dir)

    n_total = len(img_dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    batch_size = 10
    num_epochs = 5

    dataset_train, dataset_test = torch.utils.data.random_split(img_dataset, [n_train, n_test])

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    model = ToyUNet(n_channels=1, n_classes=1)
    model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # scheduler =

    print('Training...')
    for epoch in range(num_epochs):
        model.train()

        for batch, (images, masks) in enumerate(loader_train):
            images, masks = images.cuda(), masks.cuda()

            # Compute prediction error
            pred = model(images)
            loss = criterion(pred, masks)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                print(f'Epoch {epoch} of {num_epochs}, batch {batch} of {len(loader_train)}')
                print(f'Training loss: {loss.item()}')

                test_loss = test_all(loader_test, model, criterion)

                print(f'Test loss: {test_loss}')

    print('Saving model...')

    print('Model saved. Exiting...')



if __name__ == '__main__':
    main()

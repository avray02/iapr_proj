import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CoinDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (list): List of all images.
            labels (list): List of labels corresponding to each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
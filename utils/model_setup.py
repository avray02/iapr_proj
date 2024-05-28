import random
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CoinDataset(Dataset):
    def __init__(self, images, masks, labels=None, is_validation=True, augment=False):
        """
        Args:
            images (list): List of all images.
            labels (list): List of labels corresponding to each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        # if labels is None:
        #     self.labels = [0 for _ in range(len(images))]
        # else:
        self.labels = labels

        self.masks = masks
        self.is_validation = is_validation
        self.augment = augment

        self.normalize = transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],#[186.56360392/255, 171.46413211/255, 144.73335904/255],#
                        std=[0.2023, 0.1994, 0.2010],# [26.83046842/255, 27.42081317/255, 44.70402812/255],#
                    )

        self.resize = transforms.Resize((400, 400))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            self.resize,
            transforms.ToTensor(),
            self.normalize])
        
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            self.resize,
            transforms.ToTensor()
        ])
            
    def add_noise(img):
        if random.random() < 0.5:  # Apply noise with 50% probability
            noise = torch.randn(img.size()) * 0.05  # Adjust intensity as necessary
            noisy_img = img + noise
            noisy_img = torch.clamp(noisy_img, 0, 255)  # Ensure pixel values stay in [0, 1]
            return noisy_img
        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        
        if not self.is_validation and self.augment:
            # Generate random transformation parameters
            angle = 30
            translate = (0.05,0.05)

            transform_params = transforms.RandomAffine(degrees=(-angle,angle), translate=translate)
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                self.resize,
                transform_params,
                # transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2))], p=0.2),  # Gaussian blur
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.3), # Random color changes
                transforms.ToTensor(),
                # transforms.Lambda(self.add_noise),  # Add noise to the image
                self.normalize
                ])

            mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.resize,
                transform_params,
                transforms.ToTensor()
            ])

            image = transform(image)
            mask = mask_transform(mask)

        else:
            image = self.transform(image)
            mask = self.mask_transform(mask)
        
        image = image * mask

        return image, torch.tensor(label, dtype=torch.long)
    
    
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

class ImageSteganoDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=256):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Use current image as cover and next image as secret
        cover_path = self.image_files[idx]
        secret_idx = (idx + 1) % len(self.image_files)
        secret_path = self.image_files[secret_idx]
        
        cover_image = Image.open(cover_path).convert('RGB')
        secret_image = Image.open(secret_path).convert('RGB')
        
        if self.transform:
            cover_image = self.transform(cover_image)
            secret_image = self.transform(secret_image)
        
        return cover_image, secret_image

def prepare_dataloaders(data_dir, batch_size=32, image_size=256, num_workers=4):
    """Prepare train, validation, and test dataloaders"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Split dataset
    dataset = ImageSteganoDataset(data_dir, transform)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def download_sample_dataset(root_dir='data/raw'):
    """Download a sample dataset (CIFAR-10) for testing"""
    os.makedirs(root_dir, exist_ok=True)
    
    dataset = datasets.CIFAR10(
        root=root_dir,
        train=True,
        download=True,
        transform=None
    )
    
    # Save images
    for idx, (img, _) in enumerate(tqdm(dataset, desc="Saving images")):
        img_path = os.path.join(root_dir, f'image_{idx:05d}.png')
        img.save(img_path)
    
    return root_dir
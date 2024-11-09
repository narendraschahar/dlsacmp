import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from einops import rearrange
import numpy as np
from tqdm import tqdm

class SteganographyDataset(Dataset):
    def __init__(self, cover_images, secret_images, transform=None):
        self.cover_images = cover_images
        self.secret_images = secret_images
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.cover_images)
    
    def __getitem__(self, idx):
        cover = self.cover_images[idx]
        secret = self.secret_images[idx]
        
        if self.transform:
            cover = self.transform(cover)
            secret = self.transform(secret)
        return cover, secret

class CNNSteganography(nn.Module):
    def __init__(self):
        super(CNNSteganography, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(6, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(32, 3, 3, padding=1)
        
        # Decoder
        self.dec_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(64, 3, 3, padding=1)
    
    def encode(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        return torch.tanh(self.enc_conv4(x))
    
    def decode(self, stego):
        x = F.relu(self.dec_conv1(stego))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        return torch.tanh(self.dec_conv4(x))
    
    def forward(self, cover, secret):
        stego = self.encode(cover, secret)
        recovered = self.decode(stego)
        return stego, recovered

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for cover, secret in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            cover, secret = cover.to(device), secret.to(device)
            
            optimizer.zero_grad()
            stego, recovered = model(cover, secret)
            
            loss = criterion(stego, cover) + criterion(recovered, secret)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cover, secret in val_loader:
                cover, secret = cover.to(device), secret.to(device)
                stego, recovered = model(cover, secret)
                loss = criterion(stego, cover) + criterion(recovered, secret)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        
    return model

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    metrics = {'psnr': [], 'ssim': []}
    
    with torch.no_grad():
        for cover, secret in test_loader:
            cover, secret = cover.to(device), secret.to(device)
            stego, recovered = model(cover, secret)
            
            # Calculate PSNR
            mse = torch.mean((stego - cover) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            metrics['psnr'].append(psnr.item())
            
            # Calculate SSIM
            metrics['ssim'].append(calculate_ssim(stego, cover))
    
    return {k: np.mean(v) for k, v in metrics.items()}

def calculate_ssim(img1, img2):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data for testing
    batch_size = 32
    img_size = 64
    dummy_data = [(torch.randn(batch_size, 3, img_size, img_size), 
                   torch.randn(batch_size, 3, img_size, img_size)) 
                  for _ in range(10)]
    
    train_loader = DataLoader(dummy_data, batch_size=batch_size)
    val_loader = DataLoader(dummy_data[:2], batch_size=batch_size)
    
    # Initialize and train model
    model = CNNSteganography()
    model = train_model(model, train_loader, val_loader, num_epochs=2)
    
    # Evaluate
    metrics = evaluate_model(model, val_loader)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
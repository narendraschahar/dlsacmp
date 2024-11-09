import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class UNetSteganography(nn.Module):
    def __init__(self, in_channels=6):
        super(UNetSteganography, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decoding with skip connections
        d3 = self.dec3(e3)
        d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear')
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))
        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear')
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))
        
        return d1

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
    def forward(self, x):
        x = x.permute(1, 0, 2)  # (B,N,D) -> (N,B,D)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        x = x.permute(1, 0, 2)  # (N,B,D) -> (B,N,D)
        return x

class TransformerSteganography(nn.Module):
    def __init__(self, image_size=256, patch_size=16, dim=256, depth=6, heads=8):
        super(TransformerSteganography, self).__init__()
        
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(6, dim, patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])
        
        # Output projection
        self.to_pixels = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_size//patch_size),
            nn.ConvTranspose2d(dim, 3, patch_size, stride=patch_size),
            nn.Tanh()
        )
        
    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        x = self.to_patch_embedding(x)
        
        for transformer in self.transformer_blocks:
            x = transformer(x)
            
        return self.to_pixels(x)

class HybridSteganography(nn.Module):
    def __init__(self, image_size=256):
        super(HybridSteganography, self).__init__()
        
        # CNN encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Transformer middle
        self.transformer = TransformerBlock(128, heads=4)
        
        # CNN decoder
        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        
        # CNN encoding
        x = self.cnn_encoder(x)
        
        # Reshape for transformer
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Reshape back
        x = x.permute(0, 2, 1).view(b, c, h, w)
        
        # CNN decoding
        x = self.cnn_decoder(x)
        
        return x
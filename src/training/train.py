# src/training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.models.steganography import CNNSteganography
from src.models.advanced_models import UNetSteganography, TransformerSteganography, HybridSteganography
from src.data.data_loader import prepare_dataloaders
from src.utils.evaluation import evaluate_model

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.get_model()
        self.model.to(self.device)
        
        # Setup training
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
    def setup_directories(self):
        """Create necessary directories"""
        self.log_dir = Path(self.config['log_dir'])
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model(self):
        """Initialize model based on configuration"""
        model_name = self.config['model'].lower()
        if model_name == 'cnn':
            return CNNSteganography()
        elif model_name == 'unet':
            return UNetSteganography()
        elif model_name == 'transformer':
            return TransformerSteganography()
        elif model_name == 'hybrid':
            return HybridSteganography()
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def train(self, train_loader, val_loader):
        """Training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_loss = 0
            train_psnr = 0
            train_ssim = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
            for cover, secret in pbar:
                cover = cover.to(self.device)
                secret = secret.to(self.device)
                
                self.optimizer.zero_grad()
                
                if isinstance(self.model, CNNSteganography):
                    stego, recovered = self.model(cover, secret)
                    loss = self.criterion(stego, cover) + self.criterion(recovered, secret)
                else:
                    stego = self.model(cover, secret)
                    loss = self.criterion(stego, cover)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            val_metrics = evaluate_model(
                self.model,
                val_loader,
                self.device,
                self.criterion
            )
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('PSNR/val', val_metrics['psnr'], epoch)
            self.writer.add_scalar('SSIM/val', val_metrics['ssim'], epoch)
            
            # Save checkpoint if best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics['loss'])
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f}')
            print(f'Val PSNR: {val_metrics["psnr"]:.2f}')
            print(f'Val SSIM: {val_metrics["ssim"]:.4f}')
            print('-' * 50)
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

def main():
    parser = argparse.ArgumentParser(description='Train steganography model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Prepare data
    train_loader, val_loader, _ = prepare_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size']
    )
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()

'''
# src/training/train.py
from src.utils.evaluation import ModelEvaluator

class Trainer:
    def __init__(self, config):
        # ... other initialization code ...
        self.evaluator = ModelEvaluator(self.device)

    def train(self, train_loader, val_loader):
        for epoch in range(self.config['epochs']):
            # ... training code ...

            # Evaluate after each epoch
            metrics = self.evaluator.evaluate_model(
                self.model, 
                val_loader,
                self.criterion
            )

            # Log metrics
            self.writer.add_scalar('PSNR', metrics['psnr'], epoch)
            self.writer.add_scalar('SSIM', metrics['ssim'], epoch)
            
            # Save visualization periodically
            if epoch % self.config['viz_interval'] == 0:
                self.evaluator.visualize_results(
                    self.model,
                    val_loader,
                    save_path=f'results/epoch_{epoch}.png'
                )
'''
# src/utils/evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from src.models.steganography import CNNSteganography

class ModelEvaluator:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate Structural Similarity Index"""
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
    
    def evaluate_model(self, 
                      model: torch.nn.Module, 
                      dataloader: torch.utils.data.DataLoader,
                      criterion: torch.nn.Module) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        Returns dict with metrics: loss, psnr, ssim, capacity, and robustness
        """
        model.eval()
        metrics = {
            'loss': 0,
            'psnr': [],
            'ssim': [],
            'capacity': [],
            'robustness': []
        }
        
        with torch.no_grad():
            for cover, secret in tqdm(dataloader, desc='Evaluating'):
                cover = cover.to(self.device)
                secret = secret.to(self.device)
                
                # Forward pass
                if isinstance(model, CNNSteganography):
                    stego, recovered = model(cover, secret)
                    loss = criterion(stego, cover) + criterion(recovered, secret)
                    
                    # Calculate recovery accuracy
                    recovery_quality = self.calculate_psnr(recovered, secret)
                    metrics['robustness'].append(recovery_quality)
                else:
                    stego = model(cover, secret)
                    loss = criterion(stego, cover)
                
                # Calculate metrics
                metrics['loss'] += loss.item()
                metrics['psnr'].append(self.calculate_psnr(stego, cover))
                metrics['ssim'].append(self.calculate_ssim(stego, cover))
                
                # Calculate capacity (bits per pixel)
                capacity = self.calculate_capacity(stego, cover)
                metrics['capacity'].append(capacity)
        
        # Average metrics
        metrics['loss'] /= len(dataloader)
        for key in ['psnr', 'ssim', 'capacity', 'robustness']:
            if metrics[key]:
                metrics[key] = np.mean(metrics[key])
        
        return metrics
    
    def calculate_capacity(self, stego: torch.Tensor, cover: torch.Tensor) -> float:
        """Calculate embedding capacity in bits per pixel"""
        diff = torch.abs(stego - cover)
        return torch.mean(torch.log2(1 + diff)).item()
    
    def evaluate_robustness(self, 
                          model: torch.nn.Module, 
                          image: torch.Tensor,
                          attacks: List[str]) -> Dict[str, float]:
        """Evaluate model robustness against various attacks"""
        model.eval()
        robustness_metrics = {}
        
        with torch.no_grad():
            # Original prediction
            original_output = model(image)
            
            for attack in attacks:
                if attack == 'noise':
                    perturbed_image = image + torch.randn_like(image) * 0.1
                elif attack == 'blur':
                    perturbed_image = F.gaussian_blur(image, kernel_size=(5,5))
                elif attack == 'jpeg':
                    # Simulate JPEG compression
                    perturbed_image = image * 255
                    perturbed_image = perturbed_image.clamp(0, 255).byte()
                    perturbed_image = perturbed_image.float() / 255
                
                # Get prediction on perturbed image
                perturbed_output = model(perturbed_image)
                
                # Calculate similarity
                similarity = self.calculate_ssim(original_output, perturbed_output)
                robustness_metrics[attack] = similarity
        
        return robustness_metrics
    
    def visualize_results(self, 
                         model: torch.nn.Module, 
                         dataloader: torch.utils.data.DataLoader,
                         num_samples: int = 5,
                         save_path: str = None):
        """Visualize model predictions"""
        model.eval()
        fig, axes = plt.subplots(num_samples, 4, figsize=(15, 3*num_samples))
        
        with torch.no_grad():
            for idx, (cover, secret) in enumerate(dataloader):
                if idx >= num_samples:
                    break
                    
                cover = cover.to(self.device)
                secret = secret.to(self.device)
                
                if isinstance(model, CNNSteganography):
                    stego, recovered = model(cover, secret)
                else:
                    stego = model(cover, secret)
                    recovered = None
                
                # Plot results
                axes[idx, 0].imshow(self._tensor_to_image(cover[0]))
                axes[idx, 0].set_title('Cover Image')
                axes[idx, 0].axis('off')
                
                axes[idx, 1].imshow(self._tensor_to_image(secret[0]))
                axes[idx, 1].set_title('Secret Image')
                axes[idx, 1].axis('off')
                
                axes[idx, 2].imshow(self._tensor_to_image(stego[0]))
                axes[idx, 2].set_title('Stego Image')
                axes[idx, 2].axis('off')
                
                if recovered is not None:
                    axes[idx, 3].imshow(self._tensor_to_image(recovered[0]))
                    axes[idx, 3].set_title('Recovered Secret')
                    axes[idx, 3].axis('off')
                else:
                    axes[idx, 3].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image"""
        return tensor.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    
    def plot_metrics_comparison(self, 
                              results: Dict[str, Dict[str, float]],
                              save_path: str = None):
        """Plot comparison of metrics across models"""
        metrics = ['psnr', 'ssim', 'capacity', 'robustness']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            data = [results[model][metric] for model in results.keys()]
            sns.barplot(
                x=list(results.keys()),
                y=data,
                ax=axes[idx]
            )
            axes[idx].set_title(f'{metric.upper()} Comparison')
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def main():
    """Example usage of the evaluator"""
    import torch.nn as nn
    from src.data.data_loader import prepare_dataloaders
    from src.models.advanced_models import UNetSteganography
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator(device)
    
    # Load data
    _, _, test_loader = prepare_dataloaders('data/raw')
    
    # Load model
    model = UNetSteganography().to(device)
    criterion = nn.MSELoss()
    
    # Evaluate
    metrics = evaluator.evaluate_model(model, test_loader, criterion)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Visualize
    evaluator.visualize_results(model, test_loader, save_path='results/visual_comparison.png')

if __name__ == '__main__':
    main()
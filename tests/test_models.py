# tests/test_models.py
import unittest
import torch
from src.models.steganography import CNNSteganography
from src.models.advanced_models import UNetSteganography, TransformerSteganography, HybridSteganography

class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.channels = 3
        self.image_size = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create sample data
        self.cover = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size)
        self.secret = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size)
        
        # Initialize models
        self.models = {
            'CNN': CNNSteganography(),
            'UNet': UNetSteganography(),
            'Transformer': TransformerSteganography(),
            'Hybrid': HybridSteganography()
        }
    
    def test_output_shapes(self):
        """Test if models produce correct output shapes"""
        for name, model in self.models.items():
            model.to(self.device)
            cover = self.cover.to(self.device)
            secret = self.secret.to(self.device)
            
            with torch.no_grad():
                if name == 'CNN':
                    stego, recovered = model(cover, secret)
                    self.assertEqual(stego.shape, cover.shape)
                    self.assertEqual(recovered.shape, secret.shape)
                else:
                    stego = model(cover, secret)
                    self.assertEqual(stego.shape, cover.shape)
    
    def test_output_range(self):
        """Test if output values are in valid range (-1, 1)"""
        for name, model in self.models.items():
            model.to(self.device)
            cover = self.cover.to(self.device)
            secret = self.secret.to(self.device)
            
            with torch.no_grad():
                if name == 'CNN':
                    stego, recovered = model(cover, secret)
                else:
                    stego = model(cover, secret)
                
                self.assertTrue(torch.all(stego >= -1))
                self.assertTrue(torch.all(stego <= 1))
    
    def test_gradients(self):
        """Test if gradients can be computed"""
        criterion = torch.nn.MSELoss()
        
        for name, model in self.models.items():
            model.to(self.device)
            cover = self.cover.to(self.device)
            secret = self.secret.to(self.device)
            
            if name == 'CNN':
                stego, recovered = model(cover, secret)
                loss = criterion(stego, cover) + criterion(recovered, secret)
            else:
                stego = model(cover, secret)
                loss = criterion(stego, cover)
            
            loss.backward()
            
            # Check if gradients exist
            for param in model.parameters():
                self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()
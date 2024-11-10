# tests/test_data.py
import unittest
import torch
import os
from src.data.data_loader import (
    ImageSteganoDataset,
    prepare_dataloaders,
    download_sample_dataset,
    DataAugmentation
)

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'data/test'
        os.makedirs(self.data_dir, exist_ok=True)
        self.image_size = 256
        self.batch_size = 32
        
        # Download sample dataset for testing
        download_sample_dataset(self.data_dir)
    
    def test_dataset_creation(self):
        """Test dataset initialization"""
        dataset = ImageSteganoDataset(self.data_dir, image_size=self.image_size)
        self.assertGreater(len(dataset), 0)
        
        # Test single item
        cover, secret = dataset[0]
        self.assertEqual(cover.shape, (3, self.image_size, self.image_size))
        self.assertEqual(secret.shape, (3, self.image_size, self.image_size))
    
    def test_dataloaders(self):
        """Test dataloader creation and functionality"""
        train_loader, val_loader, test_loader = prepare_dataloaders(
            self.data_dir,
            batch_size=self.batch_size,
            image_size=self.image_size
        )
        
        # Test batch shapes
        cover, secret = next(iter(train_loader))
        self.assertEqual(cover.shape, (self.batch_size, 3, self.image_size, self.image_size))
        self.assertEqual(secret.shape, (self.batch_size, 3, self.image_size, self.image_size))
    
    def test_augmentation(self):
        """Test data augmentation"""
        augmentation = DataAugmentation()
        transform = augmentation.get_transform(self.image_size, augment=True)
        
        dataset = ImageSteganoDataset(self.data_dir, transform=transform)
        cover, secret = dataset[0]
        
        self.assertEqual(cover.shape, (3, self.image_size, self.image_size))
        self.assertTrue(torch.all(cover >= -1) and torch.all(cover <= 1))
    
    def tearDown(self):
        """Clean up test data"""
        import shutil
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

if __name__ == '__main__':
    unittest.main()
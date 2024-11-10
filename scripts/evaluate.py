# scripts/evaluate.py
import argparse
import torch
from src.utils.evaluation import ModelEvaluator
from src.data.data_loader import prepare_dataloaders
from src.models.steganography import CNNSteganography
from src.models.advanced_models import UNetSteganography, TransformerSteganography, HybridSteganography

def main():
    parser = argparse.ArgumentParser(description='Evaluate steganography models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--model_type', type=str, required=True, 
                      choices=['cnn', 'unet', 'transformer', 'hybrid'])
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator(device)

    # Load model
    if args.model_type == 'cnn':
        model = CNNSteganography()
    elif args.model_type == 'unet':
        model = UNetSteganography()
    elif args.model_type == 'transformer':
        model = TransformerSteganography()
    else:
        model = HybridSteganography()

    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # Prepare data
    _, _, test_loader = prepare_dataloaders(args.data_dir)

    # Evaluate
    results = evaluator.evaluate_model(model, test_loader, torch.nn.MSELoss())
    
    # Save results
    evaluator.visualize_results(
        model, 
        test_loader,
        save_path=f'{args.output_dir}/{args.model_type}_samples.png'
    )

    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
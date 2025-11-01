"""
Train GNN models on synthetic dataset for meaningful explanations.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import get_dataset
from models import GCN
from utils import set_seed


def train_model(model, dataset, epochs=100, lr=0.01, device='cpu'):
    """
    Train a GNN model on the dataset.
    
    Args:
        model: GNN model
        dataset: List of Data objects
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to use
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {'loss': [], 'acc': []}
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data in dataset:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Use mean pooling for graph-level prediction
            graph_pred = out.mean(dim=0).unsqueeze(0)
            
            # Loss
            loss = criterion(graph_pred, data.y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            pred_class = graph_pred.argmax(dim=-1)
            correct += (pred_class == data.y).sum().item()
            total += 1
        
        avg_loss = total_loss / len(dataset)
        accuracy = correct / total
        
        history['loss'].append(avg_loss)
        history['acc'].append(accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}')
    
    return model, history


def main():
    """Train models on synthetic dataset."""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading synthetic dataset...")
    dataset = get_dataset('SYNTHETIC')
    print(f"Dataset size: {len(dataset)} graphs")
    
    # Get dataset stats
    sample_graph = dataset[0]
    num_features = sample_graph.x.size(1)
    num_classes = 2  # Binary classification
    
    print(f"Features: {num_features}, Classes: {num_classes}")
    
    # Create and train model
    print("\nTraining GCN model...")
    model = GCN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=2,
        dropout=0.0  # No dropout for small dataset
    )
    
    trained_model, history = train_model(
        model, dataset,
        epochs=100,
        lr=0.01,
        device=device
    )
    
    # Save model
    output_dir = Path(__file__).parent.parent / 'results' / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'trained_gcn.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Print final stats
    print(f"\nFinal training loss: {history['loss'][-1]:.4f}")
    print(f"Final training accuracy: {history['acc'][-1]:.4f}")
    
    # Test model predictions
    print("\nTesting model predictions on first 3 graphs:")
    trained_model.eval()
    with torch.no_grad():
        for i in range(min(3, len(dataset))):
            data = dataset[i].to(device)
            out = trained_model(data.x, data.edge_index)
            graph_pred = out.mean(dim=0)
            probs = F.softmax(graph_pred, dim=-1)
            pred_class = graph_pred.argmax(dim=-1).item()
            true_class = data.y.item()
            print(f"  Graph {i}: Pred={pred_class} (conf={probs[pred_class]:.3f}), True={true_class}")


if __name__ == '__main__':
    main()

"""
Dataset Loaders
Load and preprocess datasets for experiments.
"""

import torch
from torch_geometric.datasets import TUDataset, BAShapes, PPI
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
import os


def load_mutag(root='./data'):
    """
    Load MUTAG dataset.
    Molecular dataset for mutagenicity prediction.
    """
    dataset = TUDataset(root=root, name='MUTAG')
    return dataset


def load_ba_shapes(root='./data'):
    """
    Load BA-Shapes dataset.
    Synthetic dataset with ground-truth structure (house motifs).
    """
    dataset = BAShapes(root=root)
    return dataset


def load_ppi(root='./data'):
    """
    Load PPI dataset.
    Protein-protein interaction network.
    """
    train_dataset = PPI(root=root, split='train')
    val_dataset = PPI(root=root, split='val')
    test_dataset = PPI(root=root, split='test')
    
    return train_dataset, val_dataset, test_dataset


def create_simple_graph():
    """
    Create a simple synthetic graph for testing.
    """
    # 6-node graph with a "house" structure
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 4],
        [1, 2, 0, 2, 0, 1, 4, 3]
    ], dtype=torch.long)
    
    # Make undirected
    edge_index = to_undirected(edge_index)
    
    # Random features
    x = torch.randn(5, 10)
    
    # Binary classification
    y = torch.tensor([1], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def get_dataset(name: str, root='./data'):
    """
    Get dataset by name.
    
    Args:
        name: Dataset name ('MUTAG', 'BA-Shapes', 'PPI', or 'simple')
        root: Root directory for datasets
        
    Returns:
        Dataset or tuple of datasets
    """
    name = name.upper()
    
    if name == 'MUTAG':
        return load_mutag(root)
    elif name == 'BA-SHAPES' or name == 'BASHAPES':
        return load_ba_shapes(root)
    elif name == 'PPI':
        return load_ppi(root)
    elif name == 'SIMPLE':
        return [create_simple_graph()]
    else:
        raise ValueError(f"Unknown dataset: {name}")


def train_simple_model(model, dataset, epochs=100, lr=0.01, device='cpu'):
    """
    Train a simple GNN model on a dataset.
    
    Args:
        model: GNN model
        dataset: PyG dataset
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')
    
    return model

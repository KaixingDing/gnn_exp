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


def create_synthetic_dataset(num_graphs=10, num_nodes_range=(5, 15), num_features=10):
    """
    Create a synthetic dataset with multiple graphs.
    
    Args:
        num_graphs: Number of graphs to generate
        num_nodes_range: Range of nodes per graph (min, max)
        num_features: Number of node features
        
    Returns:
        List of graph Data objects
    """
    import random
    dataset = []
    
    for i in range(num_graphs):
        # Random number of nodes
        num_nodes = random.randint(num_nodes_range[0], num_nodes_range[1])
        
        # Create random edges (ensure connectivity)
        edges = []
        # First create a spanning tree for connectivity
        for j in range(1, num_nodes):
            parent = random.randint(0, j-1)
            edges.append([parent, j])
            edges.append([j, parent])
        
        # Add some random edges
        num_extra_edges = random.randint(num_nodes//2, num_nodes)
        for _ in range(num_extra_edges):
            u = random.randint(0, num_nodes-1)
            v = random.randint(0, num_nodes-1)
            if u != v:
                edges.append([u, v])
                edges.append([v, u])
        
        # Remove duplicates
        edges = list(set(tuple(e) for e in edges))
        edges = [[e[0], e[1]] for e in edges]
        
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # Random features
        x = torch.randn(num_nodes, num_features)
        
        # Binary classification (balanced)
        y = torch.tensor([i % 2], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    
    return dataset


def get_dataset(name: str, root='./data'):
    """
    Get dataset by name.
    
    Args:
        name: Dataset name ('MUTAG', 'BA-Shapes', 'PPI', 'SIMPLE', or 'SYNTHETIC')
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
    elif name == 'SYNTHETIC':
        return create_synthetic_dataset(num_graphs=20, num_nodes_range=(8, 20), num_features=10)
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

"""Datasets module."""

from .loaders import (
    load_mutag,
    load_ba_shapes,
    load_ppi,
    create_simple_graph,
    get_dataset,
    train_simple_model
)

__all__ = [
    'load_mutag',
    'load_ba_shapes',
    'load_ppi',
    'create_simple_graph',
    'get_dataset',
    'train_simple_model'
]

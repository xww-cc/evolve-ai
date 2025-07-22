from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import torch
from config.global_constants import BATCH_SIZE

class CustomDataLoader(DataLoader):
    """自定义数据加载器 - 完整"""
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int = BATCH_SIZE, shuffle: bool = True):
        dataset = TensorDataset(inputs, targets)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
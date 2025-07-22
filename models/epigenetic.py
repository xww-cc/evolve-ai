import torch
import random

class EpigeneticMarkers:
    """表观遗传标记 - 完整"""
    def __init__(self):
        self.data = torch.tensor([random.uniform(-1.0, 1.0) for _ in range(2)], dtype=torch.float32)

    def clamp(self):
        self.data = torch.clamp(self.data, -1.0, 1.0)
        
    def clone(self):
        new_markers = EpigeneticMarkers()
        new_markers.data = self.data.clone()
        return new_markers
        
    def __getitem__(self, index):
        return self.data[index]
        
    def __setitem__(self, index, value):
        self.data[index] = value
        
    def __len__(self):
        return len(self.data)
        
    @property
    def shape(self):
        return self.data.shape
        
    def detach(self):
        new_markers = EpigeneticMarkers()
        new_markers.data = self.data.detach()
        return new_markers
        
    def __iadd__(self, other):
        """支持与Tensor的加法操作"""
        if isinstance(other, (int, float)):
            self.data += other
        elif isinstance(other, torch.Tensor):
            self.data += other
        else:
            raise TypeError(f"不支持的类型: {type(other)}")
        return self
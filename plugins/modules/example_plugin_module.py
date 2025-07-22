import torch.nn as nn

class ExamplePluginModule(nn.Module):
    """示例插件模块"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
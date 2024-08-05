from torch import nn
import torch


class MetaLoRALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 8
        self.r = 8
        self.scaling = self.alpha/self.r

    def forward(self, x, matrix_a, matrix_b):
        x = self.scaling * (x @ torch.t(matrix_a) @ torch.t(matrix_b))
        return x

class MetaLinearLoraController(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.linear = linear_layer
        self.lora = MetaLoRALayer()
    
    def forward(self, x, matrix_a, matrix_b):
        x = self.linear(x) + self.lora(x, matrix_a, matrix_b)
        return x
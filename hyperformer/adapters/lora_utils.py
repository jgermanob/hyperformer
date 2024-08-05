"""Implementation of different utility functions for LoRA layers."""

import torch.nn as nn
from transformers.activations import get_activation
import math


def init_lora_a_linear_layer(linear_layer):
    nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
    #nn.init.ones_(linear_layer.bias)

def init_lora_b_linear_layer(linear_layer):
    nn.init.zeros_(linear_layer.weight)
    #nn.init.zeros_(linear_layer.bias)

def linear_lora_a_layer(input_dim, output_dim):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim, bias=False)
    init_lora_a_linear_layer(linear)
    return linear

def linear_lora_b_layer(input_dim, output_dim):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim, bias=False)
    init_lora_b_linear_layer(linear)
    return linear
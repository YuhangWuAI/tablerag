import math

import torch
import torch.nn as nn


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": nn.ReLU, "swish": swish}


class FeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, act=gelu):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = act

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):
        """Apply residual connection to any sublayer with the same size."""
        return self.norm(self.dropout(x + sublayer_out))
        # x + self.dropout(sublayer(self.norm(x)))
        # self.norm(x + self.dropout(sublayer(x)))

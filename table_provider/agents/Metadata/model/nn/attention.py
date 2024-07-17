import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """
        :param h: Number of heads.
        :param d_model: Model (hidden layer) size.
        :param dropout: Dropout prob.
        """
        super().__init__()
        if d_model % h != 0:
            raise ValueError(f"The hidden size ({d_model:d}) is not a multiple of "
                             f"the number of attention heads ({h:d}).")

        # We assume d_v always equals d_k
        self.d_k = d_model // h  # Attention head size
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Compute 'Scaled Dot Product Attention'
        """
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # Apply the attention mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)

        # Normalize the attention scores to probabilities.
        p_attn = F.softmax(scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if dropout is not None:
            p_attn = dropout(p_attn)

        return p_attn.matmul(value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttention(nn.Module):
    def __init__(self, d_query: int, d_value: int, d_output: int, dropout: float = 0.1):
        """
        :param d_query: Query (hidden layer) size.
        :param d_value: Value (hidden layer) size.
        :param d_output: Output size
        :param dropout: Dropout prob.
        """
        super().__init__()

        self.linear_layer = nn.Linear(d_value, d_query)
        self.output_linear = nn.Linear(d_value, d_value)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, cell_mask=None, seq_mask=None):
        '''

        :param query:
        :param key:
        :param value:
        :param cell_mask: (batch, seq_len, seq_len), the first dim is query, the second dim is value
        :param seq_mask: (batch, seq_len)
        :return:
        '''
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # shape: (seq_len, seq_len)
        scores = query.matmul(self.linear_layer(key).transpose(-2, -1))  # / math.sqrt(query.size(-1))

        # Apply the attention mask
        if cell_mask == None:
            cell_mask = torch.ones_like(scores).to(device=scores.device)
        if seq_mask == None:
            seq_mask = torch.ones_like(scores).to(device=scores.device)
        else:
            seq_len = seq_mask.size(-1)
            seq_mask = seq_mask.unsqueeze(-2).expand(-1, seq_len, -1)
        scores = scores + torch.log(cell_mask) + torch.log(
            seq_mask)  # mask * exp(score)= exp(score + ln(mask))

        # Normalize the attention scores to probabilities.
        p_attn = F.softmax(scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # if dropout is not None:
        #     p_attn = dropout(p_attn)

        x = p_attn.matmul(value)
        return self.output_linear(x)

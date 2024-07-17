import torch.nn as nn

from .attention import MultiHeadedAttention
from .util import SublayerConnection, FeedForward


class TransformerEncoderLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer Encoder (self-attention)
    Transformer Encoder = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout
        )
        self.feed_forward = FeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.attention_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.transition_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.attention_sublayer(x, self.attention(x, x, x, mask=mask))
        x = self.transition_sublayer(x, self.feed_forward(x))
        return x

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_sublayer.reset_parameters()
        self.transition_sublayer.reset_parameters()


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder = MultiHead_Attention + MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout
        )
        self.translation = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout
        )
        self.feed_forward = FeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.attention_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.translation_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.transition_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, y, x_mask, y_mask):
        """
        :param x: encoder output
        :param y: decoder input
        :param x_mask: source mask
        :param y_mask: target mask
        """
        y = self.attention_sublayer(y, self.attention(y, y, y, mask=y_mask))
        x = self.translation_sublayer(y, self.translation(y, x, x, mask=x_mask))
        x = self.transition_sublayer(x, self.feed_forward(x))
        return x

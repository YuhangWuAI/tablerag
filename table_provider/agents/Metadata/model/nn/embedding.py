import math

import torch
import torch.nn as nn

from .config import ModelConfig


class InputEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, use_embedding=True):
        super().__init__()
        if config.use_df:
            embed_hidden = config.hidden - config.data_len
        else:
            embed_hidden = config.hidden
        self.embedding_compress = nn.Linear(config.embed_len,
                                            embed_hidden) if config.embed_len > 0 and use_embedding else None
        self.cat_embeds = nn.ModuleList([nn.Embedding(cat_num, embed_hidden) for cat_num in config.num_categories])
        self.position_embed = (PositionalEmbedding(d_model=embed_hidden) if config.positional else None)
        self.dropout = nn.Dropout(p=config.dropout)
        self.embed_hidden = embed_hidden

    def forward(self, segments, semantic_embeds, categories):
        x = torch.zeros(categories.size()[:-1] + torch.Size([self.embed_hidden]), dtype=torch.float,
                        device=categories.device)
        if self.embedding_compress is not None:
            x += self.embedding_compress(semantic_embeds)
        if self.position_embed is not None:
            x += self.position_embed(segments)
        if len(self.cat_embeds) > 0:
            for cat, embed in zip(categories.chunk(categories.size(-1), -1), self.cat_embeds):
                x += embed(cat.squeeze(dim=-1))
        return self.dropout(x)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
    sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = VocabEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


class VocabEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, segment_size=3, embed_size=512):
        super().__init__(segment_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        odd_len = d_model - div_term.size(-1)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:odd_len])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

import torch
import torch.nn as nn

from .config import TransformerConfig
from .embedding import InputEmbedding
from .layer import TransformerEncoderLayer, TransformerDecoderLayer


class TransformerEmbed(nn.Module):
    def __init__(self, input_embedding: InputEmbedding, config: TransformerConfig):
        super().__init__()
        self.layers = config.layers
        self.input_embed = input_embedding
        self.data_len = config.data_len
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.hidden, config.attn_heads, config.ff_hidden, config.dropout)
            for _ in range(config.layers)])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config.hidden, config.attn_heads, config.ff_hidden, config.dropout)
            for _ in range(config.layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, state, actions):
        # B * len_x * embed_hidden
        x = self.input_embed(state["token_types"], state["segments"], state["semantic_embeds"], state["categories"])
        # B * len_x * hidden
        if self.data_len > 0:
            x = torch.cat((x, state["data_characters"]), -1)
        # B * 1 * 1 * len_x
        x_mask = state["mask"].unsqueeze(-2).unsqueeze(-2)

        # B * len_y * embed_hidden
        y = self.input_embed(actions["token_types"], actions["segments"],
                             actions["semantic_embeds"], actions["categories"])
        # B * len_y * hidden
        if self.data_len > 0:
            y = torch.cat((y, actions["data_characters"]), -1)
        # B * 1 * 1 * len_y
        y_mask = actions["mask"].unsqueeze(-2).unsqueeze(-2)
        for i in range(self.layers):
            x = self.encoder_layers[i](x, x_mask)
            y = self.decoder_layers[i](x, y, x_mask, y_mask)
        return self.softmax(self.fc(self.dropout(y)))

    def get_embed_parameters(self):
        return self.input_embed.parameters()

    def get_encoder_parameters(self):
        encoder_parameters = []
        for i in range(self.layers):
            encoder_parameters.append(self.encoder_layers[i].parameters())
        return encoder_parameters

    def get_decoder_modules(self):
        decoder_modules = list(self.decoder_layers)
        decoder_modules.append(self.fc)
        return decoder_modules

    def get_embed_modules(self):
        return [self.input_embed]

    def get_encoder_modules(self):
        return self.encoder_layers


class Transformer(TransformerEmbed):
    def __init__(self, config: TransformerConfig):
        input_embed = InputEmbedding(config)
        super().__init__(input_embed, config)

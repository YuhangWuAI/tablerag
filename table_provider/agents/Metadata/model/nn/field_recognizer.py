from typing import Dict

import torch
import torch.nn as nn

from ...metadata.column_type import SubCategory
from .config import TransformerConfig
from .embedding import InputEmbedding
from .layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # config.hidden is actually the input and output dimension of transformer
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    config.hidden, config.attn_heads, config.ff_hidden, config.dropout
                )
                for _ in range(config.layers)
            ]
        )

    def forward(self, x, x_mask):
        """
        Forwarding pass of Transformer Encoder
        :param x: The input tensor with shape batch_size * max_num_fields * hidden
        :param x_mask: The valid mask for input tensor ([PAD] token is 0) with size batch_size * max_num_fields
        """
        # shape: batch_size * 1 * 1 * max_num_fields
        x_mask = x_mask.unsqueeze(-2).unsqueeze(-2)
        for layer in self.encoder_layers:
            x = layer(x, x_mask)
        return x


class FieldRecognizer(nn.Module):
    """
    The model use transformer encoder as backbone to extract and mix field features.
    Use the output of transformer encoder for the following tasks:
    1. Is measure or not (binary classification);
    2. Can a measure field used as dimension or not (binary classification);
    3. Can a measure field use sum as aggregation function or not (binary classification);
    4. Can a measure field use average as aggregation function or not (binary classification);
    5. Score a measure field over 9 aggregation function (9-classification);
    6. Score a field to be used as dimension (dimension / no dimension binary classification);
    7. Score a field to be used as measure (measure / no measure binary classification);
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.data_len = config.data_len

        self.input_embed = InputEmbedding(config)
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.LogSoftmax(dim=-1)

        # TODO: try more complex network structure for each task. work Items 50
        # Linear connection of different task
        self.fc_msr_task = nn.Linear(config.hidden, 2)  # Measure classification task
        self.fc_dim_task = nn.Linear(
            config.hidden, 2
        )  # Measure field used as dimension (both) task
        # self.fc_sum_task = nn.Linear(config.hidden, 2)  # Sum classification task
        # self.fc_avg_task = nn.Linear(config.hidden, 2)  # Average classification task
        self.fc_agg_score = nn.Linear(
            config.hidden, 9
        )  # Scoring aggregations classification task
        self.fc_dim_score = nn.Linear(
            config.hidden, 2
        )  # Dimension scoring classification task
        self.fc_msr_score = nn.Linear(
            config.hidden, 2
        )  # Measure scoring classification task
        self.fc_key_score = nn.Linear(
            config.hidden, 2
        )  # Primary key scoring classification task
        self.fc_msr_pair = nn.Linear(2 * config.hidden, 2)  # Measure pair task
        self.fc_msr_type = nn.Linear(config.hidden, len(SubCategory))

    def _do_task(self, linear_task: nn.Module, enc_outputs: torch.Tensor):
        """
        With the encoder output embedding, linear layer of the specific task, softmax,
        calculate the final output of the task.
        :param linear_task: the linear nn module of the specific task that map encoder outputs into the classification
                            results.
        :param enc_outputs: the outputs of the transformer encoder, shape = (batch_size, src_len, hidden)
        :return task_res: the classification result of the specific task, shape = (batch_size, src_len, n_class)
        """
        return self.softmax(linear_task(enc_outputs))

    def forward(self, x: Dict[str, torch.Tensor], return_embedding=False):
        """
        Forward propagation of metadata model.
        :param x: a dictionary of tensors represents the field_space.
        :param return_embedding: Whether return embeddings.
        """
        # shape: batch_size * src_len * embed_hidden
        y = self.input_embed(x["segments"], x["semantic_embeds"], x["categories"])
        # shape: batch_size * src_len * hidden
        if self.data_len > 0:
            y = torch.cat((y, x["data_characters"]), -1)

        # Forward passing with transformer encoder.
        # shape: batch_size * src_len * hidden
        encoder_outputs = self.dropout(self.encoder(y, x["mask"]))

        # With the outputs of transformer encoder, do each classification task
        # shape: batch_size, src_len, 2
        msr_res = self._do_task(self.fc_msr_task, encoder_outputs)
        if return_embedding:
            return encoder_outputs, msr_res
        # shape: batch_size, src_len, 2
        dim_res = self._do_task(self.fc_dim_task, encoder_outputs)
        # # shape: batch_size, src_len, 2
        # sum_res = self._do_task(self.fc_sum_task, encoder_outputs)
        # # shape: batch_size, src_len, 2
        # avg_res = self._do_task(self.fc_avg_task, encoder_outputs)
        # shape: batch_size, src_len, 9
        agg_score_res = self._do_task(self.fc_agg_score, encoder_outputs)
        # shape: batch_size, src_len, 2
        dim_score_res = self._do_task(self.fc_dim_score, encoder_outputs)
        # shape: batch_size, src_len, 2
        msr_score_res = self._do_task(self.fc_msr_score, encoder_outputs)
        # shape: batch_size, src_len, 2
        key_score_res = self._do_task(self.fc_key_score, encoder_outputs)

        # shape: batch_size, pair_len, 2
        msr_pair_index = x["msr_pair_idx"]
        # shape: batch_size, pair_len, hidden
        msr_pair1 = torch.gather(
            encoder_outputs,
            -2,
            msr_pair_index[:, :, 0]
            .unsqueeze(dim=-1)
            .expand(-1, -1, encoder_outputs.size()[-1]),
        )
        msr_pair2 = torch.gather(
            encoder_outputs,
            -2,
            msr_pair_index[:, :, 1]
            .unsqueeze(dim=-1)
            .expand(-1, -1, encoder_outputs.size()[-1]),
        )
        # shape: batch_size, pari_len, 2 * hidden
        msr_pair = torch.cat((msr_pair1, msr_pair2), -1)
        msr_pair_res = self._do_task(self.fc_msr_pair, msr_pair)

        # shape: batch_size, src_len, len(SubCategory)
        msr_type_res = self._do_task(self.fc_msr_type, encoder_outputs)

        return (
            msr_res,
            dim_res,
            agg_score_res,
            msr_score_res,
            dim_score_res,
            key_score_res,
            msr_pair_res,
            msr_type_res,
        )

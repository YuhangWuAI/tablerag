from typing import Dict

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from ...metadata.column_type import SubCategory
from .config import Metadata2Config


class MetadataTapas(nn.Module):
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

    def __init__(self, config: Metadata2Config, tapas=None):
        super().__init__()
        self.tapas = tapas

        self.softmax = nn.LogSoftmax(dim=-1)

        # Linear connection of different task
        self.fc_msr_task = nn.Linear(config.embed_len, 2)  # Measure classification task
        self.fc_agg_score = nn.Linear(
            config.embed_len, 9
        )  # Scoring aggregations classification task
        self.fc_dim_score = nn.Linear(
            config.embed_len, 2
        )  # Dimension scoring classification task
        self.fc_msr_score = nn.Linear(
            config.embed_len, 2
        )  # Measure scoring classification task
        self.fc_key_score = nn.Linear(
            config.embed_len, 2
        )  # Primary key scoring classification task
        self.fc_msr_pair = nn.Linear(2 * config.embed_len, 2)  # Measure pair task
        self.fc_msr_type = nn.Linear(config.embed_len, len(SubCategory))

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

    def forward(self, x: Dict[str, torch.Tensor], mode=None):
        """
        Forward propagation of metadata model.
        :param x: a dictionary of tensors represents the field_space.
        :param return_embedding: Whether return embeddings.
        """
        # Fine tune tapas
        tapas_inpuut = {}
        tapas_inpuut["input_ids"] = x["table_model"]["input_ids"]
        tapas_inpuut["token_type_ids"] = x["table_model"]["token_type_ids"]
        tapas_inpuut["attention_mask"] = x["table_model"]["attention_mask"]

        tapas_embedding = self.tapas(**tapas_inpuut)
        tapas_embedding = tapas_embedding.last_hidden_state

        index1 = x["table_model"]["col_ids"]
        field_embedding = scatter_mean(
            tapas_embedding,
            index1.unsqueeze(-1).expand(tapas_embedding.size()),
            dim=-2,
            dim_size=x["mask"].size()[1] + 1,
        )[:, 1:, :]

        # With the outputs of transformer encoder, do each classification task
        # shape: batch_size, src_len, 2
        msr_res = self._do_task(self.fc_msr_task, field_embedding)
        # shape: batch_size, src_len, 9
        agg_score_res = self._do_task(self.fc_agg_score, field_embedding)
        # shape: batch_size, src_len, 2
        dim_score_res = self._do_task(self.fc_dim_score, field_embedding)
        # shape: batch_size, src_len, 2
        msr_score_res = self._do_task(self.fc_msr_score, field_embedding)
        # shape: batch_size, src_len, 2
        key_score_res = self._do_task(self.fc_key_score, field_embedding)

        # shape: batch_size, pair_len, 2
        msr_pair_index = x["msr_pair_idx"]
        # shape: batch_size, pair_len, hidden
        msr_pair1 = torch.gather(
            field_embedding,
            -2,
            msr_pair_index[:, :, 0]
            .unsqueeze(dim=-1)
            .expand(-1, -1, field_embedding.size()[-1]),
        )
        msr_pair2 = torch.gather(
            field_embedding,
            -2,
            msr_pair_index[:, :, 1]
            .unsqueeze(dim=-1)
            .expand(-1, -1, field_embedding.size()[-1]),
        )
        # shape: batch_size, pari_len, 2 * hidden
        msr_pair = torch.cat((msr_pair1, msr_pair2), -1)
        msr_pair_res = self._do_task(self.fc_msr_pair, msr_pair)

        # shape: batch_size, src_len, len(SubCategory)
        msr_type_res = self._do_task(self.fc_msr_type, field_embedding)

        return_list = [
            msr_res,
            agg_score_res,
            msr_score_res,
            dim_score_res,
            key_score_res,
            msr_pair_res,
            msr_type_res,
        ]
        return return_list

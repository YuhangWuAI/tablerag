from typing import Dict

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from ...data.config import MODE2LOSS
from ...metadata.column_type import SubCategory
from ...metadata.metadata_data.df_idx_dictionary import DF_IDX_DICT
from .co_attention import CoAttention
from .config import Metadata2Config
from .embedding import InputEmbedding
from .field_recognizer import TransformerEncoder


class Metadata2(nn.Module):
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

        self.data_len = config.data_len
        self.use_df = config.use_df
        self.use_emb = config.use_emb
        self.use_entity = config.use_entity
        self.df_subset = config.df_subset
        self.hidden1 = config.emb_hidden1

        self.data_len = config.data_len
        self.embedding_compress1 = (
            nn.Linear(config.embed_len, config.emb_hidden1) if config.use_emb else None
        )
        if config.use_table_model and (config.use_emb or config.use_entity):
            if config.use_entity:
                self.entity_co_attention = CoAttention(
                    config.emb_hidden1, config.entity_len, config.entity_hidden
                )
                self.entity_compress = nn.Linear(
                    config.entity_len, config.entity_hidden
                )

            self.encoder1 = (
                TransformerEncoder(config.encoder1_config)
                if config.encoder1_config.layers != 0
                else None
            )

        if config.use_df and config.use_emb:
            self.embedding_compress2 = nn.Linear(
                config.encoder1_config.hidden, config.emb_hidden2 - config.data_len
            )
        elif config.use_emb:
            self.embedding_compress2 = nn.Linear(
                config.encoder1_config.hidden, config.emb_hidden2
            )
        else:
            self.embedding_compress2 = None

        if config.use_entity:
            self.col_type_co_attention = CoAttention(
                config.emb_hidden2, config.entity_len, config.entity_len
            )
            self.property_co_attention = CoAttention(
                config.emb_hidden2, config.entity_len, config.entity_len
            )
            self.col_type_compress = nn.Linear(config.entity_len, config.entity_hidden)
            self.property_compress = nn.Linear(config.entity_len, config.entity_hidden)
            config.hidden = config.hidden - config.entity_hidden
            self.input_embed = (
                InputEmbedding(config, use_embedding=False) if self.use_df else None
            )
            config.hidden = config.hidden + config.entity_hidden
        else:
            self.input_embed = (
                InputEmbedding(config, use_embedding=False) if self.use_df else None
            )

        self.encoder2_msr = TransformerEncoder(config.encoder2_config)
        self.encoder2_agg_score = TransformerEncoder(config.encoder2_config)
        self.encoder2_dim_score = TransformerEncoder(config.encoder2_config)
        self.encoder2_msr_score = TransformerEncoder(config.encoder2_config)
        self.encoder2_key_score = TransformerEncoder(config.encoder2_config)
        self.encoder2_msr_pair = TransformerEncoder(config.encoder2_config)
        self.encoder2_msr_type = TransformerEncoder(config.encoder2_config)

        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.LogSoftmax(dim=-1)

        # TODO: try more complex network structure for each task. work Items 50
        # Linear connection of different task
        self.fc_msr_task = nn.Linear(config.hidden, 2)  # Measure classification task
        # self.fc_dim_task = nn.Linear(config.hidden, 2)  # Measure field used as dimension (both) task
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

    def mask_df(self, df: torch.Tensor, df_subset: list):
        idx = torch.LongTensor([])
        for s in df_subset:
            if s != 5:
                idx = torch.cat((torch.LongTensor(DF_IDX_DICT[s]), idx), 0)
        mask = torch.zeros_like(df)
        mask[:, :, idx] = 1
        result = df * mask
        return result

    def linear_encoder(self, x):
        if "table_model" in x:  # TAPAS
            embedding1 = x["table_model"]["embedding"]
            index1 = x["table_model"]["col_ids"]
            field_embedding = scatter_mean(
                embedding1,
                index1.unsqueeze(-1).expand(embedding1.size()),
                dim=-2,
                dim_size=x["mask"].size()[1] + 1,
            )[:, 1:, :]
            encoder_outputs = self.dropout(self.linear_layers(field_embedding))
        else:
            field_embedding = x["semantic_embeds"]
            encoder_outputs = self.dropout(self.linear_layers(field_embedding))
        return encoder_outputs

    def emb_tf_encoder1(self, x):
        '''
        Transformer encoder for sub-token level embedding
        '''
        if "table_model" in x:
            if self.use_emb:
                embedding_compress = self.embedding_compress1(
                    x["table_model"]["embedding"]
                )
            else:
                embedding_compress = torch.zeros(
                    x["table_model"]["num_mask"].size() + torch.Size([self.hidden1]),
                    device=x["table_model"]["num_mask"].device,
                )
            embedding = embedding_compress

            if self.use_entity:
                entity_embedding = x["table_model"]["entity"]
                # Mask according to row and col
                batch, seq_len, _ = entity_embedding.size()
                col_ids = x["table_model"]["col_ids"]
                col_mask = torch.zeros(batch, seq_len, seq_len).to(col_ids.device)
                col_mask[
                    col_ids.unsqueeze(-1).expand(-1, -1, seq_len)
                    == col_ids.unsqueeze(-2).expand(-1, -1, seq_len)
                ] = 0.5  # the same col has 0.5 mask
                row_ids = x["table_model"]["row_ids"]
                row_mask = torch.zeros(batch, seq_len, seq_len).to(row_ids.device)
                row_mask[
                    row_ids.unsqueeze(-1).expand(-1, -1, seq_len)
                    == row_ids.unsqueeze(-2).expand(-1, -1, seq_len)
                ] = 0.5  # the same row has 0.5 mask
                cell_mask = col_mask + row_mask
                entity_embedding2 = self.entity_co_attention(
                    embedding,
                    entity_embedding,
                    entity_embedding,
                    cell_mask,
                    x["table_model"]["mask"],
                )
                entity_embedding2 = self.entity_compress(
                    entity_embedding2 + entity_embedding
                )
                embedding = torch.cat((embedding, entity_embedding2), dim=-1)
            index1 = x["table_model"]["col_ids"]
            mask1 = x["table_model"]["mask"]

            # encoder1 and change sub-token level 2 field level
            if self.encoder1 is not None:
                embedding1 = self.dropout(self.encoder1(embedding, mask1))
            else:
                embedding1 = embedding
            embedding2 = scatter_mean(
                embedding1,
                index1.unsqueeze(-1).expand(embedding1.size()),
                dim=-2,
                dim_size=x["mask"].size()[1] + 1,
            )[
                :, 1:, :
            ]  # index == 0 means query or pad
        else:
            embedding2 = self.embedding_compress1(x["semantic_embeds"])
        return embedding2

    def df(self, x, mask, embedding2=None):
        '''
        Data feature
        '''
        # Add DF
        # shape: batch_size * src_len * embed_hidden
        if 5 in self.df_subset:
            y = self.input_embed(x["segments"], None, x["categories"])
            if embedding2 is not None:
                y += self.embedding_compress2(embedding2)
        else:
            y = self.embedding_compress2(embedding2)
        # shape: batch_size * src_len * hidden
        df = x["data_characters"]
        masked_df = self.mask_df(df, self.df_subset)

        if self.data_len > 0:
            y = torch.cat((y, masked_df), -1)
        return y

    def forward(
        self, x: Dict[str, torch.Tensor], return_embedding=False, mode="general"
    ):
        """
        Forward propagation of metadata model.
        :param x: a dictionary of tensors represents the field_space.
        :param return_embedding: Whether return embeddings.
        """
        # Fine tune tapas
        if self.tapas is not None:
            tapas_inpuut = {}
            tapas_inpuut["input_ids"] = x["table_model"]["input_ids"]
            tapas_inpuut["token_type_ids"] = x["table_model"]["token_type_ids"]
            tapas_inpuut["attention_mask"] = x["table_model"]["attention_mask"]

            tapas_embedding = self.tapas(**tapas_inpuut)
            tapas_embedding = tapas_embedding.last_hidden_state
            x["table_model"]["embedding"] = tapas_embedding

        if self.use_emb:
            embedding2 = self.emb_tf_encoder1(x)
        else:
            embedding2 = None

        if self.use_df:
            y = self.df(x, x["mask"], embedding2)
        else:
            if embedding2 == None:
                raise NotImplementedError("Can't don't use emb and df at the same time")
            y = self.embedding_compress2(embedding2)

        if self.use_entity:
            entity_emb = self.col_type_co_attention(
                y,
                x["table_model"]["col_type"],
                x["table_model"]["col_type"],
                seq_mask=x["mask"],
            )
            relation_emb = self.col_type_co_attention(
                y,
                x["table_model"]["property"],
                x["table_model"]["property"],
                seq_mask=x["mask"],
            )
            entity_emb = self.col_type_compress(
                x["table_model"]["col_type"] + entity_emb
            )
            relation_emb = self.property_compress(
                x["table_model"]["property"] + relation_emb
            )
            y = torch.cat((y, entity_emb + relation_emb), dim=-1)

        # Forward passing with transformer encoder2.
        # shape: batch_size * src_len * hidden
        encoder_outputs_msr = self.dropout(self.encoder2_msr(y, x["mask"]))
        encoder_outputs_agg_score = self.dropout(self.encoder2_agg_score(y, x["mask"]))
        encoder_outputs_dim_score = self.dropout(self.encoder2_dim_score(y, x["mask"]))
        encoder_outputs_msr_score = self.dropout(self.encoder2_msr_score(y, x["mask"]))
        encoder_outputs_key_score = self.dropout(self.encoder2_key_score(y, x["mask"]))
        encoder_outputs_msr_pair = self.dropout(self.encoder2_msr_pair(y, x["mask"]))
        encoder_outputs_msr_type = self.dropout(self.encoder2_msr_type(y, x["mask"]))

        # With the outputs of transformer encoder, do each classification task
        # shape: batch_size, src_len, 2
        msr_res = self._do_task(self.fc_msr_task, encoder_outputs_msr)
        if return_embedding:
            return encoder_outputs_msr, msr_res
        # shape: batch_size, src_len, 2
        # dim_res = self._do_task(self.fc_dim_task, encoder_outputs)
        # # shape: batch_size, src_len, 2
        # sum_res = self._do_task(self.fc_sum_task, encoder_outputs)
        # # shape: batch_size, src_len, 2
        # avg_res = self._do_task(self.fc_avg_task, encoder_outputs)
        # shape: batch_size, src_len, 9
        agg_score_res = self._do_task(self.fc_agg_score, encoder_outputs_agg_score)
        # shape: batch_size, src_len, 2
        dim_score_res = self._do_task(self.fc_dim_score, encoder_outputs_dim_score)
        # shape: batch_size, src_len, 2
        msr_score_res = self._do_task(self.fc_msr_score, encoder_outputs_msr_score)
        # shape: batch_size, src_len, 2
        key_score_res = self._do_task(self.fc_key_score, encoder_outputs_key_score)

        # shape: batch_size, pair_len, 2
        msr_pair_index = x["msr_pair_idx"]
        # shape: batch_size, pair_len, hidden
        msr_pair1 = torch.gather(
            encoder_outputs_msr_pair,
            -2,
            msr_pair_index[:, :, 0]
            .unsqueeze(dim=-1)
            .expand(-1, -1, encoder_outputs_msr_pair.size()[-1]),
        )
        msr_pair2 = torch.gather(
            encoder_outputs_msr_pair,
            -2,
            msr_pair_index[:, :, 1]
            .unsqueeze(dim=-1)
            .expand(-1, -1, encoder_outputs_msr_pair.size()[-1]),
        )
        # shape: batch_size, pari_len, 2 * hidden
        msr_pair = torch.cat((msr_pair1, msr_pair2), -1)
        msr_pair_res = self._do_task(self.fc_msr_pair, msr_pair)

        # shape: batch_size, src_len, len(SubCategory)
        msr_type_res = self._do_task(self.fc_msr_type, encoder_outputs_msr_type)

        return_list = [
            msr_res,
            agg_score_res,
            msr_score_res,
            dim_score_res,
            key_score_res,
            msr_pair_res,
            msr_type_res,
        ]
        for i in range(len(return_list)):
            if i not in MODE2LOSS[mode]:
                return_list[i] = torch.zeros(
                    return_list[i].size(), device=return_list[i].device
                )
        return return_list

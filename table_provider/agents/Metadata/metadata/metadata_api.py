import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate
from transformers import TapasTokenizer, TapasModel

from ..data.dataset import Field
from ..data.sequence import Sequence
from ..data.token import Token, Segment, FieldType
from ..helper import construct_data_config
from .utils import Metadata_Output
from ..model import get_metadata2_config, Metadata2
from ..FeatureExtractor.data_feature_extractor import DataFeatureExtractor

MSR_DIM = ["Dimension", "Measure"]

MEASURE_TYPE = [
    "Others",
    "Spatial Length",
    "Spatial Area",
    "Spatial Volume",
    "Time TimeInterval",
    "Time Frequency",
    "Matter Mass",
    "Thermodynamics Pressure",
    "Electric Energy",
    "Electric Power",
    "Money",
    "Data",
    "Quantity",
    "Rank",
    "Score",
    "Ratio",
    "Factor",
    "Angle",
    "Thermodynamics Temperature",
    "Mechanics Speed",
]
AGG = [
    "Sum",
    "Average",
    "Max",
    "Min",
    "Product",
    "StdDev",
    "StdDevP",
    "Var",
    "VarP",
]


class pretend_args:
    def __init__(self):
        self.model_name = "metadata2"
        self.model_size = "customize"
        self.model_save_path = ""
        self.features = "metadata-tapas_display"
        self.num_workers = 2
        self.valid_batch_size = 1
        self.corpus_path = ""
        self.model_load_path = ''
        self.mode = "train-test"
        self.eval_dataset = 'simple'
        self.lang = 'en'
        self.use_df = False
        self.use_emb = True
        self.use_entity = False
        self.entity_type = 'transe100'
        self.entity_emb_path = ''
        self.entity_recognition = 'semtab'
        self.tf1_layers = 2
        self.tf2_layers = 2
        self.df_subset = [1, 2, 3, 4, 5]


class MetadataApi:
    def __init__(self, model_path):
        args = pretend_args()
        self.data_config = construct_data_config(args, False)
        model_config = get_metadata2_config(
            self.data_config,
            args.tf1_layers,
            args.tf2_layers,
            args.use_df,
            args.use_emb,
            args.df_subset,
        )
        self.model = Metadata2(model_config)
        self.device = 'cpu'
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['module_state'])
        self.tapas_tokenizer = TapasTokenizer.from_pretrained(
            'google/tapas-base', local_files_only=True
        )
        self.tapas_model = TapasModel.from_pretrained(
            'google/tapas-base', local_files_only=True
        )

    def embedding(self, table: pd.DataFrame):
        table = table.head(math.ceil(512 / 2 / table.shape[1]))
        queries = ['']
        inputs = self.tapas_tokenizer(
            table=table.astype(str),
            queries=queries,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        # See details about output of TapasTokenizer in https://huggingface.co/transformers/_modules/transformers/models/tapas/tokenization_tapas.html#TapasTokenizer.__call__
        # column_ids = inputs['token_type_ids'][0, :, 1]

        # The original tokenizer generate rank before truncation, but this will cause max rank > record length.
        # If the rank > the record length, we change it to the record length for the moment
        record_len = torch.max(inputs["token_type_ids"][0, :, 2])
        inputs["token_type_ids"][0, :, 5] = torch.where(
            inputs["token_type_ids"][0, :, 5] <= record_len,
            inputs["token_type_ids"][0, :, 5],
            record_len
            * torch.ones(
                inputs["token_type_ids"][0, :, 5].size(),
                device=inputs["token_type_ids"].device,
                dtype=inputs["token_type_ids"].dtype,
            ),
        )
        inputs["token_type_ids"][0, :, 4] = torch.where(
            inputs["token_type_ids"][0, :, 4] <= record_len,
            inputs["token_type_ids"][0, :, 4],
            record_len
            * torch.ones(
                inputs["token_type_ids"][0, :, 4].size(),
                device=inputs["token_type_ids"].device,
                dtype=inputs["token_type_ids"].dtype,
            ),
        )

        outputs = self.tapas_model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        seq_len = sum(inputs['attention_mask'][0])
        emb = last_hidden_states[0, :seq_len, :].detach().cpu().numpy()
        token_ids = inputs['input_ids'][0, :seq_len].detach().cpu().numpy()
        col_ids = inputs['token_type_ids'][0, :seq_len, 1].detach().cpu().numpy()
        row_ids = inputs['token_type_ids'][0, :seq_len, 2].detach().cpu().numpy()
        return emb, token_ids, col_ids, row_ids

    def df(self, table_dict: dict, language="en"):
        source_features = DataFeatureExtractor.ExtractTableFeatures(
            table_dict, language
        )
        source_features.delete_dt()
        return source_features.__dict__

    def predict(self, table: pd.DataFrame, embedding, table_dicts: dict = {}):
        fields = [Field(idx) for idx in range(len(table.columns))]

        # Get the data characteristics of each field
        if self.data_config.use_data_features:
            for idx, fd in enumerate(table_dicts):
                fields[idx].features = np.array(
                    self.data_config.data_cleanup_fn(fd["dataFeatures"])
                )

            # Get categorical features of each field
            for idx, fd in enumerate(table_dicts):
                if self.data_config.use_field_type:
                    fields[idx].type = FieldType.from_raw_int(fd["type"])
                # if self.data_config.use_field_role:
                #     fields[idx].role = FieldRole.from_raw_bool(fd["inHeaderRegion"])

        tokens = [
            Token(
                field_index=field.idx,
                field_type=field.type,
                semantic_embedding=field.embedding,
                data_characteristics=field.features,
            )
            for field in fields
        ]
        action_space = Sequence(tokens, [Segment.FIELD] * len(tokens))

        input = action_space.to_dict(
            len(action_space), None, True, False, self.data_config
        )
        table_model = {
            "embedding": torch.tensor(embedding[0].tolist(), dtype=torch.float),
            "col_ids": torch.tensor(embedding[2].tolist(), dtype=torch.long),
            "row_ids": torch.tensor(embedding[3].tolist(), dtype=torch.long),
            "mask": torch.tensor([1] * len(embedding[0]), dtype=torch.long),
        }
        input["table_model"] = table_model
        input["msr_pair_idx"] = torch.tensor(
            [[0, 0]], dtype=torch.int64
        )  # Note, pseudo pair
        input["msr_pair_mask"] = torch.tensor([1], dtype=torch.int64)
        pseudo_batch = default_collate([input])
        predict = self.model(pseudo_batch)
        _, gby_idx = (
            predict[Metadata_Output.Gby_score_res]
            .detach()[:, :, 1][0]
            .sort(dim=-1, descending=True)
        )
        _, gby_rank = gby_idx.sort(dim=-1)

        _, key_idx = (
            predict[Metadata_Output.Key_score_res]
            .detach()[:, :, 1][0]
            .sort(dim=-1, descending=True)
        )
        _, key_rank = key_idx.sort(dim=-1)

        _, msr_idx = (
            predict[Metadata_Output.Msr_score_res]
            .detach()[:, :, 1][0]
            .sort(dim=-1, descending=True)
        )
        _, msr_rank = msr_idx.sort(dim=-1)

        return {
            'Msr_res': [
                MSR_DIM[i]
                for i in predict[Metadata_Output.Msr_res][0]
                .detach()
                .argmax(dim=-1)
                .tolist()
            ],
            'Agg_score_res': [
                AGG[i]
                for i in predict[Metadata_Output.Agg_score_res][0]
                .detach()
                .argmax(dim=-1)
                .tolist()
            ],
            'Msr_score_res': {"rank": msr_rank.tolist(), "idx_list": msr_idx.tolist()},
            'Gby_score_res': {"rank": gby_rank.tolist(), "idx_list": gby_idx.tolist()},
            'Key_score_res': {"rank": key_rank.tolist(), "idx_list": key_idx.tolist()},
            # 'Msr_pair_res': predict[Metadata_Output.Msr_pair_res][0].detach().argmax(dim=-1).tolist(),
            'Msr_type_res': [
                MEASURE_TYPE[i]
                for i in predict[Metadata_Output.Msr_type_res][0]
                .detach()
                .argmax(dim=-1)
                .tolist()
            ],
        }

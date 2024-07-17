import os
from copy import deepcopy
from typing import List

import numpy as np

from ...data import config as d_config
from ...data.config import DataConfig, TABLE_MODELS
from ...data.dataset import Field, table_fields
from ...data.sequence import Sequence
from ...data.token import Token, Segment, AnaType
from ...data.util import load_json, get_embeddings

COUNT = 0
RIGHT = 0
TYPE_COUTN = 0
PROPERTY_COUNT = 0


def mtab_parser(config: DataConfig, tuid: str):
    config = deepcopy(config)
    config.entity_recognition = "mtab"
    if not os.path.exists(config.entity_recognition_path(tuid)):
        return {}, {}, {}
    mtab_info = load_json(
        config.entity_recognition_path(tuid), encoding=config.encoding
    )["semantic"]
    entity = {}
    for entity_info in mtab_info["cea"]:
        # In mtab, row_id == 0 means header
        entity[
            f"{entity_info['target'][0] - 1},{entity_info['target'][1]}"
        ] = entity_info["annotation"]["wikidata"]
    type = {}
    for type_info in mtab_info["cta"]:
        type[type_info["target"]] = type_info["annotation"][0]["wikidata"]
    property = {}
    for property_info in mtab_info["cpa"]:
        property[property_info["target"][1]] = property_info["annotation"][0][
            "wikidata"
        ]
    return entity, type, property


def generate_field_space(fields: List[Field]):
    """Generate the whole action space for a source."""
    tokens = [
        Token(
            field_index=field.idx,
            field_type=field.type,
            semantic_embedding=field.embedding,
            data_characteristics=field.features,
            tags=field.tags,
        )
        for field in fields
    ]
    action_space = Sequence(tokens, [Segment.FIELD] * len(tokens))
    idx2field = {t.field_index: t for t in tokens}

    # Final check the fields_index is as expected
    idx = 0
    for t in tokens:
        assert t.field_index == idx, "Field index should be its actual 0-based index."
        idx += 1
    return action_space, idx2field


class MetaDataTable:
    __slots__ = (
        "config",
        "tUID",
        "pUIDs",
        "cUIDs",
        "n_rows",
        "n_cols",
        "field_space",
        "idx2field",
        "vendor",
        "T2D",
        "semtab",
        "aggr_label",
        "dm_label",
        "msr_type_label",
        "msr_type_label_revendor",
        "headers",
        "table_model",
        "label_source",
    )

    def __init__(self, tUID: str, sampled_schema, config: DataConfig):
        self.config = config
        # Normal training and evaluation
        table = load_json(config.table_path(tUID), config.encoding)
        table_id = tUID.split(".")[1][1:]
        self.tUID = tUID
        if 'vendor' in sampled_schema:
            self.vendor = True
            self.T2D = False
            self.semtab = False
            vdr_table = load_json(config.vdr_table_path(tUID), config.encoding)
            self.aggr_label = vdr_table['aggrLabel']
            self.dm_label = vdr_table['dmLabel']
            self.msr_type_label = vdr_table["metrology"]
            self.msr_type_label_revendor = (
                [""] * table["nColumns"]
                if "revendor" not in vdr_table
                else vdr_table["revendor"]
            )
        elif 'T2D' in sampled_schema:
            self.vendor = False
            self.T2D = True
            self.semtab = False
            self.headers = [info["name"] for info in table["fields"]]
        elif "semtab" in sampled_schema:
            self.semtab = True
            self.vendor = False
            self.T2D = False
        else:
            # original pivot cache
            ana_list = sampled_schema["tableAnalysisPairs"][str(table_id)]
            pUIDs = []
            cUIDs = []
            # only keep pivot table data.
            for i, ana_info in enumerate(ana_list):
                ana_type = AnaType.from_raw_str(ana_info['anaType'])
                if ana_type == AnaType.PivotTable:
                    pUIDs.append(f"{tUID}.p{i}")
                elif ana_type in AnaType.major_chart_types():
                    cUIDs.append((f"{tUID}.c{i}", ana_type))

            self.pUIDs = pUIDs
            self.cUIDs = cUIDs
            self.vendor = False
            self.T2D = False
            self.semtab = False

        fields = table_fields(tUID, table['fields'], config)

        self.n_rows = table["nRows"]
        self.n_cols = len(fields)
        # Generate field space of the table.
        self.field_space, self.idx2field = generate_field_space(fields)

        table_model = {}
        global COUNT, RIGHT, TYPE_COUTN, PROPERTY_COUNT

        if self.config.use_semantic_embeds and self.config.embed_model in TABLE_MODELS:
            if config.use_entity:
                if self.semtab and config.entity_recognition != "mtab":
                    col_info = load_json(
                        config.semtab_label_path(tUID.split('.')[0]),
                        encoding=config.encoding,
                    )
                    type_info = (
                        col_info["column_type"] if "column_type" in col_info else {}
                    )
                    property_info = (
                        col_info["relation"] if "relation" in col_info else {}
                    )
                    if os.path.exists(config.entity_recognition_path(tUID)):
                        entity_info = load_json(
                            config.entity_recognition_path(tUID),
                            encoding=config.encoding,
                        )
                    else:
                        entity_info = {}
                else:
                    entity_info, type_info, property_info = mtab_parser(config, tUID)

                # Column type
                col_type = [0 for _ in range(self.n_cols)]
                for idx in type_info.keys():
                    col_id = int(idx)
                    entity_id = type_info[idx].split('/')[-1]
                    if col_id < self.n_cols and entity_id in d_config.ENTITY_MAP:
                        TYPE_COUTN += 1
                        col_type[col_id] = d_config.ENTITY_MAP[entity_id] + 1
                table_model["col_type"] = d_config.ENTITY_EMBEDDING.take(col_type, 0)
                # Property
                property = [0 for _ in range(self.n_cols)]
                for idx in property_info.keys():
                    col_id = int(idx)
                    entity_id = property_info[idx].split('/')[-1]
                    if col_id < self.n_cols and entity_id in d_config.RELATION_MAP:
                        PROPERTY_COUNT += 1
                        property[col_id] = d_config.RELATION_MAP[entity_id] + 1
                table_model["property"] = d_config.RELATION_EMBEDDING.take(property, 0)
            if self.config.embed_model == "tapas-fine-tune":
                input_ids = get_embeddings(tUID, self.config, "input_ids")
                token_type_ids = get_embeddings(tUID, self.config, "token_type_ids")
                attention_mask = get_embeddings(tUID, self.config, "attention_mask")
                table_model["col_ids"] = token_type_ids[:, 1]
                table_model["row_ids"] = token_type_ids[:, 2]
                table_model["input_ids"] = input_ids
                table_model["token_type_ids"] = token_type_ids
                table_model["attention_mask"] = attention_mask
            if "tapas" in self.config.embed_model:
                col_ids = (
                    get_embeddings(tUID, self.config, "col_ids")
                    if "col_ids" not in table_model
                    else table_model["col_ids"]
                )
                row_ids = (
                    get_embeddings(tUID, self.config, "row_ids")
                    if "row_ids" not in table_model
                    else table_model["row_ids"]
                )
                seq_len = len(col_ids)
                table_model["col_ids"] = col_ids
                table_model["row_ids"] = row_ids
                if self.config.use_entity:
                    entity = [0 for _ in range(seq_len)]
                    for seq_id, (row_id, col_id) in enumerate(zip(row_ids, col_ids)):
                        key = f"{row_id - 1},{col_id - 1}"
                        if key in entity_info:
                            entity_id = entity_info[key].split('/')[-1]
                            COUNT += 1
                            if entity_id in d_config.ENTITY_MAP:
                                entity[seq_id] = d_config.ENTITY_MAP[entity_id] + 1
                                RIGHT += 1
                    table_model["entity"] = d_config.ENTITY_EMBEDDING.take(entity, 0)

            elif "tabbie" in self.config.embed_model:
                max_col = self.n_cols + 1 if self.n_cols + 1 <= 21 else 21
                max_row = self.n_rows + 1 if self.n_rows + 1 <= 31 else 31
                if config.use_entity:
                    entity = [[0 for _ in range(max_col)] for _ in range(max_row)]
                    for idx in entity_info.keys():
                        row_id, col_id = idx.split(",")
                        row_id = int(row_id)
                        col_id = int(col_id)
                        COUNT += 1
                        # the first one is CLS, and the first row is header. the first entity is padding
                        if row_id >= max_row - 2 or col_id >= max_col - 1:
                            continue
                        entity_id = entity_info[idx].split('/')[-1]
                        if entity_id in d_config.ENTITY_MAP:
                            RIGHT += 1
                            entity[row_id + 2][col_id + 1] = (
                                d_config.ENTITY_MAP[entity_id] + 1
                            )
                    table_model["entity"] = d_config.ENTITY_EMBEDDING.take(
                        np.array(entity).reshape(-1).tolist(), 0
                    )

        self.table_model = table_model
        self.label_source = [None] * self.n_cols
        # print(COUNT, RIGHT, TYPE_COUTN, PROPERTY_COUNT)

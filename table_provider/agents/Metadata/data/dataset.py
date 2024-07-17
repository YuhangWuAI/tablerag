import json
import logging
import os
from collections import OrderedDict
from collections import namedtuple, defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np

from .config import DataConfig, T1_LANGUAGES, DATASET, TABLE_MODELS
from .token import FieldType, AnaType, IsPercent, IsCurrency, HasYear, HasMonth, HasDay
from .util import get_embeddings

Measure = namedtuple('Measure', ['field', 'aggregation'])
COPY_TIMES = {"vdr": {"ja": 2, "zh": 2, "de": 2, "fr": 2, "es": 3, "en": 12},
              "pvt": {"ja": 4, "zh": 4, "de": 4, "fr": 1, "es": 1, "en": 1}}


class Field:
    __slots__ = "idx", "type", "role", "tags", "embedding", "features"

    def __init__(self, idx: int, field_type: Optional[FieldType] = None,
                 semantic_embedding: Optional[np.ndarray] = None, data_features: Optional[np.ndarray] = None,
                 tags: Optional[Tuple[
                     IsPercent, IsCurrency, HasYear, HasMonth, HasDay]] = None, ):
        self.idx = idx
        self.type = field_type
        self.tags = tags
        self.embedding = semantic_embedding
        self.features = data_features


def table_fields(info: Union[str, dict], field_dicts: List[dict], config: DataConfig) -> List[Field]:
    fields = [Field(fd["index"]) for fd in field_dicts]
    s_id = ".".join(info.split(".")[:-1]) if isinstance(info, str) else None

    tUid = info if isinstance(info, str) else None
    # Get the header title embedding of each field
    if config.use_semantic_embeds and config.embed_model not in TABLE_MODELS:
        idx = 0
        for fd, embed in zip(field_dicts,
                             get_embeddings(s_id, config) if isinstance(info, str) else info["embeddings"]):
            fields[idx].embedding = embed[config.embed_layer][config.embed_reduce_type]
            idx += 1

    # Get the data characteristics of each field
    # if config.use_data_features: # All tables need load df, because we will use df to decide group by or primary key
    for idx, fd in enumerate(field_dicts):
        fields[idx].features = np.array(config.data_cleanup_fn(fd["dataFeatures"]))  # , num_feature[idx, 7:]))

    # Get categorical features of each field
    for idx, fd in enumerate(field_dicts):
        if config.use_field_type:
            fields[idx].type = FieldType.from_raw_int(fd["type"])
        if config.use_binary_tags:
            fields[idx].tags = (
                IsPercent.from_raw_bool(fd["isPercent"]),
                IsCurrency.from_raw_bool(fd["isCurrency"]),
                HasYear.from_raw_bool(fd["hasYear"]),
                HasMonth.from_raw_bool(fd["hasMonth"]),
                HasDay.from_raw_bool(fd["hasDay"])
            )
    return fields


class Index:
    # TODO: Support different ways to load and query index.
    # Such as only keep tables that contains analysis types specified by the config.
    def __init__(self, config: DataConfig, rank=0):
        logger = logging.getLogger(f"Index init()")

        self.config = config
        # After down sampling, the index information can be very huge, and stored in 1 single file
        with open(config.index_path(), "r", encoding=config.encoding) as f:
            self.index = json.load(f, object_pairs_hook=OrderedDict)

        self.total_files = len(self.index)  # here total files = total schema

        # Initialize f_end and tUIDs. Every file type has its own list.
        self.f_end = []  # table end index in self.tUIDs of current schema
        self.tUIDs = []  # [schema_id*.t*, ]
        self.ana_type_idx = defaultdict(list)  # ana_type to List of tables that contain the analysis type.
        self.langs = []  # language of each table
        self.datasets = []  # dataste of each table

        self.filter_by_openfile = 0
        self.filter_by_lang = 0
        self.filter_by_no_embedding = 0
        self.filter_by_too_many_fields = 0
        self.filter_by_no_valid_analysis = 0
        self.filter_by_no_column_type = 0
        self.filter_by_no_data_feature = 0
        self.pivot_filter_by_too_many_dimension = 0
        self.chart_filter_by_no_valnum = 0
        self.chart_filter_by_too_many_values = 0
        total_charts = 0
        total_pivots = 0

        for schema_index in self.index:
            try:
                with open(config.sample_path(schema_index), "r", encoding=config.encoding) as f:
                    sampled_schema = json.load(f)
            except Exception:
                self.filter_by_openfile += 1
                continue

            # schema_id = sampled_schema['sID']
            lang = sampled_schema['lang']
            if lang == "zh_cht" or lang == "zh_chs":
                lang = "zh"
            schema_tUIDs = []
            schema_tUID_indices = defaultdict(list)

            if config.use_semantic_embeds and config.embed_model not in TABLE_MODELS and not os.path.exists(
                    config.embedding_path(schema_index)) and "tabbie" not in config.embed_model:
                # bypass embedding file not found error
                self.filter_by_no_embedding += 1
                continue

            if not config.has_language(lang):
                self.filter_by_lang += 1
                continue

            if 'vendor' in sampled_schema:
                tUID = f"{schema_index}.t0"
                if not self.check_table_emb(tUID):
                    self.filter_by_no_embedding += 1
                    continue
                if not os.path.exists(config.table_path(tUID)):
                    self.filter_by_no_data_feature += 1
                    continue
                # if config.use_num_feature and not os.path.exists(config.field_number_feature_path(tUID)):
                #     self.filter_by_no_embedding+=1
                #     continue
                # vendor table, currently we account the table into PivotTable
                self.ana_type_idx[AnaType.from_raw_str('pivotTable')].append(len(self.tUIDs))
                self.tUIDs.append(f"{schema_index}.t0")
                self.f_end.append(len(self.tUIDs))
                self.langs.append(lang)
                self.datasets.append("vdr")
                continue

            if 'bing' in sampled_schema:
                # T2D table, currently we account the table into PivotTable
                self.ana_type_idx[AnaType.from_raw_str('pivotTable')].append(len(self.tUIDs))
                self.tUIDs.append(f"{schema_index}.t0")
                self.f_end.append(len(self.tUIDs))
                self.langs.append(lang)
                self.datasets.append("bing")
                continue

            if 'T2D' in sampled_schema:
                tUID = f"{schema_index}.t0"
                if not os.path.exists(config.column_type_path(tUID)):
                    self.filter_by_no_column_type += 1
                    continue
                if not os.path.exists(config.table_path(tUID)):
                    self.filter_by_no_data_feature += 1
                    continue
                # if schema_index in [23]:  # TODO: column type is not right
                #     continue
                # T2D table, currently we account the table into PivotTable
                # if config.use_num_feature and not os.path.exists(config.field_number_feature_path(tUID)):
                #     self.filter_by_no_embedding+=1
                #     continue
                if not self.check_table_emb(tUID):
                    self.filter_by_no_embedding += 1
                    continue
                self.ana_type_idx[AnaType.from_raw_str('pivotTable')].append(len(self.tUIDs))
                self.tUIDs.append(f"{schema_index}.t0")
                self.f_end.append(len(self.tUIDs))
                self.langs.append(lang)
                self.datasets.append("T2D")
                continue

            if "semtab" in sampled_schema:
                if not os.path.exists(config.semtab_label_path(schema_index)):
                    self.filter_by_no_column_type += 1
                    continue
                for idx in range(len(sampled_schema['tableAnalysisPairs'])):
                    tUID = f"{schema_index}.t{idx}"
                    if not self.check_table_emb(tUID):
                        self.filter_by_no_embedding += 1
                        continue
                    if not os.path.exists(config.table_path(tUID)):
                        self.filter_by_no_data_feature += 1
                        continue
                    self.ana_type_idx[AnaType.from_raw_str('pivotTable')].append(len(self.tUIDs))
                    self.tUIDs.append(tUID)
                    self.f_end.append(len(self.tUIDs))
                    self.langs.append(lang)
                    self.datasets.append("semtab")
                    break  # We use one table in each schema
                continue

            for table_id, ana_info_list in sampled_schema['tableAnalysisPairs'].items():
                # ana_info_list: List[Dict]
                if sampled_schema['nColumns'] > config.max_field_num:
                    self.filter_by_too_many_fields += 1
                    continue
                tUID = f"{schema_index}.t{table_id}"

                if not self.check_table_emb(tUID):
                    self.filter_by_no_embedding += 1
                    continue

                # if config.use_num_feature and not os.path.exists(config.field_number_feature_path(tUID)):
                #     self.filter_by_no_embedding+=1
                #     continue

                if not os.path.exists(config.table_path(tUID)):
                    self.filter_by_no_data_feature += 1
                    continue

                ana_type_list = []
                for i, ana_info in enumerate(ana_info_list):
                    ana_type = AnaType.from_raw_str(ana_info['anaType'])
                    if ana_type == AnaType.PivotTable:
                        # a pivot table analysis
                        if config.max_dim_num < 0 or (config.max_dim_num >= ana_info['nColDim'] and
                                                      config.max_dim_num >= ana_info['nRowDim']):
                            ana_type_list.append(ana_type)
                            total_pivots += 1
                        else:
                            self.pivot_filter_by_too_many_dimension += 1
                    else:
                        # a chart analysis, params is a int which is #value
                        if ana_info['nVals'] == 0:
                            self.chart_filter_by_no_valnum += 1
                        elif config.max_val_num < ana_info['nVals']:
                            self.chart_filter_by_too_many_values += 1
                        else:
                            ana_type_list.append(ana_type)
                            total_charts += 1

                if len(ana_type_list) > 0:
                    schema_tUIDs.append(tUID)
                elif not config.col_type_similarity:
                    self.filter_by_no_valid_analysis += 1
                    continue

                for cur_ana_type in ana_type_list:
                    schema_tUID_indices[cur_ana_type].append(len(schema_tUIDs) - 1)

            # Merge the table and analysis information of the current schema into the whole schema records.
            for ana_type in AnaType:  # TODO: comments
                self.ana_type_idx[ana_type].extend(
                    map(lambda x: x + len(self.tUIDs), schema_tUID_indices[ana_type]))
            self.tUIDs.extend(schema_tUIDs)
            self.f_end.append(len(self.tUIDs))
            self.langs.extend([lang] * len(schema_tUIDs))
            self.datasets.extend(["pvt"] * len(schema_tUIDs))

        if int(rank) == 0:
            logger.info(f"Total schemas is {len(self.index)}")
            logger.info(f"file filtered by language: {self.filter_by_lang}")
            logger.info(f"file filtered by openfile: {self.filter_by_openfile}")
            logger.info(f"table filtered by no embedding: {self.filter_by_no_embedding}")
            logger.info(f"table filtered by too many fields: {self.filter_by_too_many_fields}")
            logger.info(f"table filtered by no valid analysis: {self.filter_by_no_valid_analysis}")
            logger.info(f"table filtered by no column type: {self.filter_by_no_column_type}")
            logger.info(f"table filtered by no data feature: {self.filter_by_no_data_feature}")
            logger.info(f"chart filtered by no valnum: {self.chart_filter_by_no_valnum}")
            logger.info(f"chart filtered by too many values: {self.chart_filter_by_too_many_values}")
            logger.info(f"pivot filtered by too many dimensions: {self.pivot_filter_by_too_many_dimension}")
            logger.info(f"Total tables is {len(self.tUIDs)}")
            logger.info(f"Total charts is {total_charts}")
            logger.info(f"Total pivot tables is {total_pivots}")

    def train_tUIDs(self, lang=None, dataset=None, copy=False):
        train_threshold = self.config.train_ratio * len(self.f_end)
        if train_threshold < len(self.f_end):
            end_index = self.f_end[int(train_threshold)]
        else:
            end_index = self.f_end[-1]
        return self.get_tUIDs(0, end_index, copy=copy, lang=lang, dataset=dataset)

    def valid_tUIDs(self):
        train_threshold = self.config.train_ratio * len(self.f_end)
        valid_threshold = (self.config.train_ratio + self.config.valid_ratio) * len(self.f_end)
        if train_threshold < len(self.f_end):
            start_index = self.f_end[int(train_threshold)]
        else:
            start_index = self.f_end[-1]
        if valid_threshold < len(self.f_end):
            end_index = self.f_end[int(valid_threshold)]
        else:
            end_index = self.f_end[-1]
        return self.get_tUIDs(start_index, end_index)

    def test_tUIDs(self, lang=None, dataset=None):
        valid_threshold = (self.config.train_ratio + self.config.valid_ratio) * len(self.f_end)
        if valid_threshold < len(self.f_end):
            start_index = self.f_end[int(valid_threshold)]
        else:
            start_index = self.f_end[-1]
        end_index = self.f_end[-1]
        return self.get_tUIDs(start_index, end_index, lang=lang, dataset=dataset)

    def get_tUIDs(self, start_index=None, end_index=None, lang=None, dataset=None, copy=False):
        '''
        For every analysis type in config, the corresponding tUIDs within [start_index, end_index)
        will be extracted and concatenated as a big list.
        :param lang: Language that the extracted tUIDs should be, a value of None being no restriction.
        :param dataset: The type of dataset that the extracted tUIDs should come from, a value of None being no restriction.
        :param copy: Copy specific times of specific tUIDs if copy is True.
        '''
        if lang not in T1_LANGUAGES and lang is not None:
            raise NotImplementedError(f"{lang} not yet implemented.")
        if dataset not in DATASET and dataset is not None:
            raise NotImplementedError(f"{dataset} not yet implemented.")

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(self.tUIDs)

        indices = []
        for ana_type in self.ana_type_idx:
            if lang is None and dataset is None:
                indices_of_type = list(
                    filter(lambda idx: start_index <= idx < end_index, self.ana_type_idx[ana_type]))
            elif dataset is None:
                indices_of_type = list(
                    filter(lambda idx: start_index <= idx < end_index and lang == self.langs[idx],
                           self.ana_type_idx[ana_type]))
            elif lang is None:
                indices_of_type = list(
                    filter(lambda idx: start_index <= idx < end_index and dataset == self.datasets[idx],
                           self.ana_type_idx[ana_type]))
            else:
                indices_of_type = list(
                    filter(lambda idx: (start_index <= idx < end_index) and (dataset == self.datasets[idx]) and lang ==
                                       self.langs[idx],
                           self.ana_type_idx[ana_type]))
            indices.extend(indices_of_type)
        indices = list(set(indices))

        # For empirical study, we use all tables.
        if self.config.empirical_study:
            indices = list(range(start_index, end_index))

        if copy is True:
            indices_copy = []
            for idx in indices:
                if self.langs[idx] in T1_LANGUAGES:
                    indices_copy.extend([idx] * (COPY_TIMES[self.datasets[idx]][self.langs[idx]]))
            indices = indices_copy
        indices.sort()
        tUIDs = [self.tUIDs[idx] for idx in indices]
        return tUIDs

    def sUIDs2tUIDs(self, sUIDs):
        sUIDs = set(sUIDs)
        return [i for i in self.tUIDs if i.split('.')[0] in sUIDs]

    def save_dataset_split(self):
        logger = logging.getLogger(f"Index save_dataset_split()")
        with open(os.path.join(self.config.corpus_path, "train.json"), 'w', encoding="utf-8-sig") as f:
            train_tuids = self.train_tUIDs()
            json.dump(train_tuids, f)
        with open(os.path.join(self.config.corpus_path, "valid.json"), 'w', encoding="utf-8-sig") as f:
            valid_tuids = self.valid_tUIDs()
            json.dump(valid_tuids, f)
        with open(os.path.join(self.config.corpus_path, "test.json"), 'w', encoding="utf-8-sig") as f:
            test_tuids = self.test_tUIDs()
            json.dump(test_tuids, f)
        logger.info("train/valid/test tUIDs saved.")

    def check_table_emb(self, tuid):
        '''
        Check if there are table embedding
        :return: bool
        '''
        if not self.config.use_semantic_embeds:
            return True
        if self.config.embed_model not in TABLE_MODELS:
            return True
        if self.config.embed_model == "tapas-fine-tune":  # TODO: Why??
            # self.config.embed_model = "tapas-display"
            # flag = os.path.exists(self.config.embedding_path(tuid))
            # self.config.embed_model = "tapas-fine-tune"
            # flag = flag and os.path.exists(self.config.embedding_path(tuid, "input_ids"))
            return os.path.exists(self.config.embedding_path(tuid, "input_ids"))
        if "tapas" in self.config.embed_model:
            return os.path.exists(self.config.embedding_path(tuid))
        if "tabbie" in self.config.embed_model:
            return os.path.exists(self.config.embedding_path(tuid, None))
        raise NotImplementedError(f"We don't know embedding of {self.config.embed_model}")

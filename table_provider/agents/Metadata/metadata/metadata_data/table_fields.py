import csv
import json
import logging
import math
import os
import random
from collections import Counter
from collections import defaultdict
from collections import namedtuple
from copy import deepcopy
from enum import Enum, IntEnum
from itertools import combinations
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from ...data import DataConfig, append_padding, AggFunc
from ...data.config import TABLE_MODELS, FEATURE_MAP
from ...data.token import AnaType, FieldType
from ...data.util import load_json, get_embeddings
from ..column_type import (
    ColumnType,
    SubCategory,
    MsrTypeMapping,
    SubCategory2Category,
    Unuse_msr_type,
)
from .table import MetaDataTable

SEED = 100
random.seed(SEED)

DEFAULT_CORPUS = ["chart", "pivot", "vendor", "t2d", "semtab", "all"]


def get_corpus(args):
    all_corpus = {  # corpus_name: (path, copy_times)
        "chart": (args.chart_path, 1),
        "pivot": (args.pivot_path, 1),
        "vendor": (args.vendor_path, 10),
        "t2d": (args.t2d_path, 10),
        "semtab": (args.semtab_path, 1),
    }
    if args.corpus == "all":
        return all_corpus
    else:
        try:
            output_corpus = {}
            for corpus in args.corpus.split("+"):
                output_corpus[corpus] = all_corpus[corpus]
            return output_corpus
        except:
            raise NotImplementedError(f"Not Implement corpus {args.corpus}")


# The valid aggregation functions that considered int MetaData task, and the order(index) in labels
VALID_AGG_INDEX = {
    AggFunc.Sum: 0,
    AggFunc.Average: 1,
    AggFunc.Max: 2,
    AggFunc.Min: 3,
    AggFunc.Product: 4,
    AggFunc.StdDev: 5,
    AggFunc.StdDevP: 6,
    AggFunc.Var: 7,
    AggFunc.VarP: 8,
}
AGG_Weight = math.sqrt(60064) / torch.sqrt_(
    torch.tensor([60064, 9947, 1318, 786, 79, 254, 70, 29, 10], dtype=torch.float)
)

Measure = namedtuple("Measure", ["field_index", "agg_func"])
Dimension = namedtuple("Dimension", ["field_index", "col_dim"])
AGG_NUM = 9  # number of aggregation functions
LABEL_LENGTH = 17
Msr_Restrict_Type = [FieldType.String, FieldType.Year, FieldType.DateTime]


class Metadata_Label(IntEnum):
    Dim_label = 0
    Msr_label = 1
    Gby_score = 2
    Key_score = 3
    Msr_score = 4
    Msr_type_subcategory = 5
    Msr_type_category = 6
    Agg_label_start = 7
    Agg_label_end = 15


# Class DimMeasureLabel(Enum):


class DimMeasureLabel(Enum):
    OnlyDim = "OnlyDim"
    OnlyMsr = "OnlyMsr"
    Both = "Both"
    NoLabel = "No-Label"

    @classmethod
    def from_raw_label(cls, label_dim, label_msr):
        """
        Transform the label of onlyDim/onlyMsr/Both/No-Label from dimension label and meausre label
        """
        if label_dim == 1:
            if label_msr == 1:
                return cls.Both
            else:
                return cls.OnlyDim
        else:
            if label_msr == 1:
                return cls.OnlyMsr
            else:
                return cls.NoLabel


class SumAverageLabel(Enum):
    OnlySum = "OnlySum"
    OnlyAvg = "OnlyAverage"
    SumAvg = "SumAverage"
    Neither = "Neither"

    @classmethod
    def from_raw_label(cls, label_sum, label_avg):
        """
        Transform the label of onlyDim/onlyMsr/Both/No-Label from dimension label and meausre label
        """
        if label_sum == 1:
            if label_avg == 1:
                return cls.SumAvg
            elif label_avg == 0:
                return cls.OnlySum
            else:
                raise Exception("wrong aggregation label")
        elif label_sum == 0:
            if label_avg == 1:
                return cls.OnlyAvg
            elif label_avg == 0:
                return cls.Neither
            else:
                raise Exception("wrong aggregation label")
        else:
            raise Exception("wrong aggregation label")


def compact_pivot_info(
    pUID, config: DataConfig
) -> Tuple[List[Measure], List[Dimension]]:
    """
    Extract which fields are used as dimension and measure in a pivot table,
    and what is the aggregation function.
    """
    pivot_table = load_json(config.pivot_table_path(pUID), config.encoding)

    measures = []
    measure_index_set = set()
    for measure in pivot_table["values"]:
        aggregation = AggFunc.from_raw_str(measure.get("aggregation"))
        # Only consider the valid aggregation functions in VALID_AGG_INDEX
        if aggregation in VALID_AGG_INDEX:
            measures.append(Measure(measure["index"], aggregation))
            measure_index_set.add(measure["index"])

    dimensions = []
    dimension_index_set = set()
    # Exclude the grouped field.
    for dimension in pivot_table["columns"]:
        if not dimension['isGrouped']:
            dimensions.append(Dimension(dimension["index"], "column"))
            dimension_index_set.add(dimension["index"])
    for dimension in pivot_table["rows"]:
        if not dimension['isGrouped']:
            dimensions.append(Dimension(dimension["index"], "row"))
            dimension_index_set.add(dimension["index"])

    # If there are fields that are both act as dimension and measure in one table,
    # then these fields should be excluded from training (marked as no-label)
    both_field_index = dimension_index_set.intersection(measure_index_set)
    dimensions = [
        dimension
        for dimension in dimensions
        if dimension.field_index not in both_field_index
    ]
    measures = [
        measure for measure in measures if measure.field_index not in both_field_index
    ]

    # If there are no measure in the table, then clean up the dimension.
    # This aimed at avoiding only-dimension pivot table.
    if len(measures) == 0:
        dimensions = []
    return measures, dimensions


def compact_chart_info(
    cUID, ana_type: AnaType, config: DataConfig
) -> Tuple[List[Measure], List[Dimension]]:
    """
    Extract which fields are used as dimension and measure in a chart,
    and what is the aggregation function.
    """
    chart_info = load_json(config.chart_path(cUID), config.encoding)

    measures = []
    measure_index_set = set()
    dimensions = []
    dimension_index_set = set()

    if ana_type in [AnaType.PieChart, AnaType.BarChart]:
        for measure in chart_info["yFields"]:
            measures.append(Measure(measure["index"], "y"))
            measure_index_set.add(measure["index"])
        for dimension in chart_info["xFields"]:
            dimensions.append(Dimension(dimension["index"], "x"))
            dimension_index_set.add(dimension["index"])
    elif ana_type in [
        AnaType.ScatterChart,
        AnaType.LineChart,
    ]:  # x of those charts is not sure
        for measure in chart_info["yFields"]:
            measures.append(Measure(measure["index"], "y"))
            measure_index_set.add(measure["index"])
    else:
        raise NotImplementedError(
            "Unexpected chart type: {}".format(AnaType.to_raw_str(ana_type))
        )

    # If there are fields that are both act as dimension and measure in one chart,
    # then these fields should be excluded from training (marked as no-label)
    both_field_index = dimension_index_set.intersection(measure_index_set)
    dimensions = [
        dimension
        for dimension in dimensions
        if dimension.field_index not in both_field_index
    ]
    measures = [
        measure for measure in measures if measure.field_index not in both_field_index
    ]

    return measures, dimensions


def extract_vendor_labels(tb: MetaDataTable, field_labels: List[List[int]]):
    field_labels_ori = deepcopy(field_labels)
    for i, dm_label in enumerate(tb.dm_label):
        if dm_label in {0, 2}:
            field_labels[i][Metadata_Label.Dim_label] = 1
        if dm_label in {1, 2}:
            field_labels[i][Metadata_Label.Msr_label] = 1
    for i, aggr_label in enumerate(tb.aggr_label):
        if (aggr_label is not None) and (aggr_label != 3):
            field_labels[i][
                Metadata_Label.Agg_label_start : Metadata_Label.Agg_label_end + 1
            ] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if aggr_label in {0, 2}:
            field_labels[i][
                Metadata_Label.Agg_label_start + VALID_AGG_INDEX[AggFunc.Sum]
            ] = 1
        if aggr_label in {1, 2}:
            field_labels[i][
                Metadata_Label.Agg_label_start + VALID_AGG_INDEX[AggFunc.Average]
            ] = 1
    for i, msr_type_label in enumerate(tb.msr_type_label):
        if tb.msr_type_label_revendor[i] != "":
            field_labels[i][Metadata_Label.Msr_type_subcategory] = MsrTypeMapping[
                "ReVendor"
            ][tb.msr_type_label_revendor[i]]
            field_labels[i][Metadata_Label.Msr_type_category] = SubCategory2Category[
                field_labels[i][Metadata_Label.Msr_type_subcategory]
            ]
        elif msr_type_label != "":
            field_labels[i][Metadata_Label.Msr_type_subcategory] = MsrTypeMapping[
                "vendor"
            ][msr_type_label]
            field_labels[i][Metadata_Label.Msr_type_category] = SubCategory2Category[
                field_labels[i][Metadata_Label.Msr_type_subcategory]
            ]
        if field_labels[i][Metadata_Label.Msr_type_subcategory] in Unuse_msr_type:
            field_labels[i][Metadata_Label.Msr_type_subcategory] = -1
            field_labels[i][Metadata_Label.Msr_type_category] = -1
    # for i in range(len(field_labels)):
    #     field_labels[i][0] = -1
    #     field_labels[i][1] = -1
    return field_labels, field_labels_ori != field_labels


def extract_t2d_labels(
    tb: MetaDataTable, config: DataConfig, field_labels: List[List[int]]
):
    field_labels_ori = deepcopy(field_labels)
    # Measure type and primary key
    with open(config.column_type_path(tb.tUID), 'r', encoding='utf-8-sig') as f:
        info = csv.reader(f)
        has_primary_key = False
        for row in info:
            if (
                len(row) >= 4
                and int(row[3]) < tb.n_cols
                and tb.headers[int(row[3])].lower() == row[1].lower()
            ):
                if (
                    row[0] in MsrTypeMapping["T2Dv1"]
                    and tb.idx2field[int(row[3])].field_type not in Msr_Restrict_Type
                ):
                    field_labels[int(row[3])][
                        Metadata_Label.Msr_type_subcategory
                    ] = MsrTypeMapping["T2Dv1"][row[0]]
                    field_labels[int(row[3])][
                        Metadata_Label.Msr_type_category
                    ] = SubCategory2Category[
                        field_labels[int(row[3])][Metadata_Label.Msr_type_subcategory]
                    ]
                    if (
                        field_labels[int(row[3])][Metadata_Label.Msr_type_subcategory]
                        in Unuse_msr_type
                    ):
                        field_labels[int(row[3])][
                            Metadata_Label.Msr_type_subcategory
                        ] = -1
                        field_labels[int(row[3])][Metadata_Label.Msr_type_category] = -1
                    field_labels[int(row[3])][Metadata_Label.Msr_label] = 1
                if row[2]:  # primary key
                    if not has_primary_key:
                        has_primary_key = True
                        for i in range(len(field_labels)):
                            field_labels[i][Metadata_Label.Key_score] = 0
                    field_labels[int(row[3])][Metadata_Label.Key_score] = 1
                    field_labels[int(row[3])][Metadata_Label.Dim_label] = 1

    has_label = field_labels != field_labels_ori
    # for i in range(len(field_labels)):
    #     field_labels[i][Metadata_Label.Dim_label] = -1
    #     field_labels[i][Metadata_Label.Msr_label] = -1
    return field_labels, has_label


def extract_semtab_labels(
    tb: MetaDataTable, config: DataConfig, field_labels: List[List[int]]
):
    field_labels_ori = deepcopy(field_labels)
    labels = load_json(config.semtab_label_path(tb.tUID.split('.')[0]), config.encoding)
    for idx in labels["msr_type"].keys():
        idx_int = int(idx)
        msr_type = MsrTypeMapping["semtab"][labels["msr_type"][idx]]
        if msr_type not in Unuse_msr_type:
            field_labels[idx_int][Metadata_Label.Msr_type_subcategory] = msr_type
            field_labels[idx_int][
                Metadata_Label.Msr_type_category
            ] = SubCategory2Category[msr_type]
    has_label = field_labels != field_labels_ori
    for i in range(len(field_labels)):
        field_labels[i][Metadata_Label.Dim_label] = -1
        field_labels[i][Metadata_Label.Msr_label] = -1
    return field_labels, has_label


def extract_pivot_labels(
    tb: MetaDataTable, config: DataConfig, field_labels: List[List[int]]
):
    field_labels_ori = deepcopy(field_labels)
    has_common_groupby = False
    has_common_measure = False
    for UID in tb.pUIDs:
        measures, dimensions = compact_pivot_info(UID, config)
        for dim in dimensions:
            # common group by
            if (
                tb.field_space[dim.field_index].data_features[
                    FEATURE_MAP['cardinality'] - 1
                ]
                != 1
            ):
                if not has_common_groupby:
                    has_common_groupby = True
                    for i in range(len(field_labels)):
                        field_labels[i][Metadata_Label.Gby_score] = 0
                field_labels[dim.field_index][Metadata_Label.Gby_score] = 1
            # dimension
            field_labels[dim.field_index][Metadata_Label.Dim_label] = 1
        for msr in measures:
            if tb.idx2field[msr.field_index].field_type not in Msr_Restrict_Type:
                # Msr lable
                field_labels[msr.field_index][Metadata_Label.Msr_label] = 1

                # Agg
                if (
                    sum(
                        field_labels[msr.field_index][
                            Metadata_Label.Agg_label_start : Metadata_Label.Agg_label_end
                            + 1
                        ]
                    )
                    < -1
                ):
                    # Activate the aggregation function labels from -1 to 0
                    field_labels[msr.field_index][
                        Metadata_Label.Agg_label_start : Metadata_Label.Agg_label_end
                        + 1
                    ] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                if msr.agg_func != None:
                    field_labels[msr.field_index][
                        VALID_AGG_INDEX[msr.agg_func] + Metadata_Label.Agg_label_start
                    ] = 1

                # Common msr
                if not has_common_measure:
                    has_common_measure = True
                    for i in range(len(field_labels)):
                        field_labels[i][Metadata_Label.Msr_score] = 0
                field_labels[msr.field_index][Metadata_Label.Msr_score] = 1
    return field_labels, field_labels_ori != field_labels


def extract_chart_labels(
    tb: MetaDataTable, config: DataConfig, field_labels: List[List[int]]
):
    field_labels_ori = deepcopy(field_labels)
    has_common_measure = False
    has_primary_key = False
    msr_pairs_idx = []
    for UID in tb.cUIDs:
        measures, dimensions = compact_chart_info(UID[0], UID[1], config)
        # Make sure that all string, year datetime field won't labeled with measure = 1
        measures = [
            msr
            for msr in measures
            if tb.idx2field[msr.field_index].field_type not in Msr_Restrict_Type
        ]
        # primary key
        if (
            len(dimensions) == 1
            and tb.field_space[dimensions[0].field_index].data_features[
                FEATURE_MAP['cardinality'] - 1
            ]
            == 1
        ):  # We don't consider chart with multi x
            if not has_primary_key:
                has_primary_key = True
                for i in range(len(field_labels)):
                    field_labels[i][Metadata_Label.Key_score] = 0
            field_labels[dimensions[0].field_index][Metadata_Label.Key_score] = 1
        # pair
        msr_x = [msr for msr in measures if msr.agg_func == "x"]
        msr_x.sort(key=lambda msr: msr.field_index)
        msr_pairs_idx.extend(combinations([msr.field_index for msr in msr_x], 2))
        msr_y = [msr for msr in measures if msr.agg_func == "y"]
        msr_y.sort(key=lambda msr: msr.field_index)
        msr_pairs_idx.extend(combinations([msr.field_index for msr in msr_y], 2))
        # dim
        for dim in dimensions:
            field_labels[dim.field_index][Metadata_Label.Dim_label] = 1
        # msr
        for msr in measures:
            field_labels[msr.field_index][Metadata_Label.Msr_label] = 1
            if not has_common_measure:
                has_common_measure = True
                for i in range(len(field_labels)):
                    field_labels[i][Metadata_Label.Msr_score] = 0
            field_labels[msr.field_index][Metadata_Label.Msr_score] = 1
    return field_labels, msr_pairs_idx, field_labels_ori != field_labels


def extract_field_labels(
    tb: MetaDataTable, config: DataConfig
) -> Tuple[List[List[int]], List[Tuple[int, int]], List[int]]:
    """
    extract the label of each field in the table. The label of each field is a List of lenght 11.
    Each represents: isDimension, isMeasure, isSum, isAverage, isMax, isMin, isProduct, isStdDev, isStdDevP,
    isVar, isVarP.
    The first two labels tells whether a field used as dimension/measure. So they can be either 0, 1
    The last nine labels tells how a field is aggregated if it is used as measure.
    Note that the the last nine labels are -1, if the field is not used as measure.
    :param tb: input table instance.
    :param config: predefined data configuration.

    Returns:
        field_labels: List[List[int]], shape (Field_Len, len(Metadata_Label)). Labels of each field.
        msr_pairs_idx: List[Tuple[int, int]], shape (Pair_num, 2). Index pairs of measure pairs.
        msr_pairs_label: List[int], shape (Pair_num). Labels of measure pairs.
    """
    # Initialize the labels.
    field_labels = [
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        for _ in tb.idx2field
    ]
    msr_pairs_idx = []
    msr_pairs_label = []
    if tb.vendor:
        field_labels, has_label = extract_vendor_labels(tb, field_labels)
    elif tb.T2D:
        field_labels, has_label = extract_t2d_labels(tb, config, field_labels)
    elif tb.semtab:
        field_labels, has_label = extract_semtab_labels(tb, config, field_labels)
    elif len(tb.pUIDs) != 0 and len(tb.cUIDs) == 0:
        field_labels, has_label = extract_pivot_labels(tb, config, field_labels)
    elif len(tb.cUIDs) != 0 and len(tb.pUIDs) == 0:
        field_labels, msr_pairs_idx_chart, has_label = extract_chart_labels(
            tb, config, field_labels
        )
        msr_pairs_idx.extend(msr_pairs_idx_chart)
        msr_pairs_label.extend([1] * len(msr_pairs_idx_chart))
    elif len(tb.cUIDs) == 0 and len(tb.pUIDs) == 0:
        has_label = False
    else:
        raise NotImplementedError(
            f"{tb.tUID} has chart and pivot table at the same time"
        )

    if len(msr_pairs_idx) != 0:
        # Construct negative samples
        num_idx = [
            idx
            for idx in range(len(tb.idx2field))
            if tb.idx2field[idx].field_type == FieldType.Decimal
        ]
        num_pair = list(combinations(num_idx, 2))
        random.shuffle(num_pair)
        num_pair = list(set(num_pair) - set(msr_pairs_idx))

        # Add negative samples and shuffle them
        msr_pairs_idx.extend(num_pair[: len(msr_pairs_idx)])
        msr_pairs_label.extend([0] * (len(msr_pairs_idx) - len(msr_pairs_label)))
        msr_pairs = list(zip(msr_pairs_idx, msr_pairs_label))
        random.shuffle(msr_pairs)
        msr_pairs_idx, msr_pairs_label = zip(*msr_pairs)
    else:
        # a fake pair
        msr_pairs_idx.extend([[0, 0]])
        msr_pairs_label.extend([-1])

    return field_labels, list(msr_pairs_idx), list(msr_pairs_label), has_label


def extract_field_column_labels(
    tb: MetaDataTable, config: DataConfig
) -> List[List[int]]:
    with open(config.column_type_path(tb.tUID), 'r', encoding='utf-8-sig') as f:
        info = csv.reader(f)
        labels = [-1] * tb.n_cols
        types = [0] * tb.n_cols
        for col in tb.field_space:
            types[col.field_index] = col.field_type
        for row in info:
            if (
                len(row) >= 4
                and int(row[3]) < tb.n_cols
                and tb.headers[int(row[3])].lower() == row[1].lower()
            ):
                labels[int(row[3])] = ColumnType.index(row[0])
    return [labels, types]


class MetadataTableFields(Dataset):
    def __init__(self, tUIDs: List[str], config: DataConfig, is_train: bool):
        """
        Create a MetaData dataset from each data table.
        :param tUIDs: List of table IDs.
        :param config: Data configurations.
        :param is_train: This dataset is for training or testing/validation.
        :param column_type1: Whether to use column type as labels.
        """
        self.configs = [config]  # different config for different corpus
        self.is_train = is_train

        # Group tUIDs according to whether they have the same schema_id
        sID2tUIDs = defaultdict(list)
        for tUID in tUIDs:
            sUID = tUID.split(".")[0]
            sID2tUIDs[sUID].append(tUID)
        self.table_list = []
        for sID, tID_list in tqdm(sorted(sID2tUIDs.items())):
            # Load the sample information once for tables that have the same schema
            sampled_schema = load_json(config.sample_path(sID), config.encoding)
            for tID in tID_list:
                self.table_list.append(MetaDataTable(tID, sampled_schema, config))
        if config.col_type_similarity:
            field_labels = [
                extract_field_column_labels(table, config) for table in self.table_list
            ]
            self.field_labels = [i for i in field_labels]
            self.msr_pairs_idx = self.msr_pairs_label = [
                i for i in field_labels
            ]  # fake
        else:
            field_labels = [
                extract_field_labels(table, config) for table in self.table_list
            ]
            self.field_labels = [i[0] for i in field_labels]
            self.msr_pairs_idx = [i[1] for i in field_labels]
            self.msr_pairs_label = [i[2] for i in field_labels]
            has_label = [i[3] for i in field_labels]

            # Filter table without label
            self.table_list = [x for x, y in zip(self.table_list, has_label) if y]
            self.field_labels = [x for x, y in zip(self.field_labels, has_label) if y]
            self.msr_pairs_idx = [x for x, y in zip(self.msr_pairs_idx, has_label) if y]
            self.msr_pairs_label = [
                x for x, y in zip(self.msr_pairs_label, has_label) if y
            ]
            logger = logging.getLogger("MetadataTableFields")
            logger.info(
                f"There are {has_label.count(False)} tables without labels, and remaining {has_label.count(True)} tables"
            )

        self.tUIDs = [i.tUID for i in self.table_list]

        self.corpus = [0] * len(
            self.table_list
        )  # corpus number of each table, and the number corresponds to the config number

    def __len__(self):
        return len(self.tUIDs)

    def filter_schema(self, suids):
        filter_result = list(
            filter(
                lambda x: x[0].split('.')[0] in suids,
                zip(
                    self.tUIDs,
                    self.table_list,
                    self.field_labels,
                    self.msr_pairs_idx,
                    self.msr_pairs_label,
                    self.corpus,
                ),
            )
        )
        self.tUIDs = [i[0] for i in filter_result]
        self.table_list = [i[1] for i in filter_result]
        self.field_labels = [i[2] for i in filter_result]
        self.msr_pairs_idx = [i[3] for i in filter_result]
        self.msr_pairs_label = [i[4] for i in filter_result]
        self.corpus = [i[5] for i in filter_result]

    def filter_field(self, fuids):
        # filter table
        tuids = [i.split("-")[1] for i in fuids]
        filter_result = list(
            filter(
                lambda x: x[0] in tuids,
                zip(
                    self.tUIDs,
                    self.table_list,
                    self.field_labels,
                    self.msr_pairs_idx,
                    self.msr_pairs_label,
                    self.corpus,
                ),
            )
        )
        self.tUIDs = [i[0] for i in filter_result]
        self.table_list = [i[1] for i in filter_result]
        self.field_labels = [i[2] for i in filter_result]
        self.msr_pairs_idx = [i[3] for i in filter_result]
        self.msr_pairs_label = [i[4] for i in filter_result]
        self.corpus = [i[5] for i in filter_result]

        # filter field
        for i, tuid_labels_corpus in enumerate(
            zip(self.tUIDs, self.field_labels, self.corpus)
        ):
            tuid, labels, corpus = tuid_labels_corpus
            for j, label in enumerate(labels):
                if label[14] != -1 and f"{corpus}-{tuid}-{j}" not in fuids:
                    self.field_labels[i][j][14] = -1
                    self.field_labels[i][j][13] = -1

    def add(self, other):
        self.tUIDs.extend(other.tUIDs)
        self.table_list.extend(other.table_list)
        self.field_labels.extend(other.field_labels)
        self.msr_pairs_idx.extend(other.msr_pairs_idx)
        self.msr_pairs_label.extend(other.msr_pairs_label)
        self.corpus.extend([i + len(self.configs) for i in other.corpus])
        self.configs.extend(other.configs)

        data = list(
            zip(
                self.tUIDs,
                self.table_list,
                self.field_labels,
                self.field_labels,
                self.msr_pairs_idx,
                self.msr_pairs_label,
                self.corpus,
            )
        )
        random.shuffle(data)
        (
            self.tUIDs,
            self.table_list,
            self.field_labels,
            self.field_labels,
            self.msr_pairs_idx,
            self.msr_pairs_label,
            self.corpus,
        ) = [list(i) for i in zip(*data)]

    def header(self, num: int):
        self.tUIDs = self.tUIDs[:num]
        self.table_list = self.table_list[:num]
        self.field_labels = self.field_labels[:num]
        self.msr_pairs_idx = self.msr_pairs_idx[:num]
        self.msr_pairs_label = self.msr_pairs_label[:num]
        self.corpus = self.corpus[:num]

    def copy(self, copy_time):
        self.tUIDs = self.tUIDs * copy_time
        self.table_list = self.table_list * copy_time
        self.field_labels = self.field_labels * copy_time
        self.msr_pairs_idx = self.msr_pairs_idx * copy_time
        self.msr_pairs_label = self.msr_pairs_label * copy_time
        self.corpus = self.corpus * copy_time

        data = list(
            zip(
                self.tUIDs,
                self.table_list,
                self.field_labels,
                self.field_labels,
                self.msr_pairs_idx,
                self.msr_pairs_label,
                self.corpus,
            )
        )
        random.shuffle(data)
        (
            self.tUIDs,
            self.table_list,
            self.field_labels,
            self.field_labels,
            self.msr_pairs_idx,
            self.msr_pairs_label,
            self.corpus,
        ) = [list(i) for i in zip(*data)]

    def __getitem__(self, index):
        return (
            self.table_list[index].field_space,
            self.table_list[index].table_model,
            self.field_labels[index],
            self.msr_pairs_idx[index],
            self.msr_pairs_label[index],
            self.corpus[index],
            self.tUIDs[index],
        )

    def collate(self, batch):
        max_field_len = 128  # max(map(lambda x: len(x[0]), batch))
        max_subtoken_seq_len = (
            max(map(lambda x: len(x[1]["col_ids"]), batch))
            if "tapas" in self.configs[0].embed_model
            and self.configs[0].embed_model in TABLE_MODELS
            else 0
        )
        max_pair_len = max(map(lambda x: len(x[3]), batch))
        data_collate = []
        i = 0
        # time1 = time.time()
        # logger = logging.getLogger("collate")
        for (
            field_space,
            table_model0,
            field_label,
            msr_pairs_idx,
            msr_pairs_label,
            corpus,
            tuid,
        ) in batch:
            i += 1
            inputs = field_space.to_dict(
                max_field_len, None, True, False, self.configs[corpus]
            )
            # time2 = time.time()
            # logger.info(f"to_dict: {time2 - time1}")
            # table_model0 = copy.deepcopy(table_model0)
            # time3 = time.time()
            # logger.info(f"copy: {time3 - time2}")

            if self.configs[corpus].col_type_similarity:
                outputs = {}
                outputs["labels"] = torch.tensor(
                    append_padding(field_label[0], -1, max_field_len), dtype=torch.long
                )
                outputs["types"] = torch.tensor(
                    append_padding(field_label[1], -1, max_field_len), dtype=torch.long
                )
            else:
                inputs["msr_pair_idx"] = torch.tensor(
                    append_padding(
                        msr_pairs_idx,
                        (max_field_len - 1, max_field_len - 1),
                        max_pair_len,
                    ),
                    dtype=torch.int64,
                )
                inputs["msr_pair_mask"] = torch.tensor(
                    [1] * len(msr_pairs_idx)
                    + [0] * (max_pair_len - len(msr_pairs_idx)),
                    dtype=torch.uint8,
                )
                outputs = {
                    "fields": torch.tensor(
                        append_padding(
                            field_label, [-1] * (max(Metadata_Label) + 1), max_field_len
                        ),
                        dtype=torch.long,
                    ),
                    "msr_pairs": torch.tensor(
                        append_padding(msr_pairs_label, -1, max_pair_len),
                        dtype=torch.long,
                    ),
                }

                if self.configs[corpus].embed_model in TABLE_MODELS:
                    table_model = {}
                    if "tapas" in self.configs[corpus].embed_model:
                        subtoken_seq_len = len(table_model0["col_ids"])
                        if "fine-tune" not in self.configs[corpus].embed_model:
                            embedding = get_embeddings(
                                tuid, self.configs[corpus], "EMB"
                            ).tolist()
                            table_model["embedding"] = torch.tensor(
                                append_padding(
                                    embedding,
                                    [0] * len(embedding[0]),
                                    max_subtoken_seq_len,
                                ),
                                dtype=torch.float,
                            )
                        table_model["col_ids"] = torch.tensor(
                            append_padding(
                                table_model0["col_ids"].tolist(),
                                0,
                                max_subtoken_seq_len,
                            ),
                            dtype=torch.long,
                        )
                        table_model["row_ids"] = torch.tensor(
                            append_padding(
                                table_model0["row_ids"].tolist(),
                                0,
                                max_subtoken_seq_len,
                            ),
                            dtype=torch.long,
                        )
                        table_model["mask"] = torch.tensor(
                            [1] * subtoken_seq_len
                            + [0] * (max_subtoken_seq_len - subtoken_seq_len),
                            dtype=torch.long,
                        )
                        # for key in table_model0.keys():
                        #     table_model[key] = torch.cat((table_model0[key], torch.zeros(
                        #         (torch.Size([max_subtoken_seq_len - subtoken_seq_len]) + table_model0[key][0].size()),
                        #         dtype=table_model0[key].dtype)), dim=0)
                        # table_model["mask"] = torch.cat((torch.ones(subtoken_seq_len, dtype=torch.long),
                        #                                  torch.zeros(max_subtoken_seq_len - subtoken_seq_len,
                        #                                              dtype=torch.long)), dim=0)
                        # inputs["table_model"] = table_model
                    elif "tabbie" in self.configs[corpus].embed_model:
                        max_subtoken_seq_len = 31 * 21
                        embedding = get_embeddings(tuid, self.configs[corpus], None)
                        row_len = len(embedding)
                        col_len = len(embedding[0])
                        embedding = embedding.reshape(
                            -1, self.configs[corpus].embed_len * 2
                        ).tolist()
                        table_model["embedding"] = torch.tensor(
                            append_padding(
                                embedding, [0] * len(embedding[0]), max_subtoken_seq_len
                            ),
                            dtype=torch.float,
                        )
                        table_model["col_ids"] = torch.tensor(
                            append_padding(
                                [i for _ in range(row_len) for i in range(col_len)],
                                0,
                                max_subtoken_seq_len,
                            ),
                            dtype=torch.long,
                        )
                        table_model["row_ids"] = torch.tensor(
                            append_padding(
                                [j for j in range(row_len) for i in range(col_len)],
                                0,
                                max_subtoken_seq_len,
                            ),
                            dtype=torch.long,
                        )
                        table_model["mask"] = torch.tensor(
                            [1] * col_len * row_len
                            + [0] * (max_subtoken_seq_len - col_len * row_len),
                            dtype=torch.long,
                        )
                    if self.configs[corpus].embed_model == "tapas-fine-tune":
                        table_model["input_ids"] = torch.tensor(
                            append_padding(
                                table_model0["input_ids"].tolist(),
                                0,
                                max_subtoken_seq_len,
                            ),
                            dtype=torch.long,
                        )
                        table_model["token_type_ids"] = torch.tensor(
                            append_padding(
                                table_model0["token_type_ids"].tolist(),
                                [0] * 7,
                                max_subtoken_seq_len,
                            ),
                            dtype=torch.long,
                        )
                        table_model["attention_mask"] = torch.tensor(
                            append_padding(
                                table_model0["attention_mask"].tolist(),
                                0,
                                max_subtoken_seq_len,
                            ),
                            dtype=torch.long,
                        )
                    if self.configs[corpus].use_entity:
                        table_model["entity"] = torch.tensor(
                            append_padding(
                                table_model0["entity"].tolist(),
                                [0] * self.configs[corpus].entity_len,
                                max_subtoken_seq_len,
                            ),
                            dtype=torch.float,
                        )
                        table_model["col_type"] = torch.tensor(
                            append_padding(
                                table_model0["col_type"].tolist(),
                                [0] * self.configs[corpus].entity_len,
                                max_field_len,
                            ),
                            dtype=torch.float,
                        )
                        table_model["property"] = torch.tensor(
                            append_padding(
                                table_model0["property"].tolist(),
                                [0] * self.configs[corpus].entity_len,
                                max_field_len,
                            ),
                            dtype=torch.float,
                        )
                    inputs['table_model'] = table_model

            # time4 = time.time()
            # logger.info(f"table_model: {time4 - time3}")
            # time1 = time4
            data_collate.append({"inputs": inputs, "outputs": outputs})
        # time2 = time.time()
        # logger.info(f" {time2 - time1}")

        return default_collate(data_collate)

    def statistics(self):
        schema_num = len(set([i.split('.')[0] for i in self.tUIDs]))
        table_num = len(self.tUIDs)
        field_num = sum([i.n_cols for i in self.table_list])

        col_num = Counter([i.n_cols for i in self.table_list])
        row_num = Counter([i.n_rows for i in self.table_list])

        # [dimension label, measure label, aggregation label * 9, primary key label, vendor label, category label,
        #  subCategory label]
        dim_num = Counter(
            [
                i[Metadata_Label.Dim_label]
                for table_label in self.field_labels
                for i in table_label
            ]
        )
        msr_num = Counter(
            [
                i[Metadata_Label.Msr_label]
                for table_label in self.field_labels
                for i in table_label
            ]
        )
        agg0_num = Counter(
            [
                idx
                for table_label in self.field_labels
                for field_label in table_label
                for idx, label in enumerate(
                    field_label[
                        Metadata_Label.Agg_label_start : Metadata_Label.Agg_label_end
                        + 1
                    ]
                )
                if label == 1
            ]
        )
        primary_key_num = Counter(
            [
                i[Metadata_Label.Key_score]
                for table_label in self.field_labels
                for i in table_label
            ]
        )
        common_breakdown_num = Counter(
            [
                i[Metadata_Label.Gby_score]
                for table_label in self.field_labels
                for i in table_label
            ]
        )
        common_msr_num = Counter(
            [
                i[Metadata_Label.Msr_score]
                for table_label in self.field_labels
                for i in table_label
            ]
        )
        category_num = Counter(
            [
                i[Metadata_Label.Msr_type_category]
                for table_label in self.field_labels
                for i in table_label
            ]
        )
        subcategory_num = Counter(
            [
                i[Metadata_Label.Msr_type_subcategory]
                for table_label in self.field_labels
                for i in table_label
            ]
        )

        msr_pair_num = Counter(
            [i for pair_label in self.msr_pairs_label for i in pair_label]
        )
        with open('statistics.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)  # , delimiter=' ',
            # quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['schema_num', schema_num])
            spamwriter.writerow(['table_num', table_num])
            spamwriter.writerow(['field_num', field_num])
            spamwriter.writerow([])
            spamwriter.writerow(['col_num'] + list(col_num.keys()))
            spamwriter.writerow([''] + list(col_num.values()))
            spamwriter.writerow(['row_num'] + list(row_num.keys()))
            spamwriter.writerow([''] + list(row_num.values()))
            spamwriter.writerow([])
            spamwriter.writerow(['dim_num'] + list(dim_num.keys()))
            spamwriter.writerow([''] + list(dim_num.values()))
            spamwriter.writerow(['msr_num'] + list(msr_num.keys()))
            spamwriter.writerow([''] + list(msr_num.values()))
            spamwriter.writerow(['agg0_num'] + list(agg0_num.keys()))
            spamwriter.writerow([''] + list(agg0_num.values()))
            spamwriter.writerow(['primary_key_num'] + list(primary_key_num.keys()))
            spamwriter.writerow([''] + list(primary_key_num.values()))
            spamwriter.writerow(
                ['common_breakdown_num'] + list(common_breakdown_num.keys())
            )
            spamwriter.writerow([''] + list(common_breakdown_num.values()))
            spamwriter.writerow(['common_msr_num'] + list(common_msr_num.keys()))
            spamwriter.writerow([''] + list(common_msr_num.values()))
            spamwriter.writerow(['category_num'] + list(category_num.keys()))
            spamwriter.writerow([''] + list(category_num.values()))
            spamwriter.writerow(['subcategory_num'] + list(subcategory_num.keys()))
            spamwriter.writerow([''] + list(subcategory_num.values()))
            print([i[1] for i in sorted(subcategory_num.items(), key=lambda x: x[0])])
            spamwriter.writerow(['pair_num'] + list(msr_pair_num.keys()))
            spamwriter.writerow([''] + list(msr_pair_num.values()))

    def save(self, save_path):
        os.makedirs(os.path.join(save_path))
        for tuid, table, label, corpus, pair_idx, pair_label in tqdm(
            zip(
                self.tUIDs,
                self.table_list,
                self.field_labels,
                self.corpus,
                self.msr_pairs_idx,
                self.msr_pairs_label,
            )
        ):
            inputs = table.field_space.to_dict(
                table.n_cols, None, True, False, self.configs[corpus]
            )
            inputs = {key: inputs[key].tolist() for key in inputs.keys()}
            outputs = label
            feature = {
                "inputs": inputs,
                "outputs": outputs,
                "pair_idx": pair_idx,
                "pair_label": pair_label,
            }
            with open(
                os.path.join(save_path, f"{corpus}-{tuid}.json"),
                'w',
                encoding='utf-8-sig',
            ) as f:
                json.dump(feature, f)
        print("Successfully save")

    def rule(self):
        msr = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        msr_pair = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        msr_type = {"t": 0, "f": 0}
        primary_gby = {"R@1": 0, "R@3": 0, "total": 0}
        primary_msr = {"R@1": 0, "R@3": 0, "total": 0}
        primary_key = {"R@1": 0, "R@3": 0, "total": 0}
        agg = {"R@1": 0, "R@3": 0, "total": 0}
        for table_label, table in zip(self.field_labels, self.table_list):
            for idx, field_label in enumerate(table_label):
                # msr
                if (
                    field_label[Metadata_Label.Dim_label] == 1
                    or field_label[Metadata_Label.Msr_label] == 1
                ):
                    if table.field_space[idx].field_type in {
                        FieldType.String,
                        FieldType.Year,
                        FieldType.DateTime,
                        FieldType.Unknown,
                    }:
                        if field_label[Metadata_Label.Msr_label] == 0:
                            msr["tn"] += 1
                        else:
                            msr["fn"] += 1
                    else:
                        if field_label[Metadata_Label.Msr_label] == 1:
                            msr["tp"] += 1
                        else:
                            msr["fp"] += 1
                # msr type
                if field_label[Metadata_Label.Msr_type_subcategory] != -1:
                    if (
                        field_label[Metadata_Label.Msr_type_subcategory]
                        == SubCategory.Money
                    ):
                        msr_type["t"] += 1
                    else:
                        msr_type["f"] += 1
                # agg
                if (
                    sum(
                        field_label[
                            Metadata_Label.Agg_label_start : Metadata_Label.Agg_label_end
                        ]
                    )
                    > 0
                ):
                    agg["total"] += 1
                    if field_label[Metadata_Label.Agg_label_start] == 1:
                        agg["R@1"] += 1
                    if (
                        sum(
                            field_label[
                                Metadata_Label.Agg_label_start : Metadata_Label.Agg_label_start
                                + 3
                            ]
                        )
                        > 0
                    ):
                        agg["R@3"] += 1
            # primary
            if table_label[0][Metadata_Label.Key_score] != -1:
                key_idx = [
                    f.field_index
                    for f in table.field_space
                    if f.field_type
                    in {
                        FieldType.String,
                        FieldType.Year,
                        FieldType.DateTime,
                        FieldType.Unknown,
                    }
                    and f.data_features[FEATURE_MAP['cardinality'] - 1] == 1
                ]
                key_recall = []
                for idx in key_idx:
                    if table_label[idx][Metadata_Label.Key_score] == 1:
                        key_recall.append(1)
                    else:
                        key_recall.append(0)
                primary_key["total"] += 1
                if sum(key_recall[:1]) > 0:
                    primary_key["R@1"] += 1
                if sum(key_recall[:3]) > 0:
                    primary_key["R@3"] += 1
            if table_label[0][Metadata_Label.Gby_score] != -1:
                gby_idx = [
                    f.field_index
                    for f in table.field_space
                    if f.field_type
                    in {
                        FieldType.String,
                        FieldType.Year,
                        FieldType.DateTime,
                        FieldType.Unknown,
                    }
                    and f.data_features[FEATURE_MAP['cardinality'] - 1] <= 0.4
                ]
                gby_recall = []
                for idx in gby_idx:
                    if table_label[idx][Metadata_Label.Gby_score] == 1:
                        gby_recall.append(1)
                    else:
                        gby_recall.append(0)
                primary_gby["total"] += 1
                if sum(gby_recall[:1]) > 0:
                    primary_gby["R@1"] += 1
                if sum(gby_recall[:3]) > 0:
                    primary_gby["R@3"] += 1
            if table_label[0][Metadata_Label.Msr_score] != -1:
                msr_idx = [
                    f.field_index
                    for f in table.field_space
                    if f.field_type in {FieldType.Decimal}
                ]
                msr_idx.reverse()
                msr_recall = []
                for idx in msr_idx:
                    if table_label[idx][Metadata_Label.Msr_score] == 1:
                        msr_recall.append(1)
                    else:
                        msr_recall.append(0)
                primary_msr["total"] += 1
                if sum(msr_recall[:1]) > 0:
                    primary_msr["R@1"] += 1
                if sum(msr_recall[:3]) > 0:
                    primary_msr["R@3"] += 1
        # msr pair
        for table_pair_label, table_pair_idx in zip(
            self.msr_pairs_label, self.msr_pairs_idx
        ):
            for label, idx in zip(table_pair_label, table_pair_idx):
                if abs(idx[0] - idx[1]) == 1:
                    if label == 0:
                        msr_pair["fp"] += 1
                    else:
                        msr_pair["tp"] += 1
                else:
                    if label == 0:
                        msr_pair["tn"] += 1
                    else:
                        msr_pair["fn"] += 1
        with open('rule.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['msr'] + list(msr.keys()))
            spamwriter.writerow([''] + [str(i) for i in msr.values()])
            spamwriter.writerow(["msr_pair"] + list(msr_pair.keys()))
            spamwriter.writerow([""] + [str(i) for i in msr_pair.values()])
            spamwriter.writerow(["msr_type"] + list(msr_type.keys()))
            spamwriter.writerow([""] + [str(i) for i in msr_type.values()])
            spamwriter.writerow(["primary_gby"] + list(primary_gby.keys()))
            spamwriter.writerow([""] + [str(i) for i in primary_gby.values()])
            spamwriter.writerow(["primary_msr"] + list(primary_msr.keys()))
            spamwriter.writerow([""] + [str(i) for i in primary_msr.values()])
            spamwriter.writerow(["primary_key"] + list(primary_key.keys()))
            spamwriter.writerow([""] + [str(i) for i in primary_key.values()])
            spamwriter.writerow(["agg"] + list(agg.keys()))
            spamwriter.writerow([""] + [str(i) for i in agg.values()])

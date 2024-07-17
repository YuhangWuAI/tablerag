import logging
import math
import random
from collections import defaultdict
from typing import List

from metadata.column_type.measure_types import SubCategory, Category, SubCategory2Category
from metadata.metadata_data import MetadataTableFields
from metadata.metadata_data.table_fields import Metadata_Label

random.seed(10)


def split_train_valid_test_schema(dataset: MetadataTableFields, train_valid_test: List[float]):
    '''
    split schema by msr_type
    :param dataset: MetadataTableFields
    :param train_valid_test: splitting ratio, [train_ratio, valid_ratio, test_ratio]
    '''
    if not (train_valid_test[0] >= train_valid_test[2] >= train_valid_test[1] and sum(train_valid_test) == 1):
        raise NotImplementedError(f"Not yet implement splitting ratio {train_valid_test}")
    # Generate msr_type2suid: {msr_type: [xxx, xxx...]}, number of suids in one list = number of fields with the type in the tables
    msr_type2suid = defaultdict(list)
    existed_suid = []
    for field_space, table_model0, field_label, msr_pairs_idx, msr_pairs_label, corpus, tuid in dataset:
        tuid = f"{corpus}-{tuid}"
        if tuid.split('.')[0] in existed_suid:
            continue
        for label in field_label:
            if label[Metadata_Label.Msr_type_subcategory] != -1:
                msr_type2suid[label[Metadata_Label.Msr_type_subcategory]].append(tuid.split('.')[0])
                existed_suid.append(tuid.split('.')[0])
    msr_types = sorted(msr_type2suid.keys(), key=lambda i: len(msr_type2suid[i]))
    train = []
    valid = []
    test = []
    for msr_type in msr_types:
        suids = msr_type2suid[msr_type]
        valid_idx = math.floor((len(suids) - 1) * train_valid_test[0])
        test_idx = math.floor((len(suids) - 1) * (train_valid_test[0] + train_valid_test[1]))
        # Filter suids already in train/valid/test dataset
        suids_dedup_train = list(filter(lambda x: x not in train, suids))
        valid_idx -= len(suids) - len(suids_dedup_train)
        suids_dedup_valid = list(filter(lambda x: x not in valid, suids_dedup_train))
        test_idx -= len(suids) - len(suids_dedup_valid)
        suids = list(filter(lambda x: x not in test, suids_dedup_valid))
        # Make sure the same table in the same dataset
        enough_train = False
        enough_valid = False
        enough_test = False
        if valid_idx < 0 or train_valid_test[0] == 0:  # There are enough train suids in train
            enough_train = True
            valid_idx = 0
        if valid_idx >= test_idx or train_valid_test[1] == 0:
            enough_valid = True
            valid_idx = test_idx
            if valid_idx < 0:
                valid_idx = test_idx = 0
            elif valid_idx >= len(suids):
                valid_idx = test_idx = len(suids)
        if test_idx >= len(suids) or train_valid_test[2] == 0:
            enough_test = True
            test_idx = len(suids)
        while not enough_test and test_idx >= 1 and suids[test_idx] == suids[test_idx - 1]:
            test_idx -= 1
        if enough_valid:
            valid_idx = test_idx
        else:
            if valid_idx >= test_idx:
                valid_idx = test_idx - 1
            while valid_idx >= 1 and suids[valid_idx] == suids[valid_idx - 1]:
                valid_idx -= 1
        # Add suids into dataset
        train.extend(set(suids[:valid_idx]))
        valid.extend(set(suids[valid_idx:test_idx]))
        test.extend(set(suids[test_idx:]))
    logger = logging.getLogger("Split train/valid/test")
    record = {"SubCategory": {"train": [0] * len(SubCategory),
                              "valid": [0] * len(SubCategory),
                              "test": [0] * len(SubCategory)},
              "Category": {"train": [0] * len(Category),
                           "valid": [0] * len(Category),
                           "test": [0] * len(Category)}}
    msr_type2train_suid = {}
    for type in msr_type2suid.keys():
        train_suid = [i for i in msr_type2suid[type] if i in train]
        msr_type2train_suid[type] = train_suid
        num = len(train_suid)
        record["SubCategory"]["train"][type] = num
        record["Category"]["train"][SubCategory2Category[type]] += num
        num = len([i for i in msr_type2suid[type] if i in valid])
        record["SubCategory"]["valid"][type] = num
        record["Category"]["valid"][SubCategory2Category[type]] += num
        num = len([i for i in msr_type2suid[type] if i in test])
        record["SubCategory"]["test"][type] = num
        record["Category"]["test"][SubCategory2Category[type]] += num
    for key0 in record.keys():
        for key1 in record[key0].keys():
            logger.info(f"{key0}, {key1}, {record[key0][key1]}")
    return train, valid, test, msr_type2train_suid


def split_train_valid_test_field(dataset: MetadataTableFields, train_valid_test: List[float]):
    '''
    split fields by msr_type
    :param dataset: MetadataTableFields
    :param train_valid_test: splitting ratio, [train_ratio, valid_ratio, test_ratio]
    '''
    if not (train_valid_test[0] >= train_valid_test[2] >= train_valid_test[1] and sum(train_valid_test) == 1):
        raise NotImplementedError(f"Not yet implement splitting ratio {train_valid_test}")
    # Generate msr_type2tuid: {msr_type: [xxx.tx, xxx.tx...]}, number of tuids in one list = number of fields with the type in the tables
    msr_type2fuid = defaultdict(list)
    for field_space, table_model0, field_label, msr_pairs_idx, msr_pairs_label, corpus, tuid in dataset:
        for i, label in enumerate(field_label):
            if label[Metadata_Label.Msr_type_subcategory] != -1:
                msr_type2fuid[label[Metadata_Label.Msr_type_subcategory]].append(f"{corpus}-{tuid}-{i}")
    train = []
    test = []
    valid = []
    logger = logging.getLogger("Split train/valid/test")
    record = {"SubCategory": {"train": [0] * len(SubCategory),
                              "valid": [0] * len(SubCategory),
                              "test": [0] * len(SubCategory),
                              "all": [0] * len(SubCategory)},
              "Category": {"train": [0] * len(Category),
                           "valid": [0] * len(Category),
                           "test": [0] * len(Category),
                           "all": [0] * len(Category)}}
    random.seed(10)
    for key in msr_type2fuid.keys():
        fuid = msr_type2fuid[key]
        random.shuffle(fuid)
        train.extend(fuid[:math.floor(len(fuid) * train_valid_test[0])])
        valid.extend(fuid[math.floor(len(fuid) * train_valid_test[0]):math.floor(
            len(fuid) * (train_valid_test[0] + train_valid_test[1]))])
        test.extend(fuid[math.floor(len(fuid) * (train_valid_test[0] + train_valid_test[1])):])
        # record
        num = math.floor(len(fuid) * train_valid_test[0])
        record["SubCategory"]["train"][key] = num
        record["Category"]["train"][SubCategory2Category[key]] += num
        num = math.floor(len(fuid) * (train_valid_test[0] + train_valid_test[1])) - math.floor(
            len(fuid) * train_valid_test[0])
        record["SubCategory"]["valid"][key] = num
        record["Category"]["valid"][SubCategory2Category[key]] += num
        num = len(fuid) - math.floor(len(fuid) * (train_valid_test[0] + train_valid_test[1]))
        record["SubCategory"]["test"][key] = num
        record["Category"]["test"][SubCategory2Category[key]] += num
        record["SubCategory"]["all"][key] = len(fuid)
        record["Category"]["all"][SubCategory2Category[key]] += len(fuid)
    for key0 in record.keys():
        for key1 in record[key0].keys():
            logger.info(f"{key0}, {key1}, {record[key0][key1]}")
    return train, valid, test

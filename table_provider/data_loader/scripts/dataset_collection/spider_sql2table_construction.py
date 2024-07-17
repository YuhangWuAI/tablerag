import os
import torch
import random
import re
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from third_party.miscs.bridge_content_encoder import get_database_matches

from tqdm import tqdm

def spider_get_input(
        question: str,
        serialized_schema: str,
        prefix: str,
) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()


def spider_get_target(
        query: str,
        db_id: str,
        normalize_query: bool,
        target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def spider_add_serialized_schema(ex: dict, args) -> dict:
    if getattr(args.seq2seq, "schema_serialization_with_nl"):
        serialized_schema = serialize_schema_natural_language(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            db_primary_keys=ex["db_primary_keys"],
            db_foreign_keys=ex["db_foreign_keys"],
            schema_serialization_with_db_content=args.seq2seq.schema_serialization_with_db_content,
            normalize_query=True,
        )
    else:
        serialized_schema = serialize_schema(
            question=ex["question"],
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            schema_serialization_type="peteshaw",
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=args.seq2seq.schema_serialization_with_db_content,
            normalize_query=True,
        )
    return {"serialized_schema": serialized_schema}


def spider_pre_process_function(batch: dict, args):
    prefix = ""

    inputs = [
        spider_get_input(
            question=question, serialized_schema=serialized_schema, prefix=prefix
        )
        for question, serialized_schema in zip(
            batch["question"], batch["serialized_schema"]
        )
    ]

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=True,
            target_with_db_id=args.seq2seq.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    return zip(inputs, targets)


def spider_pre_process_one_function(item: dict, args):
    prefix = ""

    seq_out = spider_get_target(
        query=item["query"],
        db_id=item["db_id"],
        normalize_query=True,
        target_with_db_id=args.seq2seq.target_with_db_id,
    )

    return prefix + item["question"].strip(), seq_out


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(
            r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s
        )

    return comma_fix(white_space_fix(lower(query)))


def serialize_schema_natural_language(
        question: str,
        db_path: str,
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        db_primary_keys,
        db_foreign_keys,
        schema_serialization_with_db_content: bool = False,
        normalize_query: bool = True,
) -> str:
    overall_description = f'{db_id} contains tables such as ' \
                          f'{", ".join([table_name.lower() if normalize_query else table_name for table_name in db_table_names])}.'
    table_description_primary_key_template = lambda table_name, primary_key: \
        f'{primary_key} is the primary key.'
    table_description = lambda table_name, column_names: \
        f'Table {table_name} has columns such as {", ".join(column_names)}.'
    value_description = lambda column_value_pairs: \
        f'{"".join(["The {} contains values such as {}.".format(column, value) for column, value in column_value_pairs])}'
    foreign_key_description = lambda table_1, column_1, table_2, column_2: \
        f'The {column_1} of {table_1} is the foreign key of {column_2} of {table_2}.'

    db_primary_keys = db_primary_keys["column_id"]
    db_foreign_keys = list(zip(db_foreign_keys["column_id"], db_foreign_keys["other_column_id"]))


    descriptions = [overall_description]
    db_table_name_strs = []
    db_column_name_strs = []
    value_sep = ", "
    for table_id, table_name in enumerate(db_table_names):
        table_name_str = table_name.lower() if normalize_query else table_name
        db_table_name_strs.append(table_name_str)
        columns = []
        column_value_pairs = []
        primary_keys = []
        for column_id, (x, y) in enumerate(zip(db_column_names["table_id"], db_column_names["column_name"])):
            if column_id == 0:
                continue
            column_str = y.lower() if normalize_query else y
            db_column_name_strs.append(column_str)
            if x == table_id:
                columns.append(column_str)
                if column_id in db_primary_keys:
                    primary_keys.append(column_str)
                if schema_serialization_with_db_content:
                    matches = get_database_matches(
                        question=question,
                        table_name=table_name,
                        column_name=y,
                        db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
                    )
                    if matches:
                        column_value_pairs.append((column_str, value_sep.join(matches)))

        table_description_columns_str = table_description(table_name_str, columns)
        descriptions.append(table_description_columns_str)
        table_description_primary_key_str = table_description_primary_key_template(table_name_str, ", ".join(primary_keys))
        descriptions.append(table_description_primary_key_str)
        if len(column_value_pairs) > 0:
            value_description_str = value_description(column_value_pairs)
            descriptions.append(value_description_str)


    for x, y in db_foreign_keys:
        # get the table and column of x
        x_table_name = db_table_name_strs[db_column_names["table_id"][x]]
        x_column_name = db_column_name_strs[x]
        # get the table and column of y
        y_table_name = db_table_name_strs[db_column_names["table_id"][y]]
        y_column_name = db_column_name_strs[y]
        foreign_key_description_str = foreign_key_description(x_table_name, x_column_name, y_table_name, y_column_name)
        descriptions.append(foreign_key_description_str)
    return " ".join(descriptions)

def serialize_schema(
        question: str,
        db_path: str,
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        schema_serialization_type: str = "peteshaw",
        schema_serialization_randomized: bool = False,
        schema_serialization_with_db_id: bool = True,
        schema_serialization_with_db_content: bool = False,
        normalize_query: bool = True,
) -> str:
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError

    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(
                    column=column_name_str, values=value_sep.join(matches)
                )
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema




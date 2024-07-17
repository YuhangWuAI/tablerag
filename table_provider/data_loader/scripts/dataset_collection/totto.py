# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ToTTo: A Controlled Table-To-Text Generation Dataset"""


import json
import datasets
from utils import dump_db_json_schema

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{parikh2020totto,
  title={ToTTo: A Controlled Table-To-Text Generation Dataset},
  author={Parikh, Ankur P and Wang, Xuezhi and Gehrmann, Sebastian and Faruqui, Manaal and Dhingra, Bhuwan and Yang, Diyi and Das, Dipanjan},
  journal={arXiv preprint arXiv:2004.14373},
  year={2020}
"""

_DESCRIPTION = """\
ToTTo is an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/ToTTo"

_LICENSE = "Creative Commons Share-Alike 3.0"

_URL = "https://storage.googleapis.com/totto-public/totto_data.zip"


def check_alignment(matrix):
    row_lengths = len(matrix[0])
    if row_lengths == 0:
        return False
    if len(matrix[1:]) == 0:
        return False
    for row in matrix:
        if len(row) != row_lengths:
            return False

    return True


class ToTTo(datasets.GeneratorBasedBuilder):
    """ToTTo dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "table_page_title": datasets.Value("string"),
                    "table_section_title": datasets.Value("string"),
                    "table": [
                        [
                            {
                                "column_span": datasets.Value("int32"),
                                "is_header": datasets.Value("bool"),
                                "row_span": datasets.Value("int32"),
                                "value": datasets.Value("string"),
                            }
                        ]
                    ],
                    "table_rows": datasets.features.Sequence(
                        datasets.features.Sequence(datasets.Value("string"))
                    ),
                    "highlighted_cells": datasets.features.Sequence(
                        datasets.features.Sequence(datasets.Value("int32"))
                    ),
                    "final_sentences": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "table_header_hierarchy": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath
                    + "/totto_data/totto_train_data.jsonl",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath
                    + "/totto_data/totto_dev_data.jsonl",
                },
            ),
        ]

    def _generate_examples(self, data_filepath):
        with open(data_filepath, encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f.readlines()]
            for example_idx, example in enumerate(lines):
                highlight_idx = example["highlighted_cells"]
                table = example["table"]
                highlight_header, highlight_info = [], []

                # generate table header hierarchy
                header_hierarchy = []

                table_rows = []
                flag_row_for_row_span = []
                for r_idx in range(len(table)):
                    row = []
                    for c_idx in range(len(table[r_idx])):
                        if (
                            table[r_idx][c_idx]["column_span"] > 1
                            and table[r_idx][c_idx]["row_span"] == 1
                        ):
                            row.extend(
                                [table[r_idx][c_idx]["value"]]
                                * table[r_idx][c_idx]["column_span"]
                            )
                        elif (
                            table[r_idx][c_idx]["row_span"] > 1
                            and table[r_idx][c_idx]["column_span"] == 1
                        ):
                            row.extend([table[r_idx][c_idx]["value"]])
                            flag_row_for_row_span.append(
                                {
                                    "index": r_idx + 1,
                                    "value": table[r_idx][c_idx]["value"],
                                }
                            )
                        else:
                            row.extend([table[r_idx][c_idx]["value"]])
                            flag_row_for_row_span.append(
                                {
                                    "index": r_idx + 1,
                                    "value": table[r_idx][c_idx]["value"],
                                }
                            )
                        if table[r_idx][c_idx]["is_header"]:
                            header_hierarchy.append(
                                {
                                    "row_index": r_idx,
                                    "column_index": c_idx,
                                    "value": table[r_idx][c_idx]["value"],
                                    "column_span": table[r_idx][c_idx]["column_span"],
                                    "row_span": table[r_idx][c_idx]["row_span"],
                                }
                            )
                    table_rows.append(row)

                if check_alignment(table_rows) == False:
                    continue

                parsed_header = list(x.replace("|", "") for x in table_rows[0])

                for h_idx in highlight_idx:
                    if h_idx[1] < len(parsed_header):
                        highlight_info.append(
                            str(h_idx)
                            + ": "
                            + str(parsed_header[h_idx[1]])
                            + "|"
                            + table[h_idx[0]][h_idx[1]]["value"]
                            + "\n"
                        )
                    else:
                        highlight_info.append(
                            str(h_idx)
                            + ": "
                            + "-"
                            + "|"
                            + table[h_idx[0]][h_idx[1]]["value"]
                            + "\n"
                        )

                yield example_idx, {
                    "table_page_title": example["table_page_title"],
                    "table_section_title": example["table_section_title"],
                    "table": example["table"],
                    "table_rows": table_rows,
                    "highlighted_cells": example["highlighted_cells"],
                    "final_sentences": [
                        anno["final_sentence"]
                        for anno in example["sentence_annotations"]
                    ],
                    "table_header_hierarchy": header_hierarchy,
                }

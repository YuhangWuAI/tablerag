"""
This code file contains functions that borrow certain logic from an anonymous repository associated with the paper:
"TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning" (arXiv:2312.09039).
Original source: https://anonymous.4open.science/r/TableProvider-4CC3/README.md.
The repository does not list an author, but it is linked to the above paper.

Specifically, portions of the code related to data loading, data packing, and evaluation logic have been borrowed and integrated into this project.

Current author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

If you believe that any content in this file infringes your rights or if you have any concerns,
please contact me at the email address above.
"""

from datasets import load_dataset
from src.llm.llm_generator.llm_generating import LLM_Generator
from src.table_loader.data_loader.table_parser.type_sets import TableSerializationType, TaskName
from .table_linearizer import StructuredDataLinearizer
import warnings

warnings.filterwarnings("ignore")

class TableParser:
    """
    A class to parse tables from different datasets based on the task type.
    It supports tasks such as 'feverous', 'hybridqa', 'sqa', 'tabfact', 'totto', and 'spider'.
    """

    def __init__(
        self, task_name: str, split: str = "None", use_small_sample_list: bool = False
    ):
        """
        Initialize the TableParser class by loading the dataset for a specific task.

        :param task_name: The task name, selected from ["feverous", "hybridqa", "sqa", "tabfact", "totto"].
        :param split: The dataset split to load ('train', 'validation', or 'test').
        :param use_small_sample_list: Whether to load only a small subset of the data (for testing purposes).
        """
        if task_name not in [task.value for task in TaskName]:
            raise ValueError(f"Task name {task_name} is not supported")
        self.task_name = task_name
        self.split = split
        self.call_llm = LLM_Generator()

        # Load the dataset, optionally with a small sample list
        self.dataset = (
            self.load_table(use_small_sample_list)
            if split == "None"
            else self.load_table(split, use_small_sample_list)
        )

    def load_table(self, use_small_sample_list: bool):
        """
        Load the full dataset table.

        :param use_small_sample_list: Whether to load a small subset of the dataset.
        :return: The loaded dataset.
        """
        self.dataset = load_dataset(
            f"src/table_loader/data_downloader/{self.task_name}.py",
            verification_mode="no_checks",
        )
        if use_small_sample_list and len(self.dataset) >= 100:
            shuffled_dataset = self.dataset.shuffle(seed=42)
            return shuffled_dataset.select(range(100))
        else:
            return self.dataset

    def load_table(self, split: str, use_small_sample_list: bool):
        """
        Load the dataset table with a specific split (train, validation, or test).

        :param split: The dataset split to load ('train', 'validation', or 'test').
        :param use_small_sample_list: Whether to load a small subset of the dataset.
        :return: The loaded dataset.
        """
        self.dataset = load_dataset(
            f"src/table_loader/data_loader/data_downloader/{self.task_name}.py",
            split=split,
            verification_mode="no_checks",
        )
        if use_small_sample_list and len(self.dataset) >= 100:
            shuffled_dataset = self.dataset.shuffle(seed=42)
            return shuffled_dataset.select(range(100))
        else:
            return self.dataset

    def parse_table(self, _example: dict) -> dict:
        """
        Parse a table example to the specific format required by the task.

        :param _example: The table example to parse.
        :return: A dictionary representing the parsed table.
        """
        if self.task_name == "feverous":
            label = self._map_feverous_label(_example["label"])
            return {
                "title": "",
                "context": _example["context"],
                "table": {
                    "header": _example['table']['header'][0],
                    "rows": _example['table']['rows'][0],
                    "caption": "",
                },
                "query": _example["statement"],
                "label": label,
            }
        elif self.task_name == "hybridqa":
            return {
                "title": "",
                "context": [_example["context"], _example["passage"]],
                "table": {
                    "header": _example['table']['header'],
                    "rows": _example['table']['rows'],
                    "caption": "",
                },
                "query": _example["question"],
                "label": _example["answer_text"],
            }
        elif self.task_name == "sqa":
            return {
                "title": "",
                "context": "",
                "table": {
                    "header": _example['table_header'],
                    "rows": _example['table_data'],
                    "caption": "",
                },
                "query": _example["question"],
                "label": _example["answer_text"],
            }
        elif self.task_name == "tabfact":
            label = self._map_tabfact_label(_example["label"])
            return {
                "title": "",
                "context": "",
                "table": {
                    "header": _example['table']['header'],
                    "rows": _example['table']['rows'],
                    "caption": _example['table']['caption'],
                },
                "query": _example["statement"],
                "label": label,
            }
        elif self.task_name == "totto":
            return {
                "title": _example['table_page_title'],
                "context": "",
                "table": {
                    "header": _example['table_rows'][0],
                    "rows": _example['table_rows'][1:],
                    "caption": _example['table_section_title'],
                    "header_hierarchy": _example['table_header_hierarchy'],
                },
                "query": f"Produce a one-sentence description for each highlighted cells ({str(_example['highlighted_cells'])}) of the table.",
                "label": _example["final_sentences"],
            }
        elif self.task_name == "spider":
            return {
                "title": _example['db_table_names'],
                "context": "",
                "table": {
                    "header": _example['db_table']['header'],
                    "rows": _example['db_table']['rows'],
                    "caption": "",
                },
                "db_path": _example["db_path"],
                "db_id": _example["db_id"],
                "question": _example["question"],
                "query": _example["query"],
            }
        else:
            raise ValueError(f"Task name {self.task_name} is not supported")

    def linearization(self, _example: dict, func=TableSerializationType.html):
        """
        Linearize the parsed table into a specific format.

        :param _example: The parsed table example.
        :param func: The desired format for linearization (e.g., HTML, JSON).
        :return: The linearized table as a string.
        """
        linearizer = StructuredDataLinearizer()
        linearized_data = linearizer.retrieve_linear_function(
            func, structured_data_dict=_example
        )
        return linearized_data

    def _map_feverous_label(self, label: str) -> str:
        """
        Map FEVEROUS dataset labels to specific string values.

        :param label: The original label from the dataset.
        :return: Mapped label value.
        """
        if str(label) == "NOT ENOUGH INFO":
            return "2"
        elif str(label) == "REFUTES":
            return "0"
        else:
            return "1"

    def _map_tabfact_label(self, label: str) -> str:
        """
        Map TabFact dataset labels to specific string values.

        :param label: The original label from the dataset.
        :return: Mapped label value.
        """
        if str(label) == "0":
            return "0"
        elif str(label) == "1":
            return "1"
        else:
            return "2"

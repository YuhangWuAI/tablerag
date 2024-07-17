from datasets import load_dataset
from .table_linearizer import StructuredDataLinearizer
from ..contract.enum_type import TaskName, TableSerializationType
from ..agents.call_llm import CallLLM


class TableLoader:
    def __init__(
        self, task_name: str, split: str = "None", use_small_sample_list: bool = False
    ):
        """
        Load table from dataset
        Args:
            task_name (str): valid task name should be selected from ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
            split (str): train, validation, or test
        """
        if task_name not in [task.value for task in TaskName]:
            raise ValueError(f"Task name {task_name} is not supported")
        self.task_name = task_name
        # if split not in ["train", "validation", "test", "validation[:30%]"]:
        #     raise ValueError(f"Split {split} is not supported")
        self.split = split

        self.call_llm = CallLLM()

        self.dataset = (
            self.load_table(use_small_sample_list)
            if split == "None"
            else self.load_table(split, use_small_sample_list)
        )

    def load_table(self, use_small_sample_list: bool):
        """
        Load table from full dataset
        Returns:
            dict: dataset
        """
        self.dataset = load_dataset(
            f"table_provider/data_loader/scripts/dataset_collection/{self.task_name}.py",
            verification_mode="no_checks",
        )
        if use_small_sample_list and len(self.dataset) >= 1000:
            shuffled_dataset = self.dataset.shuffle(seed=42)
            return shuffled_dataset.select(range(1000))
        else:
            return self.dataset

    def load_table(self, split: str, use_small_sample_list: bool):
        """
        Load table from dataset with split
        Args:
            split (str): train, validation, or test
        Returns:
            dict: dataset
        """
        # if self.task_name == "hybridqa":
        #     split = "dev" if split == "validation" else split
        self.dataset = load_dataset(
            f"table_provider/data_loader/scripts/dataset_collection/{self.task_name}.py",
            split=split,
            verification_mode="no_checks",
        )
        if use_small_sample_list and len(self.dataset) >= 1000:
            shuffled_dataset = self.dataset.shuffle(seed=42)
            return shuffled_dataset.select(range(1000))
        else:
            return self.dataset

    def parse_table(self, _example: dict) -> dict:
        """
        Parse table to the format of each task
        Args:
            _example (dict): table example
            Returns:
                dict: parsed table
        """
        if self.task_name == "feverous":
            if str(_example["label"]) == "NOT ENOUGH INFO":
                label = "2"
            elif str(_example["label"]) == "REFUTES":
                label = "0"
            else:
                label = "1"
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
            if str(_example["label"]) == "0":
                label = "0"
            elif str(_example["label"]) == "1":
                label = "1"
            else:
                label = "2"
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
        Linearize table
        Args:
            _example (dict): table example
            Returns:
                dict: linearized table
        """
        linearizer = StructuredDataLinearizer()
        linearized_data = linearizer.retrieve_linear_function(
            func, structured_data_dict=_example
        )
        return linearized_data

    def get_k_shot_example(self, k: int):
        """
        Get K shot examples for better in-context-learning
        Args:
            k (int): number of examples
        """
        # dataset = self.load_table(split="train[:10%]", use_small_sample_list=True)
        dataset = self.load_table(split="train", use_small_sample_list=False)
        k_shot_examples = []
        for i in range(len(dataset)):
            shot_info = self.parse_table(dataset[i])

            # Get the linearized example table
            shot_example = "\n".join(
                [
                    "Example table is:",
                    self.linearization(shot_info, func=TableSerializationType.html),
                    "Example query of the following table is: ",
                    shot_info["query"],
                    "Example answer is: ",
                    "|".join(shot_info["label"])
                    if isinstance(shot_info["label"], list)
                    else shot_info["label"],
                ]
            )

            # Make sure the example is less than EXAMPLE_TOKEN_LIMIT tokens
            if (
                self.call_llm.num_tokens(shot_example)
                < self.call_llm.EXAMPLE_TOKEN_LIMIT
            ):
                k_shot_examples.append(shot_example)
                if len(k_shot_examples) == k:
                    break

            # To prevent the case that there is no example with less than EXAMPLE_TOKEN_LIMIT tokens
            if i == len(dataset) - 1 and len(k_shot_examples) < k:
                print(
                    f"Warning: There is no example with less than {self.call_llm.EXAMPLE_TOKEN_LIMIT} tokens."
                )
                k_shot_examples.append(
                    self.call_llm.truncated_string(
                        shot_example, self.call_llm.EXAMPLE_TOKEN_LIMIT
                    )
                )

        return k_shot_examples

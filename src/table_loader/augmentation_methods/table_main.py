
from src.llm.llm_generator.llm_generating import LLM_Generator
from src.table_loader.data_loader.table_parser.table_parsing import TableParser
from src.table_loader.augmentation_methods.table_clarifier import TableAugmentation

from .table_filter import TableFilter



import warnings
warnings.filterwarnings("ignore")

class TableProvider:
    def __init__(self) -> None:
        pass

    def __init__(
        self,
        task_name: str,
        split: str,
        table_filter_name: str,
        table_clarifier_name: str,
        top_k: int,
        embedding_type: str = "spacy",
        whether_column_grounding: bool = False,
    ):
        self.call_llm = LLM_Generator()
        self.table_loader = TableParser(
            task_name, split="validation", use_small_sample_list=True
        )
        self.table_filter = TableFilter(
            self.call_llm,
            task_name=task_name,
            split=split,
            table_filter_name=table_filter_name,
            embedding_type=embedding_type,
            top_k=top_k,
            whether_column_grounding=whether_column_grounding,
        )
        self.table_clarification = TableAugmentation(
            self.call_llm,
            task_name=task_name,
            split=split,
            table_clarifier_name=table_clarifier_name,
        )

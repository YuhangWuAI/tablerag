from src.table_loader.data_loader.table_parser.table_parsing import TableLoader
from .table_sampling import TableSampling
from .table_augmentation import TableAugmentation

from ..agents import CallLLM
import warnings
warnings.filterwarnings("ignore")

class TableProvider:
    def __init__(self) -> None:
        pass

    def __init__(
        self,
        task_name: str,
        split: str,
        table_sampling_type: str,
        table_augmentation_type: str,
        n_cluster: int,
        top_k: int,
        embedding_type: str = "spacy",
        whether_column_grounding: bool = False,
    ):
        self.call_llm = CallLLM()
        self.table_loader = TableLoader(
            task_name, split="validation", use_small_sample_list=True
        )
        self.table_sampler = TableSampling(
            self.call_llm,
            task_name=task_name,
            split=split,
            table_sampling_type=table_sampling_type,
            embedding_type=embedding_type,
            n_cluster=n_cluster,
            top_k=top_k,
            whether_column_grounding=whether_column_grounding,
        )
        self.table_augmentation = TableAugmentation(
            self.call_llm,
            task_name=task_name,
            split=split,
            table_augmentation_type=table_augmentation_type,
        )

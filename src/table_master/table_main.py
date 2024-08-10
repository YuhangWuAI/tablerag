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
from src.llm.llm_generator.llm_generating import LLM_Generator
from src.table_loader.data_loader.table_parser.table_parsing import TableParser
from src.table_master.table_clarifier import TableClarification
from .table_filter import TableFilter

import warnings
warnings.filterwarnings("ignore")

class TableProvider:
    """
    The TableProvider class is responsible for managing the process of table parsing, filtering, 
    and clarification using various components such as LLMs (Large Language Models).
    """

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
        """
        Initialize the TableProvider class.

        :param task_name: The name of the task for which the table processing is being conducted.
        :param split: The dataset split being used (e.g., train, validation, test).
        :param table_filter_name: The type of table filtering to apply (e.g., LLM-based or semantics-based).
        :param table_clarifier_name: The type of table clarification to apply (e.g., providing explanations or summaries).
        :param top_k: The number of top rows or columns to select based on relevance.
        :param embedding_type: The type of embeddings to use for semantic filtering (default is "spacy").
        :param whether_column_grounding: A flag indicating whether to perform column grounding (default is False).
        """
        # Initialize the LLM generator for use in table filtering and clarification
        self.call_llm = LLM_Generator()

        # Load and parse the table data using the specified task name and split
        self.table_loader = TableParser(
            task_name, split="validation", use_small_sample_list=True
        )

        # Initialize the table filtering mechanism
        self.table_filter = TableFilter(
            self.call_llm,
            task_name=task_name,
            split=split,
            table_filter_name=table_filter_name,
            embedding_type=embedding_type,
            top_k=top_k,
            whether_column_grounding=whether_column_grounding,
        )

        # Initialize the table clarification mechanism
        self.table_clarification = TableClarification(
            self.call_llm,
            task_name=task_name,
            split=split,
            table_clarifier_name=table_clarifier_name,
        )

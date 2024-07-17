import os
import pandas as pd
from ..agents import CallLLM, Embedder
from utils import select_top_k_samples


class TableGrounding:
    def __init__(
        self,
        call_llm: CallLLM,
        user_query: str,
        task_name: str,
        split: str,
        index_loop: int,
        embedding_type: str = "spacy",
        whether_empirical_study: bool = False,
    ) -> None:
        """
        Args:
            user_query: the user query
            task_name: the task name
            split: the split
            index_loop: the index of the loop
        """
        self.call_llm = call_llm
        self.user_query = user_query
        self.task_name = task_name
        self.split = split
        self.index_loop = index_loop
        self.embedder = Embedder(
            embedding_type=embedding_type,
            whether_empirical_study=whether_empirical_study,
        )

    def column_grounding(self, table: pd.DataFrame, user_query: str) -> pd.DataFrame:
        """
        Grounding columns based on the user query and return the grounded table.
        Args:
            table: the table to be grounded
            user_query: the user query
        Returns:
            the grounded table
        """

        columns = table.columns

        # call embeddings generator
        columns_embeddings, user_query_embedding = self.embedder.call_embeddings(
            user_query=user_query, value_list=columns
        )

        # column candidates
        candidate_columns = [
            columns[index]
            for index in select_top_k_samples(
                columns_embeddings, user_query_embedding, k=self.call_llm.Max_COLUMNS
            )
        ]

        # only keep the columns that are in the candidate columns
        grounded_table = table.loc[:, candidate_columns]

        return grounded_table

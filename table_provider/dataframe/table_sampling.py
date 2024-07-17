import os
import random
import pandas as pd
import numpy as np
from typing import List
from collections import Counter
from ..agents import CallLLM, Embedder
from utils import select_top_k_samples
from ..contract.enum_type import TableSamplingType
from sklearn.cluster import KMeans


def n_gram_overlap(txt_a: str, txt_b: str, n: int = 3) -> float:
    tokens_a, tokens_b = txt_a.split(), txt_b.split()
    n_grams_a = Counter(
        [' '.join(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)]
    )
    n_grams_b = Counter(
        [' '.join(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)]
    )
    intersection = sum((n_grams_a & n_grams_b).values())
    total = sum(n_grams_a.values()) + sum(n_grams_b.values())

    return intersection / total if total > 0 else 0


class TableSampling:
    def __init__(self) -> None:
        pass

    def __init__(
        self,
        call_llm: CallLLM,
        task_name: str,
        split: str,
        table_sampling_type: str,
        embedding_type: str,
        n_cluster: int = 5,
        top_k: int = 3,
        whether_column_grounding: bool = False,
    ):
        """
        args:
            task_name: str, task name
            split: str, train, dev, or test
            table_sampling_type: str, row filter type
        """
        self.task_name = task_name
        self.split = split
        self.call_llm = call_llm  # index of the loop for embedding generation/saving
        self.n_cluster = n_cluster
        self.top_k = top_k
        self.loop_index = 0
        self.whether_column_grounding = whether_column_grounding

        # Check row filter type
        if table_sampling_type not in [
            sampling_type.value for sampling_type in TableSamplingType
        ] + ["default"]:
            raise ValueError(
                f"Table sampling type {table_sampling_type} is not supported"
            )
        # set the default sampling type
        if table_sampling_type == "default":
            table_sampling_type = "clustering_sample"
        self.table_sampling_type = table_sampling_type

        # Initialize the embedder
        self.embedder = Embedder(
            task_name=task_name,
            embedding_tag="rows_embeddings",
            embedding_type=embedding_type,
        )

    def run(self, query: str, parsed_example: dict):
        assert parsed_example is not None, "Table is None"
        assert len(parsed_example["table"]["rows"]) > 0, parsed_example
        assert len(parsed_example["table"]["header"]) > 0, parsed_example
        self.user_query = query
        self.loop_index += 1  # Increment the loop index for embedding generation/saving
        # Run the row filter
        return self.func_set()[self.table_sampling_type](parsed_example)

    def evenly_sampling(self, _example: dict) -> pd.DataFrame:
        """
        row wise insert, header, row 1, row n, row 2, row n-1, ..., row n/2, row n/2+1, until token_size_threshold is reached.
        args:
            _example: dict, parsed table
        return:
            df: pd.DataFrame, filtered table
        """
        total_token_count = 0

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        # Insert rows from the head and tail of the DataFrame until the token size threshold is reached
        head_index, tail_index = 0, len(rows) - 1
        rows_count = 0
        while (
            total_token_count <= self.call_llm.MAX_TRUNCATE_TOKENS
            and rows_count < self.call_llm.MAX_ROWS
        ):
            head_row = rows[head_index]
            tail_row = rows[tail_index]
            head_token_count = self.call_llm.num_tokens_list(head_row)
            tail_token_count = self.call_llm.num_tokens_list(tail_row)

            # If the head and tail meet, add it and break the loop
            if (
                head_index >= tail_index
                or total_token_count + head_token_count
                > self.call_llm.MAX_TRUNCATE_TOKENS
                or total_token_count + tail_token_count
                > self.call_llm.MAX_TRUNCATE_TOKENS
            ):
                break

            # If adding the head row does not exceed the token size threshold, add it to the new DataFrame
            if (
                total_token_count + head_token_count
                <= self.call_llm.MAX_TRUNCATE_TOKENS
            ):
                df.loc[len(df.index)] = head_row
                total_token_count += head_token_count
                head_index += 1
                rows_count += 1

            # Otherwise, if adding the tail row does not exceed the token size threshold, add it to the new DataFrame
            if (
                total_token_count + tail_token_count
                <= self.call_llm.MAX_TRUNCATE_TOKENS
            ):
                df.loc[len(df.index)] = tail_row
                total_token_count += tail_token_count
                tail_index -= 1
                rows_count += 1

        # Set the index of the new DataFrame to be sequential integers
        df.reset_index(drop=True, inplace=True)
        return df

    def clustering_sampling(
        self,
        _example: dict,
    ) -> pd.DataFrame:
        """
        Cluster rows into n clusters, and sample top k rows from each cluster.
        args:
            _example: dict, parsed table
            n_cluster: int, number of clusters
            top_k: int, number of rows to sample from each cluster
        return:
            df: pd.DataFrame, filtered table
        """
        total_token_count = 0
        n_cluster = self.n_cluster
        top_k = self.top_k

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        # Generate embeddings of each rows and the user query
        rows_embeddings, user_query_embeddings = self.embedder.call_embeddings(
            user_query=self.user_query,
            row_column_list=['|'.join(row) for row in rows],
            file_dir_name=self.task_name + "_" + str(self.loop_index),
        )

        if n_cluster > len(rows):
            n_cluster = len(rows)

        # Cluster rows
        k_means = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(
            rows_embeddings
        )
        cluster_labels = k_means.labels_

        # Candidate rows from clustering closet to the user query
        candidate_rows = []
        for cluster_id in range(n_cluster):
            cluster_indices = np.where(cluster_labels == cluster_id)[
                0
            ]  # get the indices of the rows in the cluster
            rows_embeddings = np.array(rows_embeddings)  # convert to np
            valid_indices = select_top_k_samples(
                rows_embeddings[cluster_indices], user_query_embeddings, k=top_k
            )
            candidate_rows.extend(
                [np.array(rows)[cluster_indices][i] for i in valid_indices]
            )
        # Sampling based on the user query matching
        for row in candidate_rows:
            row_token_count = self.call_llm.num_tokens_list(row)
            if total_token_count + row_token_count <= self.call_llm.MAX_TRUNCATE_TOKENS:
                df.loc[len(df.index)] = row
                total_token_count += row_token_count
            else:
                break

        # Set the index of the new DataFrame to be sequential integers
        df.reset_index(drop=True, inplace=True)
        # print("Sampled Tables:\n {}".format(df))
        return df

    def embedding_sampling(self, _example: dict) -> pd.DataFrame:
        """
        Generate embeddings of each rows and the user query, and sample rows based on the user query matching.
        args:
            _example: dict, parsed table
        return:
            df: pd.DataFrame, filtered table
        """
        total_token_count = 0

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        # Generate embeddings of each rows and the user query
        rows_embeddings, user_query_embeddings = self.embedder.call_embeddings(
            user_query=self.user_query,
            row_column_list=['|'.join(row) for row in rows],
            file_dir_name=self.task_name + "_" + str(self.loop_index),
        )

        # Select the top k rows based on the user query matching
        top_k_rows = select_top_k_samples(
            rows_embeddings, user_query_embeddings, k=self.call_llm.MAX_ROWS
        )

        if self.whether_column_grounding:
            columns = df.columns

            # call embeddings generator
            self.embedder.modify_embedding_tag("columns_embeddings")
            columns_embeddings, user_query_embedding = self.embedder.call_embeddings(
                user_query=self.user_query,
                value_list=['|'.join(column) for column in columns],
                file_dir_name=self.task_name + "_" + str(self.loop_index),
            )

            # column candidates
            candidate_columns = [
                columns[index]
                for index in select_top_k_samples(
                    columns_embeddings,
                    user_query_embedding,
                    k=self.call_llm.Max_COLUMNS,
                )
            ]

            # only keep the columns that are in the candidate columns
            df = df.loc[:, candidate_columns]

        self.embedder.modify_embedding_tag("rows_embeddings")
        # Add the top k rows to the new DataFrame
        while total_token_count <= self.call_llm.MAX_TRUNCATE_TOKENS:
            for top_i in top_k_rows:
                top_row = rows[top_i]
                total_token_count += self.call_llm.num_tokens_list(top_row)
                df.loc[len(df.index)] = top_row
            break

        # Set the index of the new DataFrame to be sequential integers
        df.reset_index(drop=True, inplace=True)
        # print("Sampled Tables:\n {}".format(df))
        return df

    def random_sampling(self, _example: dict) -> pd.DataFrame:
        """
        Random sampling the rows.
        args:
            _example: dict, parsed table
        return:
            df: pd.DataFrame, filtered table
        """
        total_token_count = 0

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        # Random sampling the rows
        visited = set()
        while total_token_count <= self.call_llm.MAX_TRUNCATE_TOKENS:
            random_index, random_row = random.choice(list(enumerate(rows)))
            if random_index not in visited:
                total_token_count += self.call_llm.num_tokens_list(random_row)
                try:
                    df.loc[len(df.index)] = random_row
                except ValueError:
                    print("ValueError: {}".format(random_row))
                    print(rows)
                    print(df)
                    break
                visited.add(random_index)
            if len(visited) == len(rows) or len(df.index) >= self.call_llm.MAX_ROWS:
                break

        # Set the index of the new DataFrame to be sequential integers
        df.reset_index(drop=True, inplace=True)
        # print("Sampled Tables:\n {}".format(df))
        return df

    def content_snapshot(self, _example: dict) -> pd.DataFrame:
        total_token_count = 0
        utterance = self.user_query

        # Create a new DataFrame to hold the sampled rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        while total_token_count <= self.call_llm.MAX_TRUNCATE_TOKENS:
            if self.call_llm.MAX_ROWS > 1:
                overlap_scores = [
                    max(self.n_gram_overlap(utterance, ' '.join(row)) for row in rows)
                    for row in rows
                ]
                top_k_indices = np.argsort(overlap_scores)[-self.call_llm.MAX_ROWS :]
                df = df.append([rows[i] for i in top_k_indices], ignore_index=True)
            else:
                # Create a synthetic row for K=1
                synthetic_row = []
                for col_index in range(len(_example["table"]["header"])):
                    column_values = [row[col_index] for row in rows]
                    overlap_scores = [
                        self.n_gram_overlap(utterance, value) for value in column_values
                    ]
                    best_value = column_values[np.argmax(overlap_scores)]
                    synthetic_row.append(best_value)
                df.loc[0] = synthetic_row
        return df

    def table_to_text_sampling(self, _example: dict) -> pd.DataFrame:
        """
        Leverage GPT-3 for zero-shot table-to-text generation.
        Args:
            _example: dict, parsed table
        Return:
            df: pd.DataFrame, filtered table
        """
        df = pd.DataFrame(index=range(1), columns=_example["table"]["header"])
        df_rows = pd.DataFrame(
            _example["table"]["rows"], columns=_example["table"]["header"]
        )
        for col_index, col in enumerate(df.columns):
            column_text = df_rows[col]
            summarized_text = self.call_llm.call_llm_summarization(
                "|".join(column_text)
            )
            df.iloc[0, col_index] = summarized_text

        df.reset_index(drop=True, inplace=True)
        # print("Sampled Tables:\n {}".format(df))
        return df

    def auto_table_sampling(self, _example: dict) -> pd.DataFrame:
        """
        Leverage GPT-3 for zero-shot row filtering program generation.
        Reference: Generate, Transform, Answer: Question Specific Tool Synthesis for Tabular Data
        Args:
            _example: dict, parsed table
        Return:
            df: pd.DataFrame, filtered table
        """
        df = pd.DataFrame(
            index=_example["table"]["rows"], columns=_example["table"]["header"]
        )
        context = self.user_query + "\n\n" + df.to_string()
        code_snippet = self.call_llm.call_llm_code_generation(context)
        try:
            # if the code snippet is valid, then execute it
            eval(code_snippet)
            df = exec(code_snippet)
            # print("Sampled Tables:\n {}".format(df))
            return df
        except Exception as e:
            print("Error: {}".format(e))
            return df

    def func_set(self) -> dict:
        return {
            TableSamplingType.evenly_sample.value: self.evenly_sampling,
            TableSamplingType.clustering_sample.value: self.clustering_sampling,
            TableSamplingType.embedding_sample.value: self.embedding_sampling,
            TableSamplingType.random_sample.value: self.random_sampling,
            TableSamplingType.table_to_text_sample.value: self.table_to_text_sampling,
            TableSamplingType.auto_row_filter.value: self.auto_table_sampling,
        }

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

import pandas as pd
from src.llm.llm_embedder.llm_embedding import Embedder
from src.llm.llm_generator.llm_generating import LLM_Generator
from src.table_loader.data_loader.table_parser.type_sets import TableFilterType
from utils.cos_similarity import select_top_k_samples


class TableFilter:
    """
    A class to perform various types of table filtering, including semantic-based and LLM-based filtering.
    """

    def __init__(
        self,
        call_llm: LLM_Generator,
        task_name: str,
        split: str,
        table_filter_name: str,
        embedding_type: str,
        top_k: int = 3,
        whether_column_grounding: bool = False,
    ):
        """
        Initialize the TableFilter class.

        :param call_llm: An instance of LLM_Generator used for LLM-based filtering.
        :param task_name: The name of the task associated with this filtering operation.
        :param split: The dataset split being used (train, dev, test).
        :param table_filter_name: The type of filtering to be applied.
        :param embedding_type: The type of embedding used for semantic filtering.
        :param top_k: The number of top rows to keep after filtering.
        :param whether_column_grounding: A flag to indicate whether column grounding is required.
        """
        self.task_name = task_name
        self.split = split
        self.call_llm = call_llm
        self.top_k = top_k
        self.loop_index = 0
        self.whether_column_grounding = whether_column_grounding

        # Validate the table filter type
        if table_filter_name not in [
            filter_type.value for filter_type in TableFilterType
        ] + ["default"]:
            raise ValueError(
                f"Table filter type {table_filter_name} is not supported"
            )
        # Default to LLM-based filtering if no valid filter type is provided
        if table_filter_name == "default":
            table_filter_name = "llm_based_filter"
        self.table_filter_name = table_filter_name

        # Initialize the embedder for semantic filtering
        self.embedder = Embedder(
            task_name=task_name,
            embedding_tag="llm_based_filter",
            embedding_type=embedding_type,
        )

    def run(self, query: str, parsed_example: dict):
        """
        Execute the table filtering process.

        :param query: The user query to be used in filtering.
        :param parsed_example: The parsed example containing the table and associated information.
        :return: A DataFrame containing the filtered table.
        """
        assert parsed_example is not None, "Table is None"
        assert len(parsed_example["table"]["rows"]) > 0, parsed_example
        assert len(parsed_example["table"]["header"]) > 0, parsed_example
        self.user_query = query
        self.loop_index += 1  # Increment loop index for embedding generation/saving
        
        # Run the selected filter method
        return self.func_set()[self.table_filter_name](parsed_example)

    def semetics_based_filter(self, _example: dict) -> pd.DataFrame:
        """
        Semantic-based filtering method.
        Generate embeddings for each row and match with the user query to filter rows.

        :param _example: The parsed table example.
        :return: A DataFrame containing the filtered table.
        """
        total_token_count = 0

        # Create a DataFrame for filtered rows
        df = pd.DataFrame(columns=_example["table"]["header"])
        rows = _example["table"]["rows"]
        column_token_count = self.call_llm.num_tokens_list(_example["table"]["header"])
        total_token_count += column_token_count

        # Log the original table before filtering
        original_table = pd.DataFrame(rows, columns=_example["table"]["header"])
        print("Original Table:\n", original_table)

        # Generate embeddings for rows and user query
        rows_embeddings, user_query_embeddings = self.embedder.call_embeddings(
            user_query=self.user_query,
            row_column_list=['|'.join(row) for row in rows],
            file_dir_name=self.task_name + "_" + str(self.loop_index),
        )

        # Select top-k rows matching the user query
        top_k_rows = select_top_k_samples(
            rows_embeddings, user_query_embeddings, k=self.call_llm.MAX_ROWS
        )

        # Optional: Perform column grounding based on query
        if self.whether_column_grounding:
            columns = df.columns
            self.embedder.modify_embedding_tag("columns_embeddings")
            columns_embeddings, user_query_embedding = self.embedder.call_embeddings(
                user_query=self.user_query,
                value_list=['|'.join(column) for column in columns],
                file_dir_name=self.task_name + "_" + str(self.loop_index),
            )

            # Select candidate columns based on query
            candidate_columns = [
                columns[index]
                for index in select_top_k_samples(
                    columns_embeddings,
                    user_query_embedding,
                    k=self.call_llm.Max_COLUMNS,
                )
            ]

            # Filter DataFrame to include only candidate columns
            df = df.loc[:, candidate_columns]

        # Add top-k rows to the DataFrame while token count allows
        self.embedder.modify_embedding_tag("rows_embeddings")
        while total_token_count <= self.call_llm.MAX_TRUNCATE_TOKENS:
            for top_i in top_k_rows:
                top_row = rows[top_i]
                total_token_count += self.call_llm.num_tokens_list(top_row)
                df.loc[len(df.index)] = top_row
            break

        # Reset index for the new DataFrame
        df.reset_index(drop=True, inplace=True)
        return df

    def llm_based_filter(self, _example: dict) -> pd.DataFrame:
        """
        LLM-based filtering method.
        Leverage LLM to generate a program that filters rows based on the user query.

        :param _example: The parsed table example.
        :return: A DataFrame containing the filtered table.
        """
        print("Starting llm_based_filter")

        # Create DataFrame from parsed table example
        df = pd.DataFrame(data=_example["table"]["rows"], columns=_example["table"]["header"])

        # Extract column names for code generation context
        column_names = list(df.columns)

        # Generate context for LLM code generation
        context = f"Columns: {column_names}\n\n{self.user_query}\n\n{df.to_string()}"
        print("Generated context for code generation:")
        print(context)

        # Call LLM to generate the filtering code
        code_snippet = self.call_llm.call_llm_code_generation(context)
        
        # Clean up the code snippet by removing undesired markers
        code_snippet = code_snippet.replace('```python', '').replace('```', '').strip()
        code_snippet = code_snippet.replace('>>> ', '')
        print(f"Finally Generated code snippet: {code_snippet}")

        try:
            # Execute the generated code to filter the DataFrame
            locals_dict = {"df": df}
            exec(code_snippet, {}, locals_dict)
            filtered_df = locals_dict.get("filtered_table", df)
            
            # Check if the filtered DataFrame is empty
            if filtered_df.empty:
                print("Empty table after filtering, return origin")
                return df

            print("Filtered table:\n", filtered_df)
            return filtered_df
        except Exception as e:
            print(f"Error: {e}")
            return df

    def func_set(self) -> dict:
        """
        Return a dictionary mapping filter types to their respective methods.

        :return: A dictionary mapping TableFilterType values to methods.
        """
        return {
            TableFilterType.semetics_based_filter.value: self.semetics_based_filter,
            TableFilterType.llm_based_filter.value: self.llm_based_filter,
        }

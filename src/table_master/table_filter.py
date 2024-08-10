"""
This code file contains functions that borrow certain logic from an anonymous repository associated with the paper:
"TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning" (arXiv:2312.09039).
Original source: https://anonymous.4open.science/r/TableProvider-4CC3/README.md (MIT License).
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
    def __init__(self) -> None:
        pass

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
        args:
            task_name: str, task name
            split: str, train, dev, or test
            table_filter_name: str, row filter type
        """
        self.task_name = task_name
        self.split = split
        self.call_llm = call_llm  # index of the loop for embedding generation/saving
        self.top_k = top_k
        self.loop_index = 0
        self.whether_column_grounding = whether_column_grounding

        # Check row filter type
        if table_filter_name not in [
            filter_type.value for filter_type in TableFilterType
        ] + ["default"]:
            raise ValueError(
                f"Table filter type {table_filter_name} is not supported"
            )
        # set the default filter type /llm_based_filter/semetics_based_filter
        if table_filter_name == "default":
            table_filter_name = "llm_based_filter"
        self.table_filter_name = table_filter_name

        # Initialize the embedder
        self.embedder = Embedder(
            task_name=task_name,
            embedding_tag="llm_based_filter",
            embedding_type=embedding_type,
        )

    def run(self, query: str, parsed_example: dict):
        assert parsed_example is not None, "Table is None"
        assert len(parsed_example["table"]["rows"]) > 0, parsed_example
        assert len(parsed_example["table"]["header"]) > 0, parsed_example
        self.user_query = query
        self.loop_index += 1  # Increment the loop index for embedding generation/saving
        # Run the row filter    
        return self.func_set()[self.table_filter_name](parsed_example)


    def semetics_based_filter(self, _example: dict) -> pd.DataFrame:
        """
        Semantic method / Column and Row
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

        # Logging the table before filtering
        original_table = pd.DataFrame(rows, columns=_example["table"]["header"])
        print("Original Table:\n", original_table)

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

    def llm_based_filter(self, _example: dict) -> pd.DataFrame:
        """
        LLM-Decomposer Method
        Leverage GPT-3 for zero-shot row filtering program generation.
        Reference: Generate, Transform, Answer: Question Specific Tool Synthesis for Tabular Data
        Args:
            _example: dict, parsed table
        Return:
            df: pd.DataFrame, filtered table
        """
        print("Starting llm_based_filter")

        # Create DataFrame from parsed table example
        df = pd.DataFrame(data=_example["table"]["rows"], columns=_example["table"]["header"])

        # Extract column names from DataFrame
        column_names = list(df.columns)

        # Generate context for code generation
        context = f"Columns: {column_names}\n\n{self.user_query}\n\n{df.to_string()}"
        print("Generated context for code generation:")
        print(context)

        # Call LLM for code generation
        code_snippet = self.call_llm.call_llm_code_generation(context)
        
        # Remove undesired code block markers
        code_snippet = code_snippet.replace('```python', '').replace('```', '').strip()
        
        code_snippet = code_snippet.replace('>>> ', '')
        print(f"Finally Generated code snippet: {code_snippet}")

        try:
            # Evaluate the generated code snippet safely
            locals_dict = {"df": df}
            exec(code_snippet, {}, locals_dict)
            filtered_df = locals_dict.get("filtered_table", df)
            
            # Check if the filtered table is empty
            if filtered_df.empty:
                print("Empty table after filtering, return origin")
                return df

            print("Filtered table:\n", filtered_df)
            return filtered_df
        except Exception as e:
            print(f"Error: {e}")
            return df


    def func_set(self) -> dict:
        return {
            TableFilterType.semetics_based_filter.value: self.semetics_based_filter,
            TableFilterType.llm_based_filter.value: self.llm_based_filter,
        }

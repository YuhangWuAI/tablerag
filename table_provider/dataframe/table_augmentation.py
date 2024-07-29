import requests
from ..agents.Metadata.metadata import MetadataApi
from ..agents.call_llm import CallLLM
from ..contract.enum_type import TableAugmentationType
from ..data_loader.table_linearizer import StructuredDataLinearizer
from ..contract.enum_type import TableSerializationType
from langchain.retrievers import WikipediaRetriever
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
import pandas as pd
import json, os
import time


class TableAugmentation:
    def __init__(
        self,
        call_llm: CallLLM,
        task_name: str,
        split: str,
        table_augmentation_type: str,
    ):
        self.call_llm = call_llm
        self.task_name = task_name
        self.split = split
        self.linearizer = StructuredDataLinearizer()

        # check if the table augmentation type is supported
        if table_augmentation_type not in [
            augmentation_type.value for augmentation_type in TableAugmentationType
        ]:
            raise ValueError(
                f"Table Augmentation Type {table_augmentation_type} is not supported"
            )
        self.table_augmentation_type = table_augmentation_type

    def run(self, parsed_example: dict) -> pd.DataFrame:
        """
        Run the table augmentation.
        Args:
            query: the query
            table: the table
        Returns:
            the augmented table
        """
        assert parsed_example is not None, "Table is None"
        assert len(parsed_example["table"]["rows"]) > 0, "Table has no rows"
        assert len(parsed_example["table"]["header"]) > 0, "Table has no header"
        # Run the row filter
        if self.table_augmentation_type == "header_field_categories":
            augmentation_info = self.func_set()["metadata"](
                parsed_example, only_return_categories=True
            )
        else:
            augmentation_info = self.func_set()[self.table_augmentation_type](
                parsed_example
            )
        # print(f"Augmentation info: {augmentation_info}")
        if (
            self.call_llm.num_tokens(augmentation_info)
            < self.call_llm.AUGMENTATION_TOKEN_LIMIT
        ):
            return augmentation_info
        else:
            return (
                    self.call_llm.truncated_string(
                    augmentation_info,
                    self.call_llm.AUGMENTATION_TOKEN_LIMIT,
                    print_warning=False,
                )
            )

    def get_table_size(self, parsed_example: dict) -> str:
        """
        Get the table size
        Args:
            parsed_example: the parsed example
        Returns:
            the table size
        """
        return json.dumps(
            {
                "table_size": [
                    len(parsed_example["table"]["header"]),
                    len(parsed_example["table"]["rows"]),
                ]
            },
            indent=4,
            sort_keys=True,
        )

    def get_header_hierarchy(self, parsed_example: dict) -> str:
        """
        Get the header hierarchy
        Args:
            parsed_example: the parsed example
        Returns:
            the header hierarchy
        """
        return json.dumps(
            parsed_example["table"]["header_hierarchy"], indent=4, sort_keys=True
        )

    def get_metatdata(
        self, parsed_example: dict, only_return_categories: bool = False
    ) -> str:
        """
        Get the metadata
        Args:
            parsed_example: the parsed example
        Returns:
            the metadata
        """
        df = pd.DataFrame(
            parsed_example["table"]["rows"], columns=parsed_example["table"]["header"]
        )
        metadata_api = MetadataApi(
            'table_provider/agents/Metadata/model/model/metadata_tapas_202202_d0e0.pt'
        )
        emb = metadata_api.embedding(df)
        predict = metadata_api.predict(df, emb)
        if only_return_categories:
            return str(predict["Msr_type_res"])
        else:
            return json.dumps(
                {
                    "measure_dimension_type": predict["Msr_res"],
                    "aggregation_type": predict["Agg_score_res"],
                    "measure_type": predict["Msr_type_res"],
                },
                indent=4,
                sort_keys=True,
            )

    def get_intermediate_NL_reasoning_steps(self, parsed_example: dict) -> str:
        instruction = "\n".join(
            [
                """
            You are a brilliant table executor with the capabilities information retrieval, table parsing, 
        table partition and semantic understanding who can understand the structural information of the table.
        """,
                f"""
        Generate intermediate NL reasoning steps for better understanding the following table \n{parsed_example["table"]}.
        """,
                """Only return the reasoning steps in python string format""",
            ]
        )

        output = self.call_llm.generate_text(instruction)
        if isinstance(output, list):
            return " ".join(output)
        elif isinstance(output, str):
            return output

    def get_trunk_summarization(self, parsed_example: dict) -> str:
        sampled_table = {
            "title": parsed_example["title"],
            "context": parsed_example["context"],
            "table": {
                "header": parsed_example["table"]["header"],
                "rows": parsed_example["table"]["rows"][:5],
                "caption": parsed_example["table"]["caption"],
            },
        }
        linearized_table = self.linearizer.retrieve_linear_function(
            TableSerializationType.html, structured_data_dict=sampled_table
        )
        instruction = "\n".join(
            [
                """
             You are a brilliant table executor with the capabilities information retrieval, table parsing, 
        table partition and semantic understanding who can understand the structural information of the table.
        """,
                f"""
        Generate trunk summary of following table schema \n{linearized_table}.
        """,
            ]
        )

        output = self.call_llm.generate_text(instruction)
        if isinstance(output, list):
            return " ".join(output)
        elif isinstance(output, str):
            return output




    def get_term_explanations(self, parsed_example: dict) -> str:
        print("Starting get_term_explanations method")
        
        # Extract the table, query (statement), and caption from the parsed example
        table = {
            "header": parsed_example.get("table", {}).get("header", []),
            "rows": parsed_example.get("table", {}).get("rows", [])
        }
        statement = parsed_example.get("query", "")
        caption = parsed_example.get("table", {}).get("caption", "")
        
        # Print the extracted information
        print("Original Table:", json.dumps(table, indent=2))
        print("Statement:", statement)
        print("Caption:", caption)
        
        # Call terms explanation method
        generated_text = self.call_llm.generate_terms_explanation(table, statement, caption)
        print("Generated explanations:", generated_text)
        
        # Directly return the generated text as augmentation info
        return generated_text
        
    def get_docs_references(self, parsed_example: dict) -> str:
        print("Starting get_docs_references method")

        retriever = WikipediaRetriever(lang="en", load_max_docs=2)
        
        try:
            # Use caption for document retrieval if available
            if parsed_example["table"].get("caption"):
                print("Using caption for document retrieval:", parsed_example["table"]["caption"])
                docs = retriever.get_relevant_documents(parsed_example["table"]["caption"])
            # If caption is also not available, use table headers
            else:
                print("No caption found, using header instead:", parsed_example['table']['header'])
                docs = retriever.get_relevant_documents(" ".join(parsed_example["table"]["header"]))
            
            # Ensure this wait is required, might not be needed
            time.sleep(5)
            
            # Extract relevant metadata from the retrieved documents
            metadata_list = []
            for doc in docs:
                metadata = {
                    'title': doc.metadata.get('title', 'N/A'),
                    'summary': doc.metadata.get('summary', 'N/A'),
                    'source': doc.metadata.get('source', 'N/A')
                }
                metadata_list.append(metadata)

            # Print the metadata for debugging
            print("Retrieved metadata: ", metadata_list)

            # Extract table, statement, and caption from parsed_example
            table = {
                "header": parsed_example.get("table", {}).get("header", []),
                "rows": parsed_example.get("table", {}).get("rows", [])
            }
            statement = parsed_example.get("query", "")
            caption = parsed_example.get("table", {}).get("caption", "")

            # Call the method to generate table summary using metadata
            print("Calling generate_table_summary with metadata")
            generated_summary = self.call_llm.generate_table_summary(metadata_list, table, statement, caption)
            print("Generated summary:", generated_summary)
            
            return generated_summary
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while retrieving documents: {e}")
            return "Document retrieval failed"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "An unexpected error occurred"




    def assemble_neural_symbolic_augmentation(
        self, parsed_example: dict
    ) -> pd.DataFrame:
        return "\n".join(
            self.get_table_size(parsed_example),
            self.get_header_hierarchy(parsed_example),
            self.get_metatdata(parsed_example),
            self.get_intermediate_NL_reasoning_steps(parsed_example),
        )

    def assemble_retrieval_based_augmentation(
        self, parsed_example: dict
    ) -> pd.DataFrame:
        return "\n".join(
            self.get_term_explanations(parsed_example),
            self.get_docs_references(parsed_example),
        )
    
    # Ablation experiments
    def func_set(self) -> dict:
        return {
            TableAugmentationType.extra_structural_info_table_size.value: self.get_table_size,
            TableAugmentationType.extra_structural_info_header_hierarchy.value: self.get_header_hierarchy,
            TableAugmentationType.extra_analytical_roles.value: self.get_metatdata,
            TableAugmentationType.extra_intermediate_NL_reasoning_steps.value: self.get_intermediate_NL_reasoning_steps,
            TableAugmentationType.extra_summary_statistics_info_trunk_summary.value: self.get_trunk_summarization,
            TableAugmentationType.external_retrieved_knowledge_info_term_explanations.value: self.get_term_explanations,
            TableAugmentationType.external_retrieved_knowledge_info_docs_references.value: self.get_docs_references,
            TableAugmentationType.assemble_neural_symbolic_augmentation.value: self.assemble_neural_symbolic_augmentation,
            TableAugmentationType.assemble_retrieval_based_augmentation.value: self.assemble_retrieval_based_augmentation,
        }

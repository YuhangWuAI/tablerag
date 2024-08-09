import requests

from ..data_loader.enum_type import TableAugmentationType
from ..agents.call_llm import CallLLM

from ..data_loader.table_linearizer import StructuredDataLinearizer
from langchain_community.retrievers import WikipediaRetriever
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings("ignore")

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

    def run(self, parsed_example: dict) -> dict:
        """
        Run the table augmentation.
        Args:
            query: the query
            table: the table
        Returns:
            A dictionary with the augmented table including 'terms_explanation' and 'table_summary' fields.
        """
        assert parsed_example is not None, "Table is None"
        assert len(parsed_example["table"]["rows"]) > 0, "Table has no rows"
        assert len(parsed_example["table"]["header"]) > 0, "Table has no header"
        # Run the augmentation
        if self.table_augmentation_type == "header_field_categories":
            augmentation_info = self.func_set()["metadata"](
                parsed_example, only_return_categories=True
            )
        else:
            augmentation_info = self.func_set()[self.table_augmentation_type](
                parsed_example
            )
        
        # Return the augmentation info as a dictionary
        return augmentation_info

    def get_term_explanations(self, parsed_example: dict) -> dict:
        print("Starting get_term_explanations method")
        
        # Extract the table, query (statement), and caption from the parsed example
        table = {
            "header": parsed_example.get("table", {}).get("header", []),
            "rows": parsed_example.get("table", {}).get("rows", [])
        }
        statement = parsed_example.get("query", "")
        caption = parsed_example.get("table", {}).get("caption", "")
        
        # Call terms explanation method
        generated_text = self.call_llm.generate_terms_explanation(table, statement, caption)
        
        # Return the generated text in a dictionary under the 'terms_explanation' key
        return {"terms_explanation": generated_text}
        
    def get_docs_references(self, parsed_example: dict) -> dict:
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

            # Extract context from parsed_example
            context = parsed_example.get("context", [])
            
            # Extract table, statement, and caption from parsed_example
            table = {
                "header": parsed_example.get("table", {}).get("header", []),
                "rows": parsed_example.get("table", {}).get("rows", [])
            }
            statement = parsed_example.get("query", "")
            caption = parsed_example.get("table", {}).get("caption", "")

            # Call the method to generate table summary using individual parameters
            print("Calling generate_table_summary with individual parameters")
            generated_summary = self.call_llm.generate_table_summary(metadata_list, context, table, statement, caption)
            print("Generated summary:", generated_summary)
            
            # Return the generated summary in a dictionary under the 'table_summary' key
            return {"table_summary": generated_summary}
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while retrieving documents: {e}")
            return {"table_summary": "Document retrieval failed"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"table_summary": "An unexpected error occurred"}


    def assemble_retrieval_based_augmentation(
        self, parsed_example: dict
    ) -> dict:
        term_explanations = self.get_term_explanations(parsed_example)
        docs_references = self.get_docs_references(parsed_example)
        
        # Merge the dictionaries and return as a single dictionary
        return {**term_explanations, **docs_references}
    
    # Ablation experiments
    def func_set(self) -> dict:
        return {
            TableAugmentationType.external_retrieved_knowledge_info_term_explanations.value: self.get_term_explanations,
            TableAugmentationType.external_retrieved_knowledge_info_docs_references.value: self.get_docs_references,
            TableAugmentationType.assemble_retrieval_based_augmentation.value: self.assemble_retrieval_based_augmentation,
        }

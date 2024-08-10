"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

For any questions or further information, please feel free to reach out via the email address above.
"""

import requests


from src.llm.llm_generator.llm_generating import LLM_Generator
from src.table_loader.data_loader.table_parser.type_sets import TableClarificationType
from src.table_loader.data_loader.table_parser.table_linearizer import StructuredDataLinearizer




from langchain_community.retrievers import WikipediaRetriever
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings("ignore")

class TableClarification:
    def __init__(
        self,
        call_llm: LLM_Generator,
        task_name: str,
        split: str,
        table_clarifier_name: str,
    ):
        self.call_llm = call_llm
        self.task_name = task_name
        self.split = split
        self.linearizer = StructuredDataLinearizer()

        # check if the table clarifier_name type is supported
        if table_clarifier_name not in [
            clarifier_name.value for clarifier_name in TableClarificationType
        ]:
            raise ValueError(
                f"Table Augmentation Type {table_clarifier_name} is not supported"
            )
        self.table_clarifier_name = table_clarifier_name

    def run(self, parsed_example: dict) -> dict:
        """
        Run the table clarifier_name.
        Args:
            query: the query
            table: the table
        Returns:
            A dictionary with the augmented table including 'terms_explanation' and 'table_summary' fields.
        """
        assert parsed_example is not None, "Table is None"
        assert len(parsed_example["table"]["rows"]) > 0, "Table has no rows"
        assert len(parsed_example["table"]["header"]) > 0, "Table has no header"
        # Run the clarifier_name
        if self.table_clarifier_name == "header_field_categories":
            clarification_text = self.func_set()["metadata"](
                parsed_example, only_return_categories=True
            )
        else:
            clarification_text = self.func_set()[self.table_clarifier_name](
                parsed_example
            )
        
        # Return the clarifier_name info as a dictionary
        return clarification_text

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
        
    def table_summary(self, parsed_example: dict) -> dict:
        print("Starting table_summary method")

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


    def terms_explanation_and_summary(
        self, parsed_example: dict
    ) -> dict:
        term_explanations = self.get_term_explanations(parsed_example)
        docs_references = self.table_summary(parsed_example)
        
        # Merge the dictionaries and return as a single dictionary
        return {**term_explanations, **docs_references}
    
    # Ablation experiments
    def func_set(self) -> dict:
        return {
            TableClarificationType.external_retrieved_knowledge_info_term_explanations.value: self.get_term_explanations,
            TableClarificationType.external_retrieved_knowledge_info_docs_references.value: self.table_summary,
            TableClarificationType.terms_explanation_and_summary.value: self.terms_explanation_and_summary,
        }

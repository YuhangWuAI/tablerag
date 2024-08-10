"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

For any questions or further information, please feel free to reach out via the email address above.
"""

import json

def serialize_request(query: str, table_formatted: str, clarification_text: dict, context: str) -> dict:
    """
    Serialize the input parameters into a dictionary format for request processing.

    :param query: The query string to be processed.
    :param table_formatted: The formatted table data as a string.
    :param clarification_text: A dictionary containing terms explanation and table summary.
    :param context: Additional context related to the table.
    :return: A dictionary representing the serialized request.
    """
    try:
        # Extract terms_explanation and table_summary from clarification_text, defaulting to empty strings if absent
        terms_explanation = clarification_text.get("terms_explanation", "")
        table_summary = clarification_text.get("table_summary", "")

        # Construct and return the request dictionary, including the table_context field
        request_dict = {
            "query": query,
            "table_formatted": table_formatted,
            "terms_explanation": terms_explanation,
            "table_summary": table_summary,
            "table_context": context  # Add context to the request dictionary
        }
        
        return request_dict

    except Exception as e:
        print(f"Error in serialize_request: {e}")
        # Return an empty dictionary to ensure subsequent code can continue running
        return {}

from typing import List, Dict

def deserialize_retrieved_text(retrieved_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Deserialize the retrieved document texts into a structured format.

    :param retrieved_docs: A list of dictionaries containing the retrieved document content.
    :return: A list of dictionaries, each representing the parsed content of a document.
    """
    parsed_results = []

    for doc in retrieved_docs:
        content = doc.get('content', '')

        # Initialize the parsed data dictionary, all values are strings
        parsed_data = {
            'query_need_to_answer': '',
            'table_formatted': '',
            'terms_explanation': '',
            'table_summary': '',
            'table_context': ''  # Initialize the table_context field
        }
        
        try:
            # Extract query_need_to_answer
            query_start = content.find('query_need_to_answer:')
            table_formatted_start = content.find('table_formatted:')
            if query_start != -1 and table_formatted_start != -1:
                parsed_data['query_need_to_answer'] = content[query_start + len('query_need_to_answer:'):table_formatted_start].strip()

            # Extract table_formatted
            table_formatted_start += len('table_formatted:')
            terms_explanation_start = content.find('terms_explanation:')
            if table_formatted_start != -1 and terms_explanation_start != -1:
                parsed_data['table_formatted'] = content[table_formatted_start:terms_explanation_start].strip()

            # Extract terms_explanation
            terms_explanation_start += len('terms_explanation:')
            table_summary_start = content.find('table_summary:')
            if terms_explanation_start != -1 and table_summary_start != -1:
                terms_explanation_json = content[terms_explanation_start:table_summary_start].strip()
                try:
                    parsed_data['terms_explanation'] = json.dumps(json.loads(terms_explanation_json)) if terms_explanation_json else ''
                except json.JSONDecodeError:
                    parsed_data['terms_explanation'] = ''  # Set to empty string on parsing failure

            # Extract table_summary
            table_summary_start += len('table_summary:')
            table_context_start = content.find('table_context:')
            if table_summary_start != -1 and table_context_start != -1:
                parsed_data['table_summary'] = content[table_summary_start:table_context_start].strip()

            # Extract table_context
            table_context_start += len('table_context:')
            if table_context_start != -1:
                parsed_data['table_context'] = content[table_context_start:].strip()

        except Exception as e:
            print(f"Error in deserializing retrieved text: {e}")
            # Continue returning partially parsed results in case of error

        # Add the parsed result to the results list
        parsed_results.append(parsed_data)
    
    return parsed_results

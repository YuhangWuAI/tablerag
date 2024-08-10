"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

For any questions or further information, please feel free to reach out via the email address above.
"""

import json
import os

def save_jsonl_file(
    request: dict,
    label: str,
    file_path: str,
    pred: str = None,
):
    """
    Save a dictionary as a JSONL formatted line in the specified file.

    :param request: A dictionary containing the request details.
    :param label: The ground truth label associated with the request.
    :param file_path: The file path where the JSONL line will be saved.
    :param pred: The predicted value, if any.
    """
    # Create the directory if it does not exist
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the request string, including the table_context field
    request_str = (
        f"query_need_to_answer:\n{request['query']}\n"
        f"table_formatted:\n{request['table_formatted']}\n"
        f"terms_explanation:\n{request.get('terms_explanation', '')}\n"
        f"table_summary:\n{request.get('table_summary', '')}\n"
        f"table_context:\n{request.get('table_context', '')}"  # Ensure the context is included
    )

    # Prepare the data dictionary to be saved
    data = {'request': request_str, 'label': label}
    if pred is not None:
        data['pred'] = pred

    # Append the JSON-serialized data to the file
    with open(file_path, 'a') as file:
        json_string = json.dumps(data, ensure_ascii=False)  # Ensure no ASCII escaping and single-line output
        file.write(json_string + '\n')


def load_processed_indices(file_path: str):
    """
    Load the indices of already processed lines from a JSONL file.

    :param file_path: Path to the JSONL file.
    :return: A set of indices representing processed lines.
    """
    processed_indices = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                try:
                    json.loads(line)
                    processed_indices.add(i)
                except json.JSONDecodeError:
                    break
    return processed_indices

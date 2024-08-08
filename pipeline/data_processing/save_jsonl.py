# save_jsonl.py

import json
import os

def save_jsonl_file(
    request: dict,
    label: str,
    file_path: str,
    pred: str = None,
):
    # Create the directory if it does not exist
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the request string, now including the table_context (or context) field
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
        json_string = json.dumps(data, ensure_ascii=False)  # Ensuring no ASCII escaping and single-line output
        file.write(json_string + '\n')


def load_processed_indices(file_path: str):
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

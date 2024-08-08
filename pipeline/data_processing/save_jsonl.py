# save_jsonl.py

import json
import os

def save_jsonl_file(
    request: dict,
    label: str,
    file_path: str,
    pred: str = None,
):
    # mkdir
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 组织 request 的内容，并添加描述信息
    request_str = (
        f"query_need_to_answer:\n{request['query']}\n"
        f"table_formatted:\n{request['table_formatted']}\n"
        f"terms_explanation:\n{request.get('terms_explanation', '')}\n"
        f"table_summary:\n{request.get('table_summary', '')}"
    )

    data = {'request': request_str, 'label': label}
    if pred is not None:
        data['pred'] = pred

    # save jsonl: 将 json.dumps 的 indent 参数设置为 None 以确保输出为单行
    with open(file_path, 'a') as file:
        json_string = json.dumps(data, ensure_ascii=False)  # 不使用缩进和换行
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

import json

def serialize_request(query: str, table_html: str, explanations: dict, summary: str = None) -> str:
    request_parts = [
        f"query:\n{query}",
        f"the table needed to be answered:\n{table_html}",
        f"explanations:\n{json.dumps(explanations, indent=4)}",
    ]
    
    if summary:
        request_parts.append(f"summary:\n{summary}")
    
    # 将各部分拼接成一个字符串，中间用两个换行符分隔
    request = "\n\n".join(request_parts)
    
    return request

def deserialize_request(request: str) -> dict:
    # 按照索引名进行拆分
    parts = request.split("\n\n")
    
    parsed_data = {}
    
    for part in parts:
        if part.startswith("query:"):
            parsed_data["query"] = part[len("query:\n"):].strip()
        elif part.startswith("the table needed to be answered:"):
            parsed_data["table_html"] = part[len("the table needed to be answered:\n"):].strip()
        elif part.startswith("explanations:"):
            explanations_json = part[len("explanations:\n"):].strip()
            parsed_data["explanations"] = json.loads(explanations_json)
        elif part.startswith("summary:"):
            parsed_data["summary"] = part[len("summary:\n"):].strip()
    
    return parsed_data

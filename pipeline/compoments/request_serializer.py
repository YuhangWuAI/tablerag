import json

def serialize_request(query: str, table_html: str, explanations: dict, summary: str = None) -> str:
    request_dict = {
        "query": query,
        "table_html": table_html,
        "explanations": explanations,
    }
    
    if summary:
        request_dict["summary"] = summary
    
    # 将字典序列化为JSON字符串
    request = json.dumps(request_dict, indent=4)
    
    return request

def deserialize_request(request: str) -> dict:
    # 将JSON字符串解析为字典
    request_dict = json.loads(request)
    
    parsed_data = {
        "query": request_dict.get("query", ""),
        "table_html": request_dict.get("table_html", ""),
        "explanations": request_dict.get("explanations", {}),
        "summary": request_dict.get("summary", ""),
    }
    
    return parsed_data

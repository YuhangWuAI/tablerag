import json

def serialize_request(query: str, table_html: str, augmentation_info: dict) -> str:
    try:
        # 从 augmentation_info 中提取 summary 和 explanations
        summary = augmentation_info.get("summary", "")
        explanations = augmentation_info.get("explanations", {})

        # 构建请求字典
        request_dict = {
            "query": query,
            "table_html": table_html,
            "explanations": explanations,
            "summary": summary,
        }
        
        # 将字典序列化为 JSON 字符串
        request = json.dumps(request_dict, indent=4)
        return request

    except Exception as e:
        print(f"Error in serialize_request: {e}")
        # 返回一个空的 JSON 字符串，确保后续代码仍然可以正常运行
        return "{}"

def deserialize_request(request: str) -> dict:
    try:
        # 将 JSON 字符串解析为字典
        request_dict = json.loads(request)
        
        # 提取和解析相关字段
        parsed_data = {
            "query": request_dict.get("query", ""),
            "table_html": request_dict.get("table_html", ""),
            "explanations": request_dict.get("explanations", {}),
            "summary": request_dict.get("summary", ""),
        }
        return parsed_data

    except Exception as e:
        print(f"Error in deserialize_request: {e}")
        # 返回一个包含默认值的字典，确保后续代码仍然可以正常运行
        return {
            "query": "",
            "table_html": "",
            "explanations": {},
            "summary": "",
        }

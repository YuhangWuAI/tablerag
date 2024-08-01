import json

def serialize_request(query: str, table_html: str, augmentation_info: dict) -> dict:
    try:
        # 从 augmentation_info 中提取 terms_explanation 和 table_summary，如果不存在则为空字符串
        terms_explanation = augmentation_info.get("terms_explanation", "")
        table_summary = augmentation_info.get("table_summary", "")

        # 构建请求字典并返回
        request_dict = {
            "query": query,
            "table_html": table_html,
            "terms_explanation": terms_explanation,
            "table_summary": table_summary,
        }
        
        return request_dict

    except Exception as e:
        print(f"Error in serialize_request: {e}")
        # 返回一个空的字典，确保后续代码仍然可以正常运行
        return {}


def deserialize_request(request: str) -> dict:
    try:
        # 将 JSON 字符串解析为字典
        request_dict = json.loads(request)
        
        # 提取和解析相关字段，确保即使字段不存在也能返回默认值
        parsed_data = {
            "query": request_dict.get("query", ""),
            "table_html": request_dict.get("table_html", ""),
            "terms_explanation": request_dict.get("terms_explanation", ""),
            "table_summary": request_dict.get("table_summary", ""),
        }
        return parsed_data

    except Exception as e:
        print(f"Error in deserialize_request: {e}")
        # 返回一个包含默认值的字典，确保后续代码仍然可以正常运行
        return {
            "query": "",
            "table_html": "",
            "terms_explanation": "",
            "table_summary": "",
        }

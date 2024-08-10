"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

For any questions or further information, please feel free to reach out via the email address above.
"""

import json


def serialize_request(query: str, table_formatted: str, clarification_text: dict, context: str) -> dict:
    try:
        # 从 clarification_text 中提取 terms_explanation 和 table_summary，如果不存在则为空字符串
        terms_explanation = clarification_text.get("terms_explanation", "")
        table_summary = clarification_text.get("table_summary", "")

        # 构建请求字典并返回，添加 table_context 字段
        request_dict = {
            "query": query,
            "table_formatted": table_formatted,
            "terms_explanation": terms_explanation,
            "table_summary": table_summary,
            "table_context": context  # 添加 context 到请求字典中
        }
        
        return request_dict

    except Exception as e:
        print(f"Error in serialize_request: {e}")
        # 返回一个空的字典，确保后续代码仍然可以正常运行
        return {}



from typing import List, Dict

def deserialize_retrieved_text(retrieved_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    parsed_results = []

    for doc in retrieved_docs:
        content = doc.get('content', '')

        # 初始化解析后的字典，所有值均为字符串
        parsed_data = {
            'query_need_to_answer': '',
            'table_formatted': '',
            'terms_explanation': '',
            'table_summary': '',
            'table_context': ''  # 初始化 table_context 字段
        }
        
        try:
            # 解析出 query_need_to_answer
            query_start = content.find('query_need_to_answer:')
            table_formatted_start = content.find('table_formatted:')
            if query_start != -1 and table_formatted_start != -1:
                parsed_data['query_need_to_answer'] = content[query_start + len('query_need_to_answer:'):table_formatted_start].strip()

            # 解析出 table_formatted
            table_formatted_start += len('table_formatted:')
            terms_explanation_start = content.find('terms_explanation:')
            if table_formatted_start != -1 and terms_explanation_start != -1:
                parsed_data['table_formatted'] = content[table_formatted_start:terms_explanation_start].strip()

            # 解析出 terms_explanation
            terms_explanation_start += len('terms_explanation:')
            table_summary_start = content.find('table_summary:')
            if terms_explanation_start != -1 and table_summary_start != -1:
                terms_explanation_json = content[terms_explanation_start:table_summary_start].strip()
                try:
                    parsed_data['terms_explanation'] = json.dumps(json.loads(terms_explanation_json)) if terms_explanation_json else ''
                except json.JSONDecodeError:
                    parsed_data['terms_explanation'] = ''  # 解析失败时，设置为空字符串

            # 解析出 table_summary
            table_summary_start += len('table_summary:')
            table_context_start = content.find('table_context:')
            if table_summary_start != -1 and table_context_start != -1:
                parsed_data['table_summary'] = content[table_summary_start:table_context_start].strip()

            # 解析出 table_context
            table_context_start += len('table_context:')
            if table_context_start != -1:
                parsed_data['table_context'] = content[table_context_start:].strip()

        except Exception as e:
            print(f"Error in deserializing retrieved text: {e}")
            # 如果出现错误，继续返回当前解析的部分

        # 将解析结果添加到结果列表中
        parsed_results.append(parsed_data)
    
    return parsed_results
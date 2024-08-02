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


import json



def deserialize_retrieved_text(retrieved_docs):
    parsed_results = []

    for doc in retrieved_docs:
        content = doc.get('content', '')

        # 初始化解析后的字典
        parsed_data = {
            'query_need_to_answer': '',
            'table_html': '',
            'terms_explanation': '',
            'table_summary': ''
        }
        
        try:
            # 解析出 query_need_to_answer
            query_start = content.find('query_need_to_answer:')
            table_html_start = content.find('table_html:')
            if query_start != -1 and table_html_start != -1:
                parsed_data['query_need_to_answer'] = content[query_start + len('query_need_to_answer:'):table_html_start].strip()

            # 解析出 table_html
            table_html_start += len('table_html:')
            terms_explanation_start = content.find('terms_explanation:')
            if table_html_start != -1 and terms_explanation_start != -1:
                parsed_data['table_html'] = content[table_html_start:terms_explanation_start].strip()

            # 解析出 terms_explanation
            terms_explanation_start += len('terms_explanation:')
            table_summary_start = content.find('table_summary:')
            if terms_explanation_start != -1 and table_summary_start != -1:
                terms_explanation_json = content[terms_explanation_start:table_summary_start].strip()
                try:
                    parsed_data['terms_explanation'] = json.loads(terms_explanation_json) if terms_explanation_json else {}
                except json.JSONDecodeError:
                    parsed_data['terms_explanation'] = {}  # 解析失败时，设置为空字典

            # 解析出 table_summary
            table_summary_start += len('table_summary:')
            if table_summary_start != -1:
                parsed_data['table_summary'] = content[table_summary_start:].strip()

        except Exception as e:
            print(f"Error in deserializing retrieved text: {e}")
            # 如果出现错误，继续返回当前解析的部分

        # 将解析结果添加到结果列表中
        parsed_results.append(parsed_data)
    
    return parsed_results
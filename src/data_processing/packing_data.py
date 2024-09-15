import json

def process_jsonl(input_file, output_file):
    # 初始化id计数器
    id_counter = 1
    # 打开输入的jsonl文件
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        # 逐行读取jsonl文件
        for line in infile:
            # 加载每条json数据
            data = json.loads(line)
            
            # 将所有字段内容拼接成字符串，并赋值给request字段
            data['request'] = (
                f"table_title: {data.get('table_title', '')}\n"
                f"table_context: {data.get('table_context', [])}\n"
                f"table_formatted: {data.get('table_formatted', '')}\n"
                f"table_summary: {data.get('table_summary', '')}\n"
                f"terms_explanation: {data.get('terms_explanation', '')}\n"
                f"query_suggestions: {data.get('query_suggestions', '')}\n"
            )
            
            # 添加id字段
            data['id'] = str(id_counter)
            
            # 移除原有字段
            for key in ["table_title", "table_context", "table_formatted", "table_summary", "terms_explanation", "query_suggestions"]:
                data.pop(key, None)
            
            # 将处理后的数据写入输出jsonl文件
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            # 更新id计数器
            id_counter += 1


# 使用示例
input_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/clarified_data/e2ewq.jsonl'   # 输入的jsonl文件名
output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/clarified_data/packed_e2ewq.jsonl' # 输出的jsonl文件名

process_jsonl(input_file, output_file)

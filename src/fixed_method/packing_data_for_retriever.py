import json

def process_jsonl(input_file, output_file, pack_summary=False, pack_explanations=False, pack_suggestions=False):
    """
    处理jsonl文件，将特定字段打包为request字段，并可选择性地将summary, explanations, suggestions字段置为空。

    参数:
    - input_file: 输入jsonl文件路径
    - output_file: 输出jsonl文件路径
    - pack_summary: 是否将table_summary打包为空
    - pack_explanations: 是否将terms_explanation打包为空
    - pack_suggestions: 是否将query_suggestions打包为空
    """
    # 初始化id计数器
    id_counter = 1
    # 打开输入的jsonl文件
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        # 逐行读取jsonl文件
        for line in infile:
            # 加载每条json数据
            data = json.loads(line)
            
            # 根据参数选择是否将字段打包为空
            table_summary = '' if pack_summary else data.get('table_summary', '')
            terms_explanation = '' if pack_explanations else data.get('terms_explanation', '')
            query_suggestions = '' if pack_suggestions else data.get('query_suggestions', '')

            # 将所有字段内容拼接成字符串，并赋值给request字段
            data['request'] = (
                f"table_title: {data.get('table_title', '')}\n"
                f"table_context: {data.get('table_context', [])}\n"
                f"table_formatted: {data.get('table_formatted', '')}\n"
                f"table_summary: {table_summary}\n"
                f"terms_explanation: {terms_explanation}\n"
                f"query_suggestions: {query_suggestions}\n"
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
input_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/clarified_data/nqtables.jsonl'   # 输入的jsonl文件名
output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/packed_data/packed_nqtables_test.jsonl' # 输出的jsonl文件名

# 调用函数，pack_summary=True表示将summary打包为空，pack_explanations=False表示保留explanations
process_jsonl(input_file, output_file, pack_summary=True, pack_explanations=True, pack_suggestions=True)

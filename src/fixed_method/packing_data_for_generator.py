import json

# 定义文件路径
input_file_1 = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/llm_filtered_data/nqtables.jsonl'
input_file_2 = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/clarified_data/nqtables.jsonl'
input_file_3 = '/home/yuhangwu/Desktop/Projects/tablerag/data/raw/small_dataset/nqtables.jsonl'
output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/row_col_filtered_data/nqtables.jsonl'


# 读取第一个文件 e2ewtq_test.jsonl
with open(input_file_1, 'r') as file1:
    lines1 = file1.readlines()

# 初始化结果存储列表
results = []

# 逐行处理文件1的内容
for line_num, line in enumerate(lines1):
    data = json.loads(line)
    
    # 如果 id == passage_id，执行后续处理
    if data['id'] == data['result']['passage_id']:
        current_id = data['id']
        query = data['query']
        
        # 读取第二个文件 clarified_data 中的对应行 (id + 1 行)
        with open(input_file_2, 'r') as file2:
            clarified_lines = file2.readlines()
            
            if current_id < len(clarified_lines):  # 确保不会超出文件行数
                clarified_data = json.loads(clarified_lines[current_id])
                
                # 提取需要的字段
                table_title = clarified_data.get('table_title', '')
                table_context = clarified_data.get('table_context', [])
                table_summary = clarified_data.get('table_summary', '')
                terms_explanation = clarified_data.get('terms_explanation', '')
                
                # 从第三个文件 e2ewtq.jsonl 中读取 formatted_table 和 label 字段
                with open(input_file_3, 'r') as file3:
                    raw_lines = file3.readlines()

                    if current_id < len(raw_lines):  # 确保不会超出文件行数
                        raw_data = json.loads(raw_lines[current_id])
                        
                        # 提取 formatted_table 字段
                        formatted_table = raw_data.get('table', '')

                        # 提取 label 字段，结合 label 和 alternativeLabel
                        label = []
                        if 'label' in raw_data:
                            label.append(raw_data['label'])
                        if 'alternativeLabel' in raw_data:
                            label.append(raw_data['alternativeLabel'])
                        
                        # 如果 label 列表为空则赋予空值
                        if not label:
                            label = None

                # 打包数据
                result = {
                    'id': current_id,
                    'query': query,
                    'table_title': table_title,
                    'table_context': table_context,
                    'table_summary': table_summary,
                    'formatted_table': formatted_table,  # 新增字段
                    'terms_explanation': terms_explanation,
                    'label': label  # 新增字段
                }

                # 将结果添加到列表中
                results.append(result)

# 将结果写入新的 jsonl 文件
with open(output_file, 'w') as output:
    for result in results:
        output.write(json.dumps(result) + '\n')

print(f"数据处理完成，结果已保存到 {output_file}")

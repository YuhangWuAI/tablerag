import json

def process_data(input_file, output_file, error_log_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(error_log_file, 'w', encoding='utf-8') as error_log:

        for line_num, line in enumerate(infile, start=1):
            try:
                # 尝试解析JSON数据
                data = json.loads(line)
                processed_entry = {}

                # title 和 context 直接设置为空
                processed_entry['title'] = ""
                processed_entry['context'] = []

                # 表格处理
                table = data.get('table', {})
                
                # 获取表头
                header = [col['text'] for col in table.get('columns', [])]
                
                # 获取表格的行数据
                rows = []
                for row in table.get('rows', []):
                    row_data = [cell['text'] for cell in row.get('cells', [])]
                    rows.append(row_data)
                
                # 获取表格的标题作为caption
                caption = table.get('documentTitle', '')

                # 填充table结构
                processed_entry['table'] = {
                    'header': header,
                    'rows': rows,
                    'caption': caption
                }

                # 处理问题、答案和备用答案
                questions = data.get('questions', [])
                if questions:
                    question = questions[0]  # 获取第一个问题
                    processed_entry['query'] = question.get('originalText', '')
                    
                    # 处理主答案
                    main_answer = question.get('answer', {}).get('answerTexts', [])
                    processed_entry['label'] = main_answer[0] if main_answer else ""  # 取主答案的第一个
                    
                    # 处理备用答案
                    alt_answers = [alt['answerTexts'] for alt in question.get('alternativeAnswers', [])]
                    # 展平列表并去重
                    alternative_answers = list(set([item for sublist in alt_answers for item in sublist]))
                    processed_entry['alternativeLabel'] = alternative_answers if alternative_answers else []  # 设置备用答案
                else:
                    processed_entry['query'] = ''
                    processed_entry['label'] = ''
                    processed_entry['alternativeLabel'] = []

                # 将处理后的数据写入输出文件，每行一个JSON对象
                outfile.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')
            
            except json.JSONDecodeError as e:
                # 捕获JSON解析错误并记录到日志文件
                error_log.write(f"Line {line_num}: JSONDecodeError: {str(e)}\n")
            except KeyError as e:
                # 捕获键错误并记录到日志文件
                error_log.write(f"Line {line_num}: KeyError: {str(e)}\n")
            except Exception as e:
                # 捕获其他异常并记录到日志文件
                error_log.write(f"Line {line_num}: Unexpected error: {str(e)}\n")

# 使用示例
input_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/raw/all_dataset/nqtables.jsonl'  # 输入文件路径
output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/nqtables_processed.jsonl'  # 输出文件路径
error_log_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/nqtables_error_log.txt'  # 错误日志文件路径

process_data(input_file, output_file, error_log_file)

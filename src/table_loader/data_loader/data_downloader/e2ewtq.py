import json

def process_new_dataset(input_file, output_file, error_log_file):
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
                table = {}
                
                # 获取表头
                header = data.get('header', [])
                
                # 获取表格的行数据
                rows = data.get('rows', [])
                
                # 填充table结构，没有caption
                table['header'] = header
                table['rows'] = rows
                table['caption'] = ""  # 没有表格标题，设为空
                processed_entry['table'] = table

                # 处理问题和答案
                processed_entry['query'] = data.get('question', '')
                
                # 处理主答案，只有一个答案
                main_answer = data.get('answers', [])
                processed_entry['label'] = main_answer[0] if main_answer else ""

                # 没有备用答案，因此 alternativeLabel 设为空列表
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
input_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/raw/all_dataset/e2ewtq.jsonl'  # 输入文件路径
output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/raw/small_dataset/e2etwq.jsonl'  # 输出文件路径
error_log_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/raw/small_dataset/e2etwq_log.txt'  # 错误日志文件路径

process_new_dataset(input_file, output_file, error_log_file)

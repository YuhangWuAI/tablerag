import json
import os
import pandas as pd
from tqdm import tqdm  # 引入 tqdm 进度条库
from tabulate import tabulate
from src.llm.llm_generator.llm_generating import LLM_Generator

# 创建一个 LLM_Generator 实例
llm_generator = LLM_Generator()

# 处理一条数据并生成新的数据结构
def process_single_data(data, table_format):
    table = {
        "header": data["table"]["header"],
        "rows": data["table"]["rows"]
    }
    
    # 转换为 DataFrame
    df = pd.DataFrame(table['rows'], columns=table['header'])
    
    # 根据选择的格式生成表格
    if table_format == 'html':
        table_formatted = df.to_html(index=False)
    elif table_format == 'markdown':
        table_formatted = df.to_markdown(index=False)
    elif table_format == 'string':
        table_formatted = json.dumps(table)
    else:
        raise ValueError("Unsupported table format. Please choose 'html', 'markdown', or 'string'.")
    
    # 打印转换后的表格
    print(f"Table formatted ({table_format}):")
    print(table_formatted)
    
    context = data.get("context", [])
    caption = data["table"].get("caption", "")
    
    # 生成表格摘要
    print("Calling e2ewtq_generate_table_summary...")
    table_summary = llm_generator.generate_table_summary_2(caption, table_formatted, context)
    
    # 生成术语解释
    print("Calling generate_terminology_explanation...")
    terms_explanation = llm_generator.generate_terminology_explanation(caption, table_summary, table_formatted, context)

    # 生成查询建议
    print("Calling generate_query_suggestions...")
    query_suggestions = llm_generator.generate_query_suggestions(caption, table_summary, table_formatted, context)

    # 生成新的数据结构，只保留需要的字段
    new_data = {
        "table_title": caption,  # 使用表格的 caption 作为 title
        "table_context": context,
        "table_formatted": table_formatted,  
        "table_summary": table_summary,
        "terms_explanation": terms_explanation,
        "query_suggestions": query_suggestions 
    }

    return new_data

# 读取原始JSONL文件并逐行处理和保存
def process_jsonl(input_file, output_file, progress_file, table_format):
    # 读取进度文件，获取已处理的行号
    processed_lines = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as p_file:
            processed_lines = set(json.load(p_file))

    # 统计文件中的行数，用于显示进度
    total_lines = sum(1 for _ in open(input_file, 'r'))

    print(f"Opening input file: {input_file}")
    with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
        # 使用 tqdm 包装文件迭代，显示进度条
        for line_num, line in enumerate(tqdm(f_in, total=total_lines, desc="Processing Lines"), 1):
            if line_num in processed_lines:
                print(f"Line {line_num} already processed, skipping...")
                continue

            print(f"Processing line {line_num}...")
            data = json.loads(line.strip())  # 读取并解析每行JSON
            processed_data = process_single_data(data, table_format)  # 处理每条数据
            if processed_data is not None:
                f_out.write(json.dumps(processed_data) + '\n')  # 逐行写入新文件
                f_out.flush()  # 每处理完一条数据立即将其刷新到文件
                print(f"Processed data written for line {line_num}.")

                # 更新进度文件
                processed_lines.add(line_num)
                with open(progress_file, 'w') as p_file:
                    json.dump(list(processed_lines), p_file)
            else:
                print(f"Warning: Data on line {line_num} could not be processed.")

# 调用处理函数
input_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/raw/small_dataset/nqtables.jsonl'  # 替换为你的输入文件路径
output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/clarified_data/nqtables.jsonl'  # 替换为你的输出文件路径
progress_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/clarified_data/progress.json'  # 进度文件路径
table_format = 'markdown'  # 选择表格格式: 'html', 'markdown', 或 'string'

process_jsonl(input_file, output_file, progress_file, table_format)

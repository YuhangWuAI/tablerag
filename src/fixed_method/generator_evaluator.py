import json
import pandas as pd

from src.llm.llm_generator.llm_generating import LLM_Generator

# 创建一个 LLM_Generator 实例
llm_generator = LLM_Generator()

# 第一步：读取jsonl文件并提取query和增强信息
def read_jsonl_file(filepath):
    queries = []
    enhanced_infos = []
    dfs = []
    with open(filepath, 'r') as file:
        for line in file:
            data = json.loads(line)
            query = data.get("query")
            table_title = data.get("table_title", "")
            table_context = data.get("table_context", [])
            table_summary = data.get("table_summary", "")
            terms_explanation = data.get("terms_explanation", "")
            
            # 打包为增强信息
            enhanced_info = {
                "table_title": table_title,
                "table_context": table_context,
                "table_summary": table_summary,
                "terms_explanation": terms_explanation
            }
            
            # 解析表格数据为DataFrame
            formatted_table = data.get("formatted_table", {})
            if formatted_table:
                headers = formatted_table.get("header", [])
                rows = formatted_table.get("rows", [])
                df = pd.DataFrame(rows, columns=headers)
            else:
                df = pd.DataFrame()
            
            # 将每个query、增强信息和df添加到列表中
            queries.append(query)
            enhanced_infos.append(enhanced_info)
            dfs.append(df)
    
    return queries, enhanced_infos, dfs

# 执行过滤代码的函数
def execute_filter_code(code_snippet, df):
    # 清理无用的符号
    code_snippet = code_snippet.replace('```python', '').replace('```', '').strip()
    code_snippet = code_snippet.replace('>>> ', '')
    print(f"Finally Generated code snippet: {code_snippet}")

    try:
        # 定义locals字典来存储DataFrame
        locals_dict = {"df": df}
        
        # 执行过滤代码
        exec(code_snippet, {}, locals_dict)
        
        # 获取过滤后的表格
        filtered_df = locals_dict.get("filtered_table", df)
        
        # 检查过滤结果是否为空
        if filtered_df.empty:
            print("Empty table after filtering, returning original table.")
            return df
        
        print("Filtered table:\n", filtered_df)
        return filtered_df

    except Exception as e:
        print(f"Error during filtering execution: {e}")
        return df

# 将过滤后的表格转换为指定的格式
def format_filtered_table(filtered_df, output_format):
    if output_format == "string":
        return filtered_df.to_string()
    elif output_format == "html":
        return filtered_df.to_html()
    elif output_format == "markdown":
        return filtered_df.to_markdown()
    else:
        raise ValueError(f"Unsupported format: {output_format}")

# 主函数，用于读取文件并调用 LLM 函数处理所有行
def main(output_format="string"):
    filepath = "/home/yuhangwu/Desktop/Projects/tablerag/data/processed/row_col_filtered_data/e2ewtq.jsonl"
    
    # 读取JSONL文件，提取所有的query、增强信息和表格
    queries, enhanced_infos, dfs = read_jsonl_file(filepath)
    
    # 遍历所有条目，分别处理每一行
    for i, (query, enhanced_info, df) in enumerate(zip(queries, enhanced_infos, dfs)):
        print(f"\nProcessing entry {i + 1}...")

        # 打印DataFrame以确认内容
        print("Original Table (DataFrame):")
        print(df)
        
        # 获取表格的列名
        column_names = df.columns.tolist()
        
        # 调用LLM代码生成函数
        code_snippet = llm_generator.new_call_llm_code_generation(query, enhanced_info, column_names, df.to_string())
        
        # 打印生成的代码片段
        print("Generated Code Snippet:")
        print(code_snippet)
        
        # 执行生成的代码并过滤表格
        filtered_df = execute_filter_code(code_snippet, df)
        
        # 选择输出格式，转换表格
        formatted_filtered_table = format_filtered_table(filtered_df, output_format)
        
        # 打印最终结果
        print(f"Final Filtered DataFrame (in {output_format} format) for entry", i + 1)
        print(formatted_filtered_table)

        # 调用 e2ewtq_generate_final_answer 函数（由你实现）
        final_answer = llm_generator.e2ewtq_generate_final_answer(query, enhanced_info, formatted_filtered_table)
        
        # 打印最终生成的答案
        print(f"Final Answer for entry {i + 1}:\n", final_answer)

if __name__ == "__main__":
    # 调用时可以选择 'string', 'html', 'markdown'
    main(output_format="markdown")

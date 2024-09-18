import json
import pandas as pd
import re
import collections
import string

from src.llm.llm_generator.llm_generating import LLM_Generator

# 创建一个 LLM_Generator 实例
llm_generator = LLM_Generator()

# 数据清洗的辅助函数
def normalize_answer(s):
    """清洗答案字符串，移除标点、文章、额外空格等"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punctuation(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def str_normalize(user_input):
    """
    对字符串进行标准化，处理特殊符号和日期格式
    """
    user_input = str(user_input).replace("\\n", "; ")
    user_input = re.sub(r"(.*)-(.*)-(.*) 00:00:00", r"\1-\2-\3", user_input)  # 处理日期
    return user_input

def compute_exact(a_gold, a_pred):
    """计算两个字符串是否完全匹配"""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    """计算 F1 分数"""
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return int(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

# 对最终答案进行评估，并加入清洗步骤
def evaluate_answer(predicted_answer, label):
    """
    清洗预测的答案和标签，然后进行比对
    """
    predicted_answer = str_normalize(predicted_answer)
    label = [str_normalize(l) for l in label]

    exact_match = max(compute_exact(l, predicted_answer) for l in label)
    return exact_match > 0

# 执行过滤代码的函数
def execute_filter_code(code_snippet, df):
    code_snippet = code_snippet.replace('```python', '').replace('```', '').strip()
    code_snippet = code_snippet.replace('>>> ', '')
    print(f"Finally Generated code snippet: {code_snippet}")

    try:
        locals_dict = {"df": df}
        exec(code_snippet, {}, locals_dict)
        filtered_df = locals_dict.get("filtered_table", df)

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

# 保存错误的预测到文件
def save_wrong_predictions(wrong_predictions, output_filepath):
    with open(output_filepath, 'w') as outfile:
        for entry in wrong_predictions:
            json.dump(entry, outfile)
            outfile.write('\n')  # 写入每个 JSON 对象后换行

# 读取jsonl文件并提取query、增强信息和label
def read_jsonl_file(filepath):
    queries = []
    enhanced_infos = []
    labels = []
    dfs = []
    with open(filepath, 'r') as file:
        for line in file:
            data = json.loads(line)
            query = data.get("query")
            table_title = data.get("table_title", "")
            table_context = data.get("table_context", [])
            table_summary = data.get("table_summary", "")
            terms_explanation = data.get("terms_explanation", "")
            label = data.get("label", [])
            
            enhanced_info = {
                "table_title": table_title,
                "table_context": table_context,
                "table_summary": table_summary,
                "terms_explanation": terms_explanation
            }
            
            formatted_table = data.get("formatted_table", {})
            if formatted_table:
                headers = formatted_table.get("header", [])
                rows = formatted_table.get("rows", [])
                df = pd.DataFrame(rows, columns=headers)
            else:
                df = pd.DataFrame()
            
            queries.append(query)
            enhanced_infos.append(enhanced_info)
            labels.append(label)
            dfs.append(df)
    
    return queries, enhanced_infos, labels, dfs

# 主函数，用于读取文件并调用 LLM 函数处理所有行
def main(output_format="markdown"):
    filepath = "/home/yuhangwu/Desktop/Projects/tablerag/data/processed/row_col_filtered_data/e2ewtq.jsonl"
    output_filepath = "/home/yuhangwu/Desktop/Projects/tablerag/data/processed/prediction/e2ewtq.jsonl"

    queries, enhanced_infos, labels, dfs = read_jsonl_file(filepath)
    
    total_entries = len(queries)
    correct_count = 0
    wrong_predictions = []

    for i, (query, enhanced_info, label, df) in enumerate(zip(queries, enhanced_infos, labels, dfs)):
        print(f"\nProcessing entry {i + 1}...")

        print("Original Table (DataFrame):")
        print(df)
        
        column_names = df.columns.tolist()
        code_snippet = llm_generator.new_call_llm_code_generation(query, enhanced_info, column_names, df.to_string())
        
        print("Generated Code Snippet:")
        print(code_snippet)
        
        filtered_df = execute_filter_code(code_snippet, df)
        formatted_filtered_table = format_filtered_table(filtered_df, output_format)
        
        print(f"Final Filtered DataFrame (in {output_format} format) for entry", i + 1)
        print(formatted_filtered_table)

        final_answer = llm_generator.e2ewtq_generate_final_answer(query, enhanced_info, formatted_filtered_table)
        print(f"Final Answer for entry {i + 1}:\n", final_answer)

        is_correct = evaluate_answer(final_answer, label)
        if is_correct:
            print(f"Entry {i + 1}: Correct answer!")
            correct_count += 1
        else:
            print(f"Entry {i + 1}: Incorrect answer.")
            wrong_predictions.append({
                "entry_id": i + 1,
                "query": query,
                "predicted_answer": final_answer,
                "correct_answers": label
            })

    accuracy = correct_count / total_entries if total_entries > 0 else 0
    print(f"\nTotal Entries Processed: {total_entries}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    if wrong_predictions:
        save_wrong_predictions(wrong_predictions, output_filepath)
        print(f"Saved wrong predictions to {output_filepath}")

if __name__ == "__main__":
    main(output_format="markdown")

import traceback
import json
from src.llm.llm_generator.llm_generating import LLM_Generator

# 创建一个 LLM_Generator 实例
llm_generator = LLM_Generator()

# ------------------------
# Step 1: 数据加载和预处理
# ------------------------

def load_data(json_file):
    """加载 JSON 数据文件，返回所有数据"""
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    return data

def group_data_by_query(data):
    """将数据按 query 分组，每 5 行属于同一个 query"""
    grouped_data = []
    for i in range(0, len(data), 5):
        grouped_data.append(data[i:i+5])  # 每5行分组一次
    return grouped_data

def extract_table_data(group):
    """从每个组中提取有用的表格信息、ColBERT 分数、表格 ID、正确的表格 id"""
    query = group[0]['query']  # 同一个 query 对应5个不同的候选结果
    correct_id = group[0]['id']  # 正确表格的唯一 ID

    results = []
    for entry in group:
        result = entry['result']
        results.append({
            'table': result['content'],
            'passage_id': result['passage_id']  # 用于判断是否匹配正确的表格
        })
    
    return query, results, correct_id

# ------------------------
# Step 2: 逐步淘汰表格
# ------------------------

def iterative_table_selection(select_best_table, query, tables):
    """两两比较淘汰表格，直到选出最优表格"""
    
    # 初始将第一张表设为当前最优表
    current_best = tables[0]
    
    # 从第二张表开始逐一进行两两比较
    for i in range(1, len(tables)):
        next_table = tables[i]
        
        # 调用 select_best_table 比较当前最优表格与下一张表格
        # 在传递给 LLM 时，将表格内容和 passage_id 组合传递给 LLM 进行比较
        better_table_passage_id = select_best_table(
            query, 
            f"table1['passage_id']: {current_best['passage_id']}\nTable1 Content: {current_best['table']}",
            f"table2['passage_id']: {next_table['passage_id']}\nTable2 Content: {next_table['table']}"
        )

        print('better_table_passage_id (from LLM):', better_table_passage_id, type(better_table_passage_id))
        
        # 打印读取的 next_table 的 passage_id 及其类型
        print('next_table["passage_id"] (from data):', next_table['passage_id'], type(next_table['passage_id']))

        # 根据返回的 better_table_passage_id，更新 current_best 为更优的表格
        if str(better_table_passage_id) == str(next_table['passage_id']):
            current_best = next_table
    
    # 返回最终选出的最优表格
    return current_best

# ------------------------
# Step 3: 保存最优表格到 JSONL 文件
# ------------------------

def save_best_table_to_jsonl(best_table, original_group, output_file):
    """保存最佳表格的原始状态到新的 JSONL 文件"""
    # 在原始组中找到最佳表格的原始数据
    for entry in original_group:
        if entry['result']['passage_id'] == best_table['passage_id']:
            with open(output_file, 'a') as f:  # 追加写入
                f.write(json.dumps(entry) + '\n')  # 将原始数据写入文件
            break

# ------------------------
# 主程序逻辑
# ------------------------

if __name__ == "__main__":
    input_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/retrieval_results/e2ewtq.jsonl'
    output_file = '/home/yuhangwu/Desktop/Projects/tablerag/data/processed/llm_filtered_data/e2ewtq.jsonl'  # 保存最优表格的输出文件
    
    # 加载数据
    data = load_data(input_file)
    
    # 将数据按 query 分组，每组包含 5 个候选表格
    grouped_data = group_data_by_query(data)
    
    total_queries = len(grouped_data)  # 总查询数量
    queries_with_correct_in_top5 = 0  # ColBERT hits@5 包含正确答案的 query 数
    correct_final_selection_in_top5 = 0  # 在 hits@5 包含正确答案的情况下，我们筛选正确的次数
    correct_final_selection = 0  # 总的最终正确选择次数

    for group in grouped_data:  # 确保遍历所有查询组
        query, tables, correct_id = extract_table_data(group)
        
        # 判断 ColBERT 的 top 5 是否包含正确答案
        in_top5 = any(table['passage_id'] == correct_id for table in tables)
        if in_top5:
            queries_with_correct_in_top5 += 1
            
            # 逐步淘汰得到最优表格
            best_table = iterative_table_selection(llm_generator.select_best_table, query, tables)
            
            # 判断最终选择的表格是否正确
            if best_table['passage_id'] == correct_id:
                correct_final_selection_in_top5 += 1
            
            # 保存该组中最优表格的原始状态
            save_best_table_to_jsonl(best_table, group, output_file)
    
    # 确保代码遍历所有组，计算每组的结果
    if total_queries > 0:
        # 计算命中前5包含正确表格时的准确率
        if queries_with_correct_in_top5 > 0:
            accuracy_given_in_top5 = correct_final_selection_in_top5 / queries_with_correct_in_top5
        else:
            accuracy_given_in_top5 = 0
        
        # 计算总体准确率
        overall_accuracy = correct_final_selection_in_top5 / total_queries
        
        # 输出结果
        print(f"在 ColBERT 返回的 top 5 中包含正确表格的情况下，最终筛选出正确表格的准确率：{accuracy_given_in_top5:.2%}")
        print(f"总体最终准确率：{overall_accuracy:.2%}")
    else:
        print("没有数据可供处理。")

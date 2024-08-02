import json
import sys
from tqdm import tqdm
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from pipeline.ColBERT.ColBERT_test import ColBERT
from pipeline.evaluation.evaluator import Evaluator



def colbert_pipeline(jsonl_path: str, model_name: str, index_name: str, top_k: int = 1, force_fast: bool = False, rerank: bool = False, rerank_top_k: int = 1, output_file: str = "results.json"):
    colbert = ColBERT(jsonl_path, model_name, index_name)
    colbert.embed_and_index()
    responses = {}
    
    with open(jsonl_path, 'r') as file:
        data = [json.loads(line) for line in file]
        queries = [item['request'] for item in data]

    for query in queries:
        # 检索结果
        retrieved_docs = colbert.retrieve(query, top_k, force_fast, rerank, rerank_top_k)
        # 打印当前查询的结果
        print(f"Query: {query}")
        print("Results:")
        for doc in retrieved_docs:
            print(doc)
        print("\n")

        # 保存当前查询结果到字典中
        responses[query] = retrieved_docs

        # 每次查询后将所有查询结果保存到文件中
        with open(output_file, 'w') as outfile:
            json.dump(responses, outfile, indent=4)

    print(f"All results saved to {output_file}.")

def llm_generation_and_evaluation(file_path: str, colbert_model_name: str = "colbert-ir/colbertv2.0", index_name: str = "my_index"):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # 初始化 ColBERT 并处理查询
    output_file = file_path.replace(".jsonl", "_results.json")
    colbert_pipeline(file_path, colbert_model_name, index_name, top_k=1, force_fast=False, rerank=False, rerank_top_k=1, output_file=output_file)

    # 读取保存的结果
    with open(output_file, "r") as f:
        responses = json.load(f)
    
    # 处理生成的响应内容
    pred = []
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
        grd = [item['label'] for item in data]

    for query in responses:
        retrieved_content = responses[query]
        pred.append(retrieved_content)
    
    # 保存生成的结果
    response_file_path = file_path.replace(".jsonl", "_responses.txt")
    print(f"Saving responses to {response_file_path}...\n")
    with open(response_file_path, "w") as f:
        for item in pred:
            f.write("%s\n" % item)
    
    # 进行评估
    print("Evaluating the results...\n")
    numbers = Evaluator().run(pred, grd, task_name=None)
    print("Evaluation results:", numbers)

    # 保存评估结果
    evaluation_file_path = file_path.replace(".jsonl", "_evaluation.json")
    print(f"Saving evaluation results to {evaluation_file_path}...\n")
    with open(evaluation_file_path, "w") as f:
        json.dump(numbers, f, indent=4)

def main():
    # 指定要处理的文件路径
    file_path = "/home/yuhangwu/Desktop/Projects/TableProcess/pipeline/data/Exp-240801/table_augmentation/tabfact_default_term_explanations_1.jsonl"
    
    # 调用生成和评估函数
    llm_generation_and_evaluation(file_path)

if __name__ == "__main__":
    main()

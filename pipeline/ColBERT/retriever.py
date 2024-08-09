import json
import sys
from tqdm import tqdm
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from pipeline.ColBERT.ColBERT import ColBERT
from pipeline.compoments.request_serializer import deserialize_retrieved_text

import warnings
warnings.filterwarnings("ignore")
import torch

torch.cuda.empty_cache()

def generate_retrieval_results(
    dataset_path: str,  # 用于嵌入和检索的文档路径
    index_name: str,
    colbert_model_name: str = "colbert-ir/colbertv2.0",
    base_output_dir: str = "/home/yuhangwu/Desktop/Projects/TableProcess/pipeline/data/retrieval_results",
    use_rerank: bool = False,
    top_k: int = 1,
    rerank_top_k: int = 1,
    num_queries: int = 1  # 控制提取多少个 query 和 grd
):
    # 定义 query 和 grd 的文件路径
    query_grd_path = "/home/yuhangwu/Desktop/Projects/TableProcess/source/dataset/sqa.jsonl"
    
    # Ensure output directories exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Define the output file paths
    retrieval_results_save_path = os.path.join(base_output_dir, os.path.basename(dataset_path).replace(".jsonl", "_retrieval_results.jsonl"))
    progress_save_path = os.path.join(base_output_dir, os.path.basename(dataset_path).replace(".jsonl", "_progress.json"))
    
    # Step 1: Embed and index the JSONL file using ColBERT
    print("Embedding and indexing JSONL file using ColBERT\n")
    colbert = ColBERT(dataset_path, colbert_model_name, index_name)
    colbert.embed_and_index()

    # Step 2: Load queries and ground truths (grd) from the provided query_grd_path
    with open(query_grd_path, 'r') as f:
        queries_grds = [json.loads(line) for line in f]

    # 如果 num_queries 被设置，则提取指定数量的 query 和 grd，否则提取所有数据
    if num_queries is not None:
        queries_grds = queries_grds[:num_queries]

    # Step 3: Load progress if exists
    start_index = 0
    if os.path.exists(progress_save_path):
        with open(progress_save_path, 'r') as f:
            start_index = int(f.read().strip())
        print(f"Resuming from index {start_index}\n")

    # Step 4: Perform retrieval and save the results
    print("Retrieving documents using ColBERT and saving results\n")
    total_samples = len(queries_grds)
    successful_retrievals = 0
    
    with open(retrieval_results_save_path, "a") as f, open(progress_save_path, "w") as progress_file:
        for i, item in tqdm(enumerate(queries_grds[start_index:], start=start_index), desc="Processing samples", total=total_samples, ncols=150):
            try:
                query = item["query"]
                print("Query:", query, "\n")

                retrieved_docs = colbert.retrieve(query, top_k=top_k, force_fast=False, rerank=use_rerank, rerank_top_k=rerank_top_k)
                print("\nretrieved_docs:\n", retrieved_docs)

                parsed_content = deserialize_retrieved_text(retrieved_docs)

                # 保存检索结果和对应的查询
                retrieval_result = {
                    "query": query,
                    "retrieved_docs": parsed_content
                }

                # 逐条保存检索结果
                f.write(json.dumps(retrieval_result) + "\n")

                # Check if the retrieval was successful
                if parsed_content and parsed_content[0]["query_need_to_answer"] == query:
                    successful_retrievals += 1

                # Update progress
                progress_file.seek(0)
                progress_file.write(str(i + 1))
                progress_file.truncate()

            except Exception as e:
                print(f"Error processing sample {i}: {e}. Terminating.\n")
                break

    # Step 5: Calculate and print the recall accuracy
    recall_accuracy = successful_retrievals / total_samples if total_samples > 0 else 0
    print(f"Recall Accuracy: {recall_accuracy * 100:.2f}% ({successful_retrievals}/{total_samples})")

if __name__ == "__main__":
    dataset_path = "/home/yuhangwu/Desktop/Projects/TableProcess/pipeline/data/Exp-240809/table_augmentation/sqa_default_assemble_retrieval_based_augmentation_1.jsonl"
    index_name = "my_index"
    generate_retrieval_results(dataset_path, index_name, num_queries=5)  # 这里可以控制提取多少个 query 和 grd

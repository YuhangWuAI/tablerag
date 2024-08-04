import json
import sys
from tqdm import tqdm
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from pipeline.ColBERT.ColBERT import ColBERT
from pipeline.compoments.request_serializer import deserialize_retrieved_text
from pipeline.evaluation.evaluator import Evaluator
from table_provider import CallLLM

import warnings
warnings.filterwarnings("ignore")

def process_and_evaluate_colbert(
    dataset_path: str,  # 用于嵌入和检索的文档路径
    index_name: str,
    task_name: str = "tabfact",
    colbert_model_name: str = "colbert-ir/colbertv2.0",
    base_output_dir: str = "/home/yuhangwu/Desktop/Projects/TableProcess/pipeline/data/colbert_pipeline",
    call_llm: bool = False,
    run_evaluation: bool = False,
    top_k: int = 1,
    num_queries: int = None  # 控制提取多少个 query 和 grd

):
    # 定义 query 和 grd 的文件路径
    query_grd_path = "/home/yuhangwu/Desktop/Projects/TableProcess/source/dataset/tabfact.jsonl"
    
    # Ensure output directories exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Define the output file paths
    retrieval_results_save_path = os.path.join(base_output_dir, os.path.basename(dataset_path).replace(".jsonl", "_retrieval_results.jsonl"))
    grd_pred_save_path = os.path.join(base_output_dir, os.path.basename(dataset_path).replace(".jsonl", "_grd_pred.jsonl"))
    
    grd, pred = [], []

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

    # Step 3: Perform retrieval and save the results
    print("Retrieving documents using ColBERT and saving results\n")
    for i, item in tqdm(enumerate(queries_grds), desc="Processing samples", ncols=150):
        try:
            query = item["query"]
            grd_value = item["label"]
            grd.append(grd_value)
            print("Query:", query, "\n")

            retrieved_docs = colbert.retrieve(query, top_k=top_k, force_fast=False, rerank=False, rerank_top_k=1)
            print("\nretrieved_docs:\n", retrieved_docs)

            parsed_content = deserialize_retrieved_text(retrieved_docs)

            # 保存检索结果和对应的查询
            retrieval_result = {
                "query": query,
                "retrieved_docs": parsed_content
            }

            with open(retrieval_results_save_path, "a") as f:
                f.write(json.dumps(retrieval_result) + "\n")

        except KeyError as e:
            print(f"KeyError for sample {i}: {e}. Skipping this sample.\n")
            continue
        except Exception as e:
            print(f"Error processing sample {i}: {e}. Skipping this sample.\n")
            continue

    # Step 4: If call_llm is True, load the retrieval results and generate responses
    if call_llm:
        print("Generating responses using LLM\n")
        with open(retrieval_results_save_path, 'r') as f:
            retrieval_results = [json.loads(line) for line in f]

        for i, result in tqdm(enumerate(retrieval_results), desc="Generating LLM responses", ncols=150):
            try:
                query = result["query"]
                parsed_content = result["retrieved_docs"]

                for item in parsed_content:
                    query = item['query_need_to_answer']
                    table_html = item['table_html']
                    terms_explanation = item['terms_explanation']
                    table_summary = item['table_summary']

                    print("Calling LLM for the final answer\n")
                    # Generate the final answer
                    final_answer = CallLLM().generate_final_answer(query, table_html, terms_explanation, table_summary)

                    print("\nFinal answer is:", final_answer)
                    pred.append(final_answer)

                    # 每处理一个样本保存一次 `grd` 和 `pred`
                    with open(grd_pred_save_path, "w") as f:
                        f.write(json.dumps({"grd": grd, "pred": pred}) + "\n")

            except Exception as e:
                print(f"Error generating response for sample {i}: {e}. Skipping this sample.\n")
                continue

    # Step 5: Evaluation
    if run_evaluation:
        print("Running evaluation...\n")
        numbers = Evaluator().run(pred, grd, task_name)
        print("Evaluation results:", numbers, "\n")
        return numbers

if __name__ == "__main__":
    dataset_path = "pipeline/data/Exp-240802/table_augmentation/tabfact_default_assemble_retrieval_based_augmentation_1.jsonl"
    index_name = "my_index"
    process_and_evaluate_colbert(dataset_path, index_name, num_queries=None)  # 这里可以控制提取多少个 query 和 grd

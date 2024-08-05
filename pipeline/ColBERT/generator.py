import json
import sys
from tqdm import tqdm
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from pipeline.evaluation.evaluator import Evaluator
from table_provider import CallLLM

import warnings
warnings.filterwarnings("ignore")

def generate_and_evaluate(
    retrieval_results_save_path: str,
    dataset_path: str = "/home/yuhangwu/Desktop/Projects/TableProcess/source/dataset/tabfact.jsonl",
    task_name: str = "tabfact",
    base_output_dir: str = "/home/yuhangwu/Desktop/Projects/TableProcess/pipeline/data/prediction",
    run_evaluation: bool = True,
    remove_terms_explanation: bool = False,
    remove_table_summary: bool = False
):
    # Ensure output directories exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Define the output file paths
    grd_pred_save_path = os.path.join(base_output_dir, os.path.basename(retrieval_results_save_path).replace("_retrieval_results.jsonl", "_grd_pred.jsonl"))
    progress_save_path = os.path.join(base_output_dir, os.path.basename(retrieval_results_save_path).replace("_retrieval_results.jsonl", "_generation_progress.json"))

    # Load the original dataset to retrieve the ground truth (grd)
    with open(dataset_path, 'r') as f:
        original_data = [json.loads(line) for line in f]

    grd_dict = {item["query"]: item["label"] for item in original_data}

    grd, pred = [], []

    # Step 1: Load progress if exists
    start_index = 0
    if os.path.exists(progress_save_path):
        with open(progress_save_path, 'r') as f:
            start_index = int(f.read().strip())
        print(f"Resuming from index {start_index}\n")

    # Step 2: If call_llm is True, load the retrieval results and generate responses
    print("Generating responses using LLM\n")
    with open(retrieval_results_save_path, 'r') as f:
        retrieval_results = [json.loads(line) for line in f]

    for i, result in tqdm(enumerate(retrieval_results[start_index:], start=start_index), desc="Generating LLM responses", ncols=150):
        try:
            query = result["query"]
            grd_value = grd_dict.get(query, None)
            parsed_content = result["retrieved_docs"]

            for item in parsed_content:
                query_to_answer = item['query_need_to_answer']
                table_html = item['table_html']
                
                # 根据实验需求移除部分信息
                terms_explanation = "" if remove_terms_explanation else item['terms_explanation']
                table_summary = "" if remove_table_summary else item['table_summary']

                print("Calling LLM for the final answer\n")
                # Generate the final answer
                final_answer = CallLLM().generate_final_answer(query_to_answer, table_html, terms_explanation, table_summary)

                print("\nFinal answer is:", final_answer)
                pred.append(final_answer)
                grd.append(grd_value)

                # 每处理一个样本保存一次 `grd` 和 `pred`
                with open(grd_pred_save_path, "w") as f:
                    f.write(json.dumps({"grd": grd, "pred": pred}) + "\n")

                # Save progress
                with open(progress_save_path, "w") as progress_file:
                    progress_file.write(str(i + 1))

        except Exception as e:
            print(f"Error generating response for sample {i}: {e}. Skipping this sample.\n")
            break

    # Step 3: Evaluation
    if run_evaluation:
        print("Running evaluation...\n")
        numbers = Evaluator().run(pred, grd, task_name)
        print("Evaluation results:", numbers, "\n")
        return numbers

if __name__ == "__main__":
    retrieval_results_save_path = "/home/yuhangwu/Desktop/Projects/TableProcess/pipeline/data/retrieval_results/tabfact_default_docs_references_1_retrieval_results.jsonl"
    
    # 调用时根据实验需求设置参数
    generate_and_evaluate(retrieval_results_save_path, remove_terms_explanation=True, remove_table_summary=True)

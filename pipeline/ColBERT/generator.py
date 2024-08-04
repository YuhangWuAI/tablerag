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
    task_name: str = "tabfact",
    base_output_dir: str = "/home/yuhangwu/Desktop/Projects/TableProcess/pipeline/data/prediction",
    run_evaluation: bool = True
):
    # Define the output file paths
    grd_pred_save_path = retrieval_results_save_path.replace("_retrieval_results.jsonl", "_grd_pred.jsonl")
    
    grd, pred = [], []

    # Step 1: If call_llm is True, load the retrieval results and generate responses
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
                grd_value = result.get("label")
                grd.append(grd_value)
                with open(grd_pred_save_path, "w") as f:
                    f.write(json.dumps({"grd": grd, "pred": pred}) + "\n")

        except Exception as e:
            print(f"Error generating response for sample {i}: {e}. Skipping this sample.\n")
            continue

    # Step 2: Evaluation
    if run_evaluation:
        print("Running evaluation...\n")
        numbers = Evaluator().run(pred, grd, task_name)
        print("Evaluation results:", numbers, "\n")
        return numbers

if __name__ == "__main__":
    retrieval_results_save_path = ""
    generate_and_evaluate(retrieval_results_save_path)

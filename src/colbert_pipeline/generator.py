"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/
Copyright (C) 2024 Wu Yuhang. All rights reserved.
For any questions or further information, please feel free to reach out via the email address above.
"""

import json
import sys
from tqdm import tqdm
import os

# Set project root and append to sys.path for module import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.llm.llm_generator.llm_generating import LLM_Generator
from src.evaluator.evaluation import Evaluator

import warnings
warnings.filterwarnings("ignore")

def generate_and_evaluate(
    dataset_name: str,  
    retrieval_results_save_path: str,
    base_output_dir: str = "/home/yuhangwu/Desktop/Projects/TableProcess/data/processed/prediction",
    run_evaluation: bool = True,
    remove_terms_explanation: bool = False,
    remove_table_summary: bool = False
):
    """
    Generate predictions using LLM and evaluate the results.

    :param dataset_name: Name of the dataset being processed.
    :param retrieval_results_save_path: Path to the retrieval results JSONL file.
    :param base_output_dir: Base directory where output files will be saved.
    :param run_evaluation: Whether to run evaluation after generation.
    :param remove_terms_explanation: Whether to remove terms explanation from input.
    :param remove_table_summary: Whether to remove table summary from input.
    """
    # Set dataset path based on the dataset name
    dataset_path = f"/home/yuhangwu/Desktop/Projects/TableProcess/data/raw/small_dataset/{dataset_name}.jsonl"
    
    # Generate function name dynamically based on the dataset name
    generate_function_name = f"{dataset_name}_generate_final_answer"
    
    # Get the specific generation function from LLM_Generator class
    llm_generate_function = getattr(LLM_Generator(), generate_function_name, None)

    if llm_generate_function is None:
        raise ValueError(f"Function {generate_function_name} is not defined in LLM_Generator.")

    # Ensure the output directory exists
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Define output file paths
    grd_pred_save_path = os.path.join(base_output_dir, os.path.basename(retrieval_results_save_path).replace("_retrieval_results.jsonl", "_grd_pred.jsonl"))
    progress_save_path = os.path.join(base_output_dir, os.path.basename(retrieval_results_save_path).replace("_retrieval_results.jsonl", "_generation_progress.json"))

    # Load the original dataset to get ground truth (grd)
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

    # Step 2: Generate responses using LLM
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
                table_formatted = item['table_formatted']
                
                # Extract additional context field if present
                table_context = item.get('table_context', '')

                # Modify based on experiment requirements
                terms_explanation = "" if remove_terms_explanation else item['terms_explanation']
                table_summary = "" if remove_table_summary else item['table_summary']

                print("Calling LLM for the final answer\n")
                print("query: \n", query_to_answer)
                
                # Call the LLM generation function with necessary parameters
                final_answer = llm_generate_function(query_to_answer, table_formatted, terms_explanation, table_summary, table_context)

                print("\nFinal answer is:", final_answer)
                pred.append(final_answer)
                grd.append(grd_value)

                # Save `grd` and `pred` after each sample is processed
                with open(grd_pred_save_path, "w") as f:
                    f.write(json.dumps({"grd": grd, "pred": pred}) + "\n")

                # Save progress after each iteration
                with open(progress_save_path, "w") as progress_file:
                    progress_file.write(str(i + 1))

        except Exception as e:
            print(f"Error generating response for sample {i}: {e}. Skipping this sample.\n")
            break

    # Step 3: Run evaluation if specified
    if run_evaluation:
        print("Running evaluation...\n")
        numbers = Evaluator().run(pred, grd, dataset_name)
        print("Evaluation results:", numbers, "\n")
        return numbers

if __name__ == "__main__":
    retrieval_results_save_path = "/home/yuhangwu/Desktop/Projects/TableProcess/data/processed/retrieval_results/tabfact_default_None_markdown_retrieval_results.jsonl"
    dataset_name = "tabfact"
    
    # Call the function with specific parameters
    generate_and_evaluate(dataset_name, retrieval_results_save_path, remove_terms_explanation=True, remove_table_summary=True)

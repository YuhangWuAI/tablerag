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
import argparse

# Automatically determine the project root based on the current file's location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.colbert_pipeline.colbert_main import ColBERT
from src.data_processing.request_serializer import deserialize_retrieved_text

import warnings
warnings.filterwarnings("ignore")
import torch

torch.cuda.empty_cache()

# Default configuration settings
config = {
    "dataset_path": "/home/yuhangwu/Desktop/Projects/TableProcess/data/processed/table_outputs/tabfact_default_None_markdown.jsonl",
    "index_name": "my_index",
    "colbert_model_name": "colbert-ir/colbertv2.0",
    "base_output_dir": "/home/yuhangwu/Desktop/Projects/TableProcess/data/processed/retrieval_results",
    "use_rerank": False,
    "top_k": 1,
    "rerank_top_k": 1,
    "num_queries": 1000,
    "query_grd_path": "/home/yuhangwu/Desktop/Projects/TableProcess/data/raw/small_dataset/tabfact.jsonl"
}

def generate_retrieval_results(
    dataset_path: str, 
    index_name: str,
    colbert_model_name: str = "colbert-ir/colbertv2.0",
    base_output_dir: str = "/home/yuhangwu/Desktop/Projects/TableProcess/data/processed/retrieval_results",
    use_rerank: bool = False,
    top_k: int = 1,
    rerank_top_k: int = 1,
    num_queries: int = 1,  
    query_grd_path: str = "/home/yuhangwu/Desktop/Projects/TableProcess/data/raw/small_dataset/tabfact.jsonl"
):
    """
    Generate retrieval results using the ColBERT model.

    :param dataset_path: Path to the dataset used for embedding and retrieval.
    :param index_name: Name of the index to be created and used.
    :param colbert_model_name: Name of the ColBERT model to be used.
    :param base_output_dir: Directory to save the retrieval results.
    :param use_rerank: Whether to rerank the retrieved documents.
    :param top_k: Number of top documents to retrieve.
    :param rerank_top_k: Number of top documents to rerank.
    :param num_queries: Number of queries to process.
    :param query_grd_path: Path to the query and ground truth file.
    """
    # Ensure output directories exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Define output file paths
    retrieval_results_save_path = os.path.join(base_output_dir, os.path.basename(dataset_path).replace(".jsonl", "_retrieval_results.jsonl"))
    progress_save_path = os.path.join(base_output_dir, os.path.basename(dataset_path).replace(".jsonl", "_progress.json"))
    
    # Step 1: Embed and index the JSONL file using ColBERT
    print("Embedding and indexing JSONL file using ColBERT\n")
    colbert = ColBERT(dataset_path, colbert_model_name, index_name)
    colbert.embed_and_index()

    # Step 2: Load queries and ground truths (grd) from the provided path
    with open(query_grd_path, 'r') as f:
        queries_grds = [json.loads(line) for line in f]

    # If num_queries is set, limit the number of queries processed
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

                # Retrieve documents for the query
                retrieved_docs = colbert.retrieve(query, top_k=top_k, force_fast=False, rerank=use_rerank, rerank_top_k=rerank_top_k)
                print("\nretrieved_docs:\n", retrieved_docs)

                parsed_content = deserialize_retrieved_text(retrieved_docs)

                # Save retrieval result with the query
                retrieval_result = {
                    "query": query,
                    "retrieved_docs": parsed_content
                }

                # Save retrieval results incrementally
                f.write(json.dumps(retrieval_result) + "\n")

                # Check if the retrieval was successful
                if parsed_content and parsed_content[0]["query_need_to_answer"] == query:
                    successful_retrievals += 1

                # Update progress after each iteration
                progress_file.seek(0)
                progress_file.write(str(i + 1))
                progress_file.truncate()

            except Exception as e:
                print(f"Error processing sample {i}: {e}. Terminating.\n")
                break

    # Step 5: Calculate and print the recall accuracy
    recall_accuracy = successful_retrievals / total_samples if total_samples > 0 else 0
    print(f"Recall Accuracy: {recall_accuracy * 100:.2f}% ({successful_retrievals}/{total_samples})")


def main():
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate retrieval results using ColBERT")

    # Add arguments to the parser
    parser.add_argument('--dataset_path', type=str, default=config["dataset_path"], help="Path to the dataset JSONL file")
    parser.add_argument('--index_name', type=str, default=config["index_name"], help="Name of the index to create")
    parser.add_argument('--colbert_model_name', type=str, default=config["colbert_model_name"], help="Name of the ColBERT model to use")
    parser.add_argument('--base_output_dir', type=str, default=config["base_output_dir"], help="Base directory for output files")
    parser.add_argument('--use_rerank', action='store_true', default=config["use_rerank"], help="Flag to indicate if reranking should be used")
    parser.add_argument('--top_k', type=int, default=config["top_k"], help="Number of top documents to retrieve")
    parser.add_argument('--rerank_top_k', type=int, default=config["rerank_top_k"], help="Number of top documents to rerank")
    parser.add_argument('--num_queries', type=int, default=config["num_queries"], help="Number of queries to process")
    parser.add_argument('--query_grd_path', type=str, default=config["query_grd_path"], help="Path to the queries and ground truth file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to generate retrieval results with parsed arguments
    generate_retrieval_results(
        dataset_path=args.dataset_path,
        index_name=args.index_name,
        colbert_model_name=args.colbert_model_name,
        base_output_dir=args.base_output_dir,
        use_rerank=args.use_rerank,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        num_queries=args.num_queries,
        query_grd_path=args.query_grd_path
    )

if __name__ == "__main__":
    main()

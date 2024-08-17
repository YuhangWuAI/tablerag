"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/
Copyright (C) 2024 Wu Yuhang. All rights reserved.
For any questions or further information, please feel free to reach out via the email address above.
"""

import json
import os
import datetime
from typing import Optional
import pandas as pd
from tqdm import tqdm
from src.data_processing.request_serializer import serialize_request
from src.data_processing.save_jsonl import load_processed_indices, save_jsonl_file

import warnings

from src.table_master.table_main import TableClarifier

warnings.filterwarnings("ignore")

def table_processing_pipeline(
    task_name: str = "sqa",
    split: str = "validation",
    table_filter_name: str = "default",
    table_clarifier_name: str = "term_explanations_and_table_summary",
    embedding_type: str = "text-embedding-3-large",
    top_k: int = 5,
    save_jsonl: bool = True,
    load_local_dataset: bool = True,
    experiment_name: str = "table_clarification",
    use_sampled_table_for_augmentation: bool = False,
    sample_size: Optional[int] = 1,
    overwrite_existing: bool = False,
    table_format: str = "default",
    use_table_filter: bool = False,
):
    """
    Main pipeline function for processing tables using specified filters and clarifiers.
    
    :param task_name: The name of the task to be processed (e.g., "sqa").
    :param split: The dataset split to be used (e.g., "validation").
    :param table_filter_name: The name of the table filtering method (e.g., "default").
    :param table_clarifier_name: The name of the table clarification method (e.g., "term_explanations_and_table_summary").
    :param embedding_type: The type of embedding used in processing (e.g., "text-embedding-3-large").
    :param top_k: The number of top rows or columns to select based on relevance.
    :param save_jsonl: Whether to save the output in JSONL format (default is True).
    :param load_local_dataset: Whether to load a local dataset (default is True).
    :param experiment_name: The name of the experiment being conducted.
    :param use_sampled_table_for_augmentation: Whether to use sampled tables for augmentation (default is False).
    :param sample_size: The number of samples to process (default is 1).
    :param overwrite_existing: Whether to overwrite existing results (default is False).
    :param table_format: The format to use for the tables (e.g., "markdown").
    :param use_table_filter: Whether to use table filtering (default is True).
    """
    print("Starting table processing pipeline\n")
    
    # Define paths for saving files and progress
    file_save_path = f"/home/yuhangwu/Desktop/Projects/TableProcess/data/processed/table_outputs/{task_name}_{table_filter_name}_{table_clarifier_name}_{table_format}.jsonl"
    progress_save_path = f"/home/yuhangwu/Desktop/Projects/TableProcess/data/progressing/{task_name}_{table_filter_name}_{table_clarifier_name}_{table_format}.json"

    print("File save path: ", file_save_path, "\n")

    # Initialize the TableClarifier, which will manage the table parsing, filtering, and clarification
    print("Initializing TableClarifier\n")
    table_master = TableClarifier(
        task_name,
        split,
        table_filter_name,
        table_clarifier_name,
        top_k,
        embedding_type,
        whether_column_grounding=True,  
    )

    # Load dataset locally if required
    if load_local_dataset:
        print("Loading local dataset\n")
        with open(f"data/raw/small_dataset/{task_name}.jsonl", "r") as f:
            print("Loading dataset for ", task_name, "...\n")
            dataset = [json.loads(line) for line in f.readlines()]
    else:
        print("Loading examples from TableClarifier\n")

    # Initialize variables for progress tracking
    grd, pred = [], []

    # For managing batches in LLM calls
    batch_size = table_master.call_llm.BATCH_SIZE
    print("Batch size: ", batch_size, "\n")
    num_samples = (
        sample_size if sample_size is not None else (len(dataset) if load_local_dataset else len(table_master.table_loader.dataset))
    )
    print("Number of samples: ", num_samples, "\n")

    # Create directories if they do not exist
    progress_directory = os.path.dirname(progress_save_path)
    if not os.path.exists(progress_directory):
        os.makedirs(progress_directory)

    if not os.path.exists(os.path.dirname(file_save_path)):
        os.makedirs(os.path.dirname(file_save_path))

    # Check if the task has already been completed
    if os.path.exists(file_save_path) and not overwrite_existing:
        processed_indices = load_processed_indices(file_save_path)
        if len(processed_indices) >= num_samples:
            print("Task already done, skipping: ", file_save_path, "\n")
            return
    else:
        processed_indices = set()

    # Load progress from previous runs if available
    if os.path.exists(progress_save_path):
        with open(progress_save_path, "r") as progress_file:
            processed_indices.update(set(json.load(progress_file)))

    num_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size
    batches = []
    print("Number of batches: ", num_batches, ", Remaining samples: ", remaining_samples, "\n")

    # Initialize progress bar
    print("Initializing progress bar\n")
    with tqdm(
        total=num_samples,
        desc=f"Processing {experiment_name}_{task_name}",
        ncols=150,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
    ) as pbar:
        print("Progress bar initialized\n")

        # Process batches of data
        for batch_num in range(num_batches):
            print("Processing batch number: ", batch_num, "\n")
            batch_request = []
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            batch = (
                dataset[start_index:end_index]
                if load_local_dataset
                else table_master.table_loader.dataset[start_index:end_index]
            )
            print("Processing samples from index ", start_index, " to ", end_index, "\n")
            for i in range(batch_size):
                index = start_index + i
                if index in processed_indices:
                    continue
                
                print("=============================================================================================================================\n")
                print("Processing sample ", i, " in batch ", batch_num, "\n")
                parsed_sample = (
                    batch[i]
                    if load_local_dataset
                    else table_master.table_loader.parse_table(
                        {key: value[i] for key, value in batch.items()}
                    )
                )

                query = parsed_sample["query"]
                grd_value = parsed_sample["label"]
                context = parsed_sample.get("context", "")  # Extract context if available
                context = " ".join(context)
                print("context: \n", context)
                grd.append(grd_value)
                print("Query: ", query, "\n")

                try: 
                    if use_table_filter:
                        try:
                            filter_table = table_master.table_filter.run(query, parsed_sample)
                            print("Filtered table generated for sample ", i, ":\n", filter_table, "\n")
                        except Exception as e:
                            print("Error in table sampling for sample ", i, ": ", e, "\n")
                            continue
                    else:
                        print("Bypassing table sampling and using the original table as string\n")
                        filter_table = parsed_sample["table"]
                        
                        # if not use filter, transfer table to df type for using markdown or html
                        if isinstance(filter_table, dict) and "header" in filter_table and "rows" in filter_table:
                            try:
                                df = pd.DataFrame(data=filter_table["rows"], columns=filter_table["header"])
                                filter_table = df
                            except Exception as e:
                                print(f"Error converting table to DataFrame: {e}. Skipping this sample.\n")
                                continue

                    clarifier_inputs = parsed_sample
                    if use_sampled_table_for_augmentation and use_table_filter:
                        print("Using sampled table for augmentation\n")
                        clarifier_inputs = {
                            "query": parsed_sample["query"],
                            "table": {
                                "header": filter_table.columns.tolist(),
                                "rows": filter_table.to_dict('records'),
                                "caption": parsed_sample["table"].get("caption", "")
                            }
                        }
                    print("Augmentation input: ", clarifier_inputs, "\n")
                    clarification_text = (
                        table_master.table_clarification.run(clarifier_inputs)
                        if table_clarifier_name != "None"
                        else {}
                    )
                    print("Augmentation info for sample ", i, ": ", clarification_text, "\n")


                    try:
                        # Convert the filtered table to the specified format
                        if table_format == "html":
                            table_formatted = filter_table.to_html()
                        elif table_format == "markdown":
                            try:
                                from tabulate import tabulate
                                table_formatted = tabulate(filter_table, headers="keys", tablefmt="pipe")
                            except ImportError:
                                print("Tabulate module not installed, falling back to string format.")
                                table_formatted = filter_table.to_string()
                        else:
                            table_formatted = filter_table.to_string()
                    except AttributeError as e:
                        print(f"Error in converting table: {e}. Converting table to string instead.\n")
                        table_formatted = filter_table.to_string()

                    request = serialize_request(
                        query=query,
                        table_formatted=table_formatted,
                        clarification_text=clarification_text,
                        context=context  # Include the context in the request serialization
                    )

                    print("Request:\n", request, "\n")

                    batch_request.append(request)

                    # Save progress
                    processed_indices.add(index)
                    with open(progress_save_path, "w") as progress_file:
                        json.dump(list(processed_indices), progress_file)
                    
                    # Save jsonl
                    if save_jsonl:
                        print("Saving results as jsonl\n")
                        save_jsonl_file(
                            batch_request[-1],
                            grd[-1],
                            file_save_path
                        )
                except Exception as e:
                    print(f"Error in processing sample {i} in batch {batch_num}: {e}. Skipping this sample.\n")
                    continue
            pbar.update(batch_size)
            print("Finished processing batch number: ", batch_num, "\n")
            batches.append(batch_request)

        # Process any remaining samples that didn't fit into a full batch
        if remaining_samples > 0:
            print("Processing remaining samples\n")
            batch_request = []
            start_index = num_batches * batch_size
            end_index = start_index + remaining_samples
            batch = (
                dataset[start_index:end_index]
                if load_local_dataset
                else table_master.table_loader.dataset[start_index:end_index]
            )
            print("Processing samples from index ", start_index, " to ", end_index, "\n")
            for i in range(remaining_samples):
                index = start_index + i
                if index in processed_indices:
                    continue
                
                print("=============================================================================================================================\n")
                print("Processing remaining sample ", i, "\n")
                parsed_sample = (
                    batch[i]
                    if load_local_dataset
                    else table_master.table_loader.parse_table(
                        {key: value[i] for key, value in batch.items()}
                    )
                )
                query = parsed_sample["query"]
                grd_value = parsed_sample["label"]
                context = parsed_sample.get("context", "")  # Extract context if available
                grd.append(grd_value)
                print("Query: ", query, "\n")

                try:
                    if use_table_filter:
                        filter_table = table_master.table_filter.run(
                            query, parsed_sample
                        )
                        print("Filtered table generated for remaining sample ", i, "\n")
                    else:
                        print("Bypassing table sampling and using the original table as string\n")
                        filter_table = parsed_sample["table"]

                        if isinstance(filter_table, dict) and "header" in filter_table and "rows" in filter_table:
                            try:
                                df = pd.DataFrame(data=filter_table["rows"], columns=filter_table["header"])
                                filter_table = df
                            except Exception as e:
                                print(f"Error converting table to DataFrame: {e}. Skipping this sample.\n")
                                continue

                except Exception as e:
                    print("Error in table sampling for remaining sample ", i, ": ", e, "\n")
                    print("Skipping batch: ", i, "\n")
                    continue
                clarification_text = (
                    table_master.table_clarification.run(parsed_sample)
                    if table_clarifier_name != "None"
                    else {}
                )
                print("Augmentation info for remaining sample ", i, ": ", clarification_text, "\n")


                try:
                    # Convert the filtered table to the specified format
                    if table_format == "html":
                        table_formatted = filter_table.to_html()
                    elif table_format == "markdown":
                        try:
                            from tabulate import tabulate
                            table_formatted = tabulate(filter_table, headers="keys", tablefmt="pipe")
                        except ImportError:
                            print("Tabulate module not installed, falling back to string format.")
                            table_formatted = filter_table.to_string()
                    else:
                        table_formatted = filter_table.to_string()
                except AttributeError as e:
                    print(f"Error in converting table: {e}. Converting table to string instead.\n")
                    table_formatted = filter_table.to_string()

                request = serialize_request(
                    query=query,
                    table_formatted=table_formatted,
                    clarification_text=clarification_text,
                    context=context  # Include the context in the request serialization
                )

                print("Request:\n", request, "\n")

                batch_request.append(request)

                # Save progress
                processed_indices.add(index)
                with open(progress_save_path, "w") as progress_file:
                    json.dump(list(processed_indices), progress_file)
                
                # Save jsonl
                if save_jsonl:
                    print("Saving results as jsonl\n")
                    save_jsonl_file(
                        batch_request[-1],
                        grd[-1],
                        file_save_path
                    )

            pbar.update(remaining_samples)
            print("Finished processing remaining samples\n")
            batches.append(batch_request)


def main():
    """
    Main function to start the table processing pipeline with predefined parameters.
    """
    table_processing_pipeline(
        task_name="hybridqa",
        split="validation",
        table_filter_name="llm_based_filter", # llm_based_filter
        table_clarifier_name="None", # None, term_explanations_and_table_summary
        embedding_type="text-embedding-3-large",
        top_k=5,
        save_jsonl=True,
        load_local_dataset=True,
        experiment_name="table_clarification",
        use_sampled_table_for_augmentation=False,
        sample_size=1000,
        overwrite_existing=False,
        table_format="string",
        use_table_filter=True
    )

if __name__ == "__main__":
    main()

import json
import os
import datetime
import time
from tqdm import tqdm
from pipeline.ColBERT.ColBERT import ColBERT
from pipeline.compoments.request_serializer import deserialize_retrieved_text, serialize_request
from pipeline.data_processing.save_jsonl import load_processed_indices, save_jsonl_file
from table_provider import CallLLM, TableProvider
from .evaluation.evaluator import Evaluator
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

def end2end(
    task_name: str,
    split: str,
    table_sampling_type: str,
    table_augmentation_type: str,
    embedding_type: str,
    k_shot: int,
    n_cluster: int,
    top_k: int,
    save_jsonl: bool = False,
    azure_blob: bool = False,
    load_local_dataset: bool = False,
    experiment_name: str = None,
    use_sampled_table_for_augmentation = False,
    whether_column_grounding: bool = False,
    sample_size: Optional[int] = None,
    overwrite_existing: bool = False,
    colbert_model_name: str = "colbert-ir/colbertv2.0",  # Add this parameter for ColBERT model
    index_name: str = "my_index",  # Add this parameter for the index name
    call_llm: bool = True,  # Add this parameter to control whether to call LLM or not
    run_evaluation: bool = True,  # Add this parameter to control whether to run evaluation
):
    print("Starting end2end process\n")
    
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')

    file_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_{table_sampling_type}_{table_augmentation_type}_{k_shot}.jsonl"
    progress_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_progress.json"
    retrieval_results_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_retrieval_results.jsonl"
    grd_pred_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_grd_pred.jsonl"  # Save grd and pred together

    print("File save path: ", file_save_path, "\n")

    # if not exist create it
    progress_directory = os.path.dirname(progress_save_path)
    if not os.path.exists(progress_directory):
        os.makedirs(progress_directory)

    # whether the task is already done
    if os.path.exists(file_save_path) and not overwrite_existing:
        processed_indices = load_processed_indices(file_save_path)
        if len(processed_indices) >= sample_size:
            print("Task already done, skipping: ", file_save_path, "\n")
            return
    else:
        processed_indices = set()

    # Load progress
    if os.path.exists(progress_save_path):
        with open(progress_save_path, "r") as progress_file:
            processed_indices.update(set(json.load(progress_file)))

    # Initializing tableprovider and get instruction
    print("Initializing TableProvider\n")
    table_provider = TableProvider(
        task_name,
        split,
        table_sampling_type,
        table_augmentation_type,
        n_cluster,
        top_k,
        embedding_type,
        whether_column_grounding,
    )

    # Loading local dataset if required
    if load_local_dataset:
        print("Loading local dataset\n")
        with open(f"source/dataset/{task_name}.jsonl", "r") as f:
            print("Loading dataset for ", task_name, "...\n")
            dataset = [json.loads(line) for line in f.readlines()]
    else:
        print("Loading examples from TableProvider\n")

    # Initialising variance and progressing
    grd, pred = [], []

    # for LLM calling
    batch_size = table_provider.call_llm.BATCH_SIZE
    print("Batch size: ", batch_size, "\n")
    num_samples = (
        sample_size if sample_size is not None else (len(dataset) if load_local_dataset else len(table_provider.table_loader.dataset))
    )
    print("Number of samples: ", num_samples, "\n")

    num_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size
    batches = []
    print("Number of batches: ", num_batches, ", Remaining samples: ", remaining_samples, "\n")

    # Progress bar initialization
    print("Initializing progress bar\n")
    with tqdm(
        total=num_samples,
        desc=f"Processing {experiment_name}_{task_name}",
        ncols=150,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
    ) as pbar:
        print("Progress bar initialized\n")

        # Processing batches (storing all the information)
        for batch_num in range(num_batches):
            print("Processing batch number: ", batch_num, "\n")
            batch_request = []
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            batch = (
                dataset[start_index:end_index]
                if load_local_dataset
                else table_provider.table_loader.dataset[start_index:end_index]
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
                    else table_provider.table_loader.parse_table(
                        {key: value[i] for key, value in batch.items()}
                    )
                )

                query = parsed_sample["query"]
                grd_value = parsed_sample["label"]
                grd.append(grd_value)
                print("Query: ", query, "\n")

                try: 
                    try:
                        filter_table = table_provider.table_sampler.run(query, parsed_sample)
                        print("Filtered table generated for sample ", i, ":\n", filter_table, "\n")
                    except Exception as e:
                        print("Error in table sampling for sample ", i, ": ", e, "\n")
                        continue

                    augmentation_input = parsed_sample
                    if use_sampled_table_for_augmentation:
                        print("Using sampled table for augmentation\n")
                        augmentation_input = {
                            "query": parsed_sample["query"],
                            "table": {
                                "header": filter_table.columns.tolist(),
                                "rows": filter_table.to_dict('records'),
                                "caption": parsed_sample["table"].get("caption", "")
                            }
                        }
                    print("Augmentation input: ", augmentation_input, "\n")
                    augmentation_info = (
                        table_provider.table_augmentation.run(augmentation_input)
                        if table_augmentation_type != "None"
                        else ""
                    )
                    print("Augmentation info for sample ", i, ": ", augmentation_info, "\n")


                    try:
                        table_html = filter_table.to_html()
                    except AttributeError as e:
                        print(f"Error in converting table to HTML: {e}. Converting table to string instead.\n")
                        table_html = filter_table.to_string()

                    request = serialize_request(
                        query=query,
                        table_html=table_html,
                        augmentation_info=augmentation_info  
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

        if remaining_samples > 0:
            print("Processing remaining samples\n")
            batch_request = []
            start_index = num_batches * batch_size
            end_index = start_index + remaining_samples
            batch = (
                dataset[start_index:end_index]
                if load_local_dataset
                else table_provider.table_loader.dataset[start_index:end_index]
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
                    else table_provider.table_loader.parse_table(
                        {key: value[i] for key, value in batch.items()}
                    )
                )
                query = parsed_sample["query"]
                grd_value = parsed_sample["label"]
                grd.append(grd_value)
                print("Query: ", query, "\n")

                try:
                    filter_table = table_provider.table_sampler.run(
                        query, parsed_sample
                    )
                    print("Filtered table generated for remaining sample ", i, "\n")
                except Exception as e:
                    print("Error in table sampling for remaining sample ", i, ": ", e, "\n")
                    print("Skipping batch: ", i, "\n")
                    continue
                augmentation_info = (
                    table_provider.table_augmentation.run(parsed_sample)
                    if table_augmentation_type != "None"
                    else ""
                )
                print("Augmentation info for remaining sample ", i, ": ", augmentation_info, "\n")
                request = serialize_request(
                    query=query,
                    table_html=filter_table.to_html(),
                    augmentation_info=augmentation_info  
                )

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

    
    # Step 1: Embed and index the JSONL file using ColBERT
    if azure_blob:
        print("Embedding and indexing JSONL file using ColBERT\n")
        colbert = ColBERT(file_save_path, colbert_model_name, index_name)
        colbert.embed_and_index()

        # Step 2: Retrieve documents using ColBERT and generate responses
        print("Retrieving documents using ColBERT and generating responses\n")

        print("batch_request: ", batch_request)
        for batch_request in tqdm(batches, desc=f"Calling ColBERT for retrieval, experiment_name: {experiment_name}", ncols=150):
            for request in batch_request:
                # 从request中提取query进行检索
                query = request.get("query")
                print("\n Extracted query for retrieval: \n", query)

                retrieved_docs = colbert.retrieve(query, top_k=1, force_fast=False, rerank=False, rerank_top_k=1)
                
                print("\n retrieved_docs: \n", retrieved_docs)
                
                parsed_content = deserialize_retrieved_text(retrieved_docs)

                # 保存检索结果和对应的查询
                retrieval_result = {
                    "query": query,
                    "retrieved_docs": parsed_content
                }

                with open(retrieval_results_save_path, "a") as f:
                    f.write(json.dumps(retrieval_result) + "\n")

                if call_llm:
                    for item in parsed_content:
                        query = item['query_need_to_answer']
                        table_html = item['table_html']
                        terms_explanation = item['terms_explanation']
                        table_summary = item['table_summary']

                        print("CallLLM for the final answer \n")
                        # Generate the final answer
                        final_answer = CallLLM().generate_final_answer(query, table_html, terms_explanation, table_summary)
                        
                        print("\n Final answer is :", final_answer)
                        pred.append(final_answer)

                        # 每处理一个样本保存一次 `grd` 和 `pred`
                        with open(grd_pred_save_path, "w") as f:
                            f.write(json.dumps({"grd": grd, "pred": pred}) + "\n")

        # Step 3: Evaluation
        if run_evaluation:
            print("Running evaluation...\n")
            numbers = Evaluator().run(pred, grd, task_name)
            print("Evaluation results of ", experiment_name, "_", task_name, ": ", numbers, "\n")

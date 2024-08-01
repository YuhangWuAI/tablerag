import json
import os
import datetime
import time
from tqdm import tqdm
from pipeline.compoments.colbert import ColBERT
from pipeline.compoments.request_serializer import serialize_request, deserialize_request
from table_provider import CallLLM, TableProvider
from .evaluation.evaluator import Evaluator
from typing import List, Optional
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
def save_jsonl_file_single(
    request: str,
    label: str,
    file_path: str,
    pred: str = None,
):
    # mkdir
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    data = {
        'request': json.loads(request),  # 直接存储解析后的request字典
        'label': label,
    }
    if pred is not None:
        data['pred'] = pred

    # save jsonl
    with open(file_path, 'a') as file:
        json_string = json.dumps(data, indent=4)
        file.write(json_string + '\n')


def load_processed_indices(file_path: str):
    processed_indices = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                try:
                    json.loads(line)
                    processed_indices.add(i)
                except json.JSONDecodeError:
                    break
    return processed_indices

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
):
    logging.debug("Starting end2end process")
    logging.info("Starting end2end process")
    
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')

    file_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_{table_sampling_type}_{table_augmentation_type}_{k_shot}.jsonl"
    progress_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_progress.json"

    logging.debug(f"File save path: {file_save_path}")
    logging.info(f"File save path: {file_save_path}")

    # if not exist create it
    progress_directory = os.path.dirname(progress_save_path)
    if not os.path.exists(progress_directory):
        os.makedirs(progress_directory)

    # whether the task is already done
    if os.path.exists(file_save_path) and not overwrite_existing:
        processed_indices = load_processed_indices(file_save_path)
        if len(processed_indices) >= sample_size:
            logging.info(f"Task already done, skipping: {file_save_path}")
            logging.info(f"Task already done, skipping: {file_save_path}")
            return
    else:
        processed_indices = set()

    # Load progress
    if os.path.exists(progress_save_path):
        with open(progress_save_path, "r") as progress_file:
            processed_indices.update(set(json.load(progress_file)))

    # Initializing tableprovider and get instruction
    logging.info("Initializing TableProvider")
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

    max_truncate_tokens = table_provider.call_llm.MAX_TRUNCATE_TOKENS
    augmentation_tokens = table_provider.call_llm.AUGMENTATION_TOKEN_LIMIT

    # Loading local dataset if required
    if load_local_dataset:
        logging.debug("Loading local dataset")
        logging.info("Loading local dataset")
        with open(f"source/dataset/{task_name}.jsonl", "r") as f:
            logging.debug(f"Loading dataset for {task_name}...")
            logging.info(f"Loading dataset for {task_name}...")
            dataset = [json.loads(line) for line in f.readlines()]
    else:
        logging.debug("Loading examples from TableProvider")
        logging.info("Loading examples from TableProvider")

    # Initialising variance and progressing
    grd, pred = [], []

    # for LLM calling
    batch_size = table_provider.call_llm.BATCH_SIZE
    logging.debug(f"Batch size: {batch_size}")
    logging.info(f"Batch size: {batch_size}")
    num_samples = (
        sample_size if sample_size is not None else (len(dataset) if load_local_dataset else len(table_provider.table_loader.dataset))
    )
    logging.debug(f"Number of samples: {num_samples}")
    logging.info(f"Number of samples: {num_samples}")

    num_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size
    batches = []
    logging.debug(f"Number of batches: {num_batches}, Remaining samples: {remaining_samples}")
    logging.info(f"Number of batches: {num_batches}, Remaining samples: {remaining_samples}")

    # Progress bar initialization
    logging.debug("Initializing progress bar")
    logging.info("Initializing progress bar")
    with tqdm(
        total=num_samples,
        desc=f"Processing {experiment_name}_{task_name}",
        ncols=150,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
    ) as pbar:
        logging.debug("Progress bar initialized")
        logging.info("Progress bar initialized")

        # Processing batches (storing all the information)
        for batch_num in range(num_batches):
            logging.debug(f"Processing batch number: {batch_num}")
            logging.info(f"Processing batch number: {batch_num}")
            batch_request = []
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            batch = (
                dataset[start_index:end_index]
                if load_local_dataset
                else table_provider.table_loader.dataset[start_index:end_index]
            )
            logging.debug(f"Processing samples from index {start_index} to {end_index}")
            logging.info(f"Processing samples from index {start_index} to {end_index}")
            for i in range(batch_size):
                index = start_index + i
                if index in processed_indices:
                    continue
                
                logging.info("====================================================================================================================")
                logging.info(f"Processing sample {i} in batch {batch_num}")
                parsed_sample = (
                    batch[i]
                    if load_local_dataset
                    else table_provider.table_loader.parse_table(
                        {key: value[i] for key, value in batch.items()}
                    )
                )

                query = parsed_sample["query"]
                grd.append(parsed_sample["label"])
                logging.info(f"Query: {query}")

                try:
                    filter_table = table_provider.table_sampler.run(query, parsed_sample)
                    logging.info(f"Filtered table generated for sample {i}:\n{filter_table}")
                except Exception as e:
                    logging.info(f"Error in table sampling for sample {i}: {e}")
                    continue

                augmentation_input = parsed_sample
                if use_sampled_table_for_augmentation:
                    logging.info("Using sampled table for augmentation")
                    augmentation_input = {
                        "query": parsed_sample["query"],
                        "table": {
                            "header": filter_table.columns.tolist(),
                            "rows": filter_table.to_dict('records'),
                            "caption": parsed_sample["table"].get("caption", "")
                        }
                    }
                logging.info("Augmentation input:", augmentation_input)
                augmentation_info = (
                    table_provider.table_augmentation.run(augmentation_input)
                    if table_augmentation_type != "None"
                    else ""
                )
                logging.debug(f"Augmentation info for sample {i}: {augmentation_info}")
                logging.info(f"Augmentation info for sample {i}: {augmentation_info}")

                request = serialize_request(
                    query=query,
                    table_html=filter_table.to_html(),
                    augmentation_info=augmentation_info  # 假设summary是augmentation_info的一部分
                )

                logging.info("Request:\n", request)

                if (
                    table_provider.call_llm.num_tokens(request)
                    < table_provider.call_llm.TOTAL_TOKENS
                ):
                    batch_request.append(request)
                else:
                    truncated_request = table_provider.call_llm.truncated_string(
                        request,
                        table_provider.call_llm.TOTAL_TOKENS,
                        print_warning=False,
                    )
                    logging.debug(f"Truncated request for sample {i}")
                    logging.info(f"Truncated request for sample {i}")
                    batch_request.append(truncated_request)

                # Save progress
                processed_indices.add(index)
                with open(progress_save_path, "w") as progress_file:
                    json.dump(list(processed_indices), progress_file)
                
                # Save jsonl
                if save_jsonl:
                    logging.debug("Saving results as jsonl")
                    logging.info("Saving results as jsonl")
                    save_jsonl_file_single(
                        batch_request[-1],
                        grd[-1],
                        file_save_path
                    )

            pbar.update(batch_size)
            logging.debug(f"Finished processing batch number: {batch_num}")
            logging.info(f"Finished processing batch number: {batch_num}")
            batches.append(batch_request)

        if remaining_samples > 0:
            logging.debug("Processing remaining samples")
            logging.info("Processing remaining samples")
            batch_request = []
            start_index = num_batches * batch_size
            end_index = start_index + remaining_samples
            batch = (
                dataset[start_index:end_index]
                if load_local_dataset
                else table_provider.table_loader.dataset[start_index:end_index]
            )
            logging.debug(f"Processing samples from index {start_index} to {end_index}")
            logging.info(f"Processing samples from index {start_index} to {end_index}")
            for i in range(remaining_samples):
                index = start_index + i
                if index in processed_indices:
                    continue

                logging.debug(f"Processing remaining sample {i}")
                logging.info(f"Processing remaining sample {i}")
                parsed_sample = (
                    batch[i]
                    if load_local_dataset
                    else table_provider.table_loader.parse_table(
                        {key: value[i] for key, value in batch.items()}
                    )
                )
                query = parsed_sample["query"]
                grd.append(parsed_sample["label"])
                logging.debug(f"Query: {query}")
                logging.info(f"Query: {query}")

                try:
                    filter_table = table_provider.table_sampler.run(
                        query, parsed_sample
                    )
                    logging.debug(f"Filtered table generated for remaining sample {i}")
                    logging.info(f"Filtered table generated for remaining sample {i}")
                except Exception as e:
                    logging.error(f"Error in table sampling for remaining sample {i}: {e}")
                    logging.info(f"Error in table sampling for remaining sample {i}: {e}")
                    logging.info("Skipping batch:", i)
                    continue
                augmentation_info = (
                    table_provider.table_augmentation.run(parsed_sample)
                    if table_augmentation_type != "None"
                    else ""
                )
                logging.debug(f"Augmentation info for remaining sample {i}: {augmentation_info}")
                logging.info(f"Augmentation info for remaining sample {i}: {augmentation_info}")
                request = serialize_request(
                    query=query,
                    table_html=filter_table.to_html(),
                    explanations=parsed_sample.get("explanations", {}),
                    summary=augmentation_info  # 假设summary是augmentation_info的一部分
                )
                if (
                    table_provider.call_llm.num_tokens(request)
                    < table_provider.call_llm.TOTAL_TOKENS
                ):
                    batch_request.append(request)
                else:
                    truncated_request = table_provider.call_llm.truncated_string(
                        request, print_warning=False
                    )
                    logging.debug(f"Truncated request for remaining sample {i}")
                    logging.info(f"Truncated request for remaining sample {i}")
                    batch_request.append(truncated_request)

                # Save progress
                processed_indices.add(index)
                with open(progress_save_path, "w") as progress_file:
                    json.dump(list(processed_indices), progress_file)
                
                # Save jsonl
                if save_jsonl:
                    logging.debug("Saving results as jsonl")
                    logging.info("Saving results as jsonl")
                    save_jsonl_file_single(
                        batch_request[-1],
                        grd[-1],
                        file_save_path
                    )

            pbar.update(remaining_samples)
            logging.debug("Finished processing remaining samples")
            logging.info("Finished processing remaining samples")
            batches.append(batch_request)

    '''
    # Step 1: Embed and index the JSONL file using ColBERT
    if azure_blob:
        logging.debug("Embedding and indexing JSONL file using ColBERT")
        print("Embedding and indexing JSONL file using ColBERT")
        colbert = ColBERT(file_save_path, colbert_model_name, index_name)
        colbert.embed_and_index()

        # Step 2: Retrieve documents using ColBERT and generate responses
        logging.debug("Retrieving documents using ColBERT and generating responses")
        print("Retrieving documents using ColBERT and generating responses")


        for batch_request in tqdm(batches, desc=f"Calling LLM for {experiment_name}", ncols=150):
            for request in batch_request:
                retrieved_docs = colbert.retrieve(request, top_k=1, force_fast=False, rerank=False, rerank_top_k=1)
                
                # Instead of generating response using LLM, directly append the retrieved document content
                retrieved_content = [doc['content'] for doc in retrieved_docs]
                # Optionally, you can concatenate all retrieved documents' content into one response or store them as a list
                # For example, let's store them as a list of retrieved documents:
                pred.append(retrieved_content)


        # mkdir
        directory = os.path.dirname(file_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save the response
        with open(
            f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_{table_sampling_type}_{table_augmentation_type}.txt",
            "w",
        ) as f:
            for item in pred:
                f.write("%s\n" % item)

        # Evaluation
        numbers = Evaluator().run(pred, grd, task_name)
        logging.info(f"Evaluation results of {experiment_name}_{task_name}: {numbers}")
        print(f"Evaluation results of {experiment_name}_{task_name}: {numbers}")
        evaluation_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/output_evaluation.json"

        # mkdir
        directory = os.path.dirname(evaluation_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Read the existing data from the file
        try:
            with open(evaluation_save_path, "r") as file:
                existing_data = json.load(file)
        except:
            existing_data = {}

        # Update the existing data with the new data
        existing_data.update(
            {
                f"{experiment_name}-{task_name}-{table_sampling_type}-{table_augmentation_type}-{embedding_type}-{CallLLM().GPT_MODEL}-use_header_grounding-{whether_column_grounding}": numbers,
            }
        )

        # Write the updated data back to the file
        with open(evaluation_save_path, "w") as file:
            json.dump(existing_data, file, indent=4)

        # save the response
        save_jsonl_file_single(
            batches, grd, file_save_path, pred
        )
    '''
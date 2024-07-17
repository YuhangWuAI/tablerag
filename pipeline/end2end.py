import json
import os
import datetime
import time
from tqdm import tqdm
from table_provider import CallLLM, TableProvider
from .evaluation.evaluator import Evaluator
from .compoments import get_instruction
from typing import List


def save_jsonl_file(
    prompt_list: List,
    label_list: List,
    file_path: str,
    example_k_shots: str,
    augmentation_list: List,
    pred_list: List = None,
):
    def flatten_iterative(lst):
        """
        Flatten a list of lists iteratively.
        """
        stack = lst[::-1]
        result = []
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item[::-1])
            else:
                result.append(item)
        return result

    # mkdir
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # adjust Babel input
    data_list = []
    flatten_prompt_list = (
        flatten_iterative(prompt_list)
        if isinstance(prompt_list, list)
        and all(isinstance(sub_list, list) for sub_list in prompt_list)
        else prompt_list
    )

    flatten_augmentation_prompt_list = (
        flatten_iterative(augmentation_list)
        if isinstance(augmentation_list, list)
        and all(isinstance(sub_list, list) for sub_list in augmentation_list)
        else augmentation_list
    )

    if pred_list is None:
        for prompt, label, augmentation in zip(
            flatten_prompt_list, label_list, flatten_augmentation_prompt_list
        ):
            used_tokens = CallLLM().num_tokens(prompt)
            data_list.append(
                {
                    'example_k_shots': example_k_shots,
                    'augmentation': augmentation,
                    'prompt': prompt,
                    'label': label,
                    'used_tokens': used_tokens,
                }
            )
    else:
        flatten_pred_list = (
            flatten_iterative(pred_list)
            if isinstance(pred_list, list)
            and all(isinstance(sub_list, list) for sub_list in pred_list)
            else pred_list
        )
        for prompt, label, augmentation, pred in zip(
            flatten_prompt_list,
            label_list,
            flatten_augmentation_prompt_list,
            flatten_pred_list,
        ):
            used_tokens = CallLLM().num_tokens(prompt)
            data_list.append(
                {
                    'example_k_shots': example_k_shots,
                    'augmentation': augmentation,
                    'prompt': prompt,
                    'label': label,
                    'pred': pred,
                    'used_tokens': used_tokens,
                }
            )

    # save jsonl
    with open(file_path, 'w') as file:
        for item in data_list:
            json_string = json.dumps(item)
            file.write(json_string + '\n')


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
    whether_cot: bool = False,
    whether_self_consistency: bool = False,
    whether_column_grounding: bool = False,
):
    # log the daytime for the experiment
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')

    if experiment_name == "token_allocation":
        file_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_{table_sampling_type}_{table_augmentation_type}_{max_truncate_tokens}_{augmentation_tokens}.jsonl"
    elif experiment_name == "table_embedding_type":
        file_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_{table_sampling_type}_{table_augmentation_type}_{embedding_type}.jsonl"
    else:
        file_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/{task_name}_{table_sampling_type}_{table_augmentation_type}_{k_shot}.jsonl"

    # whether the task is already done
    if os.path.exists(file_save_path):
        return

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
    # task-specific instruction
    instruction = get_instruction(task_name)

    max_truncate_tokens = table_provider.call_llm.MAX_TRUNCATE_TOKENS
    augmentation_tokens = table_provider.call_llm.AUGMENTATION_TOKEN_LIMIT

    # example k shots for in-context learning
    if load_local_dataset:
        if whether_cot:
            with open(f"source/k_cot/{task_name}.jsonl", "r") as f:
                print(f"Loading k_cots for {task_name}...")
                example_k_shots = "\n".join(
                    [
                        instruction + "\n" + json.loads(line)["k_shot"]
                        for line in f.readlines()
                    ]
                )
        else:
            with open(f"source/k_shot/{task_name}.jsonl", "r") as f:
                print(f"Loading k_shots for {task_name}...")
                example_k_shots = (
                    "\n".join(
                        [
                            "shot_"
                            + str(json.loads(line)["id"])
                            + ":"
                            + instruction
                            + "\n"
                            + json.loads(line)["k_shot"]
                            for line in f.readlines()[:k_shot]
                        ]
                    )
                    if k_shot != 0
                    else None
                )
        with open(f"source/dataset/{task_name}.jsonl", "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
    else:
        example_k_shots = (
            instruction
            + "\n"
            + "\n".join(table_provider.table_loader.get_k_shot_example(k_shot))
            if k_shot != 0
            else None
        )
    # response & groundtruth
    grd, pred = [], []
    # for LLM calling
    batch_size = table_provider.call_llm.BATCH_SIZE
    # run the end2end with batch input
    num_samples = (
        len(dataset) if load_local_dataset else len(table_provider.table_loader.dataset)
    )

    num_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size
    batches, augmentation_batches = [], []
    with tqdm(
        total=num_batches + (1 if remaining_samples > 0 else 0),
        desc=f"Processing {experiment_name}_{task_name}",
        ncols=150,
    ) as pbar:
        for batch_num in range(num_batches):
            batch_prompt, augmentation_prompt = [], []
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            batch = (
                dataset[start_index:end_index]
                if load_local_dataset
                else table_provider.table_loader.dataset[start_index:end_index]
            )
            for i in range(batch_size):
                parsed_sample = (
                    batch[i]
                    if load_local_dataset
                    else table_provider.table_loader.parse_table(
                        {key: value[i] for key, value in batch.items()}
                    )
                )

                query = parsed_sample["query"]
                grd.append(parsed_sample["label"])

                filter_table = table_provider.table_sampler.run(query, parsed_sample)
                augmentation_info = (
                    table_provider.table_augmentation.run(parsed_sample)
                    if table_augmentation_type != "None"
                    else ""
                )
                prompt = "\n".join(
                    [
                        example_k_shots,
                        instruction,
                        "the table needed to be answered: \n",
                        filter_table.to_html(),
                        augmentation_info,
                        "claim"
                        if task_name == "tabfact" or task_name == "feverous"
                        else "query",
                        query,
                        "answer: \n",
                    ]
                )
                if (
                    table_provider.call_llm.num_tokens(prompt)
                    < table_provider.call_llm.TOTAL_TOKENS
                ):
                    batch_prompt.append(prompt)
                else:
                    batch_prompt.append(
                        table_provider.call_llm.truncated_string(
                            prompt,
                            table_provider.call_llm.TOTAL_TOKENS,
                            print_warning=False,
                        )
                    )
                augmentation_prompt.append(augmentation_info)
            pbar.update(1)
            batches.append(batch_prompt)
            augmentation_batches.append(augmentation_prompt)

        if remaining_samples > 0:
            batch_prompt = []
            start_index = num_batches * batch_size
            end_index = start_index + remaining_samples
            batch = (
                dataset[start_index:end_index]
                if load_local_dataset
                else table_provider.table_loader.dataset[start_index:end_index]
            )
            for i in range(remaining_samples):
                parsed_sample = (
                    batch[i]
                    if load_local_dataset
                    else table_provider.table_loader.parse_table(
                        {key: value[i] for key, value in batch.items()}
                    )
                )
                query = parsed_sample["query"]
                grd.append(parsed_sample["label"])
                try:
                    filter_table = table_provider.table_sampler.run(
                        query, parsed_sample
                    )
                except:
                    print("Skipping batch:", i)
                    continue
                augmentation_info = (
                    table_provider.table_augmentation.run(parsed_sample)
                    if table_augmentation_type != "None"
                    else ""
                )
                prompt = "\n".join(
                    [
                        example_k_shots,
                        instruction,
                        "the table needed to be answered: \n",
                        filter_table.to_html(),
                        augmentation_info,
                        "claim"
                        if task_name == "tabfact" or task_name == "feverous"
                        else "query",
                        query,
                        "answer is: \n",
                    ]
                )
                if (
                    table_provider.call_llm.num_tokens(prompt)
                    < table_provider.call_llm.TOTAL_TOKENS
                ):
                    batch_prompt.append(prompt)
                else:
                    batch_prompt.append(
                        table_provider.call_llm.truncated_string(
                            prompt, print_warning=False
                        )
                    )
                augmentation_prompt.append(augmentation_info)
            pbar.update(1)
            batches.append(batch_prompt)
            augmentation_batches.append(augmentation_prompt)

    # save as jsonl
    if save_jsonl:
        save_jsonl_file(
            batches,
            grd,
            file_save_path,
            example_k_shots,
            augmentation_batches,
        )

    # directly call LLM
    elif azure_blob:
        # call the LLMs
        for batch in tqdm(
            batches, desc=f"Calling LLM for {experiment_name}", ncols=150
        ):
            response = table_provider.call_llm.generate_text(
                batch, model_type=CallLLM().GPT_MODEL
            )
            pred.append(response)

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

        # evaluation
        numbers = Evaluator().run(pred, grd, task_name)
        print(f"Evaluation results of {experiment_name}_{task_name}:", numbers)
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
                f"{experiment_name}-{task_name}-{table_sampling_type}-{table_augmentation_type}-{embedding_type}-{CallLLM().GPT_MODEL}-token_allocation-A:{augmentation_tokens}T:{max_truncate_tokens}-w/ use_header_grounding-{whether_column_grounding}": numbers,
            }
        )

        # Write the updated data back to the file
        with open(evaluation_save_path, "w") as file:
            json.dump(existing_data, file, indent=4)

        # save the response
        save_jsonl_file(
            batches, grd, file_save_path, example_k_shots, augmentation_batches, pred
        )

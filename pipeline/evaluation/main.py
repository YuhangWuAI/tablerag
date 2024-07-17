from evaluator import Evaluator
from typing import List, Tuple
import json
import argparse
import os, os.path as Path


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Description of your evaluation running.'
    )

    # Add arguments
    parser.add_argument(
        '-t',
        '--task_name',
        type=str,
        help='Task name.',
    )

    parser.add_argument(
        '-e',
        '--experiment_type',
        type=str,
        help='Experiment type.',
        default="TP_table_augmentation",
    )

    parser.add_argument(
        '-o',
        '--option',
        type=str,
        help='Option.',
        default="clustering_sample",
    )

    parser.add_argument(
        '-p',
        '--file_dir_path',
        type=str,
        default="pipeline/data/Exp-20230717",
    )

    args = parser.parse_args()
    return args


def retrieve_answers(
    source_file_path: str, output_file_path: str
) -> Tuple[List[str], List[str]]:
    pred_answer, gold_answer = [], []
    with open(source_file_path, "r") as file:
        gold_data_list = file.readlines()
        for idx, data in enumerate(gold_data_list):
            gold_data_list[idx] = json.loads(data.strip())
        gold_answer.append([data["label"] for data in gold_data_list])

    with open(output_file_path, "r") as file:
        data_list = file.readlines()
        for idx, data in enumerate(data_list):
            data_list[idx] = json.loads(data.strip())
        pred_answer.append([data["samples"][0] for data in data_list])

    return pred_answer, gold_answer


def dump_number(args, numbers):
    if os.path.exists(f"{args.file_dir_path}/output") is False:
        os.mkdir(f"{args.file_dir_path}/output")
    file_path = f"{args.file_dir_path}/output/{args.experiment_type}_evaluation.json"

    # Read the existing data from the file
    try:
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    except:
        existing_data = {}

    # Update the existing data with the new data
    existing_data.update(
        {
            f"{args.option}_{args.task_name}": numbers,
        }
    )

    # Write the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)


def main():
    args = get_arguments()
    pred_answer, gold_answer = retrieve_answers(
        f"{args.file_dir_path}/{args.option}/{args.task_name}_validation.jsonl",
        f"{args.file_dir_path}/{args.option}/{args.experiment_type}_{args.option}_{args.task_name}/output/{args.task_name}_validation_samples.0.jsonl",
    )

    evaluator = Evaluator()
    numbers = evaluator.run(
        pred_answer[0],
        gold_answer[0],
        dataset=args.task_name,
        allow_semantic=False,
        question=None,
    )

    dump_number(args, numbers)


if __name__ == "__main__":
    main()

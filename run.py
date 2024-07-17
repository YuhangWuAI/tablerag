import os
import json
import unittest
import argparse
import numpy as np
from table_provider import (
    TableProvider,
    TableSamplingType,
    TableAugmentationType,
    TaskName,
)
from pipeline.end2end import end2end
from test.agents.test_agents import TestAgents


def add_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your code running.')

    # Add arguments
    parser.add_argument(
        '-t',
        '--task_name',
        choices=[task.value for task in TaskName],
        type=str,
        help='Task name.',
    )
    parser.add_argument(
        '-s',
        '--split',
        default="validation",
        choices=["train", "test", "validation"],
        type=str,
        help='Split of the dataset.',
    )
    parser.add_argument(
        '-r',
        '--table_sampling_type',
        choices=[table_sampling_type.value for table_sampling_type in TableSamplingType]
        + ["default"],
        default="evenly_sample",
        type=str,
        help='Type of row filter.',
    )
    parser.add_argument(
        '-a',
        '--table_augmentation_type',
        choices=[
            table_augmentation_type.value
            for table_augmentation_type in TableAugmentationType
        ],
        default="None",
        type=str,
    )
    parser.add_argument(
        '-e',
        '--embedding_type',
        choices=[
            "text-embedding-ada-002",
            "bge-large-en",
            "sentence-transformer",
            "text-embedding-ada-001",
        ],
        default="text-embedding-ada-002",
        type=str,
        help="Type of embedding.",
    )
    parser.add_argument(
        '-m',
        '--mode',
        choices=["End2end", "NL2Program"],
        default="End2end",
        help='Name of the task.',
    )
    parser.add_argument('--k_shot', type=int, default=1, help='Number of k-shot.')
    parser.add_argument('--n_cluster', type=int, default=3, help='Number of clusters.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top k.')
    parser.add_argument(
        '-c', '--config', default='config.json', help='Path to the config file.'
    )
    parser.add_argument(
        '--experiment_name', default='sampling', help='Name of the experiment.'
    )
    parser.add_argument('--test', action='store_true', help='Enable test mode.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--azure_blob', action='store_true', help='Enable azure blob.')
    parser.add_argument('--save_jsonl', action='store_true', help='Enable save jsonl.')
    parser.add_argument('--whether_cot', action='store_true', help='Enable cot.')
    parser.add_argument(
        '--self_consistency', action='store_true', help='Enable self consistency.'
    )
    parser.add_argument(
        '--load_local_dataset', action='store_true', help='Enable load local dataset.'
    )
    parser.add_argument(
        '--whether_column_grounding',
        action='store_true',
        help='Enable column grounding.',
    )

    # Add arguments for config
    parser.add_argument(
        "--embedding_model",
        type=str,
        help="Set or Update EMBEDDING_MODEL",
        default=None,
    )
    parser.add_argument(
        "--gpt_model", type=str, help="Set or Update GPT_MODEL", default=None
    )
    parser.add_argument(
        "--api_key", type=str, help="Set or Update api_key", default=None
    )
    parser.add_argument(
        "--batch_size", type=int, help="Set or Update batch_size", default=None
    )
    parser.add_argument(
        "--total_tokens", type=int, help="Set or Update total_tokens", default=None
    )
    parser.add_argument(
        "--table_token_limit_portion",
        type=int,
        help="Set or Update max_truncate_tokens",
        default=None,
    )
    parser.add_argument(
        "--augmentation_token_limit_portion",
        type=int,
        help="Set or Update augmentation_token_limit",
        default=None,
    )
    parser.add_argument(
        "--max_rows", type=int, help="Set or Update max_rows", default=None
    )
    parser.add_argument(
        "--max_columns", type=int, help="Set or Update max_columns", default=None
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def update_config(file_path, args):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "model": {
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "GPT_MODEL": "text-davinci-003",
            },
            "api_key": "",
            "batch_size": 16,
            "total_tokens": 4000,
            "max_truncate_tokens": 1400,
            "example_token_limit": {
                "tabfact": 627,
                "hybridqa": 1238,
                "sqa": 1439,
                "totto": 889,
                "feverous": 1261,
            },
            "augmentation_token_limit": 1000,
            "max_rows": 50,
            "max_columns": 10,
        }
    if args.embedding_model:
        config["model"]["EMBEDDING_MODEL"] = args.embedding_model
    if args.gpt_model:
        config["model"]["GPT_MODEL"] = args.gpt_model
    if args.api_key:
        config["api_key"] = args.api_key
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.total_tokens:
        config["total_tokens"] = args.total_tokens
    if args.table_token_limit_portion:
        config["max_truncate_tokens"] = int(
            config["total_tokens"] - config["example_token_limit"][args.task_name]
        ) * (args.table_token_limit_portion / 100)
    if args.augmentation_token_limit_portion:
        config["augmentation_token_limit"] = int(
            config["total_tokens"] - config["example_token_limit"][args.task_name]
        ) * (args.augmentation_token_limit_portion / 100)
    if args.max_rows:
        config["max_rows"] = args.max_rows
    if args.max_columns:
        config["max_columns"] = args.max_columns
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    args = add_arguments()
    update_config("config.json", args)

    if args.debug:
        table_provider = TableProvider(
            task_name="tabfact",
            split="validation",
            table_sampling_type="clustering_sample",
        )
        filtered_table = table_provider.table_sampler.run(
            query="What is the average age of the people who have a job?"
        )
        for _example in filtered_table:
            # Replace some cells with None
            random_row_index = np.random.randint(0, len(_example), size=2)
            random_col_index = np.random.randint(0, len(_example.columns), size=2)
            for i in range(len(random_row_index)):
                row_idx = random_row_index[i]
                col_idx = random_col_index[i]
                _example.iloc[row_idx, col_idx] = None
            print("Original Tables:\n {}".format(_example))
            _example_cleansing = table_provider.table_cleanser.fill_empty_column_name(
                _example
            )
            print("Cleansed Tables:\n {}".format(_example_cleansing))

        table_loader = table_provider.table_loader(
            task_name="tabfact", split="validation"
        )
        print(table_loader.get_k_shot_example(3))

    if args.test:
        TestAgents.test_str_normalize('2008-04-13 00:00:00', '2008-4-13 0:0:0')
        TestAgents.test_field_type_generation(lower_case=True)
        TestAgents.test_generate_numeric_range(lower_case=True)
        TestAgents.test_generate_time_series_interval(lower_case=True)
        TestAgents.test_metadata_api()

    end2end(
        task_name=args.task_name,
        split=args.split,
        table_sampling_type=args.table_sampling_type,
        table_augmentation_type=args.table_augmentation_type,
        embedding_type=args.embedding_type,
        k_shot=args.k_shot,
        top_k=args.top_k,
        n_cluster=args.n_cluster,
        experiment_name=args.experiment_name,
        save_jsonl=args.save_jsonl,
        azure_blob=args.azure_blob,
        load_local_dataset=args.load_local_dataset,
        whether_cot=args.whether_cot,
        whether_self_consistency=args.self_consistency,
        whether_column_grounding=args.whether_column_grounding,
    )

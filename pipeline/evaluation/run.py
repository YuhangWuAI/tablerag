import os


def main():
    file_dir_path = "pipeline/data/Exp-230723"
    table_augmentation_configs = [
        "docs_references",
        "metadata",
        "table_size",
    ]
    table_sampling_configs = [
        "clustering_sample",
        "embedding_sample",
        "head_tail_sample",
        "random_sample",
    ]
    embedding_configs = ["huggingface", "spacy", "openai"]
    all_datasets = ["tabfact", "sqa", "totto", "hybridqa", "feverous"]
    case_study_dataset = ["sqa", "feverous"]

    # Run table sampling configs
    for config in table_sampling_configs:
        for dataset in all_datasets:
            run_script = f'python pipeline/evaluation/main.py --option {config} --task_name {dataset} --experiment_type TP_table_sampling --file_dir_path {file_dir_path}'
            msg = os.system(run_script)
            print(
                "Running config: ", config, dataset, "Success" if msg == 0 else "Failed"
            )

    # Run table augmentation configs
    for config in table_augmentation_configs:
        for dataset in case_study_dataset:
            run_script = f'python pipeline/evaluation/main.py --option {config} --task_name {dataset} --experiment_type TP_table_augmentation --file_dir_path {file_dir_path}'
            msg = os.system(run_script)
            print(
                "Running config: ", config, dataset, "Success" if msg == 0 else "Failed"
            )

    # table header hierarchy
    run_script = f'python pipeline/evaluation/main.py --option header_hierarchy --task_name totto --experiment_type TP_table_augmentation --file_dir_path {file_dir_path}'

    # Run embedding configs
    for config in embedding_configs:
        for dataset in case_study_dataset:
            run_script = f'python pipeline/evaluation/main.py --option {config} --task_name {dataset} --experiment_type TP_table_embedding_type --file_dir_path {file_dir_path}'
            msg = os.system(run_script)
            print(
                "Running config: ", config, dataset, "Success" if msg == 0 else "Failed"
            )

    # Test sample
    config = "clustering_sample"
    dataset = "totto"
    run_script = f'python pipeline/evaluation/main.py --option {config} --task_name {dataset} --experiment_type TP_table_sampling --file_dir_path {file_dir_path}'
    msg = os.system(run_script)
    print(msg)


if __name__ == "__main__":
    main()

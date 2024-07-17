import os, json
from table_provider import TableLoader

task_name = ["feverous", "hybridqa", "sqa", "tabfact", "totto"]
for t in task_name:
    table_loader = TableLoader(
        task_name=t, split='validation', use_small_sample_list=False
    )

    # generate dataset
    dataset_path = f"source/dataset_full/{t}.jsonl"
    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))
    with open(dataset_path, 'w') as f:
        for i in table_loader.dataset:
            parsed_sample = table_loader.parse_table(i)
            json_string = json.dumps(parsed_sample)
            f.write(json_string + '\n')

    # # generate k-shot
    # k_shot_path = f"pipeline/raw/k_shot/{t}.jsonl"
    # if not os.path.exists(os.path.dirname(k_shot_path)):
    #     os.makedirs(os.path.dirname(k_shot_path))
    # with open(k_shot_path, 'w') as f:
    #     k_shots = table_loader.get_k_shot_example(10)
    #     for i in range(len(k_shots)):
    #         json_string = json.dumps({"id": i, "k_shot": k_shots[i]})
    #         f.write(json_string + '\n')

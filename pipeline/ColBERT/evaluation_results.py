import json

import os

from pipeline.evaluation.evaluator import Evaluator

def evaluate_results(
    pred: list,
    grd: list,
    task_name: str,
    experiment_name: str,
    formatted_today: str,
    table_sampling_type: str,
    table_augmentation_type: str,
    embedding_type: str,
    whether_column_grounding: bool = False,
):
    print("Evaluating results\n")
    numbers = Evaluator().run(pred, grd, task_name)
    print("Evaluation results of ", experiment_name, "_", task_name, ": ", numbers, "\n")
    
    evaluation_save_path = f"pipeline/data/Exp-{formatted_today}/{experiment_name}/output_evaluation.json"

    directory = os.path.dirname(evaluation_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        with open(evaluation_save_path, "r") as file:
            existing_data = json.load(file)
    except:
        existing_data = {}

    existing_data.update(
        {
            f"{experiment_name}-{task_name}-{table_sampling_type}-{table_augmentation_type}-{embedding_type}-{Evaluator().GPT_MODEL}-use_header_grounding-{whether_column_grounding}": numbers,
        }
    )

    with open(evaluation_save_path, "w") as file:
        json.dump(existing_data, file, indent=4)

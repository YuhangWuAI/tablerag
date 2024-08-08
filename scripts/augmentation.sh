# feverous hybridqa sqa and tabfact
task_name=("feverous")

# docs_references and term_explanations and assemble_retrieval_based_augmentation
augmentation_types=("assemble_retrieval_based_augmentation")

# Empirical study on augmentation types
for augmentation_type in "${augmentation_types[@]}"; do
    for task in "${task_name[@]}"; do
        echo ">>>>>>>>>>>>>>>>>>Start with $task $augmentation_type<<<<<<<<<<<<<<<<<<<<<<"
        python run.py -t $task -r default -a $augmentation_type --save_jsonl --experiment_name table_augmentation --load_local_dataset --azure_blob --table_token_limit_portion 50 --augmentation_token_limit_portion 30
        echo ">>>>>>>>>>>>>>>>>>Done with $task $augmentation_type<<<<<<<<<<<<<<<<<<<<<<<"
    done
done
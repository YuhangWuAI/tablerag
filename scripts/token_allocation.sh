task_name=("tabfact" "sqa" "totto" "hybridqa" "feverous")
for task in "${task_name[@]}"; do
    for shot in "${k_shot[@]}"; do
        python run.py -t $task -r clustering_sample -e spacy -a assemble_neural_symbolic_augmentation --save_jsonl --experiment_name token_allocation --load_local_dataset --k_shot 1 --augmentation_token_limit 1000 -max_truncate_tokens 1000
        echo "Done with $task $split $sampling_method"
    done
done
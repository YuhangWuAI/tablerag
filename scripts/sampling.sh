# End-to-end Jsonl Generation
sampling_methods=("evenly_sample" "random_sample" "embedding_sample" "clustering_sample")
task_name=("tabfact" "sqa" "totto" "hybridqa" "feverous")

# Empirical study on table sampling types
for task in "${task_name[@]}"; do
    for sampling_method in "${sampling_methods[@]}"; do
        echo ">>>>>>>>>>>>>>>>>>Start with $task $sampling_method<<<<<<<<<<<<<<<<<<<<<<"
        python run.py  --gpt_model gpt-3.5-turbo-1106 --total_tokens 160000 -t $task -r $sampling_method --experiment_name table_sampling --load_local_dataset --whether_cot --azure_blob --table_token_limit_portion 70 --augmentation_token_limit_portion 0
        echo ">>>>>>>>>>>>>>>>>>Done with $task $sampling_method<<<<<<<<<<<<<<<<<<<<<<<"
    done
done

# python run.py -t hybridqa -r random_sample --experiment_name table_sampling --load_local_dataset --whether_cot --azure_blob --table_token_limit_portion 70 --augmentation_token_limit_portion 0
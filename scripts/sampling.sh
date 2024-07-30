# End-to-end Jsonl Generation
# auto_row_filter and embedding_sample
sampling_methods=("auto_row_filter")
task_name=("tabfact")

# Empirical study on table sampling types
for task in "${task_name[@]}"; do
    for sampling_method in "${sampling_methods[@]}"; do
        echo ">>>>>>>>>>>>>>>>>>Start with $task $sampling_method<<<<<<<<<<<<<<<<<<<<<<"
        python run.py  --gpt_model gpt-3.5-turbo --total_tokens 160000 -t $task -r $sampling_method --sample_size 2 --experiment_name table_sampling --load_local_dataset --azure_blob --table_token_limit_portion 70 --augmentation_token_limit_portion 0
        echo ">>>>>>>>>>>>>>>>>>Done with $task $sampling_method<<<<<<<<<<<<<<<<<<<<<<<"
    done
done

# python run.py -t hybridqa -r random_sample --experiment_name table_sampling --load_local_dataset --whether_cot --azure_blob --table_token_limit_portion 70 --augmentation_token_limit_portion 0
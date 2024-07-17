task_name=("tabfact" "sqa" "totto" "hybridqa" "feverous")
embedding_types=("text-embedding-ada-002" "bge-large-en" "sentence-transformer" "text-embedding-ada-001")

# Empirical study on table sampling types
for task in "${task_name[@]}"; do
    for embedding_type in "${embedding_ty pes[@]}"; do
        echo ">>>>>>>>>>>>>>>>>>Start with $task $embedding_type<<<<<<<<<<<<<<<<<<<<<<"
        python run.py -t $task -r embedding_sample -e $embedding_type --experiment_name table_embedding_type --load_local_dataset --whether_cot --azure_blob --table_token_limit_portion 70 --augmentation_token_limit_portion 0
        echo ">>>>>>>>>>>>>>>>>>Done with $task $embedding_type<<<<<<<<<<<<<<<<<<<<<<<"
    done
done

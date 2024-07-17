# End-to-end Jsonl Generation
task_name=("tabfact" "sqa" "totto" "hybridqa" "feverous")
augmentation_types=("table_size" "metadata" "header_field_categories" "trunk_summary" "intermediate_NL_reasoning_steps" "term_explanations" "docs_references")

# Empirical study on augmentation types
for augmentation_type in "${augmentation_types[@]}"; do
    for task in "${task_name[@]}"; do
        echo ">>>>>>>>>>>>>>>>>>Start with $task $augmentation_type<<<<<<<<<<<<<<<<<<<<<<"
        python run.py -t $task -r default -a $augmentation_type --save_jsonl --experiment_name table_augmentation --load_local_dataset --whether_cot --azure_blob --table_token_limit_portion 50 --augmentation_token_limit_portion 30
        echo ">>>>>>>>>>>>>>>>>>Done with $task $augmentation_type<<<<<<<<<<<<<<<<<<<<<<<"
    done
done

# ToTTo header hierarchy
python run.py -t totto -r default -a header_hierarchy --save_jsonl --experiment_name table_augmentation --load_local_dataset --whether_cot --azure_blob --table_token_limit_portion 50 --augmentation_token_limit_portion 30

# # Test HybridQA dataset
# # python run.py -t hybridqa -s validation -r random_sample -m End2end --save_jsonl
# # python run.py -t totto -s validation -r random_sample -m End2end --save_jsonl
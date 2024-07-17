# End-to-end Jsonl Generation
sampling_methods=("evenly_sample" "random_sample" "embedding_sample" "clustering_sample")
task_name=("tabfact" "sqa" "totto" "hybridqa" "feverous")
case_study_dataset=("feverous" "sqa")
augmentation_types=("table_size" "metadata" "header_field_categories" "trunk_summary" "intermediate_NL_reasoning_steps" "term_explanations" "docs_references")
embedding_types=("spacy" "openai" "huggingface")

# Empirical study on table sampling types
for sampling_method in "${sampling_methods[@]}"; do
    for task in "${task_name[@]}"; do
        python run.py -t $task -s $split -r $sampling_method -e spacy -a None -m $mode --save_jsonl --experiment_name $sampling_method --load_local_dataset
        echo "Done with Table Sampling $task $split $sampling_method"
    done
done

# Empirical study on augmentation types
for augmentation_type in "${augmentation_types[@]}"; do
    for task in "${case_study_dataset[@]}"; do
        python run.py -t $task -s $split -r default -e spacy -a $augmentation_type -m $mode --save_jsonl --experiment_name $augmentation_type --load_local_dataset
        echo "Done with Table Augmentation $task $split $augmentation_type"
    done
done

# ToTTo header hierarchy
python run.py -t totto -s validation -r default -e spacy -a header_hierarchy -m $mode --save_jsonl --experiment_name header_hierarchy --load_local_dataset

# # Test HybridQA dataset
# # python run.py -t hybridqa -s validation -r random_sample -m End2end --save_jsonl
# # python run.py -t totto -s validation -r random_sample -m End2end --save_jsonl
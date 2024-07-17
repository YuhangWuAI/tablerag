task_name=("tabfact" "sqa" "totto" "hybridqa" "feverous")
k_shot=(1 2 3)
for task in "${task_name[@]}"; do
    for shot in "${k_shot[@]}"; do
        python run.py -t $task -s validation -r clustering_sample -e spacy -a None --save_jsonl --experiment_name top_k --load_local_dataset --k_shot $shot
        echo "Done with $task $split $sampling_method"
    done
done
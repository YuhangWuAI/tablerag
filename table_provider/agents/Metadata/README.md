# MetaData Repo
This repo contains training, prediction and evaluation code for paper Metadata (_Inferring Tabular Analysis Metadata by Infusing Distribution and Knowledge Information_).
## Environment
You can run this repo within docker container, and the dockerfile see [_dockerfile_](dockerfile).
## Pre-trained model embedding
The paper involves 2 pre-trained model embedding: TAPAS, TABBIE. You can generate the first one with [Hugging Face](https://github.com/huggingface). For TABBIE, you can generate embeddings from [TABBIE repo](TODO).

## Metadata code
Functions of codes are as following: 

+ [data](data): Load necessary data.
+ [metadata/evaluations](metadata/evaluations): Evaluation metric for each task in paper.
+ [metadata/measure_type](metadata/measure_type): Map measure type from different dataset to measure type in paper.
+ [metadata/metadata_data](metadata/metadata_data): Construct batch of input data for metadata.
+ [metadata/predict](metadata/predict): Evaluation model.
+ [metadata/train](metadata/train): Train model.
+ [metadata/find_perform.py](../../metadata_exp/metadata/find_perform.py): Find useful evaluation metric in log.
+ [metadata/run_train.py](metadata/run_train.py): Entry for training metadata.
+ [metadata/run_predict.py](metadata/run_predict.py): Entry for inference metadata.
+ [model](model): Model of metadata.

### Train
Run the following script for training metadata:
```shell
# TABBIE  
python -m metadata.run_train --model_size=customize --features=metadata-tabbie --model_name metadata2 --train_batch_size=64 --valid_batch_size=96 --msr_pos_weight=0.8 --sum_pos_weight=0.4 --avg_neg_weight=0.3 --both_neg_weight=0.5 --no_label_weight=0.2 --train_epochs=10 --save_model_fre=5 --num_layers=60 --num_hidden=128 --lang=en --num_workers=0 --corpus all --chart <chart_path> --pivot <pivot_path> --vendor <vendor_path> --t2d <t2d_path> --semtab <semtab_path> --tf1_layers 2 --tf2_layers 2 --df_subset 1 2 3 4 5  --mode general --use_emb --use_entity --entity_type transe100 --entity_emb_path <entity_emb_path> --entity_recognition semtab --use_df

# TAPAS 
python -m metadata.run_train --model_size=customize --features=metadata-tapas_display --model_name metadata2 --train_batch_size=64 --valid_batch_size=96 --msr_pos_weight=0.8 --sum_pos_weight=0.4 --avg_neg_weight=0.3 --both_neg_weight=0.5 --no_label_weight=0.2 --train_epochs=10 --save_model_fre=5 --num_layers=60 --num_hidden=128 --lang=en --num_workers=0 --corpus all --chart <chart_path> --pivot <pivot_path> --vendor <vendor_path> --t2d <t2d_path> --semtab <semtab_path> --tf1_layers 2 --tf2_layers 2 --df_subset 1 2 3 4 5  --mode general --use_emb --use_entity --entity_type transe100 --entity_emb_path <entity_emb_path> --entity_recognition semtab --use_df
```
Note,
+ If you don't need to use data feature, remove `--use_df`
+ If you don't need to use knowledge graph information, remove `--use_entity`

### Inference
Run the following script for inference metadata:
```shell
# TABBIE
python -m metadata.run_predict --model_size=customize --features=metadata-tabbie --valid_batch_size=96 --mode general  --num_layers=3 --num_hidden=128 --lang=en --model_load_path <model_load_path>  --eval_dataset test --corpus all --chart <chart_path> --pivot <pivot_path> --vendor <vendor_path> --t2d <t2d_path> --semtab <semtab_path>  --use_emb --tf1_layers 2 --tf2_layers 2 --model_name metadata2 --df_subset 1 2 3 4 5 --use_emb --use_entity --entity_type transe100 --entity_emb_path <entity_emb_path>  --entity_recognition semtab --use_df --num_workers 0

# TAPAS
python -m metadata.run_predict --model_size=customize --features=metadata-tapas_display --valid_batch_size=96 --mode general  --num_layers=3 --num_hidden=128 --lang=en --model_load_path <model_load_path>  --eval_dataset test --corpus all --chart <chart_path> --pivot <pivot_path> --vendor <vendor_path> --t2d <t2d_path> --semtab <semtab_path>  --use_emb --tf1_layers 2 --tf2_layers 2 --model_name metadata2 --df_subset 1 2 3 4 5 --use_emb --use_entity --entity_type transe100 --entity_emb_path <entity_emb_path>  --entity_recognition semtab --use_df --num_workers 0
```

Note, 
+ You need to keep the same setting as training.
+ Model tested in paper are in [model/model](model/model). We only provide one for each model (evaluation metrics in paper are average of 3 repeated experiments)

## Data
The paper involves 6 datasets: Chart, Pivot, Vendor, T2D, TURL, SemTab. For the first three datasets, we will publish part of them after the paper published. For T2D dataset, you can download from this [url](http://webdatacommons.org/webtables/goldstandard.html#toc2). For TURL dataset, you can download from this [url](https://github.com/sunlab-osu/TURL). For SemTab dataset, you can download from this [url](https://github.com/sunlab-osu/TURL).

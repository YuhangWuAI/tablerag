# FeatureExtractor
----------------

## Background
FeatureExtractor is the module that generates `<uid>.DF.json` file for all tables (identified by `<tuid>`) of a specific schema (identified by `<uid>`). The `<uid>` and `<tuid>` identifiers here are numbers. 

**Input:** `<uid>.json` and `<uid>.t<tuid>.table.json`
**Output:** `<uid>.t<tuid>.DF.json`


## Files
> FeatureExtractor
>> example -- folder for an example schema and its tables  
>> [data_feature_extractor.py](./data_feature_extractor.py) -- DataFeatureExtractor class  
>> [handle_chart.py](./handle_chart.py) -- main file for extracting features  
>> [source_features.py](./source_features.py) -- source feature class, used for managing output structure  

## Usage
See [run_feature_extractor_example.py](./run_feature_extractor_example.py) for an example

```bash
python run_feature_extractor_example.py
```


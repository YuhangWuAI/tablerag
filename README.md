## TAP4LLM: using LLMs as Query Interface for Structured Data Lake

## Introduction

Large language models (LLMs) are leveraged by more and more table-related features and products ( *e.g.* , Excel Copilot, DBCopilot, Viva Insights Copilot,  *etc.*) for their amazing capabilities on reasoning and generating natural language and code. When sending table(s) as part of input prompt to an LLM, there are many non-trivial choices that can greatly impact the output quality of the LLM. To help table-related applications better leverage LLMs, we introduce our recent work on TAP4LLM (Table Provider for LLM) that provides rich choices and optimized defaults on prompt designs of table given other context ( *e.g.* , user query, chat history,  *etc.* ) and constraints ( *e.g.* , token limitation).

![table_provider_framework](figs/TableProvider-Framework-v5.png)

Specifically, TAP4LLM involves three functionality components and one dependency component:

* Table sampling which decomposes a huge table into a sub-table with specific rows/columns based on the query semantics (sometimes called “grounding” for columns).
* Table augmentation which retrieves relevant knowledge from extra steps and external models and services,  *e.g.* , XLIM metadata, structural information detection, and the internal world/commonsense knowledge from LLMs.
* Table packing and serialization provides tabular serialization and packing functions to generation LLM prompts. The table provider aims to perform as a back-end module for all tabular-based tasks with the usage of LLMs.
* Table manager is the necessary dependency of table provider. It manages the table(s) in the context and helps execute the generated code against the table(s).

## Getting started
### Installtion

**1. Prepare the code and environment**

Git clone our repository, creating a python enviroment and activate it via the following command:

```bash
git clone https://common-analysis.visualstudio.com/TableProvider/_git/TableProvider
cd TableProvider
conda env create -f environment.yml
conda activate table_provider
```

**2. Prepare the validation/text data**

The current version of TableProvider is evaluated on five public datasets, i.e., feverous, hybridqa, sqa, tabfact and totto. Please refer to our loading scripts at [here](table_provider/data_loader/scripts/) to prepare the data.

Noted that anyone wishing to add a new dataset can follow the similar approach by refining a loading script and store it at [scripts](table_provider/data_loader/scripts/).

Then modify the [enum_type.py](table_provider/contract/enum_type.py), add the new dataset name to the class "TaskeName".


### Run

**1. Format of the GPT-x input**

To use the GPT-x models from the MS Clusters, the input should be formulated into jsonl files.

```json
{"prompt": "xxxxx", "label": "xxx"}
{"prompt": "xxxxx", "label": "xxx"}
{"prompt": "xxxxx", "label": "xxx"}
{"prompt": "xxxxx", "label": "xxx"}
```

Where the **'prompt'** refers to the input for the GPT-x, and the **'label'** refers to the groundtruth of the user query's corresponding answer.

**2. Generate the jsonl files**

TAP4LLM aims to support two pipeline for generating the jsonl files to use the GPT-x models.
- End-to-end: it aims to leverage LLMs to generate final answers directly, done by providing a task description and/or a few examples for in-context learning.
- Semantic Parsing/Code Generation: it leverages LLMs to convert natural language queries into executable code or structured representations, which enables more precise control over the output and facilitates better interpretability.

Please run the following code to use the End-to-end pipeline (Noted that modify the arguments in [end2end.sh](pipeline/end2end.sh) before run the code):
```bash
cd TableProvider
bash pipeline/end2end.sh
```
By default, the output will be stored at [pipeline/data/Exp-{TimeStamp}](pipeline/data/). If you intend to specify this parameter, please also set up in [end2end.sh](pipeline/end2end.sh).

Pipeline for semantic parsing/code generation will coming soon...

**3. Modify the TableProvider Components**
Please follow the code from [table_provider/dataframe](table_provider/dataframe/). Each file corresponds one component in TAP4LLM.

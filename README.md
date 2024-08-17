# **TableRAG**

## **Overview**

TableRAG is an advanced system designed for handling complex table-based question answering tasks. The system combines cutting-edge large language models (LLMs) with Retrieval-Augmented Generation (RAG) techniques, optimized specifically for multi-table environments. This project addresses the challenges posed by real-world scenarios, where users may need to query a set of related tables rather than a single one, by integrating advanced filtering, clarification, and retrieval mechanisms.

## **Table of Contents**

- [Background](#background)
- [System Architecture](#system-architecture)
  - [RAG-based Multi-Table QA System](#rag-based-multi-table-qa-system)
  - [Enhancement Mechanisms](#enhancement-mechanisms)
- [Datasets](#datasets)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Execution](#pipeline-execution)
- [Evaluation Experiments](#evaluation-experiments)
- [Acknowledgments](#acknowledgments)

## **Background**

Tables are a fundamental and widely used semi-structured data type, prevalent in relational databases, spreadsheet applications, and programming languages for data processing. They are utilized across various domains such as financial analysis, risk management, and healthcare analytics. Table Question Answering (TableQA) is a crucial downstream task that involves reasoning over table data to answer queries in natural language.

Recent advances in large language models (LLMs) have significantly improved the performance of TableQA. However, most existing studies focus on single-table scenarios. In real-world applications, users often encounter multiple related tables and may pose queries involving these tables. This project aims to address this gap by developing a system that can retrieve relevant tables from a large set and generate accurate answers.

## **System Architecture**

### **RAG-based Multi-Table QA System**

The core of our system is based on the Retrieval-Augmented Generation (RAG) approach, which combines retrieval mechanisms with generative models to enhance the accuracy of domain-specific QA tasks. The architecture consists of several key components:

1. **Table Processing and Text Segmentation**: Raw table data is preprocessed and segmented into multiple text fragments to facilitate efficient retrieval.
2. **Vector Database Construction**: The segmented text fragments are embedded and stored in a vector database, enabling rapid retrieval of relevant content.
3. **Query and Retrieval**: The system uses a ColBERT model to enhance retrieval precision by performing fine-grained token-level text encoding.
4. **Answer Generation**: Retrieved text fragments and the user's query are fed into an LLM, which generates the final answer in natural language.

![RAG-based Multi-Table QA System](./images/Overview.png)  <!-- Image 1 -->

### **Enhancement Mechanisms**

To improve the accuracy and efficiency of multi-table QA, our system incorporates several enhancement mechanisms:

1. **Semantic-Based Table Filtering**: The system filters large tables based on semantic analysis to reduce noise and focus on relevant rows and columns. This filtering is done using both OpenAI's Embedding models and the ColBERT model.
2. **LLM-Based Filtering**: In addition to semantic filtering, an LLM is used to analyze the deep semantic relationship between the table content and the query, ensuring that only the most relevant table fragments are selected.
3. **Table Clarifier**: This module provides additional context for ambiguous or domain-specific terms in the table, improving the model's understanding and reducing biases.
4. **Table Summary Generation**: The system generates concise summaries of the table content, leveraging Wikipedia metadata and the table's context to enhance the LLM's comprehension.

![Enhancement Mechanisms](./images/image.png)  <!-- Image 2 -->

## **Datasets**

We evaluated our system using several well-known TableQA datasets:

- **TableFact**: A large-scale dataset focused on table fact verification tasks.
- **FEVEROUS**: A dataset designed for fact verification tasks combining structured table data with unstructured text data.
- **SQA**: Evaluates models in multi-step question-answering scenarios.
- **HybridQA**: A multi-modal dataset that includes both table and text data for testing complex reasoning steps.

## **Implementation Details**

### **Table Filtering**

We implemented two main types of filters:

1. **Semantic-Based Filtering**: Uses OpenAI's Embedding models and the ColBERT model to generate semantic embeddings for the table content and the query, selecting the most relevant rows and columns.
2. **LLM-Based Filtering**: Leverages an LLM to perform intelligent filtering based on deep semantic understanding, ensuring that only the most crucial information is retained.

![Table Filtering](./images/table_filtering.png)  <!-- Image 3 -->

### **Table Clarifier**

To enhance the model's understanding of the table content:

1. **Term Clarification**: The system identifies and explains domain-specific terms and abbreviations using an LLM, ensuring accurate comprehension.
2. **Wiki-Based Summary Generation**: Metadata from Wikipedia is used to generate concise summaries of the table content, providing additional context for the LLM.

![Table Clarifier](./images/table_clarifier.png)  <!-- Image 4 -->

### **Retrieval Process Enhancement**

We integrated ColBERT into the retrieval process, offering several advantages:

1. **Late Interaction Framework**: ColBERT allows for efficient pre-computation of document representations, reducing online query processing time.
2. **MaxSim Operation**: This operation evaluates the relevance of queries and documents by considering the maximum similarity between token embeddings.
3. **Rerank Mechanism**: After initial retrieval, a rerank mechanism is employed to refine the results, ensuring the highest relevance to the query.

### **Input Format Optimization**

Based on recent research, we optimized the format in which tables are fed into the LLM:

1. **HTML Format**: Preserves complex table structures, making it suitable for multi-dimensional or nested tables.
2. **Markdown Format**: Offers a simple, human-readable format that performs well in the Retrieval-Augmented Generation (RAG) paradigm.

### **Self-Consistency**

To further enhance the robustness and reliability of the generated outputs, we implemented a self-consistency mechanism:

- **Multiple Code Generations**: For each filtering and clarifying task, the system generates multiple code snippets or outputs.
- **Majority Voting**: The most frequently generated output is selected as the final result. This approach helps mitigate errors and inconsistencies that might arise from a single model run.

## **Installation**

To install and set up the TableRAG system, follow these steps:

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/TableRAG.git
cd TableRAG
```
### **2. Set Up a Virtual Environment**

You can set up a virtual environment using either `conda` (Anaconda) or `venv` (Python's built-in tool).

#### **Linux and macOS:**

1. **Using venv**:

    1. **Create a Virtual Environment**:
       ```bash
       python -m venv venv
       source venv/bin/activate
       ```

    2. **Install Dependencies from `requirements.txt`**:
       ```bash
       pip install -r requirements.txt
       ```

#### **Windows:**

1. **Using venv**:

    1. **Create a Virtual Environment**:
       ```bash
       python -m venv venv
       venv\Scripts\activate
       ```

    2. **Install Dependencies from `requirements.txt`**:
       ```bash
       pip install -r requirements.txt
       ```

### **3. Set Up Environment Variables**

Configure any necessary API keys and environment variables as required. This may include API keys for OpenAI, if you are using their models for filtering or generation tasks.

### **4. Create `config.json` in the Root Directory**

After setting up the environment, create a `config.json` file in the root directory with the following structure:

```json
{
    "model": {
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "GPT_MODEL": "gpt-4o-mini"
    },
    "api_key": "your-api-key-here",
    "api_base": "https://your-api-base-url-here",
    "batch_size": 1,
    "use_self_consistency": false
}
```

- **EMBEDDING_MODEL**: The model name for embedding generation.
- **GPT_MODEL**: The model name for GPT-based text generation.
- **api_key**: Your API key for accessing the models.
- **api_base**: The base URL for the API you are using.
- **batch_size**: Number of queries or requests processed in a single batch.
- **use_self_consistency**: Boolean flag to determine if the self-consistency mechanism should be used.

### **5. Version Information**

Ensure that your environment meets the following version requirements:

- **Python**: 3.10.14
- **PyTorch**: 2.3.1
- **Transformers**: 4.36.2
- **FAISS-gpu**: 1.7.2
- **LangChain**: 0.1.0
- **OpenAI**: 1.14.0
- **CUDA**: 12.1

## **Dataflow Diagram**
<div style="overflow-x: auto;">
  <img src="./images/dataflow.png" alt="Data Flow Diagram" style="max-width: 100%;"/>
</div>

## **Class Diagram**

<div style="overflow-x: auto;">
  <img src="./images/class.png" alt="Class Diagram" style="max-width: none; width: 100%;"/>
</div>

# Project Structure

```bash
├── data  
│   ├── processed  
│   │   ├── prediction  
│   │   ├── retrieval_results  
│   │   ├── table_outputs  
│   │   ├── processing  
│   ├── raw  
│       ├── all_dataset  
│       ├── small_dataset  
├── src  
│   ├── colbert_pipeline  
│   │   ├── colbert_main.py  
│   │   ├── generator.py  
│   │   ├── retriever.py  
│   ├── data_processing  
│   │   ├── request_serializer.py  
│   │   ├── save_jsonl.py  
│   ├── evaluator  
│   │   ├── evaluation.py  
│   ├── llm  
│   │   ├── llm_embedding.py  
│   │   ├── llm_generating.py  
│   ├── table_loader  
│   │   ├── data_downloader  
│   ├── table_parser  
│   │   ├── table_linearizer.py  
│   │   ├── table_parsing.py  
│   │   ├── type_sets.py  
│   ├── table_master  
│   │   ├── table_clarifier.py  
│   │   ├── table_filter.py  
│   │   ├── table_main.py  
│   ├── table_processor  
│   │   ├── table_processing_pipeline.py  
│   ├── utils  
│       ├── cos_similarity.py  
├── config.json  
├── environment.yml  
├── LICENSE  
├── README.md  
├── requirements.txt
```

## **Usage**

Once the system is installed, you can run the pipelines to process tables, retrieve relevant data, and generate answers to queries.

## **Pipeline Execution**

### **Table Processing Pipeline**


```bash
python -m src.table_processor.table_processing_pipeline
```

The table processing pipeline allows for fine-grained control over various parameters through the `table_processing_pipeline` function. Below is a detailed explanation of the key parameters:

- **task_name**: Specifies the task identifier, such as `"hybridqa"`, to indicate the specific processing task being executed.
- **split**: Defines the dataset split to be used, with options including `"train"`, `"validation"`, or `"test"`.
- **table_filter_name**: Configures the table filtering mechanism. For example, `"llm_based_filter"` applies a filter based on a large language model (LLM), while `"None"` disables this functionality.
- **table_clarifier_name**: Determines the clarification approach for tables. Options include `"None"` for no clarification, or `"term_explanations_and_table_summary"` to enable explanations of terms and summaries of tables.
- **embedding_type**: Specifies the embedding model to be used, such as `"text-embedding-3-large"`.
- **top_k**: Sets the number of top results to be retrieved or processed.
- **save_jsonl**: A boolean flag indicating whether the output should be saved in JSONL format.
- **load_local_dataset**: A boolean flag indicating whether to load the dataset from a local directory.
- **experiment_name**: Provides a name for the experiment, which is useful for tracking and logging purposes.
- **use_sampled_table_for_augmentation**: A boolean flag that, if set to `True`, enables the use of sampled tables for data augmentation.
- **sample_size**: Specifies the number of samples to be used when table sampling is enabled.
- **overwrite_existing**: A boolean flag indicating whether existing files should be overwritten.
- **table_format**: Specifies the format in which the table data is processed, such as `"string"`.
- **use_table_filter**: A boolean flag indicating whether the table filter should be applied during processing.

The output files generated by this pipeline are stored in the `data/processed/table_outputs` directory. Additionally, the pipeline includes a progress-saving feature. If the process is interrupted, progress is automatically saved in the `data/progressing/` directory, enabling you to resume processing from the last saved state.

All configurable parameters can be reviewed and modified in the `type_set.py` file, allowing for tailored adjustments based on specific project requirements.

### **Retrieval Pipeline**

```bash
python -m src.colbert_pipeline.retrieval
```

The retrieval pipeline is configured using a dictionary of parameters that can be adjusted according to the specific requirements of your task. Below is the configuration along with an explanation of each parameter:

```python
config = {
    "dataset_path": "data/processed/table_outputs/hybridqa_llm_based_filter_None_string.jsonl",
    "index_name": "my_index",
    "colbert_model_name": "colbert-ir/colbertv2.0",
    "base_output_dir": "data/processed/retrieval_results",
    "use_rerank": False,
    "top_k": 1,
    "rerank_top_k": 1,
    "num_queries": 1000,
    "query_grd_path": "data/raw/small_dataset/hybridqa.jsonl"
}
```

- **dataset_path**: This specifies the path to the processed table file that will be used for retrieval. In this example, the path points to a JSONL file that contains the processed tables: `"data/processed/table_outputs/hybridqa_llm_based_filter_None_string.jsonl"`. This file should be the output of the table processing pipeline.
  
- **index_name**: The name of the index to be created or used for retrieval. This index is used to store the embeddings of the processed tables for efficient querying.
  
- **colbert_model_name**: Indicates the ColBERT model to be used for retrieval. In this case, `colbert-ir/colbertv2.0` is specified, which refers to a pre-trained ColBERT model suitable for information retrieval tasks.
  
- **base_output_dir**: The base directory where the retrieval results will be stored. The final results, along with any intermediate outputs, will be saved in this directory: `"data/processed/retrieval_results"`.
  
- **use_rerank**: A boolean flag that determines whether reranking should be applied to the retrieved results. If set to `True`, an additional reranking step will be performed on the top results.
  
- **top_k**: Specifies the number of top documents to retrieve from the index. For example, if `top_k=1`, only the single most relevant document will be retrieved.
  
- **rerank_top_k**: When reranking is enabled (`use_rerank=True`), this parameter specifies the number of top documents to rerank.
  
- **num_queries**: Indicates the number of queries to be processed during retrieval. For example, `num_queries=1000` means that up to 1000 queries will be executed.
  
- **query_grd_path**: Specifies the path to the query ground truth dataset, which is necessary for evaluating retrieval performance. The path provided, `"data/raw/small_dataset/hybridqa.jsonl"`, should point to the raw dataset in the `data/raw` directory.

### **Post-Retrieval Processing**

Once the retrieval pipeline is executed, the system will automatically calculate the recall accuracy based on the retrieved documents. The ground truth queries are used to evaluate the effectiveness of the retrieval process. The results and any necessary intermediate data will be stored in the `data/processed/retrieval_results` directory.

### **Progress Saving**

Similar to the table processing pipeline, this retrieval pipeline includes a progress-saving feature. If the process is interrupted, progress will be saved in the specified output directory (`data/processed/retrieval_results`), allowing you to resume the process from where it left off.

### **Generation and Evaluation**

```bash
python -m src.colbert_pipeline.generator
```
### **Generation and Evaluation Pipeline**

```bash
python -m src.colbert_pipeline.generator
```

In this pipeline, you configure and execute the generation and evaluation process based on the retrieval results. Here's a simplified explanation of the key parameters:

```python
retrieval_results_save_path = "data/processed/retrieval_results/sqa_llm_based_filter_term_explanations_and_table_summary_markdown_retrieval_results.jsonl"
dataset_name = "sqa"

# Execute with specific parameters
generate_and_evaluate(dataset_name, retrieval_results_save_path, remove_terms_explanation=True, remove_table_summary=True)
```

- **retrieval_results_save_path**: The path where the retrieval results are stored. Ensure the correct file is in place before running the pipeline.

- **dataset_name**: The name of the dataset being used, e.g., `"sqa"`.

- **remove_terms_explanation**: Set to `True` if you want to exclude term explanations from the final output.

- **remove_table_summary**: Set to `True` if you want to exclude table summaries from the final output.

### **Steps**

1. Place the retrieval results file in the specified path.
2. Set the appropriate dataset name.
3. Configure whether to include or exclude term explanations and table summaries.
4. Run the `generate_and_evaluate` function.


## **Evaluation Experiments**

### **Control Experiments**

We conducted extensive control experiments to validate the effectiveness of our enhancements. For each dataset, we compared the performance of the following system configurations:

1. **Baseline**: A standard RAG implementation without any enhancements.
2. **Enhanced System**: Our complete system with all proposed enhancements.

### **Ablation Experiments**

Ablation experiments were performed to assess the contribution of each component:

1. **Without Table Filtering**: Evaluates the system's performance without the table filtering module.
2. **Without LLM-Based Clarifier**: Measures the impact of removing the LLM-based clarifier.
3. **Without Retrieval Enhancement**: Tests the system's efficiency without the ColBERT integration.

## **Acknowledgments**

I would like to express my sincere gratitude to the authors of the paper [“Tap4llm: Table provider on sampling, augmenting, and packing semi-structured data for large language model reasoning”](https://arxiv.org/abs/2312.09039) for providing valuable insights that influenced some of the ideas presented in this article. I have also borrowed some of the code from this paper for data loading and other tasks, as noted at the beginning of the relevant scripts.

I would also like to thank the developers of [RAGatouille](https://github.com/AnswerDotAI/RAGatouille), as the ColBERT implementation used in this project was built upon their work.

Additionally, I would like to thank PeiMa from the University of Leeds for her significant contributions to this project. Her expertise and support were instrumental in shaping the outcome of this work.
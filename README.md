

# **TableRAG**

## **Overview**

TableRAG is an advanced system designed for handling complex table-based question answering tasks. The system combines cutting-edge large language models (LLMs) with Retrieval-Augmented Generation (RAG) techniques, optimized specifically for multi-table environments. This project addresses the challenges posed by real-world scenarios, where users may need to query a set of related tables rather than a single one, by integrating advanced filtering, clarification, and retrieval mechanisms.

## **Table of Contents**

- [Background](#background)
- [System Architecture](#system-architecture)
  - [RAG-based Multi-Table QA System](#rag-based-multi-table-qa-system)
  - [Enhancement Mechanisms](#enhancement-mechanisms)
- [Datasets](#datasets)
  - [TableFact](#tablefact)
  - [FEVEROUS](#feverous)
  - [SQA](#sqa)
  - [HybridQA](#hybridqa)
- [Implementation Details](#implementation-details)
  - [Table Filtering](#table-filtering)
  - [Table Clarifier](#table-clarifier)
  - [Retrieval Process Enhancement](#retrieval-process-enhancement)
  - [Input Format Optimization](#input-format-optimization)
  - [Self-Consistency](#self-consistency)
- [Pipeline Execution](#pipeline-execution)
  - [Table Processing Pipeline](#table-processing-pipeline)
  - [Retrieval Pipeline](#retrieval-pipeline)
  - [Generation and Evaluation](#generation-and-evaluation)
- [Evaluation Experiments](#evaluation-experiments)
  - [Control Experiments](#control-experiments)
  - [Ablation Experiments](#ablation-experiments)
- [Installation](#installation)
- [Usage](#usage)


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

### **1. Set Up a Virtual Environment**

You can set up a virtual environment using either `conda` (Anaconda) or `venv` (Python's built-in tool).

#### **Using Conda (Recommended):**

1. **Create a Conda Environment**:
   ```bash
   conda create -n tablerag python=3.10.14
   conda activate tablerag
   ```

2. **Install Dependencies from `environment.yml`**:
   ```bash
   conda env update --file environment.yml
   ```

#### **Using venv**:

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies from `requirements.txt`**:
   ```bash
   pip install -r requirements.txt
   ```

### **2. Clone the Repository**

```bash
git clone https://github.com/yourusername/TableRAG.git
cd TableRAG
```

### **3. Set Up Environment Variables**

Configure any necessary API keys and environment variables as required. This may include API keys for OpenAI, if you are using their models for filtering or generation tasks.

### **4. Version Information**

Ensure that your environment meets the following version requirements:

- **Python**: 3.10.14
- **LangChain**: 0.1.0
- **OpenAI**: 1.14.0
- **PyTorch**: 2.3.1 (with CUDA 12.1 support, if using GPU acceleration)
- **CUDA**: 12.1 (if applicable)


## **Installation**

To install and set up the TableRAG system:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/TableRAG.git
   cd TableRAG
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Configure any necessary API keys and environment variables as required.

## **Usage**

To use the TableRAG system:

1. **Prepare Your Dataset**:
   - Ensure your tables are formatted and stored as per the input requirements.

2. **Run the System**:
   - Use the provided scripts to run the system on your dataset.
   - Example:
     ```bash
     python run_tablerag.py --config config.yaml
     ```

3. **View Results**:
   - The generated answers and relevant table fragments will be stored in the output directory.


## **Pipeline Execution**

### **Table Processing Pipeline**

To process tables, execute the following pipeline:

```bash
python -m src.table_processor.table_processing_pipeline
```

This pipeline supports several parameters:

- `task_name`: The name of the task to be processed (e.g., "tabfact").
- `split`: The dataset split to be used (e.g., "validation").
- `table_filter_name`: The name of the table filtering method (e.g., "default").
- `table_clarifier_name`: The name of the table clarification method (e.g., "term_explanations_and_table_summary").
- `embedding_type`: The type of embedding used in processing (e.g., "text-embedding-3-large").
- `top_k`: The number of top rows or columns to select based on relevance.
- `save_jsonl`: Whether to save the output in JSONL format (default is True).
- `load_local_dataset`: Whether to load a local dataset (default is True).
- `experiment_name`: The name of the experiment being conducted.
- `use_sampled_table_for_augmentation`: Whether to use sampled tables for augmentation (default is False).
- `sample_size`: The number of samples to process (default is 1).
- `overwrite_existing`: Whether to overwrite existing results (default is False).
- `table_format`: The format to use for the tables (e.g., "markdown").
- `use_table_filter`: Whether to use table filtering (default is True).

Processed files will be saved in `data/processed/table_outputs`. If the process is interrupted, progress is saved in `data/progressing`, allowing you to resume from where it left off.

### **Retrieval Pipeline**

After table processing, execute the retrieval pipeline:

```bash
python src/colbert_pipeline/retriever.py
```

Modify the following parameters as needed:

- `dataset_path`: Path to the dataset used for embedding and retrieval.
- `index_name`: Name of the index to be created and used.
- `colbert_model_name`: Name of the ColBERT model to be used.
- `base_output_dir`: Directory to save the retrieval results (e.g., `data/processed/retrieval_results`).
- `use_rerank`: Whether to rerank the retrieved documents.
- `top

_k`: Number of top documents to retrieve.
- `rerank_top_k`: Number of top documents to rerank.
- `num_queries`: Number of queries to process.
- `query_grd_path`: Path to the query and ground truth file.

Retrieval results will be stored in the `data/processed/retrieval_results` directory. The system also supports saving progress, enabling you to resume the process from where it was last interrupted.

### **Generation and Evaluation**

Finally, run the generation and evaluation process:

```bash
python src/colbert_pipeline/generator.py
```

Parameters include:

- `dataset_name`: Name of the dataset being processed.
- `retrieval_results_save_path`: Path to the retrieval results JSONL file (e.g., `data/processed/retrieval_results`).
- `base_output_dir`: Base directory where output files will be saved (e.g., `data/processed/prediction`).
- `run_evaluation`: Whether to run evaluation after generation.
- `remove_terms_explanation`: Whether to remove terms explanation from input.
- `remove_table_summary`: Whether to remove table summary from input.

Generated predictions and evaluations will be stored in the `data/processed/prediction` directory. Like the previous steps, this process also saves progress and can be resumed if interrupted.

## **Evaluation Experiments**

### **Control Experiments**

We conducted control experiments to evaluate the impact of each enhancement module. We compared the performance of various configurations, including different filtering methods, clarifiers, formatters, and retrievers.

### **Ablation Experiments**

Ablation experiments were performed to assess the contribution of each module by selectively removing them from the system and observing the impact on performance.
# Acknowledgments
I would like to express my sincere gratitude to the authors of the paper “Tap4llm: Table provider on sampling, augmenting, and packing semi-structured data for large language model reasoning” for providing valuable insights that influenced some of the ideas presented in this article. 
Additionally, I would like to thank PeiMa from the University of Leeds for her significant contributions to this project. Her expertise and support were instrumental in shaping the outcome of this work.
import json
from ragatouille import RAGPretrainedModel
import warnings
warnings.filterwarnings("ignore")

class ColBERT:
    """
    Manages loading, embedding, indexing, and retrieving documents 
    using a pre-trained RAG model.
    """

    def __init__(self, jsonl_path: str, model_name: str, index_name: str):
        """
        Initialize with paths and model settings.
        """
        self.jsonl_path = jsonl_path
        self.model_name = model_name
        self.index_name = index_name
        self.RAG = RAGPretrainedModel.from_pretrained(model_name)

    def load_jsonl(self):
        """
        Load data from a JSONL file.
        
        :return: List of data entries.
        """
        with open(self.jsonl_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return data

    def embed_and_index(self):
        """
        Embed documents and index them using the RAG model.
        """
        data = self.load_jsonl()
        docs = [item['request'] for item in data]
        self.RAG.index(index_name=self.index_name, collection=docs, split_documents=False)

    def retrieve(self, query: str, top_k: int = 1, force_fast: bool = False, rerank: bool = False, rerank_top_k: int = 1):
        """
        Retrieve documents based on a query, with optional reranking.
        
        :return: List of relevant documents.
        """
        results = self.RAG.search(query, index_name=self.index_name, k=top_k, force_fast=force_fast)

        if rerank:
            documents = [result['content'] for result in results]
            reranked_results = self.RAG.rerank(query=query, documents=documents, k=rerank_top_k)
            return reranked_results
        
        return results

def colbert_pipeline(output_path: str, jsonl_query_path: str, jsonl_index_path: str, model_name: str, index_name: str, top_k: int = 1, force_fast: bool = False, rerank: bool = False, rerank_top_k: int = 1):
    """
    Pipeline to load data, embed, index, and retrieve documents.
    
    :return: None
    """
    # Initialize the ColBERT class with the data that needs to be indexed
    pipeline = ColBERT(jsonl_index_path, model_name, index_name)
    pipeline.embed_and_index()  # Embed and index the data from jsonl_index_path

    # Load queries from the specified jsonl_query_path
    with open(jsonl_query_path, 'r') as file:
        queries_data = [json.loads(line) for line in file]
    
    queries = [item['query'] for item in queries_data]
    correct_ids = list(range(len(queries)))  # Create a list of correct passage_ids based on query index

    # Open the output file and write each result as it's processed
    hits_at_k = 0
    total_queries = len(queries)
    
    # New file to store the incorrect retrievals
    error_output_path = "/home/yuhangwu/Desktop/Projects/tablerag/data/progressing/retriever_recording/e2ewtq.jsonl"
    
    with open(output_path, 'w') as file, open(error_output_path, 'w') as error_file:
        for i, query in enumerate(queries):
            retrieved_docs = pipeline.retrieve(query, top_k, force_fast, rerank, rerank_top_k)
            
            retrieved_ids = []
            for doc in retrieved_docs:
                # Update this line based on the actual structure of `doc`
                passage_id = doc.get('passage_id')  # Adjust if necessary
                if passage_id:
                    retrieved_ids.append(passage_id)
                
            # Check if the correct passage_id is in the retrieved results
            if correct_ids[i] in retrieved_ids:
                hits_at_k += 1
            else:
                # Save the incorrect retrievals
                error_info = {
                    "query": query,
                    "retrieved_docs": [doc['content'] for doc in retrieved_docs],  # Store the content of the incorrect docs
                    "correct_doc": queries_data[correct_ids[i]]['query']  # Correct doc content based on the correct ID
                }
                error_file.write(json.dumps(error_info) + '\n')

            # Save each result immediately after retrieval
            for result in retrieved_docs:
                file.write(json.dumps({
                    "id": i,  # Adding the id field
                    "query": query,
                    "result": result
                }) + '\n')
    
    # Calculate hits@k
    hits_at_k_ratio = hits_at_k / total_queries
    print(f"Hits@{top_k}: {hits_at_k_ratio * 100:.2f}%")

def main():
    # Configuration parameters
    jsonl_query_path = "/home/yuhangwu/Desktop/Projects/tablerag/data/raw/small_dataset/nqtables.jsonl"  # Queries
    jsonl_index_path = "/home/yuhangwu/Desktop/Projects/tablerag/data/processed/packed_data/packed_nqtables_test.jsonl"  # Data to be indexed and queried
    model_name = "colbert-ir/colbertv2.0"
    index_name = "my_index"
    output_path = "/home/yuhangwu/Desktop/Projects/tablerag/data/processed/retrieval_results/nqtables.jsonl"  # Save output to this file

    # Retrieve documents and save results immediately
    colbert_pipeline(
        output_path=output_path, 
        jsonl_query_path=jsonl_query_path, 
        jsonl_index_path=jsonl_index_path, 
        model_name=model_name, 
        index_name=index_name, 
        top_k=5, 
        force_fast=False, 
        rerank=False, 
        rerank_top_k=5
    )

if __name__ == "__main__":
    main()
    
"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

For any questions or further information, please feel free to reach out via the email address above.
"""


import json
from ragatouille import RAGPretrainedModel
import warnings
warnings.filterwarnings("ignore")

class ColBERT:
    def __init__(self, jsonl_path: str, model_name: str, index_name: str):
        self.jsonl_path = jsonl_path
        self.model_name = model_name
        self.index_name = index_name
        self.RAG = RAGPretrainedModel.from_pretrained(model_name)

    def load_jsonl(self):
        with open(self.jsonl_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return data

    def embed_and_index(self):
        data = self.load_jsonl()
        docs = [item['request'] for item in data]
        self.RAG.index(index_name=self.index_name, collection=docs, split_documents=False)

    def retrieve(self, query: str, top_k: int = 1, force_fast: bool = False, rerank: bool = False, rerank_top_k: int = 1):
        # 初步检索
        results = self.RAG.search(query, index_name=self.index_name, k=top_k, force_fast=force_fast)

        #for i, result in enumerate(results):
        #    print(f"Result {i+1}: {result}")

        # 如果启用重排序
        if rerank:
            # 从返回结果中提取文档内容
            documents = [result['content'] for result in results]
            # 重新排序
            reranked_results = self.RAG.rerank(query=query, documents=documents, k=rerank_top_k)
            return reranked_results
        
        return results

def colbert_pipeline(jsonl_path: str, model_name: str, index_name: str, queries: list, top_k: int = 1, force_fast: bool = False, rerank: bool = False, rerank_top_k: int = 1):
    pipeline = ColBERT(jsonl_path, model_name, index_name)
    pipeline.embed_and_index()
    responses = {}
    for query in queries:
        retrieved_docs = pipeline.retrieve(query, top_k, force_fast, rerank, rerank_top_k)
        responses[query] = retrieved_docs
    return responses

if __name__ == "__main__":
    # Example usage:
    jsonl_path = "/home/yuhangwu/Desktop/Projects/TableProcess/pipeline/data/Exp-240808/table_clarification/feverous_default_assemble_retrieval_based_augmentation_1.jsonl"
    model_name = "colbert-ir/colbertv2.0"
    index_name = "my_index"
    queries = ["the scheduled date for the farm with 17 turbine be 2012"]
    
    # 控制返回结果数量和重排序
    responses = colbert_pipeline(jsonl_path, model_name, index_name, queries, top_k=3, force_fast=False, rerank=False, rerank_top_k=1)
    print(responses)
import json
from ragatouille import RAGPretrainedModel

class RAGatouillePipeline:
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
        docs = [item['prompt'] for item in data]
        self.RAG.index(index_name=self.index_name, collection=docs)

    def retrieve(self, query: str, top_k: int = 1):
        results = self.RAG.search(query, index_name=self.index_name, k=top_k)
        return results

def ragatouille_pipeline(jsonl_path: str, model_name: str, index_name: str, queries: list, top_k: int = 1):
    pipeline = RAGatouillePipeline(jsonl_path, model_name, index_name)
    pipeline.embed_and_index()
    responses = {}
    for query in queries:
        retrieved_docs = pipeline.retrieve(query, top_k)
        responses[query] = retrieved_docs
    return responses

if __name__ == "__main__":
    # Example usage:
    jsonl_path = "pipeline/data/Exp-240730/table_augmentation/tabfact_default_term_explanations_1.jsonl"
    model_name = "colbert-ir/colbertv2.0"
    index_name = "my_index"
    queries = ["the scheduled date for the farm with 17 turbine be 2012"]
    responses = ragatouille_pipeline(jsonl_path, model_name, index_name, queries)
    print(responses)

import os, glob
from tqdm import tqdm
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import (
    SpacyEmbeddings,
    GPT4AllEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)

import warnings
warnings.filterwarnings("ignore")

class Embedder:
    def __init__(
        self,
        task_name: str,
        embedding_tag: str = "row_embeddings",
        embedding_type: str = "spacy",
    ):
        self.embedding_tag = embedding_tag
        self.embedding_type = embedding_type
        self.db = None
        self.embedder = None
        self.embedding_save_path = f"table_provider/agents/embedder/{self.embedding_type}/{self.embedding_tag}/{task_name}"

    def modify_embedding_tag(self, embedding_tag):
        self.embedding_tag = embedding_tag

    def call_embeddings(
        self,
        user_query: str,
        row_column_list: List[str],
        file_dir_name: str,
    ):
        if self.embedding_type == "text-embedding-ada-002":
            # generate column embeddings
            try:
                self.embedder = OpenAIEmbeddings(
                    model="text-embedding-ada-002", 
                    openai_api_base= "https://aigc.x-see.cn/v1"
                    )
                print("Successfully initialized OpenAIEmbeddings")
            except Exception as e:
                print(f"Error initializing OpenAIEmbeddings: {e}")
                raise
        elif self.embedding_type == "text-embedding-3-small":
            try:    
                self.embedder = OpenAIEmbeddings(
                    model="text-embedding-3-small", 
                    openai_api_base= "https://aigc.x-see.cn/v1"
                    )
                print("Successfully initialized OpenAIEmbeddings")
            except Exception as e:
                print(f"Error initializing OpenAIEmbeddings: {e}")
                raise
        elif self.embedding_type == "bge-large-en":
            # generate column embeddings
            self.embedder = HuggingFaceEmbeddings(
                model="BAAT/bge-small-en",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False},
            )
        elif self.embedding_type == "sentence-transformer":
            # generate column embeddings
            self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            raise ValueError(f"embedding_type {self.embedding_type} not supported")
        value_list_embeddings = self.embedder.embed_documents(row_column_list)
        user_query_embedding = self.embedder.embed_query(user_query)
        self.construct_vector_base(
            row_column_list=row_column_list,
            embeddings=value_list_embeddings,
            index_name=file_dir_name,
        )
        return value_list_embeddings, user_query_embedding

    def construct_vector_base(self, row_column_list, embeddings, index_name):
        # whether the folder exists
        if not (
            os.path.exists(f"{self.embedding_save_path}/{index_name}.pkl")
            or os.path.exists(f"{self.embedding_save_path}/{index_name}.faiss")
        ):
            text_embeddings = list(zip(row_column_list, embeddings))
            db = FAISS.from_embeddings(
                text_embeddings=text_embeddings, embedding=self.embedder
            )
            db.save_local(
                folder_path=self.embedding_save_path,
                index_name=index_name,
            )
        self.db = self.load_vector_base(index_name=index_name)

    def load_vector_base(self, index_name):
        db = FAISS.load_local(
            folder_path=self.embedding_save_path,
            embeddings=self.embedder,
            index_name=index_name,
        )
        return db

    def search_vector_base(self, query, k=4):
        return self.db.similarity_search(query, k)[0].page_content

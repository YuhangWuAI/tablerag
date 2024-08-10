"""
This code file contains functions that borrow certain logic from an anonymous repository associated with the paper:
"TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning" (arXiv:2312.09039).
Original source: https://anonymous.4open.science/r/TableProvider-4CC3/README.md (MIT License).
The repository does not list an author, but it is linked to the above paper.

Specifically, portions of the code related to data loading, data packing, and evaluation logic have been borrowed and integrated into this project.

Current author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

If you believe that any content in this file infringes your rights or if you have any concerns,
please contact me at the email address above.
"""
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
        if self.embedding_type == "text-embedding-3-small":
            # generate column embeddings
            try:
                self.embedder = OpenAIEmbeddings(
                    model="text-embedding-3-large", 
                    openai_api_base= "https://aigc.x-see.cn/v1"
                    )
                print("Successfully initialized OpenAIEmbeddings")
            except Exception as e:
                print(f"Error initializing OpenAIEmbeddings: {e}")
                raise
        elif self.embedding_type == "text-embedding-3-large":
            try:    
                self.embedder = OpenAIEmbeddings(
                    model="text-embedding-3-large", 
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

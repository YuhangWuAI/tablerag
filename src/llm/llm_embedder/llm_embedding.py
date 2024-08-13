"""
This code file contains functions that borrow certain logic from an anonymous repository associated with the paper:
"TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning" (arXiv:2312.09039).
Original source: https://anonymous.4open.science/r/TableProvider-4CC3/README.md.
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
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)

import warnings
warnings.filterwarnings("ignore")

class Embedder:
    """
    The Embedder class handles the embedding of text data using different embedding models.
    It supports saving and loading embeddings to/from a local vector database.
    """

    def __init__(
        self,
        task_name: str,
        embedding_tag: str = "row_embeddings",
        embedding_type: str = "spacy",
    ):
        """
        Initialize the Embedder with task-specific configurations.

        :param task_name: Name of the task for which embeddings are generated.
        :param embedding_tag: Tag to distinguish different embeddings.
        :param embedding_type: Type of embedding model to be used.
        """
        self.embedding_tag = embedding_tag
        self.embedding_type = embedding_type
        self.db = None
        self.embedder = None
        self.embedding_save_path = f"table_provider/agents/embedder/{self.embedding_type}/{self.embedding_tag}/{task_name}"

    def modify_embedding_tag(self, embedding_tag):
        """
        Modify the embedding tag used for distinguishing embeddings.

        :param embedding_tag: New embedding tag.
        """
        self.embedding_tag = embedding_tag

    def call_embeddings(
        self,
        user_query: str,
        row_column_list: List[str],
        file_dir_name: str,
    ):
        """
        Generate embeddings for the provided data and user query.

        :param user_query: The user's query string.
        :param row_column_list: List of row or column data to embed.
        :param file_dir_name: Directory name for saving the embeddings.
        :return: Tuple of (document embeddings, query embedding).
        """
        if self.embedding_type == "text-embedding-3-small":
            # Initialize OpenAIEmbeddings with a small model
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
            # Initialize OpenAIEmbeddings with a large model
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
            # Initialize HuggingFaceEmbeddings with the BGE model
            self.embedder = HuggingFaceEmbeddings(
                model="BAAT/bge-small-en",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False},
            )
        elif self.embedding_type == "sentence-transformer":
            # Initialize HuggingFaceEmbeddings with a sentence-transformer model
            self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            raise ValueError(f"embedding_type {self.embedding_type} not supported")

        # Generate embeddings for the documents and the user query
        value_list_embeddings = self.embedder.embed_documents(row_column_list)
        user_query_embedding = self.embedder.embed_query(user_query)
        
        # Construct the vector base using the generated embeddings
        self.construct_vector_base(
            row_column_list=row_column_list,
            embeddings=value_list_embeddings,
            index_name=file_dir_name,
        )
        
        return value_list_embeddings, user_query_embedding

    def construct_vector_base(self, row_column_list, embeddings, index_name):
        """
        Construct and save a vector database using the provided embeddings.

        :param row_column_list: List of row/column data.
        :param embeddings: List of embeddings corresponding to the row/column data.
        :param index_name: Name of the index for saving the vector database.
        """
        # Check if the vector database already exists
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
        """
        Load a vector database from the local storage.

        :param index_name: Name of the index to load.
        :return: Loaded FAISS database.
        """
        db = FAISS.load_local(
            folder_path=self.embedding_save_path,
            embeddings=self.embedder,
            index_name=index_name,
        )
        return db

    def search_vector_base(self, query, k=4):
        """
        Search the vector database for the most similar entries to the query.

        :param query: The query string to search for.
        :param k: Number of top results to return.
        :return: Content of the most similar page.
        """
        return self.db.similarity_search(query, k)[0].page_content

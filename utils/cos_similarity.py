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

import numpy as np

def select_top_k_samples(
    samples_embeddings: list[list[float]],
    user_query_embedding: list[float],
    k: int = 10,
) -> list[int]:
    """
    Return the top k samples that have the highest cosine similarity with the user query.

    Args:
        samples_embeddings: A list of embeddings (each embedding is a list of floats) for the samples.
        user_query_embedding: The embedding of the user query (a list of floats).
        k: The number of top similar samples to return (default is 10).

    Returns:
        A list of indices corresponding to the top k samples that are most similar to the user query.
    """
    # 计算用户查询与每个样本的余弦相似度
    similarity = np.dot(samples_embeddings, user_query_embedding) / (
        np.linalg.norm(samples_embeddings, axis=1) * np.linalg.norm(user_query_embedding)
    )
    
    # 返回最相似的前 k 个样本的索引
    return np.argsort(similarity)[-k:]

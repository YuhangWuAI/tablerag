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

import numpy as np

def select_top_k_samples(
    samples_embeddings: list[list[float]],
    user_query_embedding: list[float],
    k: int = 10,
) -> list[int]:
    """Return the top k samples that have the highest cosine similarity with the user query."""
    # Compute cosine similarity between the user query and each sample
    similarity = np.dot(samples_embeddings, user_query_embedding) / (
        np.linalg.norm(samples_embeddings, axis=1) * np.linalg.norm(user_query_embedding)
    )
    
    # Return the indices of the top k most similar samples
    return np.argsort(similarity)[-k:]

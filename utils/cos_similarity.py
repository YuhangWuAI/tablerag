import numpy as np


def cosine_similarity(
    rows_embeddings: list[list[float]], user_query_embedding: list[float]
) -> list[float]:
    """Return the cosine similarity between two embeddings."""
    return np.dot(rows_embeddings, user_query_embedding) / (
        np.linalg.norm(rows_embeddings) * np.linalg.norm(user_query_embedding)
    )


def select_top_k_samples(
    samples_embeddings: list[list[float]],
    user_query_embedding: list[float],
    k: int = 10,
) -> list[int]:
    """Return the top k samples that have the highest cosine similarity with the user query."""
    similarity = cosine_similarity(samples_embeddings, user_query_embedding)
    return np.argsort(similarity)[-k:]

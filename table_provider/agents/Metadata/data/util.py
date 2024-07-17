import json
import pickle

import numpy as np

from .config import DataConfig


def load_json(file_path: str, encoding: str):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def dump_json(file_path: str, obj, encoding: str):
    with open(file_path, "w", encoding=encoding) as f:
        return json.dump(obj, f)


def load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def dump_pickle(file_path, obj):
    with open(file_path, 'w') as f:
        return pickle.dump(obj, f)


def get_embeddings(uID: str, config: DataConfig, type="EMB"):
    if config.embed_format == "json":
        return load_json(config.embedding_path(uID), config.encoding)
    elif config.embed_format == "pickle":
        return load_pickle(config.embedding_path(uID))
    elif config.embed_format == "npy":
        return np.load(config.embedding_path(uID, type), allow_pickle=True)

from copy import copy
from typing import List, Optional

import numpy as np
import torch
from numpy import ndarray

from .config import DataConfig
from .token import Token, IsPercent, IsCurrency, HasYear, HasMonth, HasDay, Segment


def append_padding(seq: List, pad, final_length: int):
    return seq + [pad] * (final_length - len(seq))


def append_padding_np(seq: np.array, pad, final_length: int):
    if len(seq) < final_length:
        return np.concatenate((seq, np.array([pad] * (final_length - len(seq)))), axis=0)
    else:
        return seq


class Sequence:
    """A list of token, segment pair. DO NOT use this for representing states! (Use State instead.)"""
    __slots__ = 'tokens', 'segments', 'hash_value', 'current'

    def __init__(self, tokens: List[Token], segments: List[Segment]):
        if len(tokens) != len(segments):
            raise ValueError("Lengths of tokens and segments not matching!")
        self.tokens = tokens
        self.segments = segments
        self.hash_value = self._calc_hash_(tokens)

    @staticmethod
    def _calc_hash_(tokens: List[Token]):
        value = 487
        for t in tokens:
            value = value * 31 + hash(t)
        return value

    def _update_hash_(self, t: Token):
        self.hash_value = self.hash_value * 31 + hash(t)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Sequence):
            if len(self.tokens) == len(o.tokens):
                for i in range(len(o.tokens)):
                    if self.tokens[i] != o.tokens[i]:
                        return False
                return True
        return False

    def __hash__(self) -> int:
        return self.hash_value

    def __repr__(self):
        return "(" + " ".join(t.__repr__() for t in self.tokens) + ")"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item) -> Token:
        return self.tokens[item]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self) -> Token:
        if self.current >= len(self.tokens):
            raise StopIteration
        else:
            self.current += 1
            return self[self.current - 1]

    def __copy__(self):
        return Sequence(self.tokens[:], self.segments[:])

    @staticmethod
    def _compose_categories_(token: Token, config: DataConfig):
        categories = []
        # The adding order should be the same as config.cat_num order
        if config.use_field_type:
            categories.append(0 if token.field_type is None else int(token.field_type))
        if config.use_binary_tags:
            tags = tuple([None] * 5) if token.tags is None else token.tags
            categories.append(IsPercent.to_int(tags[0]))
            categories.append(IsCurrency.to_int(tags[1]))
            categories.append(HasYear.to_int(tags[2]))
            categories.append(HasMonth.to_int(tags[3]))
            categories.append(HasDay.to_int(tags[4]))
        return categories

    def to_dict(self, final_len: int, field_permutation: Optional[ndarray],
                field_indices: bool, fixed_order: bool, config: DataConfig):
        # TODO: memory efficient way to construct tensors
        if field_permutation is None:
            tokens = self.tokens
            indices = [-1 if token.field_index is None else token.field_index for token in tokens]
        else:  # field permutation should be applied
            indices = [-1] * len(self.tokens)
            if fixed_order:  # The field token order in state should be fixed
                tokens = self.tokens
                reverse = [0] * len(field_permutation)
                for idx, origin in enumerate(field_permutation):
                    reverse[origin] = idx
                for i in range(len(tokens)):
                    if tokens[i].field_index is not None:
                        indices[i] = reverse[tokens[i].field_index]
            else:  # The field token order in action space should be changed
                tokens = copy(self.tokens)
                for idx, origin in enumerate(field_permutation):
                    tokens[idx] = self.tokens[origin]
                    indices[idx] = idx

        result = {
            "segments": torch.tensor(append_padding([segment.value for segment in self.segments], 0, final_len),
                                     dtype=torch.long),
            "categories": torch.tensor(append_padding(
                [self._compose_categories_(token, config)
                 for token in tokens],
                config.empty_cat, final_len), dtype=torch.long),
            "semantic_embeds": 0 if len(config.empty_embed) == 0 else torch.tensor(append_padding(
                [config.empty_embed if token.semantic_embed is None else token.semantic_embed
                 for token in tokens],
                config.empty_embed, final_len), dtype=torch.float),
            "data_characters": 0 if len(config.empty_data) == 0 else torch.tensor(append_padding(
                [config.empty_data if token.data_features is None else token.data_features
                 for token in tokens],
                config.empty_data, final_len), dtype=torch.float),
            "mask": torch.tensor([1] * len(tokens) + [0] * (final_len - len(tokens)), dtype=torch.uint8)
        }
        if field_indices:
            result["field_indices"] = torch.tensor(append_padding(indices, -1, final_len), dtype=torch.long)
        return result

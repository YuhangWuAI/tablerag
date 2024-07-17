import os
from typing import Optional

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer


def to_device(data, device, non_blocking=False):
    if isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: to_device(value, device, non_blocking) for k, value in data.items()}
    else:
        return data


def unsqueeze(data):
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(0)
    elif isinstance(data, dict):
        return {k: unsqueeze(value) for k, value in data.items()}
    else:
        raise ValueError("Could not handle!")


def scores_from_confusion(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def save_states(epoch: int, states: dict, dir_path: str):
    """
    Saving the current states in dir_path
    See https://pytorch.org/tutorials/beginner/saving_loading_models.html
    :param epoch: current epoch number
    :param states: states dict to save
    :param dir_path: model output path
    :return: final_output_path
    """
    os.makedirs(dir_path, exist_ok=True)
    output_path = os.path.join(dir_path, "states_ep%d.pt" % epoch)
    rank = torch.cuda.current_device()
    if int(rank) == 0:
        torch.save(states, output_path)
    return output_path


def save_checkpoint(dir_path: str, epoch: int, module: Module,
                    optimizer: Optimizer = None, extra: Optional[dict] = None):
    states = {
        "epoch": epoch,
        "module_state": module.state_dict(),
    }
    if optimizer is not None:
        states["optim_state"] = optimizer.state_dict()
    if extra is not None:
        states["extra"] = extra

    return save_states(epoch, states, dir_path)

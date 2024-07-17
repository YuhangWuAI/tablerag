import logging
from datetime import datetime, timezone, timedelta
from enum import IntEnum
from os import path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn import metrics
from torch.nn.parallel import DistributedDataParallel
from transformers import TapasModel

from .metadata_data.table_fields import Metadata_Label
from ..model import FieldRecognizer, Metadata2, MetadataTapas
from ..model import get_tf_config, get_metadata2_config

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)

LABEL_MAP = {'OnlyDim': 0, 'OnlyMsr': 1, 'Both': 2, 'No-Label': 3}


class Metadata_Output(IntEnum):
    Msr_res = 0
    # Dim_res = 1
    Agg_score_res = 1
    Msr_score_res = 2
    Gby_score_res = 3
    Key_score_res = 4
    Msr_pair_res = 5
    Msr_type_res = 6


def clean_headers(header_string):
    """
    Clean the header string and return the string split.
    """
    header_words = []
    if '#' in header_string:
        header_words.append('#')
        header_string = header_string.replace('#', '')

    header_string = header_string.strip()
    header_string = header_string.lower()
    header_words.extend(header_string.split(' |_'))

    return header_words


def conf_matrix_cal(target, prediction, labels, valid_mask=None):
    """
    Given a pair of target and prediction numpy vectors, calculate the confusion matrix.

    :param target: the targets, a numpy vector
    :param prediction: the predictions, a numpy vector
    :param labels: a list of labels
    :param valid_mask: a numpy bool vector, with the same size as target, indicating which sample to be included in
                       the calculation of confusion matrix.
    :return conf_mat: the confusion matrix of the given target, prediction and labels. Note that if all target are
                      not valid, or no labels exists in target, then return n * n matrix filled with zero
                      where n is number of labels.
    """
    if len(target) != len(prediction):
        raise Exception("The length of target and prediction should be equal.")
    if valid_mask is not None:
        if len(valid_mask) != len(target):
            raise Exception("The length of mask and target should be equal.")
        valid_target = target[valid_mask]
        valid_prediction = prediction[valid_mask]
    else:
        valid_target = target
        valid_prediction = prediction
    valid_target_value = set(valid_target)
    label_value = set(labels)
    n_label = len(label_value)
    if len(valid_target_value.intersection(label_value)) == 0:
        return np.zeros((n_label, n_label), dtype=np.int)
    else:
        return metrics.confusion_matrix(valid_target, valid_prediction, labels=labels)


def build_model(args, data_config, device, rank=0):
    logger = logging.getLogger("Build-Model")
    if args.model_name == "tf":
        model_config = get_tf_config(
            data_config,
            args.model_size,
            args.num_layers,
            args.num_hidden,
            args.num_attention,
        )
        model = FieldRecognizer(model_config)
    elif args.model_name == "metadata2":
        model_config = get_metadata2_config(
            data_config,
            args.tf1_layers,
            args.tf2_layers,
            args.use_df,
            args.use_emb,
            args.df_subset,
        )
        if data_config.embed_model == "tapas-fine-tune":
            tapas_model = TapasModel.from_pretrained('google/tapas-base')
            model = Metadata2(model_config, tapas_model)
        else:
            model = Metadata2(model_config)
    elif args.model_name == "metadata_tapas":
        model_config = get_metadata2_config(
            data_config,
            args.tf1_layers,
            args.tf2_layers,
            args.use_df,
            args.use_emb,
            args.df_subset,
        )
        if data_config.embed_model == "tapas-fine-tune":
            tapas_model = TapasModel.from_pretrained('google/tapas-base')
            model = MetadataTapas(model_config, tapas_model)
        else:
            raise NotImplementedError(
                "Must fine tune tapas when use MetadataTapas model"
            )
    model.to(device)
    mode_prefix = args.mode.split('-')[0]
    if mode_prefix == "tune":
        # Build model and load parameters
        if not path.isfile(args.model_load_path):
            raise Exception("Invalid model load path.")
        logger.info('Loading model parameters...')
        checkpoint = torch.load(args.model_load_path)
        model.load_state_dict(checkpoint['module_state'])
        logger.info('Load successfully!')

        if args.mode == "tune-both" or args.mode == "tune-both-cross-valid":
            # Freeze parameter except both head
            # First freeze all parameters
            for para in model.parameters():
                para.requires_grad = False
            # Next set the "both" module's require grad to true
            for para in model.fc_both_task.parameters():
                para.requires_grad = True
        elif args.mode == "tune-sum":
            # Freeze parameter except sum head
            # First freeze all parameters
            for para in model.parameters():
                para.requires_grad = False
            # Next set the "sum" module's require grad to true
            for para in model.fc_sum_task.parameters():
                para.requires_grad = True
        elif args.mode == "tune-measure":
            # Freeze parameter except measure head
            # First freeze all parameters
            for para in model.parameters():
                para.requires_grad = False
            # Next set the "sum" module's require grad to true
            for para in model.fc_msr_task.parameters():
                para.requires_grad = True
        elif args.mode == "tune-score":
            # Freeze parameter except measure head
            # First freeze all parameters
            for para in model.parameters():
                para.requires_grad = False
            # Next set the "score" module's require grad to true
            for para in model.fc_agg_Task.parameters():
                para.requires_grad = True
            for para in model.fc_dim_score.parameters():
                para.requires_grad = True
            for para in model.fc_msr_score.parameters():
                para.requires_grad = True
        else:
            raise ValueError("Unknown tuning mode.")

    logger.info(
        "Model #parameters = {}".format(
            sum([param.nelement() for param in model.parameters()])
        )
    )
    logger.info("Model Config: {}".format(vars(model_config)))
    model_path = path.join(
        args.model_save_path,
        datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M-")
        + str(model_config),
    )
    if rank == 0:
        wandb.config["model_config"] = vars(model_config)
        wandb.config["model_path"] = model_path
    return (
        DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=True,
        ),
        model_path,
    )
    # return model, model_path


def generate_field_records(
    target, prediction, mask, pair_idx, pair_target, pair_mask
) -> (pd.DataFrame, pd.DataFrame):
    # Shape: batch_size, src_len
    label = target.clone().detach()

    # Shape: batch_size, src_len
    msr_pred = prediction[Metadata_Output.Msr_res].argmax(dim=-1).detach()
    agg_scoring = prediction[Metadata_Output.Agg_score_res].detach().argmax(dim=-1)
    gby_scoring = prediction[Metadata_Output.Gby_score_res][:, :, 1].detach()
    gby_scoring.masked_fill_(~mask, -1e9)
    _, gby_idx = gby_scoring.sort(dim=-1, descending=True)
    _, gby_rank = gby_idx.sort(dim=-1)

    msr_scoring = prediction[Metadata_Output.Msr_score_res][:, :, 1].detach()
    msr_scoring.masked_fill_(~mask, -1e9)
    _, msr_idx = msr_scoring.sort(dim=-1, descending=True)
    _, msr_rank = msr_idx.sort(dim=-1)

    key_scoring = prediction[Metadata_Output.Key_score_res][:, :, 1].detach()
    key_scoring.masked_fill_(~mask, -1e9)
    _, key_idx = key_scoring.sort(dim=-1, descending=True)
    _, key_rank = key_idx.sort(dim=-1)

    msr_type = prediction[Metadata_Output.Msr_type_res].detach().argmax(dim=-1)

    # Only keep the valid fields (excl. padding) for recording.
    label_valid = label[
        mask.unsqueeze(-1).expand(-1, -1, max(list(Metadata_Label)) + 1)
    ].view(-1, max(list(Metadata_Label)) + 1)
    msr_pred_valid = msr_pred[mask]
    agg_scoring_valid = agg_scoring[mask]
    gby_rank_valid = gby_rank[mask]
    msr_rank_valid = msr_rank[mask]
    key_rank_valid = key_rank[mask]
    msr_type_valid = msr_type[mask]

    # Transform to numpy array
    label_valid = label_valid.cpu().numpy()
    msr_pred_valid = msr_pred_valid.cpu().numpy()
    agg_scoring_valid = agg_scoring_valid.cpu().numpy()
    gby_rank_valid = gby_rank_valid.cpu().numpy()
    msr_rank_valid = msr_rank_valid.cpu().numpy()
    key_rank_valid = key_rank_valid.cpu().numpy()
    msr_type_valid = msr_type_valid.cpu().numpy()

    records = {}
    name = [
        "dimension label",
        "measure label",
        "common groupby label",
        "primary key label",
        "common measure label",
        "subCategory label",
        "category label",
        "aggregation label0",
        "aggregation label1",
        "aggregation label2",
        "aggregation label3",
        "aggregation label4",
        "aggregation label5",
        "aggregation label6",
        "aggregation label7",
        "aggregation label8",
    ]

    for i in range(max(list(Metadata_Label)) + 1):
        records[name[i]] = label_valid[:, i]
    records["MsrPred"] = msr_pred_valid
    records["GbyScoring"] = gby_rank_valid
    records["MsrScoring"] = msr_rank_valid
    records["KeyScoring"] = key_rank_valid
    records["MsrTypePred"] = msr_type_valid
    records["AggPred"] = agg_scoring_valid

    records = pd.DataFrame(records)

    # Msr pair
    # Shape: batch_size, src_len
    pair_label = pair_target.clone().detach()
    pair_pred = prediction[Metadata_Output.Msr_pair_res].argmax(dim=-1).detach()
    table_idx = (
        torch.arange(len(pair_label), device=pair_label.device)
        .unsqueeze(-1)
        .expand(pair_label.size())
    )

    # Shape: batch_size, src_len, 2
    pair_idx = pair_idx.clone().detach()

    # Only keep the valid fields (excl. padding) for recording.
    pair_label_valid = pair_label[pair_mask].cpu().numpy()
    pair_pred_valid = pair_pred[pair_mask].cpu().numpy()
    pair_idx0_valid = pair_idx[:, :, 0][pair_mask].cpu().numpy()
    pair_idx1_valid = pair_idx[:, :, 1][pair_mask].cpu().numpy()
    table_idx = table_idx[pair_mask].cpu().numpy()

    pair_record = {}
    pair_record["table_idx"] = table_idx
    pair_record["pair_field0_idx"] = pair_idx0_valid
    pair_record["pair_field1_idx"] = pair_idx1_valid
    pair_record["pair_label"] = pair_label_valid
    pair_record["pair_pred"] = pair_pred_valid

    pair_record = pd.DataFrame(pair_record)
    return records, pair_record

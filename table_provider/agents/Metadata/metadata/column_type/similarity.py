import argparse
import csv
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import pairwise
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Index, DEFAULT_LANGUAGES, DEFAULT_FEATURE_CHOICES, DEFAULT_ANALYSIS_TYPES, TRAIN_MODE
from helper import construct_data_config
from metadata.column_type.column_types import ColumnType
from metadata.metadata_data import MetadataTableFields
from model import get_tf_config, DEFAULT_MODEL_SIZES, DEFAULT_MODEL_NAMES, FieldRecognizer
from util import to_device

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def get_arguments():
    parser = argparse.ArgumentParser()

    # Model loading configurations
    parser.add_argument('-p', '--pre_model_file', type=str, metavar='PATH',
                        help='file path to a previously trained model (as the starting point)')

    # Model choose configurations
    parser.add_argument("--model_name", choices=DEFAULT_MODEL_NAMES, default="tf", type=str)

    # Data and BERT configurations
    parser.add_argument("--model_size", choices=DEFAULT_MODEL_SIZES, required=True, type=str)
    parser.add_argument('-m', "--model_save_path", default="/storage/models/", type=str)
    parser.add_argument('--features', choices=DEFAULT_FEATURE_CHOICES, default="metadata-mul_bert", type=str,
                        help="Limit the data loading and control the feature ablation.")
    parser.add_argument('-s', '--search_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, default="pivotTable",
                        help="Determine which data to load and what types of analysis to search.")
    parser.add_argument('--input_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, default='all',
                        help="Determine which data to load. This parameter is prior to --search_type.")
    parser.add_argument('--previous_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, default='all',
                        help="Tell the action space information of pre_model_file/model_file."
                             "Bar grouping should be the same as in --features.")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers (per GPU) for data loading.")
    parser.add_argument("--valid_batch_size", default=64, type=int, help="Batch size (per GPU) for validating.")
    parser.add_argument('--field_permutation', default=False, dest='field_permutation', action='store_true',
                        help="Whether to randomly permutate table fields when training.")
    parser.add_argument('--unified_ana_token', default=False, dest='unified_ana_token', action='store_true',
                        help="Whether to use unified analysis token [ANA] instead of concrete type tokens.")

    # MetaData parameters: model loading path
    parser.add_argument("--corpus_path", type=str, required=True, help="The corpus path for metadata task.")
    parser.add_argument("--model_load_path", type=str, required=True, help="The path to load metadata model")
    parser.add_argument("--mode", type=str, default="train-test", choices=TRAIN_MODE, help="The mode of training.")
    # Choose evaluation dataset
    parser.add_argument("--eval_dataset", type=str, required=True, choices=['test', 'valid', 'all', 'simple'],
                        default='test', help="The dataset to be evaluate.")
    # parser.add_argument("--table_name", default=None, type=str,
    #                     help="If eval_dataset is simple, can test the specific table.")
    # Whether required prediction records
    parser.add_argument("--require_records", type=int, default=0, help="Whether need prediction records, "
                                                                       "0 for no, 1 for yes")
    parser.add_argument("--require_df", default=False, action='store_true',
                        help="Whether keep data features of records")
    parser.add_argument("--eval_res_file", type=str, default="/data/statistics/")

    # Predict language
    parser.add_argument("--lang", "--specify_lang", choices=DEFAULT_LANGUAGES, default='en', type=str,
                        help="Specify the header language(s) to load tables.")

    parser.add_argument("--num_layers", type=int, default=2, help="Size of layers.")
    parser.add_argument("--num_hidden", type=int, default=96, help="Size of hidden layers.")
    parser.add_argument("--num_attention", type=int, default=8, help="Size of attention heads.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    logger = logging.getLogger("MetaDataSimilarity")
    logger.info(f"MetaDataTraining Args: {args}")
    data_config = construct_data_config(args, False, True)
    logger.info("DataConfig: {}".format(vars(data_config)))
    logger.info("Loading index...")
    index = Index(data_config)
    tuids = index.tUIDs

    logger.info("Loading data...")
    dataset = MetadataTableFields(tuids, data_config, False)
    data_loader = DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=dataset.collate)
    logger.info(f"Test set contains {len(dataset)} tables.")

    logger.info("Constructing transformer model...")
    model_config = get_tf_config(data_config, args.model_size, args.num_layers, args.num_hidden, args.num_attention)
    model = FieldRecognizer(model_config)
    logger.info("Transformer #parameters = {}".format(sum([param.nelement() for param in model.parameters()])))
    logger.info("TFConfig: {}".format(vars(model_config)))
    logger.info('Loading model parameters...')
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    model = model.to(device)
    checkpoint = torch.load(args.model_load_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['module_state'])
    logger.info('Load successfully!')

    column_labels = np.array([])
    column_types = np.array([])
    embedding = np.array([])
    predict = np.array([])
    model.eval()
    with torch.set_grad_enabled(False):
        for i, data in enumerate(data_loader):
            data = to_device(data, device, non_blocking=True)
            prediction_embedding, prediction_msr = model(data['inputs'], True)
            column_labels = np.append(column_labels, data['outputs']["labels"].detach().cpu().numpy())
            column_types = np.append(column_types, data['outputs']["types"].detach().cpu().numpy())
            embedding = np.append(embedding, prediction_embedding.detach().cpu().numpy())
            predict = np.append(predict, prediction_msr.detach().argmax(dim=-1).cpu().numpy())
    embedding = embedding.reshape((column_labels.shape[0], 128))
    print(column_labels.shape)
    print(embedding.shape)

    # Calculate distance
    similarity = [[0] * len(ColumnType) for _ in range(len(ColumnType))]
    labels_types = np.array([])
    labels_predict = np.array([])
    for i in tqdm(range(len(ColumnType))):
        labels_types = np.append(labels_types, ' '.join([str(type) for type in column_types[column_labels == i]]))
        labels_predict = np.append(labels_predict, ' '.join([str(label) for label in predict[column_labels == i]]))
        for j in range(i, len(ColumnType)):
            x1 = embedding[column_labels == i]
            x2 = embedding[column_labels == j]
            if len(x1) == 0 or len(x2) == 0:
                similarity_ij = 0
            else:
                similarity_ij = pairwise.pairwise_distances(x1, x2, metric='cosine').sum() / len(x1) / len(x2)
            similarity[i][j] = similarity_ij
            similarity[j][i] = similarity_ij

    # Figure
    logger.info("Begin to draw cluster map")
    plt.figure()
    a = sns.clustermap(pd.DataFrame(similarity), method='ward', metric='euclidean')
    print("index", list(a.data2d.axes[0]))
    print("column type", list(labels_types[list(a.data2d.axes[0])]))
    print("label name", {i: ColumnType[i][28:] for i in list(a.data2d.axes[0])})
    print("predict label", list(labels_predict[list(a.data2d.axes[0])]))
    # sns.heatmap(data=pd.DataFrame(similarity), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title('Distance')
    plt.savefig('Distance.png')
    logger.info("Successfully save figure")

    # save embedding
    logger.info("Begin to save embedding for T-sne")
    with open('embedding.tsv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        for i in embedding[column_labels != -1]:
            spamwriter.writerow(list(i))
    with open('label.tsv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(["label", "label_name", "type", "prediction"])
        for i, j, k in zip(column_labels[column_labels != -1], column_types[column_labels != -1],
                           predict[column_labels != -1]):
            spamwriter.writerow([int(i), ColumnType[int(i)], int(j), int(k)])

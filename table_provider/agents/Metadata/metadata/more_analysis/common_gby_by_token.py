import argparse
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TapasModel

import data.config as d_config
from data import Index, DEFAULT_LANGUAGES, DEFAULT_FEATURE_CHOICES, DEFAULT_ANALYSIS_TYPES, \
    TRAIN_MODE, ENTITY_TYPE, ENTITY_RECOGNITION
from data.util import load_json
from helper import construct_data_config
from metadata.metadata_data import MetadataTableFields, DEFAULT_CORPUS, get_corpus
from metadata.predict import predict
from model import get_metadata2_config, Metadata2, MetadataTapas
from model import get_tf_config, FieldRecognizer, DEFAULT_MODEL_SIZES, DEFAULT_MODEL_NAMES

tqdm.pandas(desc="progress")
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
    parser.add_argument("--model_load_path", type=str, required=True, help="The path to load metadata model")
    parser.add_argument("--mode", type=str, default="train-test", choices=TRAIN_MODE, help="The mode of training.")
    # Choose evaluation dataset
    parser.add_argument("--eval_dataset", type=str, required=True, choices=['test', 'valid', 'all', 'simple'],
                        default='test', help="The dataset to be evaluate.")
    # parser.add_argument("--table_name", default=None, type=str,
    #                     help="If eval_dataset is simple, can test the specific table.")

    parser.add_argument("--exp_name", type=str, default="", help="Experiment name for wandb")

    # Whether required prediction records
    parser.add_argument("--require_records", type=int, default=0, help="Whether need prediction records, "
                                                                       "0 for no, 1 for yes")
    parser.add_argument("--require_df", default=False, action='store_true',
                        help="Whether keep data features of records")
    parser.add_argument("--eval_res_file", type=str, default="/data/statistics/")
    parser.add_argument("--tsne", default=False, action='store_true',
                        help="Whether to tsne. ")

    # Predict language
    parser.add_argument("--lang", "--specify_lang", choices=DEFAULT_LANGUAGES, default='en', type=str,
                        help="Specify the header language(s) to load tables.")

    parser.add_argument("--num_layers", type=int, default=2, help="Size of layers.")
    parser.add_argument("--num_hidden", type=int, default=96, help="Size of hidden layers.")
    parser.add_argument("--num_attention", type=int, default=8, help="Size of attention heads.")

    # Whether to use data features and number features in the model
    parser.add_argument("--use_df", default=False, action='store_true',
                        help="Whether to use data features in the model. ")
    parser.add_argument("--use_num", default=False, action='store_true',
                        help="Whether to use number features in the model. ")
    parser.add_argument("--use_emb", default=False, action='store_true',
                        help="Whether to use embedding features in the model. ")
    parser.add_argument("--use_linear", default=False, action='store_true',
                        help="Whether to use linear layers in the model. ")
    parser.add_argument("--df_subset", type=int, nargs='*',
                        help="specify the subset of Data Features to use. ")
    parser.add_argument("--use_entity", default=False, action='store_true',
                        help="Whether to use entity in the model. ")
    parser.add_argument("--entity_type", choices=list(ENTITY_TYPE.keys()), default='transe100', type=str,
                        help="Specify the entity type.")
    parser.add_argument("--entity_emb_path", default='/storage1/Wikidata', type=str,
                        help="Entity embedding path.")
    parser.add_argument("--entity_recognition", choices=list(ENTITY_RECOGNITION.keys()), default='semtab', type=str,
                        help="Specify the entity recognition.")
    # Corpus
    parser.add_argument("--corpus", type=str, required=True, choices=DEFAULT_CORPUS,
                        help="The corpus for metadata task.")
    parser.add_argument("--chart_path", type=str, default="/storage/chart-202011/chart-202011",
                        help="The chart corpus path for metadata task, only take effect when --corpus == all")
    parser.add_argument("--pivot_path", type=str, default="/storage/pivot-20201029",
                        help="The pivot corpus path for metadata task, only take effect when --corpus == all")
    parser.add_argument("--vendor_path", type=str, default="/storage/vendor-202108",
                        help="The chart corpus path for metadata task, only take effect when --corpus == all")
    parser.add_argument("--t2d_path", type=str, default="/storage/t2d-202108",
                        help="The T2D corpus path for metadata task, only take effect when --corpus == all")
    parser.add_argument("--semtab_path", type=str, default="/storage/semtab_2110",
                        help="The semtab corpus path for metadata task, only take effect when --corpus == all")
    parser.add_argument("--corpus_path", type=str, default="/storage/chart-202011/chart-202011",
                        help="The corpus path for metadata task, only take effect when --corpus != all")

    # hyper-parameter for metadata2
    parser.add_argument("--tf1_layers", type=int, default=2, help="Layers for the first transformer")
    parser.add_argument("--tf2_layers", type=int, default=2, help="Layers for the second transformer")

    return parser.parse_args()


def draw_time_distribution(time, args, table_len):
    time.sort()
    time_distribution = [time[int(len(time) * i / 100)] for i in range(0, 100, 5)]
    time_distribution.append(time[-1])
    plt.figure()
    plt.plot(time_distribution, np.linspace(0, 1, 21))
    for x, y in zip(time_distribution, np.linspace(0, 1, 21)):
        plt.text(x, y, "%.2f" % x)

    plt.ylabel('Cumulative Distribution Probability')
    plt.xlabel('Cost time(s)')
    plt.savefig(args.eval_res_file + "time_distribution.jpg")

    plt.figure()

    plt.xlabel('time')
    plt.ylabel('table headers')
    plt.scatter(time, table_len)
    plt.savefig(args.eval_res_file + "time_headers.jpg")
    print(time_distribution)


# @profile
def run_predict(test_dataset, data_config, args, logger, wandb_log=False, dataset_name="all"):
    test_dataloader = DataLoader(test_dataset, batch_size=args.valid_batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=test_dataset.collate)
    logger.info(f"Test set contains {len(test_dataset)} tables.")

    logger.info("Constructing transformer model...")
    if args.model_name == "tf":
        model_config = get_tf_config(data_config, args.model_size, args.num_layers, args.num_hidden, args.num_attention)
        model = FieldRecognizer(model_config)
    elif args.model_name == "metadata2":
        model_config = get_metadata2_config(data_config, args.tf1_layers, args.tf2_layers,
                                            args.use_df, args.use_emb, args.df_subset)
        if data_config.embed_model == "tapas-fine-tune":
            tapas_model = TapasModel.from_pretrained('google/tapas-base')
            model = Metadata2(model_config, tapas_model)
        else:
            model = Metadata2(model_config)
    elif args.model_name == "metadata_tapas":
        model_config = get_metadata2_config(data_config, args.tf1_layers, args.tf2_layers,
                                            args.use_df, args.use_emb, args.df_subset)
        if data_config.embed_model == "tapas-fine-tune":
            tapas_model = TapasModel.from_pretrained('google/tapas-base')
            model = MetadataTapas(model_config, tapas_model)
        else:
            raise NotImplementedError("Must fine tune tapas when use MetadataTapas model")
    logger.info("Transformer #parameters = {}".format(sum([param.nelement() for param in model.parameters()])))
    logger.info("TFConfig: {}".format(vars(model_config)))
    if wandb_log:
        wandb.config["model_config"] = vars(model_config)
        wandb.config["model_path"] = args.model_load_path

    df_pred, pair_pred, accuracy, predict_time = predict(test_dataloader, model, args.model_load_path,
                                                         simple=(True if args.eval_dataset == 'simple' else False),
                                                         use_df=args.use_df, use_num=args.use_num, tsne=args.tsne,
                                                         embed_model=data_config.embed_model, dataset_name=dataset_name)

    token_len = [len(test_dataset.table_list[i].table_model["col_ids"]) for i in range(len(test_dataset.table_list))]
    row_num = [test_dataset.table_list[i].n_rows for i in range(len(test_dataset.table_list))]
    total_gby_res = accuracy["total_gby_res"]
    total_gby_res["token_len"] = token_len
    total_gby_res["n_rows"] = row_num
    print(len(token_len))
    print(len(total_gby_res['numerator_dim_R@1']))
    dataframe = pd.DataFrame(total_gby_res)
    dataframe.to_csv("total_gby_res.csv", index=False, sep=',')


if __name__ == "__main__":
    args = get_arguments()
    logger = logging.getLogger("MetaDataTraining")
    logger.info(f"MetaDataTraining Args: {args}")
    data_config = construct_data_config(args, False)
    if data_config.use_entity:
        d_config.ENTITY_EMBEDDING = data_config.get_entity_embedding(data_config.entity_path)
        d_config.ENTITY_MAP = data_config.get_entity_map(data_config.entity_map)
        d_config.RELATION_EMBEDDING = data_config.get_entity_embedding(data_config.relation_path)
        d_config.RELATION_MAP = data_config.get_entity_map(data_config.relation_map)
    logger.info("DataConfig: {}".format(vars(data_config)))
    # wandb init
    project = 'Metadata'
    entity = 'dki'
    group = 'debug'
    display_name = f'{args.model_name}-{args.features.split("-")[-1]}-entity{1 if data_config.use_entity else 0}-df{1 if data_config.use_data_features else 0}-{args.exp_name}'
    wandb.init(reinit=True, project=project, entity=entity,
               name=display_name, group=group, tags=["predict"])
    wandb.config["dlvm"] = 16
    wandb.config["data_config"] = vars(data_config)
    wandb.config["args"] = vars(args)

    corpus = get_corpus(args)
    test_dataset = None
    for dataset, _ in corpus.values():
        data_config = deepcopy(data_config)
        data_config.corpus_path = dataset

        logger.info("Loading index...")
        index = Index(data_config)
        if args.eval_dataset == 'test':
            eval_suids = load_json(data_config.test_path(), encoding=data_config.encoding)
        elif args.eval_dataset == 'valid':
            eval_suids = load_json(data_config.valid_path(), encoding=data_config.encoding)
        elif args.eval_dataset == 'simple':
            if args.table_name is None:
                eval_suids = load_json(data_config.test_path(), encoding=data_config.encoding)
                eval_suids = [eval_suids[0]]
            else:
                eval_suids = [args.table_name]
            args.valid_batch_size = 1
        else:
            eval_suids = load_json(data_config.index_path(), encoding=data_config.encoding)

        logger.info("Loading test data...")
        if test_dataset == None:
            test_dataset = MetadataTableFields(index.sUIDs2tUIDs(eval_suids), data_config, False)
        else:
            test_dataset.add(MetadataTableFields(index.sUIDs2tUIDs(eval_suids), data_config, False))

    run_predict(test_dataset, data_config, args, logger, True)
    wandb.log({"finish": True})
    wandb.finish()

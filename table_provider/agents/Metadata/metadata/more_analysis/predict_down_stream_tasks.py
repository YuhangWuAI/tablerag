import argparse
import logging
from collections import defaultdict
from copy import deepcopy

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TapasModel

import data.config as d_config
from data import Index, DEFAULT_LANGUAGES, DEFAULT_FEATURE_CHOICES, DEFAULT_ANALYSIS_TYPES, \
    TRAIN_MODE, ENTITY_TYPE, ENTITY_RECOGNITION
from data import SpecialTokens
from data.dataset import Chart, PivotTable
from data.token import AnaType
from data.util import load_json
from helper import construct_data_config
from metadata.metadata_data import MetadataTableFields, DEFAULT_CORPUS, get_corpus
from metadata.metadata_data.table_fields import VALID_AGG_INDEX
from metadata.predict import predict
from metadata.utils import Metadata_Output
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

    predictions, masks, _, _ = predict(test_dataloader, model, args.model_load_path,
                                       simple=False,
                                       use_df=args.use_df, use_num=args.use_num, tsne=args.tsne,
                                       embed_model=data_config.embed_model, dataset_name=dataset_name,
                                       return_prediction=True)

    agg_scoring = predictions[Metadata_Output.Agg_score_res].argmax(dim=-1)
    gby_scoring = predictions[Metadata_Output.Gby_score_res][:, :, 1].detach()
    gby_scoring.masked_fill_(~masks, -1e9)
    _, gby_idx = gby_scoring.sort(dim=-1, descending=True)
    _, gby_rank = gby_idx.sort(dim=-1)

    msr_scoring = predictions[Metadata_Output.Msr_score_res][:, :, 1].detach()
    msr_scoring.masked_fill_(~masks, -1e9)
    _, msr_idx = msr_scoring.sort(dim=-1, descending=True)
    _, msr_rank = msr_idx.sort(dim=-1)

    key_scoring = predictions[Metadata_Output.Key_score_res][:, :, 1].detach()
    key_scoring.masked_fill_(~masks, -1e9)
    _, key_idx = key_scoring.sort(dim=-1, descending=True)
    _, key_rank = key_idx.sort(dim=-1)

    chart_table_total = 0
    recall_field_selection = 0
    recall_field_selection1 = 0
    recall_type = defaultdict(int)

    pivot_table_total = 0
    recall_pivot = 0
    recall_pivot1 = 0
    recall_pivot_gby = 0
    recall_pivot_msr = 0
    recall_pivot_agg = 0

    for idx, table in tqdm(enumerate(test_dataset.table_list)):
        # print(f"New table~~~~~~~~chart, {recall_field_selection}, {recall_field_selection1}, {chart_table_total}~~~~"
        #       f"pivot, {recall_pivot}, {recall_pivot1}, {pivot_table_total}")
        chart_fields_major = [int(msr_rank[idx][0]), int(key_rank[idx][0])]
        chart_fields_scatter = [int(msr_rank[idx][0])]
        chart_fields_major.sort()
        chart_fields_scatter.sort()

        pivot_msr = (int(msr_rank[idx][0]), int(agg_scoring[idx][msr_rank[idx][0]]))
        pivot_gby = int(gby_rank[idx][0])

        tUid = table.tUID
        has_chart = False
        recall_flag = False
        for cuid in table.cUIDs:
            try:
                chart = Chart(cuid[0], cuid[1], table.idx2field, test_dataset.configs[test_dataset.corpus[idx]],
                              False)  # Fake chart type
                has_chart = True
            except:
                continue

            # print(cuid[1], [f.field_index for f in chart.x_fields],
            #       [v.field_index for v in chart.values],
            #       key_rank[idx][:int(sum(masks[idx]))],
            #       msr_rank[idx][:int(sum(masks[idx]))])

            if chart.type == AnaType.ScatterChart or chart.type == AnaType.LineChart:
                chart_fields = chart_fields_scatter
                true_fields = [f.field_index for f in chart.values]
                if chart_fields[0] in true_fields + [f.field_index for f in chart.x_fields] and recall_flag == False:
                    recall_field_selection1 += 1
                    recall_flag = True
                    recall_type[cuid[1]] += 1
                if true_fields == chart_fields:
                    recall_field_selection += 1
                    break

            else:
                chart_fields = chart_fields_major
                true_fields = [v.field_index for v in chart.values] + [f.field_index for f in chart.x_fields]
                true_fields.sort()
                if chart_fields[0] in true_fields and chart_fields[1] in true_fields and recall_flag == False:
                    recall_field_selection1 += 1
                    recall_flag = True
                    recall_type[cuid[1]] += 1
                if true_fields == chart_fields:
                    recall_field_selection += 1
                    break

        if has_chart:
            chart_table_total += 1

        recall_flag = False
        recall_flag_gby = False
        recall_flag_msr = False
        recall_flag_agg = False
        log = []
        if len(table.pUIDs) != 0:
            pivot_table_total += 1
        for puid in table.pUIDs:
            pivot = PivotTable(puid, table.idx2field,
                               SpecialTokens(test_dataset.configs[test_dataset.corpus[idx]], False),
                               test_dataset.configs[test_dataset.corpus[idx]])  # Fake chart type
            measures = [(m[0].field_index, VALID_AGG_INDEX[m[1].agg_func] if m[1].agg_func in VALID_AGG_INDEX else None)
                        for m in
                        pivot.measures]
            true_gby = [f.field_index for f in pivot.rows + pivot.columns]
            log.append([true_gby, pivot_gby, measures, pivot_msr])
            if pivot_gby in true_gby and recall_flag_gby == False:
                recall_pivot_gby += 1
                recall_flag_gby = True
            if pivot_msr[0] in [m[0] for m in measures] and recall_flag_msr == False:
                recall_pivot_msr += 1
                recall_flag_msr = True
            if pivot_msr in measures and recall_flag_agg == False:
                recall_pivot_agg += 1
                recall_flag_agg = True
            if pivot_msr in measures and pivot_gby in true_gby and recall_flag == False:
                recall_pivot1 += 1
                recall_flag = True
            if len(measures) == 1 and pivot_msr == measures[0] and len(true_gby) == 1 and pivot_gby == true_gby[0]:
                recall_pivot += 1
                break
        if not recall_flag:
            for l in log:
                print(l)
    print(
        f"Chart field selection: {recall_field_selection} / {chart_table_total} = {recall_field_selection / chart_table_total if chart_table_total != 0 else 0}")
    print(
        f"Chart field selection1: {recall_field_selection1} / {chart_table_total} = {recall_field_selection1 / chart_table_total if chart_table_total != 0 else 0}")

    print(
        f"Pivot: {recall_pivot} / {pivot_table_total} = {recall_pivot / pivot_table_total if pivot_table_total != 0 else 0}")
    print(
        f"Pivot1: {recall_pivot1} / {pivot_table_total} = {recall_pivot1 / pivot_table_total if pivot_table_total != 0 else 0}")

    print(f"Pivot {recall_pivot_gby}, {recall_pivot_msr}, {recall_pivot_agg}")

    print(recall_type)


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
               name=display_name, group=group, tags=["down_stream_tasks"])
    wandb.config["dlvm"] = 16
    wandb.config["data_config"] = vars(data_config)
    wandb.config["args"] = vars(args)

    corpus = get_corpus(args)
    test_dataset = None
    for dataset, _ in corpus.values():
        if "chart" not in dataset and "pivot" not in dataset:
            continue
        data_config = deepcopy(data_config)
        data_config.corpus_path = dataset

        logger.info("Loading index...")
        index = Index(data_config)
        eval_suids = load_json(data_config.test_path(), encoding=data_config.encoding)

        logger.info("Loading test data...")
        if test_dataset == None:
            test_dataset = MetadataTableFields(index.sUIDs2tUIDs(eval_suids), data_config, False)
        else:
            test_dataset.add(MetadataTableFields(index.sUIDs2tUIDs(eval_suids), data_config, False))

    run_predict(test_dataset, data_config, args, logger, True)
    wandb.log({"finish": True})
    wandb.finish()

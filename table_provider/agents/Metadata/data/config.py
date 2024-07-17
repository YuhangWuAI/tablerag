import math
from os import path

import numpy as np

from .token import AnaType, FieldType, IsPercent, IsCurrency, HasYear, HasMonth, HasDay

EMB_FORMAT = ["json", "pickle", "npy"]

EMBED_MODELS = [
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "bert-large-cased",
    "bert-base-multilingual-cased",
    "glove.6B.50d",
    "glove.6B.100d",
    "glove.6B.200d",
    "glove.6B.300d",
    "fasttext",
    "xlm-roberta-base",
    "tapas",
    "tapas-display",  # Generate tapas embedding from display string
    "tabbie",
    "tapas-fine-tune"
]

TABLE_MODELS = [
    "tapas",
    "tapas-display",
    "tabbie",
    "tapas-fine-tune"
]  # They have different storage format

MODEL_LANG = {
    "bert-base-uncased": "en",
    "bert-base-cased": "en",
    "bert-large-uncased": "en",
    "bert-large-cased": "en",
    "bert-base-multilingual-cased": "mul",
    "glove.6B.50d": "en",
    "glove.6B.100d": "en",
    "glove.6B.200d": "en",
    "glove.6B.300d": "en",
    "fasttext": "mul",
    "xlm-roberta-base": "mul",
    "tapas": "mul",
    "tapas-display": "mul",
    "tabbie": "mul",
    "tapas-fine-tune": "en"
}

EMBED_LEN = {
    "bert-base-uncased": 768,
    "bert-base-cased": 768,
    "bert-large-uncased": 1024,
    "bert-large-cased": 1024,
    "bert-base-multilingual-cased": 768,
    "glove.6B.50d": 50,
    "glove.6B.100d": 100,
    "glove.6B.200d": 200,
    "glove.6B.300d": 300,
    "fasttext": 50,
    "xlm-roberta-base": 768,
    "tapas": 768,
    "tapas-display": 768,
    "tabbie": 768,
    "tapas-fine-tune": 768
}

EMBED_LAYERS = {
    "bert-base-uncased": {-2, -12},
    "bert-base-cased": {-2, -12},
    "bert-large-uncased": {-2, -12},
    "bert-large-cased": {-2, -12},
    "bert-base-multilingual-cased": {-2, -12},
    "glove.6B.50d": {0},
    "glove.6B.100d": {0},
    "glove.6B.200d": {0},
    "glove.6B.300d": {0},
    "fasttext": {0},
    "xlm-roberta-base": {-2, -12},
}

DF_FEATURE_NUM = 31

FEATURE_MAP = {
    'aggrPercentFormatted': 1,
    'aggr01Ranged': 2,
    'aggr0100Ranged': 3,  # Proportion of values ranged in 0-100
    'aggrIntegers': 4,  # Proportion of integer values
    'aggrNegative': 5,  # Proportion of negative values
    # 'aggrBayesLikeSum': 6,  # Aggregated Bayesian feature
    # 'dmBayesLikeDimension': 7,  # Bayes feature for dimension and measure
    'commonPrefix': 6,  # Proportion of most common prefix digit
    'commonSuffix': 7,  # Proportion of most common suffix digit
    'keyEntropy': 8,  # Entropy by values
    'charEntropy': 9,  # Entropy by digits/chars
    'range': 10,  # Values range
    'changeRate': 11,  # Proportion of different adjacent values
    'partialOrdered': 12,  # Maximum proportion of increasing or decreasing adjacent values
    'variance': 13,  # Standard deviation
    'cov': 14,  # Coefficient of variation
    'cardinality': 15,  # Proportion of distinct values
    'spread': 16,  # Cardinality divided by range
    'major': 17,  # Proportion of the most frequent value
    'benford': 18,  # Distance of the first digit distribution to real-life average
    'orderedConfidence': 19,  # Indicator of sequentiality
    'equalProgressionConfidence': 20,  # confidence for a sequence to be equal progression
    'geometircProgressionConfidence': 21,  # confidence for a sequence to be geometric progression
    'medianLength': 22,  # median length of fields' records, 27.5 is 99% value
    'lengthVariance': 23,  # transformed length stdDev of a sequence
    'sumIn01': 24,
    'sumIn0100': 25,
    'absoluteCardinality': 26,
    'skewness': 27,
    'kurtosis': 38,
    'gini': 29,
    'nRows': 30,
    'averageLogLength': 31,

}
INDEX_FEATURE = {v: k for k, v in FEATURE_MAP.items()}

TYPE_MAP = {
    FieldType.Unknown: 0,
    FieldType.String: 1,
    FieldType.DateTime: 3,
    FieldType.Decimal: 5,
    FieldType.Year: 7
}


def cleanup_data_features_nn(data_features: dict):
    """
    Clean up data features that used in neural network models.
    Features like 'range', 'variance', 'cov', 'lengthStdDev' are in range [-inf, inf] or [0, inf].
    These features may cause problems in NN model and need to be normalized.
    We adopt normalization by distribution here.
    To take range as an example, this feature distributes in [0, inf]. We first square root this feature.
    Then examining the distribution (CDF) of the feature, we find that 99% square-rooted values less than 25528.5.
    Therefore, we normalize it by 25528.5. If the value is greater than this threshold (25528.5), they are set to 1.
    """
    # Normalize range, var and cov
    raw_range = data_features.get('range', 0.0)
    norm_range = 1 if isinstance(raw_range, str) else min(1.0, math.sqrt(raw_range) / 25528.5)
    raw_var = data_features.get('variance', 0.0)
    norm_var = 1 if isinstance(raw_var, str) else min(1.0, math.sqrt(raw_var) / 38791.2)
    raw_cov = data_features.get('cov', 0.0)
    if isinstance(raw_cov, str):
        norm_cov = 1
    else:
        norm_cov = min(1.0, math.sqrt(raw_cov) / 55.2) if raw_cov >= 0 else \
            max(-1.0, -1.0 * math.sqrt(abs(raw_cov)) / 633.9)
    # Use standard deviation rather than variance of feature 'lengthVariance'
    # 99% length stdDev of fields' records is less than 10
    lengthStdDev = min(1.0, math.sqrt(data_features.get('lengthVariance', 0.0)) / 10.0)

    # There are NAN or extremely large values in skewness and kurtosis, so we set:
    # skewness: NAN -> 0.0, INF/large values -> 1.0
    # kurtosis: NAN -> 0.0, INF/large values -> 1.0
    # skewness 99%ile = 3.844
    # kurtosis 99%ile = 0.7917 (no normalization)
    skewness_99ile = 3.844
    skewness = data_features.get('skewness', 0.0)
    if skewness == "NAN":
        skewness = 0.0
    elif isinstance(skewness, str) or abs(skewness) > skewness_99ile:
        skewness = skewness_99ile
    skewness = skewness / skewness_99ile

    kurtosis = data_features.get('kurtosis', 0.0)
    if kurtosis == "NAN":
        kurtosis = 0.0
    elif isinstance(kurtosis, str) or abs(kurtosis) > 1.0:
        kurtosis = 1.0

    gini = data_features.get('gini', 0.0)
    if gini == "NAN":
        gini = 0.0
    elif isinstance(gini, str) or abs(gini) > 1.0:
        gini = 1.0

    benford = data_features.get('benford', 0.0)
    if benford == "NAN":
        benford = 0.0
    elif isinstance(benford, str) or abs(benford) > 1.036061:
        benford = 1.036061

    features = [
        data_features.get('aggrPercentFormatted', 0),  # Proportion of cells having percent format
        data_features.get('aggr01Ranged', 0),  # Proportion of values ranged in 0-1
        data_features.get('aggr0100Ranged', 0),  # Proportion of values ranged in 0-100
        data_features.get('aggrIntegers', 0),  # Proportion of integer values
        data_features.get('aggrNegative', 0),  # Proportion of negative values
        # data_features.get('aggrBayesLikeSum', 0),  # Aggregated Bayes feature
        # data_features.get('dmBayesLikeDimension', 0),  # Bayes feature for dimension measure
        data_features['commonPrefix'],  # Proportion of most common prefix digit
        data_features['commonSuffix'],  # Proportion of most common suffix digit
        data_features['keyEntropy'],  # Entropy by values
        data_features['charEntropy'],  # Entropy by digits/chars
        norm_range,  # data_features.get('range', 0),  # Values range
        data_features['changeRate'],  # Proportion of different adjacent values
        data_features.get('partialOrdered', 0),  # Maximum proportion of increasing or decreasing adjacent values
        norm_var,  # data_features.get('variance', 0),  # Standard deviation
        norm_cov,  # data_features.get('cov', 0),  # Coefficient of variation
        data_features['cardinality'],  # Proportion of distinct values
        data_features.get('spread', 0),  # Cardinality divided by range
        data_features['major'],  # Proportion of the most frequent value
        benford,  # Distance of the first digit distribution to real-life average
        data_features.get('orderedConfidence', 0),  # Indicator of sequentiality
        data_features.get('equalProgressionConfidence', 0),  # confidence for a sequence to be equal progression
        data_features.get('geometircProgressionConfidence', 0),  # confidence for a sequence to be geometric progression
        min(1, data_features.get('medianLength', 0) / 27.5),  # median length of fields' records, 27.5 is 99% value
        lengthStdDev,  # transformed length stdDev of a sequence
        data_features.get('sumIn01', 0.0),  # Sum the values when they are ranged 0-1
        data_features.get('sumIn0100', 0.0) / 100,  # Sum the values when they are ranged 0-100
        min(1, data_features.get('absoluteCardinality', 0.0) / 344),  # Absolute Cardinality, 344 is 99% value
        skewness,
        kurtosis,
        gini,
        data_features.get('nRows', 0.0) / 576,  # Number of rows, 576 is 99% value
        data_features.get('averageLogLength', 0.0),
    ]
    for i, f in enumerate(features):
        if isinstance(f, str) or abs(f) > 10000:
            print("WARNING: feature[{}] is {}".format(i, f))
    return [0 if isinstance(f, str) else f for f in features]


ENTITY_TYPE = {
    "transe100": {"len": 100,
                  "entity_path": "embeddings/dimension_100/transe/entity2vec.bin",
                  "relation_path": "embeddings/dimension_100/transe/relation2vec.bin",
                  "entity_map": "knowledge_graphs/entity2id.txt",
                  "relation_map": "knowledge_graphs/relation2id.txt"},
    "transe50": {"len": 50,
                 "entity_path": "embeddings/dimension_50/transe/entity2vec.bin",
                 "relation_path": "embeddings/dimension_50/transe/relation2vec.bin",
                 "entity_map": "knowledge_graphs/entity2id.txt",
                 "relation_map": "knowledge_graphs/relation2id.txt"
                 },
}

ENTITY_RECOGNITION = {"semtab": "CEA_labels",
                      "mtab": "mtab/labels"}

ENTITY_EMBEDDING = None
ENTITY_MAP = None
RELATION_EMBEDDING = None
RELATION_MAP = None


class DataConfig:
    """Data configurations to specify data loading and representation formats"""

    def __init__(self, corpus_path: str = "/storage/chart-20200830/", encoding: str = "utf-8-sig",
                 max_field_num: int = 128, max_val_num: int = 4, max_dim_num: int = 4, use_field_type: bool = True,
                 use_binary_tags: bool = True, use_data_features: bool = True, use_semantic_embeds: bool = True,
                 embed_model: str = "bert-base-multilingual-cased", embed_format: str = "pickle", embed_layer: int = -2,
                 embed_reduce_type: str = "mean", train_ratio: float = 0.7, valid_ratio: float = 0.1,
                 test_ratio: float = 0.2, empirical_study: bool = False, lang: str = 'en', col_type_similarity=False,
                 df_subset: list = [0], use_entity=False, entity_type: str = "transe100",
                 entity_recognition: str = "semtab", entity_emb_path: str = ""):

        """
        :param corpus_path: the root dir of the corpus
        :param encoding: default json encoding
        :param max_field_num: the max number of fields allowed in a table, only has effect when allow_multiple_values
        :param max_val_num: the max number of values/series in a chart, only useful when allow_multiple_values
        :param max_dim_num: the max number of field as column/row dimension in a pivot table.
        :param use_field_type: if FieldType is used as a categorical feature of a token
        :param use_data_features: if data characteristics vector is used as part of a token
        :param use_semantic_embeds: if semantic header embedding vector is used as part of a token
        :param embed_model: which type of header embedding to adopt
        :param embed_format: "json" or "pickle"
        :param embed_layer: choose a layer from EMBED_LAYERS wrt embed_model
        :param embed_reduce_type: "mean" or "max"
        :param lang: only keep the tables with headers in the specified language(s).
        :param col_type_similarity: Whether to use column types as labels,
        """
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.lang = lang
        self.english_only = (self.lang == "en")

        self.max_field_num = max_field_num
        self.max_val_num = max_val_num
        self.max_dim_num = max_dim_num

        self.use_field_type = use_field_type
        self.use_binary_tags = use_binary_tags
        self.use_data_features = use_data_features
        self.use_semantic_embeds = use_semantic_embeds
        self.use_entity = use_entity

        self.entity_len = ENTITY_TYPE[entity_type]["len"]
        self.entity_path = path.join(entity_emb_path, ENTITY_TYPE[entity_type]["entity_path"])
        self.relation_path = path.join(entity_emb_path, ENTITY_TYPE[entity_type]["relation_path"])
        self.entity_map = path.join(entity_emb_path, ENTITY_TYPE[entity_type]["entity_map"])
        self.relation_map = path.join(entity_emb_path, ENTITY_TYPE[entity_type]["relation_map"])
        self.entity_recognition = entity_recognition

        self.col_type_similarity = col_type_similarity

        if embed_model not in EMBED_MODELS:
            raise ValueError("{} is not a valid model name.".format(embed_model))
        self.embed_model = embed_model
        model_lang = MODEL_LANG[embed_model]
        if model_lang != "mul" and not self.english_only:
            raise ValueError("Model language is {} while english_only = {}".format(model_lang, self.english_only))
        self.embed_len = EMBED_LEN[embed_model] if use_semantic_embeds else 0

        if embed_format in EMB_FORMAT:
            self.embed_format = embed_format
        else:
            raise ValueError("Embedding format {} is unrecognizable.".format(embed_format))

        if embed_model not in TABLE_MODELS and embed_layer in EMBED_LAYERS[embed_model]:
            self.embed_layer = str(embed_layer) if "fast" in embed_model else embed_layer
        elif embed_model not in TABLE_MODELS:
            raise ValueError("Embedding layer {} not available for model {}".format(embed_layer, embed_model))

        if embed_reduce_type == "mean" or embed_reduce_type == "max":
            self.embed_reduce_type = embed_reduce_type
        else:
            raise ValueError("Embedding type {} is unrecognizable.".format(embed_reduce_type))

        self.data_len = DF_FEATURE_NUM if use_data_features else 0
        self.df_subset = df_subset
        self.data_cleanup_fn = cleanup_data_features_nn

        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.empirical_study = empirical_study

        self.cat_nums = []  # Notice: The order in categories impacts sequence.to_dict() !!
        if use_field_type:
            self.cat_nums.append(FieldType.cat_num())
        if use_binary_tags:
            self.cat_nums.append(IsPercent.cat_num())
            self.cat_nums.append(IsCurrency.cat_num())
            self.cat_nums.append(HasYear.cat_num())
            self.cat_nums.append(HasMonth.cat_num())
            self.cat_nums.append(HasDay.cat_num())

        self.cat_len = len(self.cat_nums)

        self.empty_embed = [0.] * self.embed_len
        self.empty_cat = [0] * self.cat_len
        self.empty_data = [0.] * self.data_len

    def has_language(self, language: str):
        if self.lang == "mul" or language == self.lang:
            return True
        if self.lang == "t1":
            for lang in T1_LANGUAGES:
                if language.startswith(lang):
                    return True
        if self.lang == "t2":
            for lang in T2_LANGUAGES:
                if language.startswith(lang):
                    return True
        return False

    def index_path(self):
        # return path.join(self.corpus_path, "index", "merged-unique.json")
        return path.join(self.corpus_path, "index", "schema_ids.json")

    def train_path(self):
        return path.join(self.corpus_path, "index", "train_suid.json")

    def valid_path(self):
        return path.join(self.corpus_path, "index", "valid_suid.json")

    def test_path(self):
        return path.join(self.corpus_path, "index", "test_suid.json")

    def sample_path(self, sUID: str):
        return path.join(self.corpus_path, "sample-new", f"{sUID}.sample.json")

    def file_info_path(self, fUID: str):
        return path.join(self.corpus_path, "data", f"{fUID}.json")

    def table_path(self, tUID: str):
        return path.join(self.corpus_path, "data", f"{tUID}.DF.json")

    def vdr_table_path(self, tUID: str):
        return path.join(self.corpus_path, "tables", f"{tUID}.table.json")

    def mutual_info_path(self, tUID: str):
        return path.join(self.corpus_path, "data", f"{tUID}.MI.json")

    def embedding_path(self, uID: str, type="EMB"):
        if self.embed_format == "json":
            return path.join(self.corpus_path, "embeddings", self.embed_model, f"{uID}.EMB.json")
        elif self.embed_format == "pickle":
            return path.join(self.corpus_path, "embeddings", self.embed_model, f"{uID}.pickle")
        elif self.embed_format == "npy" and type != None:
            return path.join(self.corpus_path, "embeddings", self.embed_model, f"{uID}.{type}.npy")
        elif self.embed_format == "npy":
            return path.join(self.corpus_path, "embeddings", self.embed_model, f"{uID}.npy")

    def pivot_table_path(self, pUID: str):
        return path.join(self.corpus_path, "data", f"{pUID}.json")

    def chart_path(self, cUID: str):
        return path.join(self.corpus_path, "data", f"{cUID}.json")

    def column_type_path(self, tUID: str):
        return path.join(self.corpus_path, "column_type", f"{tUID}.csv")

    def semtab_label_path(self, sUID: str):
        return path.join(self.corpus_path, "labels", f"{sUID}.json")

    def number_feature_path(self, tUID: str):
        return path.join(self.corpus_path, "number_features", f"{tUID}.NF.json")

    def field_number_feature_path(self, tUID: str):
        return path.join(self.corpus_path, "new_df_v3", f"{tUID}.newDF.npy")

    def entity_recognition_path(self, tUID: str):
        return path.join(self.corpus_path, ENTITY_RECOGNITION[self.entity_recognition], f"{tUID}.json")

    def get_entity_embedding(self, path):
        embedding = np.memmap(path, dtype='float32', mode='r')
        embedding = np.concatenate((np.zeros((1, self.entity_len)), embedding.reshape((-1, self.entity_len))),
                                   axis=0)  # consider 0 as padding
        return embedding

    def get_entity_map(self, path):
        map = {}
        with open(path, "r") as f:
            f.readline()
            for line in f.readlines():
                line = line.split()
                map[line[0]] = int(line[1])
        return map


# NOTICE: value-first (reverse) is now the only supported format.

# Languages specifications (t1 = tier 1 languages, mul = all languages)
T1_LANGUAGES = ["ja", "zh", "de", "fr", "es", "en"]
T2_LANGUAGES = ["nl", "it", "ko", "pt", "ru", "tr"]
DEFAULT_LANGUAGES = T1_LANGUAGES + T2_LANGUAGES + ["t1", "t2", "mul"]

# Dataset for metadata
DATASET = ["pvt", "vdr"]

# Feature choices "ablation-embedding[-(single|x_group)]"
DEFAULT_FEATURE_CHOICES = ["metadata-mul_bert", "metadata-fast-single", "metadata-turing",
                           "metadata-tapas", "metadata-tapas_display", "metadata-tabbie", "metadata-tapas_tune"]

# Analysis types for Table2Analysis/Charts
DEFAULT_ANALYSIS_TYPES = [AnaType.to_raw_str(ana_type) for ana_type in AnaType] + ["all", "allCharts"]

# Train modes for Metadata tasks
# "train-test" means that dataset split is train:valid:test = 8:0:2, while the first 8 modes are 7:1:2. ("k-fold" is 8:0:2)
TRAIN_MODE = ["general", "tune-both", "tune-sum", "tune-both-cross-valid", "tune-measure", "tune-score", "only-sum",
              "only-measure", "k-fold", "train-test",
              "+loss0", "+loss1", "+loss2", "+loss3", "+loss4", "+loss5", "+loss6",
              "-loss0", "-loss1", "-loss2", "-loss3", "-loss4",
              "dump_split_index"]

MODE2LOSS = {
    "general": [0, 1, 2, 3, 4, 5, 6], "tune-both": [0], "tune-sum": [1], "tune-both-cross-valid": [0],
    "tune-measure": [0],
    "tune-score": [2, 3, 4], "only-sum": [1],
    "only-measure": [0], "k-fold": [0, 1, 2, 3, 4, 5, 6], "train-test": [0, 1, 2, 3, 4, 5, 6],
    "+loss0": [0], "+loss1": [1], "+loss2": [2], "+loss3": [3], "+loss4": [4], "+loss5": [5], "+loss6": [6],
    "-loss0": [1, 2, 3, 4, 5, 6], "-loss1": [0, 2, 3, 4, 5, 6], "-loss2": [0, 1, 3, 4, 5, 6],
    "-loss3": [0, 1, 2, 4, 5, 6], "-loss4": [0, 1, 2, 3, 5, 6]
}


def get_data_config(corpus_path: str, constraint: str, lang: str = 'en', mode: str = None,
                    empirical_study: bool = False, col_type_similarity=False,
                    df_subset: list = [0],
                    use_entity=False, entity_type: str = "transe100", entity_recognition: str = "semtab",
                    entity_emb_path: str = "", use_emb=None, use_df=None):
    # TODO: put unified_ana_token and field_permutation into constraint str
    configs = constraint.split('-')

    if configs[0] == 'metadata':
        use_data_features = True
        use_semantic_embeds = True
        use_field_type = True
        use_binary_tags = True
    else:
        raise NotImplementedError(f"Data config for {configs[0]} not yet implemented.")

    if use_df == False:
        use_data_features = False
    if use_emb == False:
        use_semantic_embeds = False

    if len(configs) == 1:
        configs.append('fast')
    if configs[1] == 'en_bert':
        embed_model = "bert-base-uncased"
        embed_format = "pickle"
        embed_layer = -2
        embed_reduce_type = "mean"
    elif configs[1] == 'mul_bert':
        embed_model = "bert-base-multilingual-cased"
        embed_format = "pickle"
        embed_layer = -2
        embed_reduce_type = "mean"
    elif configs[1] == 'glove':
        embed_model = "glove.6B.300d"
        embed_format = "pickle"
        embed_layer = 0
        embed_reduce_type = "mean"
    elif configs[1] == 'fast':
        embed_model = "fasttext"
        embed_format = "json"
        embed_layer = 0
        embed_reduce_type = "mean"
    elif configs[1] == "turing":
        embed_model = "xlm-roberta-base"
        embed_format = "pickle"
        embed_layer = -2
        embed_reduce_type = "mean"
    elif configs[1] == "tapas":
        embed_model = "tapas"
        embed_format = "npy"
        embed_layer = -2  # fake
        embed_reduce_type = "mean"  # fake
    elif configs[1] == "tapas_display":
        embed_model = "tapas-display"
        embed_format = "npy"
        embed_layer = -2  # fake
        embed_reduce_type = "mean"  # fake
    elif configs[1] == "tabbie":
        embed_model = "tabbie"
        embed_format = "npy"
        embed_layer = -2  # fake
        embed_reduce_type = "mean"  # fake
    elif configs[1] == "tapas_tune":
        embed_model = "tapas-fine-tune"
        embed_format = "npy"
        embed_layer = -2  # fake
        embed_reduce_type = "mean"  # fake
    else:
        raise NotImplementedError(f"Data config for {configs[1]} not yet implemented.")

    if mode in TRAIN_MODE[8:10]:
        train_ratio = 0.8
        valid_ratio = 0.0
        test_ratio = 0.2
    else:
        train_ratio = 0.7
        valid_ratio = 0.1
        test_ratio = 0.2

    return DataConfig(corpus_path=corpus_path, use_field_type=use_field_type,
                      use_binary_tags=use_binary_tags,
                      use_data_features=use_data_features, use_semantic_embeds=use_semantic_embeds,
                      embed_model=embed_model, embed_format=embed_format,
                      embed_layer=embed_layer, embed_reduce_type=embed_reduce_type,
                      lang=lang,
                      train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio,
                      empirical_study=empirical_study,
                      col_type_similarity=col_type_similarity, df_subset=df_subset, use_entity=use_entity,
                      entity_type=entity_type, entity_recognition=entity_recognition, entity_emb_path=entity_emb_path)

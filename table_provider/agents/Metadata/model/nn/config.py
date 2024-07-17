from typing import Optional

from ...data import DataConfig
from ...data.config import TABLE_MODELS


class ModelConfig:
    """Hyper-parameters for the Input Embedding part"""

    def __init__(
        self,
        data_config: DataConfig,
        hidden_size: int = 128,
        dropout: float = 0.1,
        position_sensitive: bool = True,
    ):
        self.embed_len = data_config.embed_len
        self.data_len = data_config.data_len
        self.cat_len = len(data_config.cat_nums)
        self.positional = position_sensitive
        self.num_categories = data_config.cat_nums

        self.hidden = hidden_size
        self.dropout = dropout


class TransformerConfig(ModelConfig):
    """Hyper-parameters for the Transformer neural net architecture"""

    def __init__(
        self,
        data_config: DataConfig,
        layers: int = 3,
        hidden_size: int = 128,
        attention_heads: int = 8,
        feed_forward_hidden: int = 256,
        dropout: float = 0.1,
        position_sensitive: bool = True,
    ):
        self.layers = layers
        self.attn_heads = attention_heads
        self.ff_hidden = feed_forward_hidden

        super().__init__(
            data_config,
            hidden_size=hidden_size,
            dropout=dropout,
            position_sensitive=position_sensitive,
        )

    def __str__(self):
        return "{}l{}h{}a{}ff".format(
            self.layers, self.hidden, self.attn_heads, self.ff_hidden
        )


class Metadata2Config(ModelConfig):
    """Hyper-parameters for the Transformer neural net architecture"""

    def __init__(
        self,
        data_config: DataConfig,
        hidden_size: int = 128,
        dropout: float = 0.1,
        position_sensitive: bool = True,
        use_df: bool = True,
        use_emb=True,
        tf1_layers=2,
        tf2_layers=2,
        df_subset: list = [0],
    ):
        self.use_entity = data_config.use_entity
        self.entity_len = data_config.entity_len
        self.entity_path = data_config.entity_path
        self.entity_hidden = 64

        # To make sure hidden size in transformer is a multiple of the number of attention heads (8)
        self.emb_hidden1 = 192
        encoder1_hidden = (
            self.emb_hidden1
            if not self.use_entity
            else self.emb_hidden1 + self.entity_hidden
        )
        self.encoder1_config = get_tf_config(
            data_config, "customize", tf1_layers, encoder1_hidden, 8
        )
        self.emb_hidden2 = (
            hidden_size
            if not use_df
            else int((hidden_size + data_config.data_len) / 8) * 8
        )
        encoder2_hidden = (
            self.emb_hidden2
            if not self.use_entity
            else self.emb_hidden2 + self.entity_hidden
        )
        self.encoder2_config = get_tf_config(
            data_config, "customize", tf2_layers, encoder2_hidden, 8
        )
        self.use_df = use_df
        self.use_emb = use_emb
        self.use_table_model = data_config.embed_model in TABLE_MODELS
        self.df_subset = df_subset
        super().__init__(
            data_config,
            hidden_size=encoder2_hidden,
            dropout=dropout,
            position_sensitive=position_sensitive,
        )

    def __str__(self):
        return "{}l{}h{}a{}ff".format(
            self.encoder2_config.layers,
            self.hidden,
            self.encoder2_config.attn_heads,
            self.encoder2_config.ff_hidden,
        )


DEFAULT_MODEL_SIZES = [
    "small",
    "medium",
    "large",
    "super",
    "resize_small",
    "shallow_large",
    "shallower_large",
    "shallowest_large",
    "customize",
]
DEFAULT_MODEL_NAMES = ["tf", "cp", "metadata2", "metadata_tapas"]


def small_tf_config(data_config: DataConfig, position_sensitive: bool = True):
    return TransformerConfig(
        data_config,
        layers=4,
        hidden_size=96,
        attention_heads=8,
        feed_forward_hidden=192,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def medium_tf_config(data_config: DataConfig, position_sensitive: bool = True):
    return TransformerConfig(
        data_config,
        layers=5,
        hidden_size=128,
        attention_heads=8,
        feed_forward_hidden=256,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def large_tf_config(data_config: DataConfig, position_sensitive: bool = True):
    return TransformerConfig(
        data_config,
        layers=6,
        hidden_size=192,
        attention_heads=12,
        feed_forward_hidden=384,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def super_tf_config(data_config: DataConfig, position_sensitive: bool = True):
    return TransformerConfig(
        data_config,
        layers=7,
        hidden_size=256,
        attention_heads=16,
        feed_forward_hidden=512,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def resize_small_tf_config(data_config: DataConfig, position_sensitive: bool = True):
    return TransformerConfig(
        data_config,
        layers=2,
        hidden_size=144,
        attention_heads=8,
        feed_forward_hidden=192,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def shallow_large_tf_config(data_config: DataConfig, position_sensitive: bool = True):
    return TransformerConfig(
        data_config,
        layers=3,
        hidden_size=288,
        attention_heads=12,
        feed_forward_hidden=384,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def shallower_large_tf_config(data_config: DataConfig, position_sensitive: bool = True):
    return TransformerConfig(
        data_config,
        layers=2,
        hidden_size=360,
        attention_heads=12,
        feed_forward_hidden=384,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def shallowest_large_tf_config(
    data_config: DataConfig, position_sensitive: bool = True
):
    return TransformerConfig(
        data_config,
        layers=1,
        hidden_size=528,
        attention_heads=12,
        feed_forward_hidden=384,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def customize_tf_config(
    data_config: DataConfig,
    num_layers: int,
    num_hidden: int,
    num_attention: int,
    position_sensitive: bool = True,
):
    return TransformerConfig(
        data_config,
        layers=num_layers,
        hidden_size=num_hidden,
        attention_heads=num_attention,
        feed_forward_hidden=2 * num_hidden,
        dropout=0.1,
        position_sensitive=position_sensitive,
    )


def get_tf_config(
    data_config: DataConfig,
    size: str,
    num_layers: Optional[int] = None,
    num_hidden: Optional[int] = None,
    num_attention: Optional[int] = None,
):
    if size == "small":
        return small_tf_config(data_config)
    elif size == "resize_small":
        return resize_small_tf_config(data_config)
    elif size == "medium":
        return medium_tf_config(data_config)
    elif size == "large":
        return large_tf_config(data_config)
    elif size == "shallow_large":
        return shallow_large_tf_config(data_config)
    elif size == "shallower_large":
        return shallower_large_tf_config(data_config)
    elif size == "shallowest_large":
        return shallowest_large_tf_config(data_config)
    elif size == "super":
        return super_tf_config(data_config)
    elif (
        size == "customize"
        and num_layers is not None
        and num_hidden is not None
        and num_attention is not None
    ):
        return customize_tf_config(data_config, num_layers, num_hidden, num_attention)
    else:
        raise NotImplementedError(
            f"Transformer config for [size: {size}, num_layers: {num_layers}, num_hidden: {num_hidden}, num_attention: {num_layers}] not yet implemented."
        )


def metadata2_config(
    data_config: DataConfig,
    tf1_layers=2,
    tf2_layers=2,
    use_df: bool = True,
    use_emb=True,
    position_sensitive: bool = True,
    df_subset: list = [0],
):
    return Metadata2Config(
        data_config,
        dropout=0.1,
        position_sensitive=position_sensitive,
        tf1_layers=tf1_layers,
        tf2_layers=tf2_layers,
        use_df=use_df,
        use_emb=use_emb,
        df_subset=df_subset,
    )


def get_metadata2_config(
    data_config: DataConfig,
    tf1_layers=2,
    tf2_layers=2,
    use_df=True,
    use_emb=True,
    df_subset=[0],
):
    return metadata2_config(
        data_config, tf1_layers, tf2_layers, use_df, use_emb, df_subset=df_subset
    )  # TODO: Add different size config

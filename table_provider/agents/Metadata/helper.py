try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    pass

from .data import get_data_config, DataConfig


def construct_data_config(args, is_train=True, col_type_similarity=False) -> DataConfig:
    if is_train:
        data_config = get_data_config(
            args.corpus_path,
            args.features,
            lang=args.lang,
            mode=args.mode,
            col_type_similarity=col_type_similarity,
            df_subset=args.df_subset,
            use_entity=args.use_entity,
            entity_type=args.entity_type,
            entity_emb_path=args.entity_emb_path,
            entity_recognition=args.entity_recognition,
            use_emb=args.use_emb,
            use_df=args.use_df,
        )
    else:
        data_config = get_data_config(
            args.corpus_path,
            args.features,
            lang=args.lang,
            mode=args.mode,
            col_type_similarity=col_type_similarity,
            df_subset=args.df_subset,
            use_entity=args.use_entity,
            entity_type=args.entity_type,
            entity_emb_path=args.entity_emb_path,
            entity_recognition=args.entity_recognition,
            use_emb=args.use_emb,
            use_df=args.use_df,
        )

    return data_config

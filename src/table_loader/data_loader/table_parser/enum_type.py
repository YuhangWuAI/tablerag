from enum import Enum, unique


@unique
class TaskName(Enum):
    feverous = "feverous"
    hybridqa = "hybridqa"
    sqa = "sqa"
    tabfact = "tabfact"


@unique
class TableSerializationType(Enum):
    markdown = "markdown"
    markdown_grid = "markdown_grid"
    xml = "xml"
    html = "html"
    json = "json"
    latex = "latex"
    nl_sep = "nl_sep"


@unique
class TableSamplingType(Enum):
    embedding_sample = "embedding_sample"
    auto_row_filter = "auto_row_filter"


@unique
class TableCleansingType(Enum):
    remove_empty_rows = "remove_empty_rows"
    remove_empty_rows_with_threshold = "remove_empty_rows_with_threshold"
    remove_empty_columns = "remove_empty_columns"
    remove_empty_cells = "remove_empty_cells"
    remove_empty_columns_with_threshold = "remove_empty_columns_with_threshold"
    remove_duplicate_rows = "remove_duplicate_rows"
    remove_duplicate_columns = "remove_duplicate_columns"
    fill_empty_cells = "fill_empty_cells"
    fill_empty_column_name = "fill_empty_column_name"
    parse_text_into_structured_data = "parse_text_into_structured_data"
    remove_specific_string = "remove_specific_string"
    replace_specific_pattern = "replace_specific_pattern"


@unique
class TableAugmentationType(Enum):
    external_retrieved_knowledge_info_term_explanations = "term_explanations"
    external_retrieved_knowledge_info_docs_references = "docs_references"
    terms_explanation_and_summary = "terms_explanation_and_summary"
    invalid = "None"

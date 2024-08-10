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
class TableFilterType(Enum):
    semetics_based_filter = "semetics_based_filter"
    llm_based_filter = "llm_based_filter"



@unique
class TableAugmentationType(Enum):
    external_retrieved_knowledge_info_term_explanations = "term_explanations"
    external_retrieved_knowledge_info_docs_references = "docs_references"
    terms_explanation_and_summary = "terms_explanation_and_summary"
    invalid = "None"

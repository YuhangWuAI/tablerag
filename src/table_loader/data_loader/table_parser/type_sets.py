"""
This code file contains functions that borrow certain logic from an anonymous repository associated with the paper:
"TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning" (arXiv:2312.09039).
Original source: https://anonymous.4open.science/r/TableProvider-4CC3/README.md (MIT License).
The repository does not list an author, but it is linked to the above paper.

Specifically, portions of the code related to data loading, data packing, and evaluation logic have been borrowed and integrated into this project.

Current author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

If you believe that any content in this file infringes your rights or if you have any concerns,
please contact me at the email address above.
"""

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
class TableClarificationType(Enum):
    external_retrieved_knowledge_info_term_explanations = "term_explanations"
    external_retrieved_knowledge_info_docs_references = "docs_references"
    terms_explanation_and_summary = "terms_explanation_and_summary"
    invalid = "None"

"""
This code file contains functions that borrow certain logic from an anonymous repository associated with the paper:
"TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning" (arXiv:2312.09039).
Original source: https://anonymous.4open.science/r/TableProvider-4CC3/README.md.
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
    """
    Enum representing different task names used in the project.

    Attributes:
        feverous (str): Task related to the FEVEROUS dataset.
        hybridqa (str): Task related to the HybridQA dataset.
        sqa (str): Task related to the SQA dataset.
        tabfact (str): Task related to the TabFact dataset.
    """
    feverous = "feverous"
    hybridqa = "hybridqa"
    sqa = "sqa"
    tabfact = "tabfact"

@unique
class TableSerializationType(Enum):
    """
    Enum representing different serialization formats for tables.

    Attributes:
        markdown (str): Serialization in Markdown format.
        markdown_grid (str): Serialization in Markdown Grid format.
        xml (str): Serialization in XML format.
        html (str): Serialization in HTML format.
        json (str): Serialization in JSON format.
        latex (str): Serialization in LaTeX format.
        nl_sep (str): Serialization in Natural Language (separated) format.
    """
    markdown = "markdown"
    markdown_grid = "markdown_grid"
    xml = "xml"
    html = "html"
    json = "json"
    latex = "latex"
    nl_sep = "nl_sep"

@unique
class TableFilterType(Enum):
    """
    Enum representing different types of filters applied to tables.

    Attributes:
        semetics_based_filter (str): Filter based on semantic analysis.
        llm_based_filter (str): Filter based on Large Language Model (LLM) analysis.
    """
    semetics_based_filter = "semetics_based_filter"
    llm_based_filter = "llm_based_filter"

@unique
class TableClarificationType(Enum):
    """
    Enum representing different types of clarifications that can be applied to tables.

    Attributes:
        term_explanations (str): Clarifications based on external knowledge, focusing on term explanations.
        table_summary (str): Clarifications based on external knowledge, referencing documents.
        term_explanations_and_table_summary (str): Clarifications that include both term explanations and a summary.
        invalid (str): Represents an invalid or undefined clarification type.
    """
    term_explanations = "term_explanations"
    table_summary = "docs_references"
    term_explanations_and_table_summary = "term_explanations_and_table_summary"
    invalid = "None"

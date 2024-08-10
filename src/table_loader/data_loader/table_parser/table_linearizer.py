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

import json
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.table_loader.data_loader.table_parser.type_sets import TableSerializationType

def to_xml(df):
    """
    Convert a pandas DataFrame to XML format.

    :param df: DataFrame to convert.
    :return: XML formatted string.
    """
    def row_to_xml(row):
        xml = ["<item>"]
        for i, col_name in enumerate(row.index):
            xml.append(f'  <field name="{col_name}">{row.iloc[i]}</field>')
        xml.append("</item>")
        return "\n".join(xml)

    res = "\n".join(df.apply(row_to_xml, axis=1))
    return res

class StructuredDataLinearizer:
    """
    Class for converting structured data into various linearized formats such as Markdown, XML, HTML, JSON, LaTeX, etc.
    
    The input data is expected to be a dictionary with keys 'title', 'context', and 'table' where 'table' contains 'header' and 'rows'.
    """

    def __init__(self):
        # Add the custom XML conversion function to pandas DataFrame
        pd.DataFrame.to_xml = to_xml

    def retrieve_linear_function(
        self,
        func,
        structured_data_dict: dict,
        use_structure_mark=False,
        add_grammar=False,
        change_order=False,
    ):
        """
        Retrieve the appropriate linearization function based on the specified format.

        :param func: The linearization function type from TableSerializationType.
        :param structured_data_dict: The structured data to be linearized.
        :param use_structure_mark: Whether to include structure markers in the output.
        :param add_grammar: Whether to add a grammar description in the output.
        :param change_order: Whether to change the table from row-major to column-major order.
        :return: The linearized text in the specified format.
        """
        self.structured_data_dict = structured_data_dict
        self.use_structure_mark = use_structure_mark
        self.add_grammar = add_grammar
        self.change_order = change_order

        # Dictionary mapping serialization types to their corresponding linearization methods
        dict = {
            TableSerializationType.markdown: self.linearize_markdown,
            TableSerializationType.markdown_grid: self.linearize_markdown_grid,
            TableSerializationType.xml: self.linearize_xml,
            TableSerializationType.html: self.linearize_html,
            TableSerializationType.json: self.linearize_json,
            TableSerializationType.latex: self.linearize_latex,
            TableSerializationType.nl_sep: self.linear_nl_sep,
        }
        return dict[func]()

    def linearize_markdown(self):
        """
        Convert the structured data into Markdown format.

        :return: The linearized text in Markdown format.
        """
        additional_knowledge = self._construct_additional_knowledge()

        # Convert rows from column-major to row-major order if change_order is True
        structured_data = self._get_structured_data()
        structured_data_markdown = tabulate(structured_data, tablefmt="pipe", showindex=True)

        if self.add_grammar:
            grammar = "<Markdown grammar>\n To add a table, use three or more hyphens (---) to create each column’s header, and use pipes (|) to separate each column, every cell is separated by pipe \n"
            return additional_knowledge + grammar + structured_data_markdown + "\n"
        else:
            return additional_knowledge + structured_data_markdown + "\n"

    def linearize_markdown_grid(self):
        """
        Convert the structured data into Markdown Grid format.

        :return: The linearized text in Markdown Grid format.
        """
        additional_knowledge = self._construct_additional_knowledge()

        # Convert rows from column-major to row-major order if change_order is True
        structured_data = self._get_structured_data()
        structured_data_markdown = tabulate(
            structured_data,
            headers=self.structured_data_dict["table"]["header"],
            tablefmt="grid",
            showindex=True,
        )

        if self.add_grammar:
            grammar = (
                "<Markdown grammar>\n To add a table, use three or more hyphens (---) to create each column’s header, and use pipes (|) to separate each column, every cell is separated by pipe \n"
                "Grid is like tables formatted by Emacs' table.el package. It corresponds to grid_tables in Pandoc Markdown extensions\n"
            )
            return additional_knowledge + grammar + structured_data_markdown + "\n"
        else:
            return additional_knowledge + structured_data_markdown + "\n"

    def linearize_xml(self):
        """
        Convert the structured data into XML format.

        :return: The linearized text in XML format.
        """
        additional_knowledge = self._construct_additional_knowledge()

        header = self.structured_data_dict["table"]["header"]
        # Replace spaces in header names with underscores
        for i in range(len(header)):
            header[i] = "_".join(header[i].split())

        structured_data = self._get_structured_data()
        structured_data_xml = structured_data.to_xml()

        if self.add_grammar:
            grammar = "<XML grammar>\n <?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <index>0</index>\n    <column_1>2</<column_1>>\n  </row>\n  <row>\n    <index>1</index>\n    <column_2>4</column_2>\n  </row>\n</data>"
            return additional_knowledge + grammar + structured_data_xml + "\n"
        else:
            return additional_knowledge + structured_data_xml + "\n"

    def linearize_html(self):
        """
        Convert the structured data into HTML format.

        :return: The linearized text in HTML format.
        """
        additional_knowledge = self._construct_additional_knowledge()

        rows = len(self.structured_data_dict["table"]["rows"])
        columns = len(self.structured_data_dict["table"]["header"])
        additional_knowledge += f"\nThe table has {rows} rows and {columns} columns \n"

        structured_data = self._get_structured_data()
        structured_data_html = structured_data.to_html(header=True)

        if self.add_grammar:
            grammar = "<HTML grammar>\n Each table cell is defined by a <td> and a </td> tag.\n Each table row starts with a <tr> and ends with a </tr> tag.\n th stands for table header.\n"
            return additional_knowledge + grammar + structured_data_html + "\n"
        else:
            return additional_knowledge + structured_data_html + "\n"

    def linearize_json(self):
        """
        Convert the structured data into JSON format.

        :return: The linearized text in JSON format.
        """
        if self.add_grammar:
            grammar = "<JSON grammar>\n JSON is built of a collection of name/value pairs. Each pair is key-value\n"
            return grammar + json.dumps(self.structured_data_dict)
        else:
            return json.dumps(self.structured_data_dict)

    def linearize_latex(self):
        """
        Convert the structured data into LaTeX format.

        :return: The linearized text in LaTeX format.
        """
        additional_knowledge = self._construct_additional_knowledge()

        structured_data = self._get_structured_data()
        structured_data_latex = structured_data.to_latex()

        if self.add_grammar:
            grammar = (
                "<Latex grammar>\n \\begin{tabular} starts the table environment and the curly braces denote the alignment of the columns.\n |c|c|c| means that the table has three columns and each column is center-aligned.\n "
                "\\hline creates a horizontal line.\n The text in between the & symbols is the content of the cells.\n '\\' is used to end a row.\n \\end{tabular} ends the table environment.\n"
            )
            return additional_knowledge + grammar + structured_data_latex + "\n"
        else:
            return additional_knowledge + structured_data_latex + "\n"

    def linear_nl_sep(self):
        """
        Convert the structured data into a natural language format with separators.

        :return: The linearized text in a natural language format.
        """
        additional_knowledge = self._construct_additional_knowledge()

        if self.change_order:
            header = self.structured_data_dict["table"]["header"]
            reversed_table = np.array(self.structured_data_dict["table"]["rows"]).T
            cells = []
            for i in range(len(reversed_table)):
                cells.append(header[i] + "|" + "|".join(reversed_table[i]) + "\n")
            structured_data_nl_sep = "".join(cells)
        else:
            header = "|".join(self.structured_data_dict["table"]["header"]) + "\n"
            cells = []
            for row in self.structured_data_dict["table"]["rows"]:
                cells.append("|".join(row) + "\n")
            structured_data_nl_sep = header + "".join(cells)

        if self.add_grammar:
            grammar = "<Grammar>\n Each table cell is separated by | , the column idx starts from 0, .\n"
            return additional_knowledge + grammar + structured_data_nl_sep + "\n"
        else:
            return additional_knowledge + structured_data_nl_sep + "\n"

    def _construct_additional_knowledge(self) -> str:
        """
        Construct additional knowledge text based on title, context, and caption.

        :return: Constructed additional knowledge text.
        """
        if self.use_structure_mark:
            additional_knowledge = (
                f"<title>\n{self.structured_data_dict['title']}\n"
                f"<context>\n{self.structured_data_dict['context']}\n"
                f"<caption>\n{self.structured_data_dict['table']['caption']}\n"
            )
        else:
            additional_knowledge = (
                f"{self.structured_data_dict['title']}\n"
                f"{self.structured_data_dict['context']}\n"
                f"{self.structured_data_dict['table']['caption']}\n"
            )
        return additional_knowledge

    def _get_structured_data(self) -> pd.DataFrame:
        """
        Retrieve structured data as a pandas DataFrame, potentially with order changed.

        :return: Structured data as a DataFrame.
        """
        if self.change_order:
            return pd.DataFrame(
                np.array(self.structured_data_dict["table"]["rows"]).T,
                columns=self.structured_data_dict["table"]["header"],
            )
        else:
            return pd.DataFrame(
                self.structured_data_dict["table"]["rows"],
                columns=self.structured_data_dict["table"]["header"],
            )

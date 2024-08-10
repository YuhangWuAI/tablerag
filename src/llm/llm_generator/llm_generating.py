"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

For any questions or further information, please feel free to reach out via the email address above.
"""

import ast  # for converting embeddings saved as strings back to arrays
from collections import Counter
import json
import logging
import os
import time
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from typing import List, Tuple, Union
from scipy import spatial  # for calculating vector similarities for search
from tenacity import retry, stop_after_attempt, wait_random_exponential
import warnings
warnings.filterwarnings("ignore")

class Config:
    def __init__(self, config_path: str = "config.json"):
        config = json.loads(open(config_path).read())
        self.EMBEDDING_MODEL = config["model"]["EMBEDDING_MODEL"]
        self.GPT_MODEL = config["model"]["GPT_MODEL"]
        self.OPENAI_API_KEY = config["api_key"]
        self.API_BASE = config["api_base"]
        self.BATCH_SIZE = config["batch_size"]  # 2048 embedding inputs per request
        self.USE_SELF_CONSISTENCY = config.get('use_self_consistency', False)

        self.client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY, 
            base_url=self.API_BASE
        )

class LLM_Generator:
    """Class for calling the OpenAI Language Model API."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config("config.json")
        self.EMBEDDING_MODEL = config.EMBEDDING_MODEL
        self.GPT_MODEL = config.GPT_MODEL
        self.OPENAI_API_KEY = config.OPENAI_API_KEY
        self.API_BASE = config.API_BASE
        self.BATCH_SIZE = config.BATCH_SIZE
        self.USE_SELF_CONSISTENCY = config.USE_SELF_CONSISTENCY
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["API_BASE"] = self.API_BASE
        self.client = config.client

    def num_tokens_list(self, text: List[str]) -> int:
        """Return the number of tokens in a list of strings."""
        encoding = tiktoken.encoding_for_model(self.GPT_MODEL)
        return len(encoding.encode("".join([str(item) for item in text])))


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def call_llm_code_generation(self, context: str) -> str:
        """Synthesize code snippet from the table context."""
        prompt = f"""
        Example: Synthesize code snippet from the table context to select the proper rows and columns for verifying a statement / answering query.
        The generated code must use the exact column names provided, including spaces, capitalization, and punctuation.
        The generated code should treat all data as strings, even if they look like numbers.
        Only filter out rows and columns that are definitely not needed to verify the statement / answering query.

        User 1:
        I need an expert to help me verify the statement by filtering the table to make it smaller. Statement: The scheduled date for the farm with 17 turbines be 2012.
        Columns: ['wind farm', 'scheduled', 'capacity (mw)', 'turbines', 'type', 'location']
        df = pd.DataFrame({{
            'wind farm': ['codling', 'carrowleagh', 'dublin array', 'glenmore', 'glenough', 'gortahile', 'grouse lodge', 'moneypoint', 'mount callan', 'oriel', 'skerd rocks', 'shragh', 'garracummer', 'knockacummer', 'monaincha', 'gibbet hill', 'glenough extension'],
            'scheduled': ['unknown', '2012', '2015', '2009 summer', '2010 winter', '2010 autumn', '2011 summer', 'unknown', 'unknown', '2013', 'unknown', 'planning submitted oct 2011', '2012', '2013', '2013', '2013', '2013'],
            'capacity (mw)': [1100, 36.8, 364, 30, 32.5, 20, 20, 22.5, 90, 330, 100, 135, 42.5, 87.5, 36, 15, 2.5],
            'turbines': [220, 16, 145, 10, 13, 8, 8, 9, 30, 55, 20, 45, 17, 35, 15, 6, 1],
            'type': ['unknown', 'enercon e - 70 2.3', 'unknown', 'vestas v90', 'nordex n80 / n90', 'nordex n90', 'nordex n90', 'unknown', '3 mw', 'unknown', '5 mw', 'enercon e82 3.0 mw', 'nordex n90 2.5 mw', 'nordex n90 2.5 mw', 'nordex n117 2.4 mw', 'nordex n90 2.5 mw', 'nordex n90 2.5 mw'],
            'location': ['county wicklow', 'county cork', 'county dublin', 'county clare', 'county tipperary', 'county laois', 'county tipperary', 'county clare', 'county clare', 'county louth', 'county galway', 'county clare', 'county tipperary', 'county cork', 'county tipperary', 'county wexford', 'county tipperary']
        }})
        User 2:
        To verify the statement 'The scheduled date for the farm with 17 turbines be 2012', we need to filter the rows and columns to focus on relevant information. 
        Since we are interested in the 'wind farm', 'scheduled', and 'turbines' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['wind farm', 'scheduled', 'turbines']].query("turbines == '17'")

        User 1:
        I need an expert to help me verify the statement by filtering the table to make it smaller. Statement: All 12 club play a total of 22 game for the wru division one east.
        Columns: ['club', 'played', 'drawn', 'lost', 'points for', 'points against', 'tries for', 'tries against', 'try bonus', 'losing bonus', 'points']
        df = pd.DataFrame({{
            'club': ['pontypool rfc', 'caerphilly rfc', 'blackwood rfc', 'bargoed rfc', 'uwic rfc', 'llanharan rfc', 'newbridge rfc', 'rumney rfc', 'newport saracens rfc', 'beddau rfc', 'fleur de lys rfc', 'llantrisant rfc'],
            'played': [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
            'drawn': [2, 2, 2, 0, 2, 1, 2, 2, 0, 0, 1, 0],
            'lost': [2, 4, 6, 8, 7, 12, 11, 12, 14, 15, 16, 18],
            'points for': [648, 482, 512, 538, 554, 436, 355, 435, 344, 310, 300, 402],
            'points against': [274, 316, 378, 449, 408, 442, 400, 446, 499, 483, 617, 592],
            'tries for': [81, 56, 60, 72, 71, 44, 36, 56, 45, 32, 34, 55],
            'tries against': [32, 37, 42, 52, 50, 51, 47, 52, 64, 61, 77, 77],
            'try bonus': [12, 7, 8, 10, 6, 1, 2, 5, 2, 2, 2, 4],
            'losing bonus': [1, 3, 3, 4, 2, 7, 3, 3, 3, 4, 4, 6],
            'points': [89, 78, 71, 70, 64, 46, 45, 44, 37, 34, 28, 26]
        }})
        User 2:
        To verify the statement 'All 12 club play a total of 22 game for the wru division one east', we need to filter the rows and columns to focus on relevant information. 
        Since we are interested in the 'club' and 'played' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['club', 'played']].query("played == '22'")

        User 1:
        I need an expert to help me verify the statement by filtering the table to make it smaller. Statement: Touchdown Atlantic, in the category of sporting, be established in 2010.
        Columns: ['event name', 'established', 'category', 'sub category', 'main venue']
        df = pd.DataFrame({{
            'event name': ['dieppe kite international', 'the frye festival', 'hubcap comedy festival', 'touchdown atlantic', 'atlantic nationals automotive extravaganza', 'world wine & food expo', 'shediac lobster festival', 'mosaïq multicultural festival'],
            'established': [2001, 2000, 2000, 2010, 2000, 1990, 1950, 2004],
            'category': ['sporting', 'arts', 'arts', 'sporting', 'transportation', 'arts', 'arts', 'festival'],
            'sub category': ['kite flying', 'literary', 'comedy', 'football', 'automotive', 'food & drink', 'food & drink', 'multicultural'],
            'main venue': ['dover park', 'university of moncton', 'various', 'moncton stadium', 'moncton coliseum', 'moncton coliseum', 'shediac festival grounds', 'moncton city hall plaza']
        }})
        User 2:
        To verify the statement 'Touchdown Atlantic, in the category of sporting, be established in 2010', we need to filter the rows and columns to focus on relevant information. 
        Since we are interested in the 'event name' and 'established' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['event name', 'established']].query("`event name` == 'touchdown atlantic' and established == '2010'")

        Now, generate a code snippet from the table context to select the proper rows and columns to verify the given statement / answering query.
        Use the existing column names from the provided DataFrame.
        The column names in the generated code must match the provided column names exactly, including spaces, capitalization, and punctuation.
        Only filter out rows and columns that are definitely not needed to verify the statement.
        Only return the code. 
        {context}
        \n\n:
        """

        if self.USE_SELF_CONSISTENCY:
            generated_codes = [self.generate_text(prompt) for _ in range(5)]
            print("Generated codes:", generated_codes)
            
            # Find the most common code
            code_counter = Counter(generated_codes)
            most_common_code, count = code_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_code
            else:
                return generated_codes[0]
        else:
            return self.generate_text(prompt)

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_terms_explanation(self, table: dict, statement: str, caption: str) -> str:
        prompt = f"""
        Example: You will be given a table, a statement, and the table's caption. Your task is to identify difficult to understand column names, terms, or abbreviations in the table and provide simple explanations for each. Only explain terms related to the statement.

        User 1:
        I need an expert to help me explain the terms in this table. Here is the statement: The scheduled date for the farm with 17 turbines be 2012.
        Here is the table caption: Wind Farm Details in Ireland
        Here is the table:
        {{
            "wind farm": ["codling", "carrowleagh", "dublin array", "glenmore", "glenough", "gortahile", "grouse lodge", "moneypoint", "mount callan", "oriel", "skerd rocks", "shragh", "garracummer", "knockacummer", "monaincha", "gibbet hill", "glenough extension"],
            "scheduled": ["unknown", "2012", "2015", "2009 summer", "2010 winter", "2010 autumn", "2011 summer", "unknown", "unknown", "2013", "unknown", "planning submitted oct 2011", "2012", "2013", "2013", "2013", "2013"],
            "capacity (mw)": [1100, 36.8, 364, 30, 32.5, 20, 20, 22.5, 90, 330, 100, 135, 42.5, 87.5, 36, 15, 2.5],
            "turbines": [220, 16, 145, 10, 13, 8, 8, 9, 30, 55, 20, 45, 17, 35, 15, 6, 1],
            "type": ["unknown", "enercon e - 70 2.3", "unknown", "vestas v90", "nordex n80 / n90", "nordex n90", "nordex n90", "unknown", "3 mw", "unknown", "5 mw", "enercon e82 3.0 mw", "nordex n90 2.5 mw", "nordex n90 2.5 mw", "nordex n117 2.4 mw", "nordex n90 2.5 mw", "nordex n90 2.5 mw"],
            "location": ["county wicklow", "county cork", "county dublin", "county clare", "county tipperary", "county laois", "county tipperary", "county clare", "county clare", "county louth", "county galway", "county clare", "county tipperary", "county cork", "county tipperary", "county wexford", "county tipperary"]
        }}

        User 2:
        Explanations:
        "scheduled": "The planned date for the wind farm to be operational.",
        "turbines": "The number of wind turbines in the wind farm."

        User 1:
        I need an expert to help me explain the terms in this table. Here is the statement: All 12 clubs play a total of 22 games for the WRU Division One East.
        Here is the table caption: WRU Division One East Standings
        Here is the table:
        {{
            "club": ["pontypool rfc", "caerphilly rfc", "blackwood rfc", "bargoed rfc", "uwic rfc", "llanharan rfc", "newbridge rfc", "rumney rfc", "newport saracens rfc", "beddau rfc", "fleur de lys rfc", "llantrisant rfc"],
            "played": [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
            "drawn": [2, 2, 2, 0, 2, 1, 2, 2, 0, 0, 1, 0],
            "lost": [2, 4, 6, 8, 7, 12, 11, 12, 14, 15, 16, 18],
            "points for": [648, 482, 512, 538, 554, 436, 355, 435, 344, 310, 300, 402],
            "points against": [274, 316, 378, 449, 408, 442, 400, 446, 499, 483, 617, 592],
            "tries for": [81, 56, 60, 72, 71, 44, 36, 56, 45, 32, 34, 55],
            "tries against": [32, 37, 42, 52, 50, 51, 47, 52, 64, 61, 77, 77],
            "try bonus": [12, 7, 8, 10, 6, 1, 2, 5, 2, 2, 2, 4],
            "losing bonus": [1, 3, 3, 4, 2, 7, 3, 3, 3, 4, 4, 6],
            "points": [89, 78, 71, 70, 64, 46, 45, 44, 37, 34, 28, 26]
        }}

        User 2:
        Explanations:
        "played": "The number of games played by the club.",
        "points for": "The total points scored by the club.",
        "points against": "The total points scored against the club."

        Now, explain the terms in the following table.

        Table caption:
        {caption}

        Statement:
        {statement}

        Table:
        {json.dumps(table, indent=2)}

        Please return the result in the following format:
        {{
            "explanations": {{
                "term1": "explanation1",
                "term2": "explanation2",
                ...
            }}
        }}
        """

        generated_text = self.generate_text(prompt)
        return generated_text  


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_table_summary(self, metadata_list: list, context: list, table: dict, query: str, caption: str) -> str:
        prompt = f"""
        Example: You will be given a table, a query (which can be either a statement or a question), the table's caption, metadata from related Wikipedia documents, and the context of the table. Your task is to generate a concise summary for the table that directly addresses the query, using the Wikipedia metadata and the context to enhance understanding. Ensure the summary starts with the phrase 'This table is used to answer the query: [query]' and includes only content related to the query. Do not directly reveal the answer, but guide the reader to make an informed decision based on the provided information.

        User 1:
        I need an expert to help me create a concise summary of the table. Here is the query: The scheduled date for the farm with 17 turbines be 2012.
        Here is the table caption: Wind Farm Details in Ireland
        Here is the table:
        {{
            "wind farm": ["codling", "carrowleagh", "dublin array", "glenmore", "glenough", "gortahile", "grouse lodge", "moneypoint", "mount callan", "oriel", "skerd rocks", "shragh", "garracummer", "knockacummer", "monaincha", "gibbet hill", "glenough extension"],
            "scheduled": ["unknown", "2012", "2015", "2009 summer", "2010 winter", "2010 autumn", "2011 summer", "unknown", "unknown", "2013", "unknown", "planning submitted oct 2011", "2012", "2013", "2013", "2013", "2013"],
            "capacity (mw)": [1100, 36.8, 364, 30, 32.5, 20, 20, 22.5, 90, 330, 100, 135, 42.5, 87.5, 36, 15, 2.5],
            "turbines": [220, 16, 145, 10, 13, 8, 8, 9, 30, 55, 20, 45, 17, 35, 15, 6, 1],
            "type": ["unknown", "enercon e - 70 2.3", "unknown", "vestas v90", "nordex n80 / n90", "nordex n90", "nordex n90", "unknown", "3 mw", "unknown", "5 mw", "enercon e82 3.0 mw", "nordex n90 2.5 mw", "nordex n90 2.5 mw", "nordex n117 2.4 mw", "nordex n90 2.5 mw", "nordex n90 2.5 mw"],
            "location": ["county wicklow", "county cork", "county dublin", "county clare", "county tipperary", "county laois", "county tipperary", "county clare", "county clare", "county louth", "county galway", "county clare", "county tipperary", "county cork", "county tipperary", "county wexford", "county tipperary"]
        }}
        Here is the metadata from related Wikipedia documents:
        [
            {{
                "title": "Wind power in Ireland",
                "summary": "As of 2021 the island of Ireland has 5,585 megawatt and the Republic of Ireland has 4,309 MW of installed wind power nameplate capacity...",
                "source": "https://en.wikipedia.org/wiki/Wind_power_in_Ireland"
            }},
            {{
                "title": "List of wind farms in the Republic of Ireland",
                "summary": "This is a list of wind farms in the Republic of Ireland...",
                "source": "https://en.wikipedia.org/wiki/List_of_wind_farms_in_the_Republic_of_Ireland"
            }}
        ]
        Here is the context of the table: The table is part of a broader analysis of wind power development in Ireland, comparing different wind farms' schedules, capacities, and locations.

        User 2:
        Summary:
        "This table is used to answer the query: The scheduled date for the farm with 17 turbines be 2012. The farm with 17 turbines, Garracummer, is scheduled for 2012 according to the table. Wind power in the Republic of Ireland is significant, with over 300 wind farms generating electricity. As of 2021, the Republic of Ireland has 4,309 MW of installed wind power capacity, contributing to a high wind power penetration in the country."

        Now, generate a summary for the given table, addressing the query and using the Wikipedia metadata and the context provided for enhanced understanding. Ensure the summary starts with the phrase 'This table is used to answer the query: [query]' and includes only content related to the query. Please avoid directly revealing the answer.

        Query:
        {query}

        This table is used to answer the query: {query}

        Table caption:
        {caption}

        Table:
        {json.dumps(table, indent=2)}

        Wikipedia metadata:
        {json.dumps(metadata_list, indent=2)}

        Context:
        {json.dumps(context, indent=2)}

        Please return the result in the following format:
        {{
            "summary": "The summary that includes the query, context from the caption, and relevant Wikipedia information."
        }}
        """

        generated_text = self.generate_text(prompt)
        return generated_text

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def tabfact_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        print("\nCalling OpenAI API for generating the final answer !!!\n")

        # Check if terms explanation, table summary, or table context are empty, and use a placeholder if they are
        if not terms_explanation.strip():
            terms_explanation = "[No terms explanation provided]"

        if not table_summary.strip():
            table_summary = "[No table summary provided]"

        if not table_context.strip():
            table_context = "[No additional context provided]"

        prompt = f"""
        Example: You will be given a statement, a table summary, the full table content, terms explanations, and possibly additional context.
        The table content may be provided in string format, Markdown format, or HTML format.
        Your task is to determine whether the statement is true or false based on the table, provided information, and any additional context.
        Return 1 if the statement is true, and 0 if it is false or if you cannot determine the answer based on the provided information.
        Provide only the number '1' or '0' as the answer without any additional text.

        User 1:
        Statement: "The scheduled date for the farm with 17 turbines is 2012."
        Table Context: [No additional context provided]
        Table Summary: "This table is used to answer the query: the scheduled date for the farm with 17 turbines is 2012. The Garracummer wind farm, which has 17 turbines, is indeed scheduled for 2012 as indicated in the table. Wind power is a significant energy source in Ireland, contributing to a high percentage of the country's electricity needs, with the Republic of Ireland boasting a total installed capacity of 4,309 MW as of 2021."
        table_formatted:
        <table border="1" class="dataframe">
        <thead>
            <tr style="text-align: right;">
            <th></th>
            <th>wind farm</th>
            <th>scheduled</th>
            <th>capacity (mw)</th>
            <th>turbines</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <th>12</th>
            <td>garracummer</td>
            <td>2012</td>
            <td>42.5</td>
            <td>17</td>
            </tr>
        </tbody>
        </table>
        Terms Explanation:
        {{
            "scheduled": "The planned date for the wind farm to be operational.",
            "turbines": "The number of wind turbines in the wind farm."
        }}

        User 2:
        1

        User 1:
        Statement: "The most recent locomotive to be manufactured was made more than 10 years after the first was manufactured."
        Table Context: "This context may provide additional details about the locomotives or their manufacturing process."
        Table Summary: "This table is used to answer the query: the most recent locomotive to be manufacture was made more than 10 years after the first was manufactured. The first locomotives listed in the table were manufactured between 1889 and 1907, while the most recent locomotive was manufactured in 1923. This indicates that the most recent locomotive was made 16 years after the first ones, thus supporting the truth of the statement. The list provides an overview of locomotives from the Palatinate Railway, highlighting the historical context of railway development in the region."
        Table_fomatted:
        <table border="1" class="dataframe">
        <thead>
            <tr style="text-align: right;">
            <th></th>
            <th>class</th>
            <th>year (s) of manufacture</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <th>1</th>
            <td>l 2</td>
            <td>1903 - 1905</td>
            </tr>
            <!-- more rows -->
        </tbody>
        </table>
        Terms Explanation:
        {{
            "year (s) of manufacture": "The years when the locomotives or railbuses were built.",
            "axle arrangement ( uic ) bauart": "The configuration of the wheels on the locomotive or railbus, as defined by the UIC (International Union of Railways) classification system."
        }}

        User 2:
        0

        Now, verify the following statement and return only '1' or '0' as the result.

        Statement: "{query_need_to_answer}"
        Table Context: "{table_context}"
        Table Summary: "{table_summary}"
        Table_formatted: {table_formatted}
        Terms Explanation: {terms_explanation}

        If you cannot determine whether the statement is true or false based on the provided information, return '0'. Otherwise, return '1' for true or '0' for false.
        Return only '1' or '0'.
        """

        # This is where the LLM is called to generate the answer
        generated_text = self.generate_text(prompt)
        return generated_text.strip()  # Ensure any whitespace is removed, returning only the digit

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def feverous_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        print("\nCalling OpenAI API for generating the final answer for FEVEROUS dataset!!!\n")

        # Check if terms explanation, table summary, or table context are empty, and use a placeholder if they are
        if not terms_explanation.strip():
            terms_explanation = "[No terms explanation provided]"

        if not table_summary.strip():
            table_summary = "[No table summary provided]"

        if not table_context.strip():
            table_context = "[No additional context provided]"

        prompt = f"""
        Example: You will be given a statement, a table summary, the full table content, terms explanations, and possibly additional context.
        The table content may be provided in string format, Markdown format, or HTML format.
        Your task is to determine whether the statement is true, false, or if the evidence provided is insufficient.
        Return 1 if the statement is true, 0 if it is false, and 2 if you cannot determine the answer based on the provided information.
        Provide only the number '1', '0', or '2' as the answer without any additional text.

        User 1:
        Statement: "All the ethnic groups in the Urmiri Municipality have the same population."
        Table Context: [No additional context provided]
        Table Summary: "This table is used to answer the query: All the ethnic groups in the Urmiri Municipality have the same population. The table presents the population percentages of various ethnic groups in the Urmiri Municipality, indicating that the Quechua and Aymara groups have significant representations at 49.3% and 46.6%, respectively, while other groups have negligible populations. This disparity suggests that not all ethnic groups in the municipality share the same population size. Understanding ethnic composition is important as it reflects cultural diversity and social dynamics within the community."
        Table_formatted:
        |    | Ethnic group              |    % |
        |---:|:--------------------------|-----:|
        |  0 | Quechua                   | 49.3 |
        |  1 | Aymara                    | 46.6 |
        |  2 | Guarani, Chiquitos, Moxos |  0   |
        |  3 | Not indigenous            |  4.1 |
        |  4 | Other indigenous groups   |  0.1 |
        Terms Explanation:
        {{
            "%": "The percentage representation of each ethnic group's population compared to the total population of the Urmiri Municipality."
        }}

        User 2:
        0

        User 1:
        Statement: "Grammy Award for Best Immersive Audio Album (open to both classical and non-classical recordings) happened almost every year between 2005 and 2021, one of which was for the title Genius Loves Company."
        Table Context: "It is one of a few categories which are open to both classical and non-classical recordings, new or re-issued. The Grammy Award for Best Immersive Audio Album (until 2018: Best Surround Sound Album) was first awarded in 2005, as the first category in a new 'Surround Sound' field. On 24 November 2020 during the announcement of the nominations for the 63rd Grammy Awards, to be presented on 31 January 2021, the Recording Academy said there will be no winner or nominees in this category."
        Table Summary: "This table is used to answer the query: Grammy Award for Best Immersive Audio Album (open to both classical and non-classical recordings) happened almost every year between 2005 and 2021, one of which was for the title Genius Loves Company. The Grammy Award for Best Immersive Audio Album, established in 2005, recognizes excellence in both classical and non-classical recordings. The table lists winners from 2005 to 2021, confirming that 'Genius Loves Company' by Ray Charles & Various Artists won the award in 2005. This category has been significant in the evolution of audio recording standards, reflecting the growing importance of immersive sound in the music industry."
        Table_formatted:
        <table border="1" class="dataframe">
        <thead>
            <tr style="text-align: right;">
            <th></th>
            <th>Year</th>
            <th>Winner(s)</th>
            <th>Title</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <th>0</th>
            <td>2005</td>
            <td>Al Schmitt*, Robert Hadley & Doug Sax**, John Burk, Phil Ramone & Herbert Walf***</td>
            <td>Genius Loves Company</td>
            </tr>
        </tbody>
        </table>
        Terms Explanation:
        {{
            "Winner(s)": "The individuals or groups who won the Grammy Award for that year.",
            "Title": "The name of the album or recording that won the award."
        }}

        User 2:
        1

        Now, verify the following statement and return only '1', '0', or '2' as the result.

        Statement: "{query_need_to_answer}"
        Table Context: "{table_context}"
        Table Summary: "{table_summary}"
        Table_formatted: {table_formatted}
        Terms Explanation: {terms_explanation}

        If you cannot determine whether the statement is true or false based on the provided information, return '2'. Otherwise, return '1' for true, '0' for false, or '2' for insufficient evidence.
        Return only '1', '0', or '2'.
        """

        # This is where the LLM is called to generate the answer
        generated_text = self.generate_text(prompt)
        return generated_text.strip()  # Ensure any whitespace is removed, returning only the digit


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def hybridqa_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        print("\nCalling OpenAI API for generating the final answer for HybridQA dataset!!!\n")

        # Check if terms explanation, table summary, or table context are empty, and use a placeholder if they are
        if not terms_explanation.strip():
            terms_explanation = "[No terms explanation provided]"

        if not table_summary.strip():
            table_summary = "[No table summary provided]"

        if not table_context.strip():
            table_context = "[No additional context provided]"

        prompt = f"""
        Example: You will be given a query, a table summary, the full table content, terms explanations, and additional context.
        The table content may be provided in string format, Markdown format, or HTML format.
        Your task is to determine the answer to the query based on the table, provided information, and any additional context.
        Pay close attention to the specific details and conditions mentioned in the query.
        Make sure to match all the given conditions in the query to ensure the answer is accurate.
        Return the answer as a single string without any additional text.

        Example 1:
        Query: "How many years did constructor AGS compete in Formula One?"
        Table Context: "1991 Portuguese Grand Prix | Classification -- Pre-Qualifying |  | The 1991 Portuguese Grand Prix was a Formula One motor race held at the Autódromo do Estoril on 22 September 1991. It was the thirteenth race of the 1991 FIA Formula One World Championship."
        Table Summary: "This table is used to answer the query: How many years did constructor AGS compete in Formula One? The table provides results from a specific Formula One event, showcasing drivers and their constructors, including AGS - Ford. While the table does not directly state the duration of AGS's participation in Formula One, it highlights their presence in the competitive landscape of the sport. For further context, AGS was one of several constructors that participated in Formula One during the late 1980s and early 1990s, contributing to the diversity of teams in the championship."
        Table_formatted:
        |    |   Pos | Driver            | Constructor   |
        |---:|------:|:------------------|:--------------|
        |  2 |     3 | Gabriele Tarquini | AGS - Ford    |
        |  4 |     5 | Fabrizio Barbazza | AGS - Ford    |
        Terms Explanation:
        {{
            "Constructor": "The team that builds and enters the car in Formula One races.",
            "Time": "The time taken by the driver to complete the race or qualifying session.",
            "Gap": "The time difference between the driver and the driver ahead of them in the race or qualifying session."
        }}

        User 2:
        5 years

        Example 2:
        Query: "What year did the team with 4,499 officially registered fan clubs lose 2-0 to lose out on the title?"
        Table Context: "DFL-Supercup | Performances -- Performance by team |  | The DFL-Supercup or German Super Cup is a one-off football match in Germany that features the winners of the Bundesliga championship and the DFB-Pokal."
        Table Summary: "This table is used to answer the query: What year did the team with 4,499 officially registered fan clubs lose 2-0 to lose out on the title? The table provides details about various football teams, their titles won, and years they lost in competitions. While it does not specify a team with 4,499 fan clubs, it indicates that Bayern Munich and Borussia Dortmund have been prominent teams in the DFL-Supercup, with Bayern Munich losing in specific years. The context surrounding the DFL-Supercup highlights the competitive nature of these matches, which feature top teams in German football, thus providing a backdrop for understanding the significance of the losses mentioned in the query."
        Table_formatted:
        |    | Team                     | Winners   | Runners-up   | Years won                                      | Years lost                              |
        |---:|:-------------------------|:----------|:-------------|:-----------------------------------------------|:----------------------------------------|
        |  0 | Bayern Munich            | 7         | 6            | 1987 , 1990 , 2010 , 2012 , 2016 , 2017 , 2018 | 1989 , 1994 , 2013 , 2014 , 2015 , 2019 |
        |  1 | Borussia Dortmund        | 6         | 4            | 1989 , 1995 , 1996 , 2013 , 2014 , 2019        | 2011 , 2012 , 2016 , 2017               |
        |  2 | Werder Bremen            | 3         | 1            | 1988 , 1993 , 1994                             | 1991                                    |
        Terms Explanation:
        {{
            "Winners": "The number of times the team has won the title.",
            "Runners-up": "The number of times the team has finished in second place.",
            "Years won": "The specific years in which the team won the title.",
            "Years lost": "The specific years in which the team lost the title."
        }}

        User 2:
        2014

        Now, answer the following query based on the provided information.

        Query: "{query_need_to_answer}"
        Table Context: "{table_context}"
        Table Summary: "{table_summary}"
        Table_formatted: {table_formatted}
        Terms Explanation: {terms_explanation}

        Carefully match all specific conditions mentioned in the query.
        Provide only the answer as a single string without any additional text.
        """

        # This is where the LLM is called to generate the answer
        generated_text = self.generate_text(prompt)
        return generated_text.strip()  # Ensure any whitespace is removed, returning only the answer

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def sqa_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        print("\nCalling OpenAI API for generating the final answer for SQA dataset!!!\n")

        # Check if terms explanation, table summary, or table context are empty, and use a placeholder if they are
        if not terms_explanation.strip():
            terms_explanation = "[No terms explanation provided]"

        if not table_summary.strip():
            table_summary = "[No table summary provided]"

        if not table_context.strip():
            table_context = "[No additional context provided]"

        prompt = f"""
        You will be given a query, a table summary, the full table content, terms explanations, and possibly additional context.
        The table content may be provided in string format, Markdown format, or HTML format.
        Your task is to determine the answer to the query based on the table, provided information, and any additional context.
        Pay close attention to the specific details and conditions mentioned in the query.
        Make sure to match all the given conditions in the query to ensure the answer is accurate.
        Return the answer as a single string without any additional text.

        Example 1:
        Query: "who published this game in 2011"
        Table Context: "[No additional context provided]"
        Table Summary: "This table is used to answer the query: who published this game in 2011. The game released in 2011 titled 'Alice: Madness Returns' was published by Electronic Arts. The table provides details on various games, their release years, developers, and publishers, highlighting the contributions of Spicy Horse, the developer known for its notable title, Alice: Madness Returns. For more insights into Spicy Horse, which was founded by American McGee, you can refer to the provided Wikipedia link."
        Table_formatted:
        |    | Title                  |   Year | Publisher       |
        |---:|:-----------------------|-------:|:----------------|
        |  4 | Alice: Madness Returns |   2011 | Electronic Arts |
        Terms Explanation:
        {{
            "Publisher": "The company or organization that publishes the game, making it available for sale or distribution."
        }}

        User 2:
        Electronic Arts

        User 1:
        Query: "and which model costs the most?"
        Table Context: ""
        Table Summary: "This table is used to answer the query: and which model costs the most? The table lists various models of vehicles along with their specifications and starting prices. By comparing the starting prices, one can determine which model is the most expensive. The context of the table indicates that it provides a detailed overview of different vehicle models, which helps in evaluating their cost and features."
        Table_formatted:
        |    | Model      | Class   | Length   | Fuel   | Starting Price   |
        |---:|:-----------|:--------|:---------|:-------|:-----------------|
        |  0 | Tour       | Class A | 42'      | Diesel | $362,285         |
        |  1 | Journey    | Class A | 35'-43'  | Diesel | $246,736         |
        |  2 | Adventurer | Class A | 32'-37'  | Gas    | $150,711         |
        |  3 | Via        | Class A | 25'      | Diesel | $126,476         |
        |  4 | Sightseer  | Class A | 31'-37'  | Gas    | $126,162         |
        |  5 | Vista      | Class A | 26'-35'  | Gas    | $107,717         |
        |  6 | View       | Class C | 24'-25'  | Diesel | $100,955         |
        |  7 | Aspect     | Class C | 29'-31'  | Gas    | $95,948          |
        |  8 | Access     | Class C | 25'-31'  | Gas    | $74,704          |
        Terms Explanation:
        {{
            "Model": "The name of the vehicle model.",
            "Starting Price": "The initial cost of the vehicle model, before any additional options or fees."
        }}

        User 2:
        Tour

        Now, answer the following query based on the provided information.

        Query: "{query_need_to_answer}"
        Table Context: "{table_context}"
        Table Summary: "{table_summary}"
        Table_formatted: {table_formatted}
        Terms Explanation: {terms_explanation}

        Carefully match all specific conditions mentioned in the query.
        Provide only the answer as a single string without any additional text.
        """

        # This is where the LLM is called to generate the answer
        generated_text = self.generate_text(prompt)
        return generated_text.strip()  # Ensure any whitespace is removed, returning only the answer


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_text(self, prompt: str) -> str:
        """Generate text based on the prompt and instruction."""
        try:
            print("Calling OpenAI API for text generation")
            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            result = response.choices[0].message.content.strip()
            print("Generated text successfully!")
            return result
        except Exception as e:
            print("Error in generate_text:", e)
            raise

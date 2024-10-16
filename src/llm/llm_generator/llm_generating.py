"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/
Copyright (C) 2024 Wu Yuhang. All rights reserved.
For any questions or further information, please feel free to reach out via the email address above.
"""
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
import traceback



from datetime import datetime
import ast  # For converting embeddings saved as strings back to arrays
from collections import Counter
import json
import logging
import os
import time
import openai  # For calling the OpenAI API
import pandas as pd  # For storing text and embeddings data
import tiktoken  # For counting tokens
from typing import List, Tuple, Union
from scipy import spatial  # For calculating vector similarities for search
from tenacity import retry, stop_after_attempt, wait_random_exponential
import warnings
warnings.filterwarnings("ignore")

class Config:
    """
    Configuration class for managing settings related to the LLM, including model selection and API keys.
    """
    def __init__(self, config_path: str = "config.json"):
        config = json.loads(open(config_path).read())
        self.EMBEDDING_MODEL = config["model"]["EMBEDDING_MODEL"]
        self.GPT_MODEL = config["model"]["GPT_MODEL"]
        self.OPENAI_API_KEY = config["api_key"]
        self.API_BASE = config["api_base"]
        self.BATCH_SIZE = config["batch_size"]  # Number of embedding inputs per request
        self.USE_SELF_CONSISTENCY = config.get('use_self_consistency', False)

        # Initialize OpenAI client with the provided API key and base URL
        self.client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY, 
            base_url=self.API_BASE
        )

class LLM_Generator:
    """
    Class for interacting with the OpenAI Language Model API to generate text, code snippets, and other outputs.
    """

    def __init__(self, config: Config = None):
        """
        Initialize the LLM_Generator with a configuration object.

        :param config: Configuration object containing API keys, model settings, etc.
        """
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
        """
        Return the number of tokens in a list of strings.

        :param text: List of strings to be tokenized.
        :return: Total number of tokens.
        """
        encoding = tiktoken.encoding_for_model(self.GPT_MODEL)
        return len(encoding.encode("".join([str(item) for item in text])))


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


    def new_call_llm_code_generation(self, query, enhanced_info, column_names, table) -> str:

        prompt = f"""
        You will receive a query, a table, and the column names of the table. Additionally, you may receive information such as the table title, table summary, or terms explanation to provide more context.

        Your task:
        Generate a code snippet from the table context that selects the necessary rows and columns to answer the query. Specifically, you need to filter out rows and columns that are not relevant to the query.

        Instructions:
        1. Use the exact column names provided, including spaces, capitalization, and punctuation.
        2. Treat all data as strings, even if they appear to be numbers.
        3. Only remove rows and columns that are clearly unnecessary for answering the query or verifying the statement.

        Example 1:
        User's Statement: "The scheduled date for the farm with 17 turbines is 2012."
        Columns: ['wind farm', 'scheduled', 'capacity (mw)', 'turbines', 'type', 'location']
        Filtered code:
        filtered_table = df[['wind farm', 'scheduled', 'turbines']].query("turbines == '17'")

        Example 2:
        User's Statement: "All 12 clubs played a total of 22 games for the WRU division one east."
        Columns: ['club', 'played', 'drawn', 'lost', 'points for', 'points against', 'tries for', 'tries against', 'try bonus', 'losing bonus', 'points']
        Filtered code:
        filtered_table = df[['club', 'played']].query("played == '22'")
        
        Now, based on the table context provided, generate the appropriate code snippet to filter and select the rows and columns needed to answer the query.
        
        Query:
        {query}
        
        Enhanced Info:
        {enhanced_info}

        Column Names:
        {column_names}

        Table:
        {table}

        
        Use the existing column names from the provided DataFrame.
        The column names in the generated code must match the provided column names exactly, including spaces, capitalization, and punctuation.
        Only filter out rows and columns that are definitely not needed to verify the statement.
        Only return the code. 

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
    def generate_terms_explanation(self, summary, table, caption):
        """
        Generate explanations for terms found in a table, focusing on those related to a given statement.

        :param table: Dictionary representing the table's data.
        :param statement: Statement related to the table.
        :param caption: Caption of the table for context.
        :return: JSON string containing the terms and their explanations.
        """
        prompt = f"""
        Example: You will be given a table, a statement, and the table's caption. Your task is to identify difficult to understand column names, terms, or abbreviations in the table and provide simple explanations for each. Only explain terms related to the statement.

        Now, explain the terms in the following table.

        Table caption:
        {caption}

        Table Summary:
        {summary}

        Table:
        {table}

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
    

    def generate_terminology_explanation(self, caption, summary, table: str, context: list) -> str:

        prompt = f"""
        You might receive tables in various formats, including Markdown, HTML, or plain text strings. Along with the table data, you might also receive a table summary, additional context, and a table caption, if available.

        Your task is to generate detailed explanations for the terms used in the table. The explanations should include definitions, synonyms, and query suggestions to aid in understanding and retrieving relevant information from the table. 

        Please provide the following for each term identified in the table:

        1. **Term**: Identify and list all possible terms and abbreviations found in the table.
        
        2. **Definition**: Provide a concise definition or explanation for each term. Ensure the definition is clear and directly relevant to the context of the table.

        3. **Synonyms and Related Concepts**: List several synonyms or similar concepts for each term. This helps in understanding the term’s usage in different contexts.

        4. **Query Suggestions**: Generate 3 example queries that someone might use to search for information about the term in this table. Ensure the queries are diverse, using different sentence structures, and cover a range of potential user needs or scenarios. Each query should address a unique aspect of the term within the context of the table.

        **Example 1: Swimming Competition Results Table**
        
        User 1:
        Table:
        | Rank   |   Lane | Name                   | Nationality   | Time    | Notes   |
        |:-------|-------:|:-----------------------|:--------------|:--------|:--------|
        |        |      4 | Sophie Pascoe          | New Zealand   | 2:25.65 | WR      |
        |        |      5 | Summer Ashley Mortimer | Canada        | 2:32.08 |         |
        |        |      3 | Zhang Meng             | China         | 2:33.95 | AS      |
        | 4      |      6 | Katherine Downie       | Australia     | 2:34.64 |         |
        | 5      |      2 | Nina Ryabova           | Russia        | 2:35.65 |         |
        | 6      |      8 | Aurelie Rivard         | Canada        | 2:37.70 |         |
        | 7      |      7 | Harriet Lee            | Great Britain | 2:39.42 |         |
        | 8      |      1 | Gemma Almond           | Great Britain | 2:42.16 |         |
        
        User 2:
        **Term Explanations**:

        **1. WR (World Record)**
        - **Definition**: A world record, denoting the highest achievement in a specific event or discipline recognized globally.
        - **Synonyms and Related Concepts**: Best record, global record, all-time record.
        - **Query Suggestions**:
        1. "Who holds the world record for this swimming event according to the table?"
        2. "What is the time listed for the world record swimmer?"
        3. "How does the world record time compare to the other times in the table?"

        **2. AS (Asian Record)**
        - **Definition**: An Asian record, marking the highest achievement in a specific event or discipline within the Asian continent.
        - **Synonyms and Related Concepts**: Best Asian performance, continental record, Asian best.
        - **Query Suggestions**:
        1. "Which swimmer set the Asian record in this competition?"
        2. "What time is recorded for the Asian record in this table?"
        3. "How does the Asian record time compare with the world record?"

        **3. Lane**
        - **Definition**: The designated track or path that a swimmer or competitor is assigned during the race.
        - **Synonyms and Related Concepts**: Track, position, route.
        - **Query Suggestions**:
        1. "Which lane had the swimmer with the fastest time?"
        2. "How are the lanes assigned in this table's competition?"
        3. "What is the lane assignment for the swimmer with the Asian record?"

        **Example 2: Fight Record Table**
        User 1:
        Table:
        | Date       | Result   | Opponent             | Event                                                                                 | Location                     | Method                       | Round   | Time   |
        |:-----------|:---------|:---------------------|:--------------------------------------------------------------------------------------|:-----------------------------|:-----------------------------|:--------|:-------|
        | 2013-12-14 | Loss     | Mohamed Diaby        | Victory, Semi Finals                                                                  | Paris, France                | Decision                     | 3       | 3:00   |
        | 2013-03-09 |          | Juanma Chacon        | Enfusion Live: Barcelona                                                              | Barcelona, Spain             |                              |         |        |
        | 2012-05-27 | Loss     | Murthel Groenhart    | K-1 World MAX 2012 World Championship Tournament Final 16                             | Madrid, Spain                | KO (punches)                 | 3       | 3:00   |
        | 2012-02-11 | Win      | Francesco Tadiello   | Sporthal De Zandbergen                                                                | Sint-Job-in-'t-Goor, Belgium | KO                           | 1       |        |
        | 2012-01-28 | Win      | Chris Ngimbi         | It's Showtime 2012 in Leeuwarden                                                      | Leeuwarden, Netherlands      | TKO (cut)                    | 2       | 1:22   |
        | 2011-09-24 | Loss     | Andy Souwer          | BFN Group & Music Hall presents: It's Showtime "Fast & Furious 70MAX", Quarter Finals | Brussels, Belgium            | Extra round decision (split) | 4       | 3:00   |
        | 2011-04-09 | Win      | Lahcen Ait Oussakour | Le Grande KO XI                                                                       | Liege, Belgium               | KO                           | 1       |        |
        | 2011-03-19 | Loss     | Gino Bourne          | Fight Night Turnhout                                                                  | Turnhout, Belgium            | DQ                           |         |        |
        | 2011-02-12 | Win      | Henri van Opstal     | War of the Ring                                                                       | Amsterdam, Netherlands       | Decision (unanimous)         | 3       | 3:00   |
        | 2010-12-04 | Win      | Alessandro Campagna  | Janus Fight Night 2010                                                                | Padua, Italy                 | Decision                     | 3       | 3:00   |
        | 2010-09-10 | Win      | Edson Fortes         | Ring Sensation Gala                                                                   | Utrecht, Netherlands         | Decision                     | 3       | 3:00   |
        | 2010-03-21 | Loss     | Mohamed Khamal       | K-1 World MAX 2010 West Europe Tournament, Final                                      | Utrecht, Netherlands         | KO (punch)                   | 2       |        |
        | 2010-03-21 | Win      | Anthony Kane         | K-1 World MAX 2010 West Europe Tournament, Semi Finals                                | Utrecht, Netherlands         | Decision                     | 3       | 3:00   |
        | 2010-03-21 | Win      | Bruno Carvalho       | K-1 World MAX 2010 West Europe Tournament, Quarter Finals                             | Utrecht, Netherlands         | Decision                     | 3       | 3:00   |
        | 2009-11-28 | Win      | Davy Deraedt         | Battle of the Kings 2009                                                              | Brussels, Belgium            | KO (punches)                 | 1       |        |
        | 2009-09-27 | Win      | Freddy Kemayo        | Almelo Fight for Delight                                                              | Almelo, Netherlands          | TKO                          |         |        |
        | 2009-03-14 | Win      | Viktor Sarezki       | War of the Ring                                                                       | Belgium                      | KO (punch to the body)       | 1       |        |
        | 2009-02-21 | Win      | Pedro Sedarous       | Turnhout Gala                                                                         | Turnhout, Belgium            | Decision                     | 5       | 3:00   |
        | 2009-01-31 | Win      | Dahou Naim           | Tielrode Gala                                                                         | Tielrode, Belgium            | 2nd extra round decision     | 5       | 3:00   |
        | 2008-09-20 | Win      | Abdallah Mabel       | S-Cup Europe 2008, Reserve Bout                                                       | Gorinchem, Netherlands       | Decision                     | 3       | 3:00   |
        | 2008-09-14 | Win      | Jordy Sloof          | The Outland Rumble                                                                    | Rotterdam, Netherlands       | KO (Right cross)             | 1       |        |
        | 2008-03-08 | Win      | Naraim Ruben         | Lommel Gala                                                                           | Lommel, Belgium              | TKO (retirement)             | 3       |        |
        | 2008-02-23 | Win      | Pierre Petit         | St. Job Gala                                                                          | St. Job, Belgium             | KO (Right punch)             | 2       |        |
        | 2008-01-26 | Win      | Yildiz Bullut        | Tielrode Gala                                                                         | Tielrode, Belgium            | TKO                          | 2       |        |
        | 2007-11-28 | Win      | Ibrahim Benazza      | Lint Gala                                                                             | Lint, Belgium                | Decision                     | 5       | 2:00   |
        | 2007-10-27 | Win      | Anthony Kane         | One Night in Bangkok                                                                  | Antwerp, Belgium             | Decision                     | 5       |        |

        User 2:
        **Term Explanations**:

        **1. KO (Knockout)**
        - **Definition**: A victory where the opponent is rendered unconscious or unable to continue the fight.
        - **Synonyms and Related Concepts**: Knockout, KO, knockout victory.
        - **Query Suggestions**:
        1. "Which fights ended in a knockout according to this table?"
        2. "How many knockout victories did each fighter achieve?"
        3. "What was the method used in the KO victories listed?"

        **2. TKO (Technical Knockout)**
        - **Definition**: A victory awarded when the referee stops the fight due to one fighter's inability to continue, but the opponent is not necessarily knocked out.
        - **Synonyms and Related Concepts**: Technical knockout, referee stoppage, TKO.
        - **Query Suggestions**:
        1. "Which bouts were concluded by TKO?"
        2. "How does a TKO affect the fight’s result compared to a KO?"
        3. "What are the circumstances leading to TKO victories in this data?"

        **3. DQ (Disqualification)**
        - **Definition**: A loss awarded when a fighter is disqualified for breaking the rules of the competition.
        - **Synonyms and Related Concepts**: Disqualification, disallowed, no contest.
        - **Query Suggestions**:
        1. "Which fight resulted in a disqualification?"
        2. "What were the reasons for disqualification in this table?"
        3. "How often does disqualification occur in the fights listed?"

        **4. Method**
        - **Definition**: The technique or reason by which a fighter won or lost the match, such as KO, TKO, or decision.
        - **Synonyms and Related Concepts**: Victory method, fight outcome, result.
        - **Query Suggestions**:
        1. "What methods were most common in the fights recorded in this table?"
        2. "Which method resulted in the most victories?"
        3. "How does the method of victory affect overall fighter performance?"

        **5. Round**
        - **Definition**: The segments or phases in a fight, where each round represents a period of competition followed by a break.
        - **Synonyms and Related Concepts**: Fight round, period, round number.
        - **Query Suggestions**:
        1. "In which round did the majority of fights end?"
        2. "How many rounds were fought in the longest match listed?"
        3. "What was the round distribution for fights ending in KO?"

        Now, generate terminology explanations for the following table:
        Table caption (if available): {caption}
        Table summary: {summary}
        Table data (in its original format): {table}
        Context (if provided): {context}
        """

        generated_text = self.generate_text(prompt)
        
        print("Generated terminology explanations: ")
        print(generated_text)

        return generated_text


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_table_summary(self, metadata_list: list, context: list, table: dict, query: str, caption: str) -> str:
        """
        Generate a summary for a table that directly addresses a given query, using metadata and context.

        :param metadata_list: List of metadata from related Wikipedia documents.
        :param context: Additional context about the table.
        :param table: Dictionary representing the table's data.
        :param query: The query or statement to be addressed by the summary.
        :param caption: Caption of the table for context.
        :return: JSON string containing the generated summary.
        """
        prompt = f"""
        Example: You will be given a table, a query, the table's caption, metadata from related Wikipedia documents, and the context of the table. 
        Your task is to generate a concise summary for the table that directly addresses the query, using the Wikipedia metadata and the context to enhance understanding. 
        Ensure the summary begins by rephrasing or summarizing the query in a way that naturally introduces the purpose of the table. 
        Do not directly reveal the answer, but guide the reader to make an informed decision based on the provided information.

        Now, generate a summary for the given table, addressing the query and using the Wikipedia metadata and the context provided for enhanced understanding. 
        Ensure the summary starts by rephrasing or summarizing the query to introduce the table's purpose and includes only content related to the query. 
        Please avoid directly revealing the answer.

        Query:
        {query}

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
            "summary": "The summary that rephrases the query, includes context from the caption, and incorporates relevant Wikipedia information."
        }}

        """

        generated_text = self.generate_text(prompt)
        return generated_text



    def generate_table_summary_2(self, caption: str, table: str, context: list) -> str:
        """
        Generate a summary for a table that directly addresses a given query, using metadata and context.

        :param table: String representing the table's data in various formats (Markdown, HTML, or plain text).
        :param context: Additional context about the table.
        :param caption: Caption of the table for context.
        :return: JSON string containing the generated summary.
        """

        try:

            prompt = f"""

            You might receive tables in various formats, including Markdown, HTML, or plain text strings. Along with the table data, you might also receive additional context and a table caption, if available.

            Your task is generating a detailed summary for the table based on the provided content. The summary should include:

            1. **Table Title and Keywords**: Create a descriptive title for the table and extract relevant keywords. The title and keywords should be detailed enough to capture the essence of the table and its primary data.

            2. **Table Content Overview**: Provide a clear and concise summary of the table's contents. Include key information such as data fields, structure, and the overall purpose or focus of the table. The overview should offer enough detail to understand the table's main data without requiring specific domain knowledge.

            3. **Data Patterns and Common Trends**: Summarize any observable patterns, trends, or key data points in the table. Highlight significant statistics, recurring themes, or other notable insights present in the data.

            **Example 1: Swimming Competition Results Table**
            User 1:
            Table:
            | Rank   |   Lane | Name                   | Nationality   | Time    | Notes   |
            |:-------|-------:|:-----------------------|:--------------|:--------|:--------|
            |        |      4 | Sophie Pascoe          | New Zealand   | 2:25.65 | WR      |
            |        |      5 | Summer Ashley Mortimer | Canada        | 2:32.08 |         |
            |        |      3 | Zhang Meng             | China         | 2:33.95 | AS      |
            | 4      |      6 | Katherine Downie       | Australia     | 2:34.64 |         |
            | 5      |      2 | Nina Ryabova           | Russia        | 2:35.65 |         |
            | 6      |      8 | Aurelie Rivard         | Canada        | 2:37.70 |         |
            | 7      |      7 | Harriet Lee            | Great Britain | 2:39.42 |         |
            | 8      |      1 | Gemma Almond           | Great Britain | 2:42.16 |         |

            User 2:
            **Title**: International Swimming Competition Results with World and Asian Record Highlights
            **Keywords**: Swimmer name, nationality, time, ranking, world record, Asian record, lane number
            **Content Overview**: This table contains the final results of an international swimming competition, showing the rank, name, nationality, lane, and final time for each swimmer. It also highlights world records (WR) and Asian records (AS) set during the event. The data allows comparison of swimmer performances across different nations.
            **Data Patterns and Trends**: Swimmers with the fastest times tend to occupy the top ranks, with two notable record-breaking performances. New Zealand’s Sophie Pascoe set a world record, and China’s Zhang Meng set an Asian record.

            **Example 2: Fight Record Table**
            User 1:
            Table:
            | Date       | Result   | Opponent             | Event                                                                                 | Location                     | Method                       | Round   | Time   |
            |:-----------|:---------|:---------------------|:--------------------------------------------------------------------------------------|:-----------------------------|:-----------------------------|:--------|:-------|
            | 2013-12-14 | Loss     | Mohamed Diaby        | Victory, Semi Finals                                                                  | Paris, France                | Decision                     | 3       | 3:00   |
            | 2013-03-09 |          | Juanma Chacon        | Enfusion Live: Barcelona                                                              | Barcelona, Spain             |                              |         |        |
            | 2012-05-27 | Loss     | Murthel Groenhart    | K-1 World MAX 2012 World Championship Tournament Final 16                             | Madrid, Spain                | KO (punches)                 | 3       | 3:00   |
            | 2012-02-11 | Win      | Francesco Tadiello   | Sporthal De Zandbergen                                                                | Sint-Job-in-'t-Goor, Belgium | KO                           | 1       |        |
            | 2012-01-28 | Win      | Chris Ngimbi         | It's Showtime 2012 in Leeuwarden                                                      | Leeuwarden, Netherlands      | TKO (cut)                    | 2       | 1:22   |
            | 2011-09-24 | Loss     | Andy Souwer          | BFN Group & Music Hall presents: It's Showtime "Fast & Furious 70MAX", Quarter Finals | Brussels, Belgium            | Extra round decision (split) | 4       | 3:00   |
            | 2011-04-09 | Win      | Lahcen Ait Oussakour | Le Grande KO XI                                                                       | Liege, Belgium               | KO                           | 1       |        |
            | 2011-03-19 | Loss     | Gino Bourne          | Fight Night Turnhout                                                                  | Turnhout, Belgium            | DQ                           |         |        |
            | 2011-02-12 | Win      | Henri van Opstal     | War of the Ring                                                                       | Amsterdam, Netherlands       | Decision (unanimous)         | 3       | 3:00   |
            | 2010-12-04 | Win      | Alessandro Campagna  | Janus Fight Night 2010                                                                | Padua, Italy                 | Decision                     | 3       | 3:00   |
            | 2010-09-10 | Win      | Edson Fortes         | Ring Sensation Gala                                                                   | Utrecht, Netherlands         | Decision                     | 3       | 3:00   |
            | 2010-03-21 | Loss     | Mohamed Khamal       | K-1 World MAX 2010 West Europe Tournament, Final                                      | Utrecht, Netherlands         | KO (punch)                   | 2       |        |
            | 2010-03-21 | Win      | Anthony Kane         | K-1 World MAX 2010 West Europe Tournament, Semi Finals                                | Utrecht, Netherlands         | Decision                     | 3       | 3:00   |
            | 2010-03-21 | Win      | Bruno Carvalho       | K-1 World MAX 2010 West Europe Tournament, Quarter Finals                             | Utrecht, Netherlands         | Decision                     | 3       | 3:00   |
            | 2009-11-21 | Win      | Seo Doo Won          | It's Showtime 2009 Barneveld                                                          | Barneveld, Netherlands       | TKO (referee stoppage)       | 1       |        |
            | 2009-09-24 | Win      | Chris Ngimbi         | It's Showtime 2009 Lommel                                                             | Lommel, Belgium              | Extra round decision         | 4       | 4:00   |
            | 2009-04-11 | Win      | Farid Riffi          | Almelo Fight for Delight                                                              | Almelo, Netherlands          | TKO                          |         |        |
            | 2009-03-14 | Win      | Viktor Sarezki       | War of the Ring                                                                       | Belgium                      | KO (punch to the body)       | 1       |        |
            | 2009-02-21 | Win      | Pedro Sedarous       | Turnhout Gala                                                                         | Turnhout, Belgium            | Decision                     | 5       | 3:00   |
            | 2009-01-31 | Win      | Dahou Naim           | Tielrode Gala                                                                         | Tielrode, Belgium            | 2nd extra round decision     | 5       | 3:00   |
            | 2008-09-20 | Win      | Abdallah Mabel       | S-Cup Europe 2008, Reserve Bout                                                       | Gorinchem, Netherlands       | Decision                     | 3       | 3:00   |
            | 2008-09-14 | Win      | Jordy Sloof          | The Outland Rumble                                                                    | Rotterdam, Netherlands       | KO (Right cross)             | 1       |        |
            | 2008-03-08 | Win      | Naraim Ruben         | Lommel Gala                                                                           | Lommel, Belgium              | TKO (retirement)             | 3       |        |
            | 2008-02-23 | Win      | Pierre Petit         | St. Job Gala                                                                          | St. Job, Belgium             | KO (Right punch)             | 2       |        |
            | 2008-01-26 | Win      | Yildiz Bullut        | Tielrode Gala                                                                         | Tielrode, Belgium            | TKO                          | 2       |        |
            | 2007-11-28 | Win      | Ibrahim Benazza      | Lint Gala                                                                             | Lint, Belgium                | Decision                     | 5       | 2:00   |
            | 2007-10-27 | Win      | Anthony Kane         | One Night in Bangkok                                                                  | Antwerp, Belgium             | Decision                     | 5       |        |

            User 2:
            **Title**: Detailed Fighter Records and Outcomes from 2007 to 2013
            **Keywords**: Fight date, opponent, result, method (KO, decision), event location, fight rounds
            **Content Overview**: This table provides an extensive record of fights from 2007 to 2013, showing the date, result (win or loss), opponent, event location, method of victory/defeat (KO, TKO, decision), and the number of rounds. The table helps track fighter performance over time, as well as notable fight outcomes.
            **Data Patterns and Trends**: The table shows a mix of wins and losses for different fighters, with a notable trend of knockouts (KO) in several matches. Some fighters appear multiple times, indicating repeat matchups. The table also indicates consistent performance from specific fighters in certain locations.

            The final output should be structured in the following format, without using any extra wrapping or JSON-like syntax:

            **Title**: [Generated title]
            **Keywords**: [Generated keywords]
            **Content Overview**: [Generated content overview]
            **Data Patterns and Trends**: [Generated patterns and trends]

            Now, please generate summary for the following table as requirements:

            Table caption (if available):
            {caption}

            Table data (in its original format):
            {table}

            Context (if provided):
            {context}

            """
            

            generated_text = self.generate_text(prompt)
            
            print("Generated summary: ")
            print(generated_text)

            return generated_text
        
        except Exception as e:
            print(f"Error occurred in e2ewtq_generate_table_summary: {e}")
            traceback.print_exc()  
            return ""

    def generate_query_suggestions(self, caption: str,  summary: str, table: str, context: list) -> str:

        try:
            prompt = f"""

            You might receive tables in various formats, including Markdown, HTML, or plain text strings. Along with the table data, you might also receive table summary, additional context and a table caption, if available.
            Your task is generating 5 example queries that someone might use to search for information in this table.
            Ensure the queries are diverse, using different sentence structures, and cover a range of potential user needs or scenarios. Avoid repetition and ensure each query addresses a unique aspect of the table.

            Example 1 (Swimming Competition Results):
            User 1:
            Table data:
            | Rank   |   Lane | Name                   | Nationality   | Time    | Notes   |
            |:-------|-------:|:-----------------------|:--------------|:--------|:--------|
            |        |      4 | Sophie Pascoe          | New Zealand   | 2:25.65 | WR      |
            |        |      5 | Summer Ashley Mortimer | Canada        | 2:32.08 |         |
            |        |      3 | Zhang Meng             | China         | 2:33.95 | AS      |
            | 4      |      6 | Katherine Downie       | Australia     | 2:34.64 |         |
            | 5      |      2 | Nina Ryabova           | Russia        | 2:35.65 |         |
            | 6      |      8 | Aurelie Rivard         | Canada        | 2:37.70 |         |
            | 7      |      7 | Harriet Lee            | Great Britain | 2:39.42 |         |
            | 8      |      1 | Gemma Almond           | Great Britain | 2:42.16 |         |
            
            Table Summary:
            **Title**: International Swimming Competition Results with World and Asian Record Highlights
            **Keywords**: Swimmer name, nationality, time, ranking, world record, Asian record, lane number
            **Content Overview**: This table contains the final results of an international swimming competition, showing the rank, name, nationality, lane, and final time for each swimmer. It also highlights world records (WR) and Asian records (AS) set during the event. The data allows comparison of swimmer performances across different nations.
            **Data Patterns and Trends**: Swimmers with the fastest times tend to occupy the top ranks, with two notable record-breaking performances. New Zealand’s Sophie Pascoe set a world record, and China’s Zhang Meng set an Asian record.
            
            User 2:
            Query Suggestions:
            1. "Who set the World Record in this swimming competition?"
            2. "Which swimmer finished in the top three positions in this event?"
            3. "What was the finishing time of the swimmer from China?"
            4. "How did swimmers from Great Britain perform in this competition?"
            5. "how long did it take aurelie rivard to finish?"

            Example 2 (Filmography Table):
            User 1:

            Table data:
            |   Year | Title                                    | Role                          | Notes           |
            |-------:|:-----------------------------------------|:------------------------------|:----------------|
            |   1995 | Polio Water                              | Diane                         | Short film      |
            |   1996 | New York Crossing                        | Drummond                      | Television film |
            |   1997 | Lawn Dogs                                | Devon Stockard                |                 |
            |   1999 | Pups                                     | Rocky                         |                 |
            |   1999 | Notting Hill                             | 12-Year-Old Actress           |                 |
            |   1999 | The Sixth Sense                          | Kyra Collins                  |                 |
            |   2000 | Paranoid                                 | Theresa                       |                 |
            |   2000 | Skipped Parts                            | Maurey Pierce                 |                 |
            |   2000 | Frankie & Hazel                          | Francesca 'Frankie' Humphries | Television film |
            |   2001 | Lost and Delirious                       | Mary 'Mouse' Bedford          |                 |
            |   2001 | Julie Johnson                            | Lisa Johnson                  |                 |
            |   2001 | Tart                                     | Grace Bailey                  |                 |
            |   2002 | A Ring of Endless Light                  | Vicky Austin                  | Television film |
            |   2003 | Octane                                   | Natasha 'Nat' Wilson          |                 |
            |   2006 | The Oh in Ohio                           | Kristen Taylor                |                 |
            |   2007 | Closing the Ring                         | Young Ethel Ann               |                 |
            |   2007 | St Trinian's                             | JJ French                     |                 |
            |   2007 | Virgin Territory                         | Pampinea                      |                 |
            |   2008 | Assassination of a High School President | Francesca Fachini             |                 |
            |   2009 | Walled In                                | Sam Walczak                   |                 |
            |   2009 | Homecoming                               | Shelby Mercer                 |                 |
            |   2010 | Don't Fade Away                          | Kat                           |                 |
            |   2011 | You and I                                | Lana                          |                 |
            |   2012 | Into the Dark                            | Sophia Monet                  |                 |
            |   2012 | Ben Banks                                | Amy                           |                 |
            |   2012 | Apartment 1303 3D                        | Lara Slate                    |                 |
            |   2012 | Cyberstalker                             | Aiden Ashley                  | Television film |
            |   2013 | Bhopal: A Prayer for Rain                | Eva Gascon                    |                 |
            |   2013 | A Resurrection                           | Jessie                        | Also producer   |
            |   2013 | L.A. Slasher                             | The Actress                   |                 |
            |   2013 | Gutsy Frog                               | Ms. Monica                    | Television film |

            Table summary:
            **Title**: Filmography of Notable Roles from 1995 to 2013  
            **Keywords**: Year, Title, Role, Notes, Film, Television, Short film, Producer  
            **Content Overview**: This table presents a chronological filmography detailing various roles played by an actress from 1995 to 2013. It includes columns for the year of release, title of the film or television project, the role played by the actress, and any additional notes, such as whether the project was a short film or a television film. The table serves as a comprehensive overview of the actress's career, showcasing her diverse roles across different genres and formats.  
            **Data Patterns and Trends**: The actress has participated in a wide range of projects, including short films, television films, and feature films. Notably, the late 1990s and early 2000s show a concentration of roles, indicating a peak in her career during that time. Additionally, the table reveals her involvement in several television films, highlighting a trend towards this format in her later years. The presence of producer credits in some entries indicates her expanding role in the industry beyond acting.  
            
            Query Suggestions:
            1. "What roles did the actress play in films released in 2001?"
            2. "Can you list all the television films that the actress was involved in?"
            3. "Which film marked the actress's debut in 1995?"
            4. "What are the notable projects the actress produced during her career?"
            5. "In which year did the actress appear in the film 'Notting Hill'?"

            Now, Please generate 5 example queries that someone might use to search for information in this table. 
            Ensure the queries are diverse, using different sentence structures, and cover a range of potential user needs or scenarios. Avoid repetition and ensure each query addresses a unique aspect of the table.
            
            Table caption (if available):
            {caption}

            Table data (in its original format):
            {table}

            Table summary:
            {summary}

            Context (if available):
            {context}

            return the query suggestions directly, without any other information. 
            """

            generated_text = self.generate_text(prompt)
            
            print("Generated queries: ")
            print(generated_text)

            return generated_text
        
        except Exception as e:
            print(f"Error occurred in generate_query_suggestions: {e}")
            traceback.print_exc()  
            return ""

    def select_best_table(self, query: str, table1: str, table2: str) -> str:
        try:
            prompt = f"""

            You will receive a query along with two tables provided in various formats, such as Markdown, HTML, or plain text. These tables may include elements like titles, summaries, query suggestions, terminology, and abbreviation explanations.

            Your task is to determine, based on the provided query, which table is most likely to answer the question and return **only the passage_id** from the most appropriate table. You MUST choose between the two given passage_ids and return ONLY the passage_id.

            To make the best judgment, consider the following:

            1. **Matching of key terms, entities, and time**  
            Does the table contain the key terms from the query (such as people names, place names, organization names, or specific terms) and the specified time or date? These matches are often crucial in determining whether the table can answer the question.

            2. **Completeness of information matching**  
            Evaluate how well the information in the table matches the query. Does the table contain all or the most relevant content needed to answer the question?
s
            3. **Overall judgment**  
            If neither table is a perfect match, use the above two points as guidance, along with your own judgment, to choose the table most likely to answer the question.

            You must return **only the numeric value** of the passage_id from the table you determine to be the most suitable. 
            Choose between the two passage_ids: table1['passage_id'] or table2['passage_id'].

            Return only one of these two numbers without any additional text or formatting.

            Query: {query}

            Table 1:
            {table1}

            Table 2:
            {table2}

            You must return **only the numeric value** of the passage_id from the table you determine to be the most suitable. 
            Choose between the two passage_ids: table1['passage_id'] or table2['passage_id'].

            Return only one of these two numbers without any additional text or formatting.

            """

            generated_text = self.generate_text(prompt)

            return generated_text

        except Exception as e:
            print(f"Error occurred in select_best_table: {e}")
            traceback.print_exc()
            return ""


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

        if self.USE_SELF_CONSISTENCY:
            generated_texts = [self.generate_text(prompt) for _ in range(5)]
            print("Generated texts:", generated_texts)
            
            # Find the most common generated text
            text_counter = Counter(generated_texts)
            most_common_text, count = text_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_text
            else:
                return generated_texts[0]
        else:
            return self.generate_text(prompt).strip()  # Ensure any whitespace is removed, returning only the digit

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

        if self.USE_SELF_CONSISTENCY:
            generated_texts = [self.generate_text(prompt) for _ in range(5)]
            print("Generated texts:", generated_texts)
            
            # Find the most common generated text
            text_counter = Counter(generated_texts)
            most_common_text, count = text_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_text
            else:
                return generated_texts[0]
        else:
            return self.generate_text(prompt).strip()  # Ensure any whitespace is removed, returning only the digit

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

        if self.USE_SELF_CONSISTENCY:
            generated_texts = [self.generate_text(prompt) for _ in range(5)]
            print("Generated texts:", generated_texts)
            
            # Find the most common generated text
            text_counter = Counter(generated_texts)
            most_common_text, count = text_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_text
            else:
                return generated_texts[0]
        else:
            return self.generate_text(prompt).strip()  # Ensure any whitespace is removed, returning only the answer

    def nqtables_generate_final_answer(self, query, enhanced_info, formatted_filtered_table) -> str:

        # Check if terms explanation, table summary, or table context are empty, and use a placeholder if they are

        prompt = f"""
        You will receive a query and a table. Additionally, you may receive information such as the table title, table summary, or terms explanation to provide more context.
        The table content may be provided in string format, Markdown format, or HTML format.

        Your task is to determine the answer to the query based on the table, provided information, and any additional enhanced context.
        Pay close attention to the specific details and conditions mentioned in the query.
        Make sure to match all the given conditions in the query to ensure the answer is accurate.

        When answering the query, **strictly follow these rules**:
        
        1. **Always return only one answer**: Regardless of whether the query starts with "who", "which", or another question word, you should always return only one, most likely answer. Even if there are multiple possible answers, return just one unless the query explicitly asks for multiple answers.

        2. **For special formats (e.g., time, date)**: Always extract the exact format from the table. Do not attempt to reformat or interpret time, date, or other specific values differently. Simply provide the value exactly as it appears in the table.

        3. **Remove references and footnotes**: If a cell contains references, footnotes, or additional notations (e.g., "(1)", "(*)"), make sure to remove them in your final answer. Provide only the core information.

        4. **Do not add any additional explanation, justification, or extra information**: The answer should be as concise as possible, containing only the necessary details to answer the query.

        Now, based on the provided information, answer the following query. Ensure that your answer is concise, relevant, and avoids including unnecessary details.

        Query: 
        {query}
        
        Enhanced Info:
        {enhanced_info}

        Table: 
        {formatted_filtered_table}

        Provide only the answer as a single string without any additional text or formatting changes.
        """

        if self.USE_SELF_CONSISTENCY:
            generated_texts = [self.generate_text(prompt) for _ in range(5)]
            print("Generated texts:", generated_texts)
            
            # Find the most common generated text
            text_counter = Counter(generated_texts)
            most_common_text, count = text_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_text
            else:
                return generated_texts[0]
        else:
            return self.generate_text(prompt).strip()  # Ensure any whitespace is removed, returning only the answer


    def e2ewtq_generate_final_answer(self, query, enhanced_info, formatted_filtered_table) -> str:

        # Check if terms explanation, table summary, or table context are empty, and use a placeholder if they are

        prompt = f"""
        You will receive a query, a table. Additionally, you may receive information such as the table title, table summary, or terms explanation to provide more context.
        The table content may be provided in string format, Markdown format, or HTML format.

        Your task is to determine the answer to the query based on the table, provided information, and any additional enhanced context.
        Pay close attention to the specific details and conditions mentioned in the query.
        Make sure to match all the given conditions in the query to ensure the answer is accurate.

        When answering the query:
        1. **For questions involving "or" (e.g., "Was it X or Y?"),** make sure to **strictly choose between the provided options (X or Y)** without adding any extra words or creating your own response. Provide only one of the options as the answer unless the query explicitly requires otherwise.
        2. **For time-related questions (e.g., "How long did it take?"),** ensure you **strictly follow the format provided in the table** (e.g., hours, minutes, seconds). Avoid reformatting or interpreting the time differently.
        3. **Do not add any additional explanation, justification, or extra information.** Return the answer as a single string with no surrounding text.

        Now, answer the following query based on the provided information.

        Query: 
        {query}
        
        Enhanced Info:
        {enhanced_info}

        Table: 
        {formatted_filtered_table}

        Carefully match all specific conditions mentioned in the query.
        Provide only the answer as a single string without any additional text.
        """

        if self.USE_SELF_CONSISTENCY:
            generated_texts = [self.generate_text(prompt) for _ in range(5)]
            print("Generated texts:", generated_texts)
            
            # Find the most common generated text
            text_counter = Counter(generated_texts)
            most_common_text, count = text_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_text
            else:
                return generated_texts[0]
        else:
            return self.generate_text(prompt).strip()  # Ensure any whitespace is removed, returning only the answer
        

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

        Example 2:
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

        if self.USE_SELF_CONSISTENCY:
            generated_texts = [self.generate_text(prompt) for _ in range(5)]
            print("Generated texts:", generated_texts)
            
            # Find the most common generated text
            text_counter = Counter(generated_texts)
            most_common_text, count = text_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_text
            else:
                return generated_texts[0]
        else:
            return self.generate_text(prompt).strip()  # Ensure any whitespace is removed, returning only the answer








    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the prompt and instruction. If the prompt exceeds the maximum allowed length, it will be truncated,
        and the original (untruncated) prompt will be saved to a JSONL file for further analysis.

        :param prompt: The prompt to guide the text generation.
        :return: Generated text as a string.
        """
        MAX_PROMPT_LENGTH = 1048500  # 设置最大允许的prompt长度（略小于API最大字符限制）
        JSONL_FILE_PATH = "/home/yuhangwu/Desktop/Projects/tablerag/data/processed/truncated_data/truncated_prompts.jsonl"  # 保存截断内容的jsonl文件

        try:
            # 如果 prompt 长度超过最大限制，进行截断并保存原始内容
            if len(prompt) > MAX_PROMPT_LENGTH:
                print(f"Prompt is too long ({len(prompt)} characters). Truncating to {MAX_PROMPT_LENGTH} characters.")
                
                # 保存原始 prompt 到 jsonl 文件
                original_prompt_data = {
                    "timestamp": datetime.now().isoformat(),
                    "original_prompt": prompt,
                    "original_length": len(prompt)
                }
                
                with open(JSONL_FILE_PATH, 'a') as f:
                    f.write(json.dumps(original_prompt_data) + '\n')
                
                # 截断 prompt
                prompt = prompt[:MAX_PROMPT_LENGTH]

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



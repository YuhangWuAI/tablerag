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
        self.TOTAL_TOKENS = config["total_tokens"]
        self.MAX_TRUNCATE_TOKENS = config[
            "max_truncate_tokens"
        ]  # truncate text to this many tokens before calling the API
        self.EXAMPLE_TOKEN_LIMIT = config['example_token_limit']
        self.AUGMENTATION_TOKEN_LIMIT = config['augmentation_token_limit']
        self.MAX_ROWS = config[
            "max_rows"
        ]  # truncate tables to this many rows before calling the API
        self.MAX_COLUMNS = config[
            "max_columns"
        ]  # truncate tables to this many columns before calling the API
        self.USE_SELF_CONSISTENCY = config.get('use_self_consistency', False)

        self.client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY, 
            base_url=self.API_BASE
        )

class CallLLM:
    """Class for calling the OpenAI Language Model API."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config("config.json")
        self.EMBEDDING_MODEL = config.EMBEDDING_MODEL
        self.GPT_MODEL = config.GPT_MODEL
        self.OPENAI_API_KEY = config.OPENAI_API_KEY
        self.API_BASE = config.API_BASE
        self.BATCH_SIZE = config.BATCH_SIZE
        self.MAX_TRUNCATE_TOKENS = config.MAX_TRUNCATE_TOKENS
        self.MAX_ROWS = config.MAX_ROWS
        self.MAX_COLUMNS = config.MAX_COLUMNS
        self.EXAMPLE_TOKEN_LIMIT = config.EXAMPLE_TOKEN_LIMIT
        self.AUGMENTATION_TOKEN_LIMIT = config.AUGMENTATION_TOKEN_LIMIT
        self.TOTAL_TOKENS = config.TOTAL_TOKENS
        self.USE_SELF_CONSISTENCY = config.USE_SELF_CONSISTENCY
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["API_BASE"] = self.API_BASE
        self.client = config.client

    def call_llm_embedding(self, text: str) -> List[float]:
        """Return an embedding for a string."""
        openai.api_key = self.OPENAI_API_KEY
        response = openai.Embedding.create(model=self.EMBEDDING_MODEL, input=text)
        return response["data"][0]["embedding"]

    def call_llm_embeddings(self, text: List[str], file_path: str) -> List[List[float]]:
        """Return a list of embeddings for a list of strings."""
        openai.api_key = self.OPENAI_API_KEY
        embeddings = []
        for batch_start in range(0, len(text), self.BATCH_SIZE):
            batch_end = batch_start + self.BATCH_SIZE
            batch = text[batch_start:batch_end]
            # print(f"Batch {batch_start} to {batch_end-1}")
            response = openai.Embedding.create(model=self.EMBEDDING_MODEL, input=batch)
            for i, be in enumerate(response["data"]):
                assert (
                    i == be["index"]
                )  # double check embeddings are in same order as input
            batch_embeddings = [e["embedding"] for e in response["data"]]
            embeddings.extend(batch_embeddings)
        df = pd.DataFrame({"text": text, "embedding": embeddings})
        df.to_json(file_path, index=True)
        return embeddings

    def load_llm_embeddings(self, text: List[str], file_path: str) -> List[List[float]]:
        """Load a list of embedddigs with its origin text to a csv file."""
        if not os.path.exists(file_path):
            print("Embeddings file not found. Calling the API to generate embeddings.")
            return self.call_llm_embeddings(text, file_path)
        df = pd.read_json(file_path)
        try:
            df["embedding"] = df["embedding"].apply(lambda x: ast.literal_eval(x))
        except:
            df["embedding"] = df["embedding"]
        embeddings = df["embedding"].tolist()
        return embeddings

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.GPT_MODEL)
        return len(encoding.encode(text))

    def num_tokens_list(self, text: List[str]) -> int:
        """Return the number of tokens in a list of strings."""
        encoding = tiktoken.encoding_for_model(self.GPT_MODEL)
        return len(encoding.encode("".join([str(item) for item in text])))

    def truncated_string(
        self, string: str, token_limit: int, print_warning: bool = True
    ) -> str:
        """Truncate a string to a maximum number of tokens."""
        encoding = tiktoken.encoding_for_model(self.GPT_MODEL)
        encoded_string = encoding.encode(string)
        truncated_string = encoding.decode(encoded_string[:token_limit])
        if print_warning and len(encoded_string) > token_limit:
            print(
                f"Warning: Truncated string from {len(encoded_string)} tokens to {token_limit} tokens."
            )
        return truncated_string

    def parse_text_into_table(self, string: str) -> str:
        """Parse strings into a table format."""
        prompt = f"Example: A table summarizing the fruits from Goocrux:\n\n\
            There are many fruits that were found on the recently discovered planet Goocrux. \
            There are neoskizzles that grow there, which are purple and taste like candy. \
            There are also loheckles, which are a grayish blue fruit and are very tart, \
            a little bit like a lemon. Pounits are a bright green color and are more savory than sweet.\
            There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy. \
            Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, \
            and a pale orange tinge to them.\n\n| Fruit | Color | Flavor | \n\n {string} \n\n "
        return self.generate_text(prompt)

    def fill_in_cell_with_context(self, context: str) -> str:
        """Fill in the blank based on the column context."""
        prompt = f"Example: Fill in the blank based on the column context. \
            \n\n remaining | 4 | 32 | 59 | 8 | 113 | none | 2 | 156. \n\n 16 \n\n \
                Fill in the blank based on the column context {context}, Only return the value: "
        return self.generate_text(prompt)

    def fill_in_column_with_context(self, context: str) -> str:
        """Fill in the blank based on the column cells context."""
        prompt = f"Example: Fill in the blank (column name) based on the cells context. \
            \n\n none | 4 | 32 | 59 | 8 | 113 | 3 | 2 | 156. \n\n remarking \n\n \
                Fill in the blank based on the column cells context {context}, Only return the column name: "
        return self.generate_text(prompt)

    def call_llm_summarization(self, context: str) -> str:
        """Summarize the table context to a single sentence."""
        prompt = f"Example: Summarize the table context to a single sentence. \
            {context} \n\n: "
        return self.generate_text(prompt)


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def call_llm_code_generation(self, context: str) -> str:
        """Synthesize code snippet from the table context."""
        prompt = f"""
        Example: Synthesize code snippet from the table context to select the proper rows and columns for verifying a statement.
        The generated code must use the exact column names provided, including spaces, capitalization, and punctuation.
        The generated code should treat all data as strings, even if they look like numbers.

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
        Since we are interested in the 'wind farm', 'scheduled', 'capacity (mw)', and 'turbines' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['wind farm', 'scheduled', 'capacity (mw)', 'turbines']].query("turbines == '17'")

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
        Since we are interested in the 'club', 'played', 'drawn', and 'lost' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['club', 'played', 'drawn', 'lost']].query("played == '22'")

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
        Since we are interested in the 'event name', 'category', and 'established' columns, the most impactful change will be to filter the rows and columns as follows:
        filtered_table = df[['event name', 'established', 'category']].query("`event name` == 'touchdown atlantic' and category == 'sporting' and established == '2010'")

        Now, generate a code snippet from the table context to select the proper rows and columns to verify the given statement.
        Use the existing column names from the provided DataFrame.
        The column names in the generated code must match the provided column names exactly, including spaces, capitalization, and punctuation.
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
    def generate_table_summary(self, metadata_list: list, table: dict, statement: str, caption: str) -> str:
        prompt = f"""
        Example: You will be given a table, a statement, the table's caption, and metadata from related Wikipedia documents. Your task is to generate a concise summary for the table that directly addresses the statement, using the Wikipedia metadata to enhance understanding. Ensure the summary starts with the phrase 'This table is used to determine the truth of the statement: [statement]' and includes only content related to the statement.

        User 1:
        I need an expert to help me create a concise summary of the table. Here is the statement: The scheduled date for the farm with 17 turbines be 2012.
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

        User 2:
        Summary:
        "This table is used to determine the truth of the statement: The scheduled date for the farm with 17 turbines be 2012. The farm with 17 turbines, Garracummer, is scheduled for 2012 according to the table. Wind power in the Republic of Ireland is significant, with over 300 wind farms generating electricity. As of 2021, the Republic of Ireland has 4,309 MW of installed wind power capacity, contributing to a high wind power penetration in the country."

        Now, generate a summary for the given table, addressing the statement and using the Wikipedia metadata for enhanced understanding. Ensure the summary starts with the phrase 'This table is used to determine the truth of the statement: [statement]' and includes only content related to the statement.

        Statement:
        {statement}

        This table is used to determine the truth of the statement: {statement}

        Table caption:
        {caption}

        Table:
        {json.dumps(table, indent=2)}

        Wikipedia metadata:
        {json.dumps(metadata_list, indent=2)}

        Please return the result in the following format:
        {{
            "summary": "The summary that includes the statement, context from the caption, and relevant Wikipedia information."
        }}
        """

        generated_text = self.generate_text(prompt)
        return generated_text   

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_final_answer(self, query_need_to_answer: str, table_html: str, terms_explanation: str, table_summary: str) -> str:
        print("\n Calling OpenAI API for generating the final answer !!! \n")

        prompt = f"""
        Example: You will be given a statement, a table summary, the full table content, and terms explanations.
        Your task is to determine whether the statement is true or false based on the table and provided information.
        Return 1 if the statement is true, and 0 if it is false or if you cannot determine the answer based on the provided information.
        Provide only the number '1' or '0' as the answer without any additional text.

        User 1:
        Statement: "The scheduled date for the farm with 17 turbines is 2012."
        Table Summary: "This table is used to determine the truth of the statement: the scheduled date for the farm with 17 turbines is 2012. The Garracummer wind farm, which has 17 turbines, is indeed scheduled for 2012 as indicated in the table. Wind power is a significant energy source in Ireland, contributing to a high percentage of the country's electricity needs, with the Republic of Ireland boasting a total installed capacity of 4,309 MW as of 2021."
        Table:
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
        Table Summary: "This table is used to determine the truth of the statement: the most recent locomotive to be manufacture was made more than 10 years after the first was manufactured. The first locomotives listed in the table were manufactured between 1889 and 1907, while the most recent locomotive was manufactured in 1923. This indicates that the most recent locomotive was made 16 years after the first ones, thus supporting the truth of the statement. The list provides an overview of locomotives from the Palatinate Railway, highlighting the historical context of railway development in the region."
        Table:
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
        Table Summary: "{table_summary}"
        Table: {table_html}
        Terms Explanation: {terms_explanation}

        If you cannot determine whether the statement is true or false based on the provided information, return '0'. Otherwise, return '1' for true or '0' for false.
        Return only '1' or '0'.
        """



        # This is where the LLM is called to generate the answer
        generated_text = self.generate_text(prompt)
        return generated_text.strip()  # Ensure any whitespace is removed, returning only the digit





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

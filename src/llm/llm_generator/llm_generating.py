"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/
Copyright (C) 2024 Wu Yuhang. All rights reserved.
For any questions or further information, please feel free to reach out via the email address above.
"""

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
        Example: You will be given a table, a query, the table's caption, metadata from related Wikipedia documents, and the context of the table. Your task is to generate a concise summary for the table that directly addresses the query, using the Wikipedia metadata and the context to enhance understanding. Ensure the summary starts with the phrase 'This table is used to answer the query: [query]' and includes only content related to the query. Do not directly reveal the answer, but guide the reader to make an informed decision based on the provided information.

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
        """
        Generate the final answer for the TabFact dataset based on a query and table content.
        """
        print("\nCalling OpenAI API for generating the final answer!!!\n")

        if not terms_explanation.strip():
            terms_explanation = "[No terms explanation provided]"

        if not table_summary.strip():
            table_summary = "[No table summary provided]"

        if not table_context.strip():
            table_context = "[No additional context provided]"

        prompt = f"""
        Example: You will be given a statement, a table summary, the full table content, terms explanations, and possibly additional context.
        Your task is to determine whether the statement is true or false based on the table, provided information, and any additional context.
        Return 1 if the statement is true, and 0 if it is false or if you cannot determine the answer based on the provided information.
        Provide only the number '1' or '0' as the answer without any additional text.

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
            generated_answers = [self.generate_text(prompt).strip() for _ in range(5)]
            print("Generated answers:", generated_answers)
            
            # Find the most common answer
            answer_counter = Counter(generated_answers)
            most_common_answer, count = answer_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_answer
            else:
                return generated_answers[0]
        else:
            return self.generate_text(prompt).strip()


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def feverous_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        """
        Generate the final answer for the FEVEROUS dataset based on a query and table content.
        """
        print("\nCalling OpenAI API for generating the final answer for FEVEROUS dataset!!!\n")

        if not terms_explanation.strip():
            terms_explanation = "[No terms explanation provided]"

        if not table_summary.strip():
            table_summary = "[No table summary provided]"

        if not table_context.strip():
            table_context = "[No additional context provided]"

        prompt = f"""
        Example: You will be given a statement, a table summary, the full table content, terms explanations, and possibly additional context.
        Your task is to determine whether the statement is true, false, or if the evidence provided is insufficient.
        Return 1 if the statement is true, 0 if it is false, and 2 if you cannot determine the answer based on the provided information.
        Provide only the number '1', '0', or '2' as the answer without any additional text.

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
            generated_answers = [self.generate_text(prompt).strip() for _ in range(5)]
            print("Generated answers:", generated_answers)
            
            # Find the most common answer
            answer_counter = Counter(generated_answers)
            most_common_answer, count = answer_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_answer
            else:
                return generated_answers[0]
        else:
            return self.generate_text(prompt).strip()


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def hybridqa_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        """
        Generate the final answer for the HybridQA dataset based on a query and table content.
        """
        print("\nCalling OpenAI API for generating the final answer for HybridQA dataset!!!\n")

        if not terms_explanation.strip():
            terms_explanation = "[No terms explanation provided]"

        if not table_summary.strip():
            table_summary = "[No table summary provided]"

        if not table_context.strip():
            table_context = "[No additional context provided]"

        prompt = f"""
        Example: You will be given a query, a table summary, the full table content, terms explanations, and additional context.
        Your task is to determine the answer to the query based on the table, provided information, and any additional context.
        Pay close attention to the specific details and conditions mentioned in the query.
        Make sure to match all the given conditions in the query to ensure the answer is accurate.
        Return the answer as a single string without any additional text.

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
            generated_answers = [self.generate_text(prompt).strip() for _ in range(5)]
            print("Generated answers:", generated_answers)
            
            # Find the most common answer
            answer_counter = Counter(generated_answers)
            most_common_answer, count = answer_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_answer
            else:
                return generated_answers[0]
        else:
            return self.generate_text(prompt).strip()


    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def sqa_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        """
        Generate the final answer for the SQA dataset based on a query and table content.
        """
        print("\nCalling OpenAI API for generating the final answer for SQA dataset!!!\n")

        if not terms_explanation.strip():
            terms_explanation = "[No terms explanation provided]"

        if not table_summary.strip():
            table_summary = "[No table summary provided]"

        if not table_context.strip():
            table_context = "[No additional context provided]"

        prompt = f"""
        You will be given a query, a table summary, the full table content, terms explanations, and possibly additional context.
        Your task is to determine the answer to the query based on the table, provided information, and any additional context.
        Pay close attention to the specific details and conditions mentioned in the query.
        Make sure to match all the given conditions in the query to ensure the answer is accurate.
        Return the answer as a single string without any additional text.

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
            generated_answers = [self.generate_text(prompt).strip() for _ in range(5)]
            print("Generated answers:", generated_answers)
            
            # Find the most common answer
            answer_counter = Counter(generated_answers)
            most_common_answer, count = answer_counter.most_common(1)[0]
            
            if count > 1:
                return most_common_answer
            else:
                return generated_answers[0]
        else:
            return self.generate_text(prompt).strip()





    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the prompt and instruction.

        :param prompt: The prompt to guide the text generation.
        :return: Generated text as a string.
        """
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

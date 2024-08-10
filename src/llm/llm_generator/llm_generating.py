"""
Author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

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
        """
        Synthesize a code snippet from the provided table context.

        :param context: Contextual information to guide code generation.
        :return: Generated code as a string.
        """
        prompt = f"""
        Example: Synthesize code snippet from the table context to select the proper rows and columns for verifying a statement / answering query.
        The generated code must use the exact column names provided, including spaces, capitalization, and punctuation.
        The generated code should treat all data as strings, even if they look like numbers.
        Only filter out rows and columns that are definitely not needed to verify the statement / answering query.
        Now, generate a code snippet from the table context to select the proper rows and columns to verify the given statement / answering query.
        Use the existing column names from the provided DataFrame.
        The column names in the generated code must match the provided column names exactly, including spaces, capitalization, and punctuation.
        Only filter out rows and columns that are definitely not needed to verify the statement.
        Only return the code.
        {context}
        \n\n:
        """

        if self.USE_SELF_CONSISTENCY:
            # Generate multiple code snippets and choose the most common one
            generated_codes = [self.generate_text(prompt) for _ in range(5)]
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

        :param query_need_to_answer: The query that needs to be answered.
        :param table_formatted: The formatted table content.
        :param terms_explanation: Explanation of terms in the table.
        :param table_summary: Summary of the table.
        :param table_context: Additional context for the table.
        :return: The generated final answer as '1' for true, '0' for false.
        """
        print("\nCalling OpenAI API for generating the final answer !!!\n")

        # Use placeholders if terms explanation, table summary, or table context are empty
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

        generated_text = self.generate_text(prompt)
        return generated_text.strip()

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def feverous_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        """
        Generate the final answer for the FEVEROUS dataset based on a query and table content.

        :param query_need_to_answer: The query that needs to be answered.
        :param table_formatted: The formatted table content.
        :param terms_explanation: Explanation of terms in the table.
        :param table_summary: Summary of the table.
        :param table_context: Additional context for the table.
        :return: The generated final answer as '1' for true, '0' for false, or '2' for insufficient information.
        """
        print("\nCalling OpenAI API for generating the final answer for FEVEROUS dataset!!!\n")

        # Use placeholders if terms explanation, table summary, or table context are empty
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

        generated_text = self.generate_text(prompt)
        return generated_text.strip()

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def hybridqa_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        """
        Generate the final answer for the HybridQA dataset based on a query and table content.

        :param query_need_to_answer: The query that needs to be answered.
        :param table_formatted: The formatted table content.
        :param terms_explanation: Explanation of terms in the table.
        :param table_summary: Summary of the table.
        :param table_context: Additional context for the table.
        :return: The generated final answer as a single string.
        """
        print("\nCalling OpenAI API for generating the final answer for HybridQA dataset!!!\n")

        # Use placeholders if terms explanation, table summary, or table context are empty
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

        generated_text = self.generate_text(prompt)
        return generated_text.strip()

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def sqa_generate_final_answer(self, query_need_to_answer: str, table_formatted: str, terms_explanation: str, table_summary: str, table_context: str) -> str:
        """
        Generate the final answer for the SQA dataset based on a query and table content.

        :param query_need_to_answer: The query that needs to be answered.
        :param table_formatted: The formatted table content.
        :param terms_explanation: Explanation of terms in the table.
        :param table_summary: Summary of the table.
        :param table_context: Additional context for the table.
        :return: The generated final answer as a single string.
        """
        print("\nCalling OpenAI API for generating the final answer for SQA dataset!!!\n")

        # Use placeholders if terms explanation, table summary, or table context are empty
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

        generated_text = self.generate_text(prompt)
        return generated_text.strip()

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

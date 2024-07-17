import ast  # for converting embeddings saved as strings back to arrays
import json
import os
import time
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from typing import List, Tuple, Union
from scipy import spatial  # for calculating vector similarities for search
from tenacity import retry, stop_after_attempt, wait_random_exponential


class Config:
    def __init__(self, config_path: str = "config.json"):
        config = json.loads(open(config_path).read())
        self.EMBEDDING_MODEL = config["model"]["EMBEDDING_MODEL"]
        self.GPT_MODEL = config["model"]["GPT_MODEL"]
        self.OPENAI_API_KEY = config["api_key"]
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


class CallLLM:
    """Class for calling the OpenAI Language Model API."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config("config.json")
        self.EMBEDDING_MODEL = config.EMBEDDING_MODEL
        self.GPT_MODEL = config.GPT_MODEL
        self.OPENAI_API_KEY = config.OPENAI_API_KEY
        self.BATCH_SIZE = config.BATCH_SIZE
        self.MAX_TRUNCATE_TOKENS = config.MAX_TRUNCATE_TOKENS
        self.MAX_ROWS = config.MAX_ROWS
        self.MAX_COLUMNS = config.MAX_COLUMNS
        self.EXAMPLE_TOKEN_LIMIT = config.EXAMPLE_TOKEN_LIMIT
        self.AUGMENTATION_TOKEN_LIMIT = config.AUGMENTATION_TOKEN_LIMIT
        self.TOTAL_TOKENS = config.TOTAL_TOKENS
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY

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
        prompt = f"\
            Example: Synthesize code snippet from the table context to select the proper rows. \n \
            User 1: \nI need an expert to help me answer the question by making the table smaller. Question: Who are all of the players on the Westchester High School club team? \n \
            table = 'Player': ['Jarrett Jack', 'Jermaine Jackson', ...\
            'No.': ['1', '8', ..., 'Nationality': ['United States', 'United States', ...\
            'Position': ['Guard', 'Guard', ... \
            'Years in Toronto': ['2009-10', '2002-03', ... \
            'School/Club Team': ['Georgia Tech', 'Detroit', ... \
            User 2: \
            For 'Who are all of the players on the Westchester High School club\
            team?' the most impactful change will be to filter the rows. Since I \
            don't know all the rows I'll use rough string matching, float casting,\
            lowering and be as broad as possible.\
            >>> new_table = table[table.apply(lambda row_dict: 'Westchester' in\
            row_dict['School/Club Team'].lower(), axis=1)] \n \
            Synthesize code snippet from the table context to select the proper rows\n Only return the code \n {context} \n\n: "
        return self.generate_text(prompt)

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_text(self, prompt: str) -> str:
        """Generate text based on the prompt and instruction."""
        openai.api_key = self.OPENAI_API_KEY
        response = openai.Completion.create(
            model=self.GPT_MODEL,
            prompt=f"{prompt} \n\n:",
            temperature=0,
            max_tokens=96,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response["choices"][0]["text"].strip()

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_text(self, prompts: List[str], model_type: str) -> List[str]:
        """Generate text based on the prompt and instruction."""
        openai.api_key = self.OPENAI_API_KEY

        if model_type in ["text-davinci-003", "gpt-3.5-turbo-instruct"]:
            # batched examples, with xx completions per request
            response = openai.Completion.create(
                model=self.GPT_MODEL,
                prompt=prompts,
                temperature=0,
                max_tokens=96,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            # match completions to prompts by index
            responses = [""] * len(prompts)
            for choice in response.choices:
                responses[choice.index] = choice.text.strip()
            return responses
        elif model_type in ["gpt-3.5-turbo", "gpt-4"]:
            responses = [""] * len(prompts)
            for i in range(len(prompts)):
                prompts[i] = {"role": "user", "content": prompts[i]}
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[prompts[i]],
                    temperature=0,
                    max_tokens=96,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                responses[i] = response["choices"][0]["message"]["content"].strip()
                time.sleep(15)
            return responses
        else:
            return f"Model type '{model_type}' not supported."

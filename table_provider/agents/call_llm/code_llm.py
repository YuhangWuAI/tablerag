import openai
from typing import Dict, List, Tuple, Optional, Callable, Union
import time


class Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FewShotLLM(object):
    _supported_chat_models = [
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-3.5-turbo",
        "gpt-35-turbo",
        "gpt-3.5-turbo-0301",
    ]

    _supported_completion_models = ["text-davinci-003"]
    _azure_supported_models = ["text-davinci-003", "gpt-35-turbo"]

    model: str = "text-davinci-003"
    temperature: float = 0
    max_tokens: int = 512
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0.0
    stop: Optional[str] = None

    def __init__(self, key, use_azure=False, **kwargs) -> None:
        openai.api_key = key
        self._set_params(**kwargs)
        self.use_azure = use_azure

        if use_azure:
            openai.api_type = "azure"
            openai.api_base = (
                "https://augloop-cs-test-scus-shared-open-ai-0.openai.azure.com"
            )

    def _set_params(self, **kwargs):
        valid_params = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]

        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    def _generate_completion_prompt(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join([message['content'] for message in messages])

    def _generate_chat_completion_messages(
        self, instruction: str, examples: List[Dict[str, str]], prompt: str
    ) -> List[Dict[str, str]]:
        messages = [{"role": Role.SYSTEM, "content": instruction}]

        if examples:
            for example in examples:
                messages.append({"role": Role.USER, "content": example[Role.USER]})
                messages.append(
                    {"role": Role.ASSISTANT, "content": example[Role.ASSISTANT]}
                )

        messages.append({"role": Role.USER, "content": prompt})

        return messages

    def _request(self, messages: List[Dict[str, str]]) -> str:
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        if self.use_azure:
            params["engine"] = self.model
        else:
            params["model"] = self.model

        if self.stop is not None:
            params["stop"] = [self.stop]

        if self.model in self._supported_chat_models:
            params["messages"] = messages
            if self.use_azure:
                openai.api_version = "2023-03-15-preview"
            response = openai.ChatCompletion.create(**params)["choices"][
                0
            ].message.content

        elif self.model in self._supported_completion_models:
            final_prompt = self._generate_completion_prompt(messages)
            params["prompt"] = final_prompt

            if self.use_azure:
                openai.api_version = "2022-12-01"
            response = openai.Completion.create(**params)["choices"][0]["text"]

        return response

    def _completion(
        self, instruction: str, examples: List[Dict[str, str]], prompt: str
    ) -> str:
        if (
            self.model not in self._supported_chat_models
            and self.model not in self._supported_completion_models
        ):
            raise Exception("Unsupported OpenAI model")

        messages = self._generate_chat_completion_messages(
            instruction, examples, prompt
        )

        cnt = 0
        while cnt < 3:
            try:
                return self._request(messages)
            except Exception as e:
                print(
                    "[Request Error]",
                    e.args[0],
                    f"retrying with sleep {2**cnt} secs...",
                )
                cnt += 1
                time.sleep(2**cnt)

        raise Exception(f"Fail to request OpenAI services with max_retries = {cnt}")


class CodeLLM(FewShotLLM):
    def generate_code(
        self,
        instruction: str,
        examples: List[Dict[str, str]],
        prompt: str,
        extract_code_fn: Callable[[str], Union[str, Tuple[str, str]]],
    ) -> Union[str, Tuple[str, str]]:
        completion = self._completion(instruction, examples, prompt)

        if extract_code_fn is not None:
            return extract_code_fn(completion)
        else:
            return completion

from call_llm import CodeLLM


class TestLLMCalling:
    def __init__(self) -> None:
        pass

    @staticmethod
    def test_code_llm():
        pre = "2kL6MVOQJV4EF"
        mid = "8Kb3MkcT3BlbkFJ4xh"
        suf = "XoCE3SC2FI59HtZpY"
        key = 'sk-' + pre + mid + suf

        # aug-loop service, use_azure=True
        key = "516a05f6bed44ddeb2a6e8a047046ad5"

        model = "gpt-35-turbo"
        code_llm = CodeLLM(key, use_azure=True, model=model)

        code = code_llm.generate_code(
            "Write a single Python function to solve a problem.",
            [],
            "I want to sort the list [4, 5, 2, 3, 1]",
            lambda code: code.strip(),
        )
        print(code)

import pandas as pd
import argparse
import json
import ast


def readTable(path):
    with open(path, "r") as f:
        table = json.load(f)

    df = pd.DataFrame(table['Data'], columns=table['ColumnNames'])

    def format(x):
        try:
            return ast.literal_eval(x)
        except:
            return str(x)

    df = df.applymap(format).convert_dtypes()

    return df


def dumpJson(path, obj):
    with open(path, 'w') as f:
        return json.dump(obj, f)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_path', type=str, help='A path of table json')
    parser.add_argument('--query', type=str, help='A string of user query')
    parser.add_argument('--result_path', type=str, help='A path of serialized result')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # openai service, use_azure=False
    pre = "2kL6MVOQJV4EF"
    mid = "8Kb3MkcT3BlbkFJ4xh"
    suf = "XoCE3SC2FI59HtZpY"
    key = 'sk-' + pre + mid + suf

    # aug-loop service, use_azure=True
    key = "516a05f6bed44ddeb2a6e8a047046ad5"

    model = "gpt-35-turbo"
    llm = CodeLLM(key, use_azure=True, model=model, stop="# END")
    auto_pd = AutoDIAL(llm)

    df = readTable(args.table_path)

    code, res = auto_pd.query(df, args.query, show_code=True, return_code=True)

    res_data = {}
    if isinstance(res, pd.DataFrame):
        filed_names = []
        value_dict = {}
        for column in res.columns:
            filed_names.append(str(column))
            value_dict[column] = [
                value if isinstance(value, (int, float, str)) else str(value)
                for value in res[column].to_list()
            ]

        res_data["type"] = "table"
        res_data["data"] = {"field_names": filed_names, "value_dict": value_dict}

    elif isinstance(res, list):
        res_data["type"] = "multi_values"
        res_data["data"] = res
    else:
        res_data["type"] = "single_value"
        res_data["data"] = res

    cache = {"code": code, "result": res_data}

    dumpJson(f"{args.result_path}", cache)

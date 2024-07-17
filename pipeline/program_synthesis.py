import pandas as pd
import json
from .evaluation.evaluator import Evaluator
from table_provider import TableLoader, CallLLM, CodeLLM, Role


class ProgramGeneration(object):
    _max_retries: int = 5

    _PYTHONLIB = [
        "import pandas as pd",
        "from typing import List",
        "from pandas.core.groupby.generic import DataFrameGroupBy",
        "import sklearn",
    ]

    _INSTRUCTION = """Write Python code to solve a query problem.
You have to follow the rules:
1. Let's finish step by step within a single function with only one argument named `table`.
2. You need to write down the thought of operations in comment. 
3. You should specify the name of function by Json string in first comment. 
4. All table operation should apply to `table columns' given in the comment exactly. Column names are case-sensitive.
5. You need to convert columns into appropriate data types first and then do the query.
6. The function must return one variable being one of pd.DataFrame, List[int], List[float], List[str], int, float and str.
7. You must take the previous messages as examplars.
8. You must directly and concisely answer the query without any additional information.
9. You should prioritize using the following library:
<s>
{0}
</s>
""".format(
        "\n".join(_PYTHONLIB)
    )

    _USER_QUERY_TEMPLATE = """Pandas dataframe (table) with {num_rows} rows and {num_columns} columns.
The output of `print(table.head({rows_to_display}).to_csv(index=False))` is:
<s>
{table_head}
</s>
table columns:
<s>
{table_columns}
</s>
column types:
<s>
{column_types}
</s>
Query:
<s>
{query}
</s>"""

    _EXAMPLES = [
        {
            Role.USER: """Pandas dataframe (table) with 20 rows and 7 columns.
The output of `print(table.head(2).to_csv(index=False))` is:
<s>
Year,Brand,Category,Model,Sales,Profit,Cost
1/1/2007,BMW,Compact,BMW 3-Series,142490,40109,12018
12/1/2007,BMW,Compact,BMW 4-Series,112464,39488,11182
1/1/2008,BMW,Compact,BMW 3-Series,112464,39488,11182
</s>
table columns:
<s>
[ 'Year', 'Brand', 'Category', 'Model', 'Sales', 'Profit', 'Cost' ]
</s>
column types:
<s>
{"Year": "string", "Brand": "string", "Category": "string", "Model": "string", "Sales": "Int64", "Profit": "Int64", "Cost": "Int64"}
</s>
Query:
<s>
sales of BMW in Jan
</s>
""",
            Role.ASSISTANT: """#  { "name": "bmw_jan_sales" }
def bmw_jan_sales(table: pd.DataFrame) -> int:
    
    # Converting Year column to date format
    table['Year'] = pd.to_datetime(table['Year'])
    
    # Filtering the data for BMW brand and Jan month
    bmw_jan_sales = table[(table['Brand']=='BMW') & (table['Year'].dt.month==1)]
    
    # Calculating the sum of sales of BMW in Jan
    bmw_jan_sales_sum = bmw_jan_sales['Sales'].sum()
    
    # Returning the sum of sales of BMW in Jan
    return int(bmw_jan_sales_sum)

# END""",
        },
        {
            Role.USER: """Pandas dataframe (table) with 20 rows and 7 columns.
The output of `print(table.head(2).to_csv(index=False))` is:
<s>
Year,Brand,Category,Model,Sales,Profit,Cost
1/1/2007,BMW,Compact,BMW 3-Series,142490,40109,12018
12/1/2007,BMW,Compact,BMW 4-Series,112464,39488,11182
1/1/2008,BMW,Compact,BMW 3-Series,112464,39488,11182
</s>
table columns:
<s>
[ 'Year', 'Brand', 'Category', 'Model', 'Sales', 'Profit', 'Cost' ]
</s>
column types:
<s>
{"Year": "string", "Brand": "string", "Category": "string", "Model": "string", "Sales": "Int64", "Profit": "Int64", "Cost": "Int64"}
</s>
Query:
<s>
sales_by_year
</s>
""",
            Role.ASSISTANT: """#  { "name": "sales_by_year" }
def sales_by_year(table: pd.DataFrame) -> pd.DataFrame:
    
    # Converting Year column to date format
    table['Year'] = pd.to_datetime(table['Year'])
    
    # Grouping the data by Year and calculating the sum of Sales, Profit and Cost columns
    grouped_data = table.groupby('Year').agg({'Sales':'sum', 'Profit':'sum', 'Cost':'sum'}).reset_index()
    
    # Returning the grouped data
    return grouped_data

# END""",
        },
        {
            Role.USER: """Pandas dataframe (table) with 10 rows and 4 columns.
The output of `print(table.head(2).to_csv(index=False))` is:
<s>
Island,Nickname,Area,Population (as of 2010)
Hawai'i, The Big Island, 4,028.0 sq mi, 185,079
Maui, The Valley Isle, 727.2 sq mi, 144,444
</s>
table columns:
<s>
[ 'Island', 'Nickname', 'Area', 'Population (as of 2010)' ]
</s>
column types:
<s>
{"Island": "string", "Nickname": "string", "Area": "string", "Population (as of 2010)": "string"}
</s>
Query:
<s>
Can you show me the island and nickname of each record?
</s>
""",
            Role.ASSISTANT: """#  { "name": "island_nickname" }
def island_nickname(table: pd.DataFrame) -> pd.DataFrame:
    
    # Selecting only the Island and Nickname columns
    island_nickname_data = table[['Island', 'Nickname']]
    
    # Returning the selected columns data
    return island_nickname_data

# END""",
        },
        {
            Role.USER: """Pandas dataframe (table) with 20 rows and 7 columns.
The output of `print(table.head(2).to_csv(index=False))` is:
<s>
Year,Brand,Category,Model,Sales,Profit,Cost
1/1/2007,BMW,Compact,BMW 3-Series,142490,40109,12018
12/1/2007,BMW,Compact,BMW 4-Series,112464,39488,11182
1/1/2008,BMW,Compact,BMW 3-Series,112464,39488,11182
</s>
table columns:
<s>
[ 'Year', 'Brand', 'Category', 'Model', 'Sales', 'Profit', 'Cost' ]
</s>
column types:
<s>
{"Year": "string", "Brand": "string", "Category": "string", "Model": "string", "Sales": "Int64", "Profit": "Int64", "Cost": "Int64"}
</s>
Query:
<s>
predict future sales of BMW in 2012.
</s>
""",
            Role.ASSISTANT: """#  { "name": "predict_bmw_sales" }
def predict_bmw_sales(table: pd.DataFrame) -> int:
    
    # Converting Year column to date format
    table['Year'] = pd.to_datetime(table['Year'])
    
    # Filtering the data for BMW brand and Jan month
    bmw_sales = table[table['Brand']=='BMW']
    
    # Select the date as the independent variable X and the sales as the dependent variable y
    X = data['Year'].values.reshape(-1, 1)
    y = data['Sales']

    # Create a linear regression model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()

    # Fit the model
    model.fit(X, y)

    # Predict future sales trends
    target_date = pd.to_datetime('2012-1-1')
    X_pred = pd.array([target_date.year]).reshape(-1, 1)
    predicted_sales = model.predict(X_pred)

    # The predictions
    predictions = pd.DataFrame({'Date': future_dates, 'Predicted Sales': predicted_sales})
    
    # Returning the predictions
    return predictions

# END""",
        },
    ]

    _DEBUGGING_INSTRUCTION = """You are a Python code debugging assistant.
You have to follow the rules:
1. You must correct the Python function and return a right one that fixes the given error.
2. You must generate the right function with the same signature as the given Python code.
3. You must take the previous messages as examplars.
4. The function must return one variable being one of pd.DataFrame, List[int], List[float], List[str], int, float and str.
5. You must only return the correct Python function without any addition explanation
"""
    _DEBUGGING_QUERY_TEMPLATE = """Pandas dataframe (table) with {num_rows} rows and {num_columns}  columns.
The output of `print(table.head({rows_to_display}).to_csv(index=False))` is:
<s>
{table_head}
</s>
table columns:
<s>
{table_columns}
</s>
column types: 
<s>
{column_types}
</s>
The user input the following query:
<s>
{query}
</s>
You generated this python code:
<s>
{function}
</s>
`{fn_name}(table)` fails with the following error:
<s>
{error_returned}
</s>
"""

    _DEBUGGING_EXAMPLES = [
        {
            Role.USER: """
Pandas dataframe (table) with 10 rows and 4 columns.
The output of `print(table.head(2).to_csv(index=False))` is:
<s>
Island,Nickname,Area,Population (as of 2010)
Hawai'i, The Big Island, 4,028.0 sq mi, 185,079
Maui, The Valley Isle, 727.2 sq mi, 144,444
</s>
table columns:
<s>
[ 'Island', 'Nickname', 'Area', 'Population (as of 2010)' ]
</s>
column types: 
<s>
{"Island": "string", "Nickname": "string", "Area": "string", "Population (as of 2010)": "string"}
</s>
The user input the following query:
<s>
Can you tell me the total population?
</s>
You generated this python code:
<s>
def total_population(table: pd.DataFrame) -> int:

    # Selecting only 'Population (as of 2010)' column
    population = table[['Population (as of 2010)']]

    # Calculating total population
    total_population = table['Population (as of 2010)'].sum()
    
    # Returning the total population
    return int(total_population)
</s>
`total_population(table)` fails with the following error:
<s>
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in total_population
ValueError: invalid literal for int() with base 10: '185,079144,444'
</s>
""",
            Role.ASSISTANT: """def total_population(table: pd.DataFrame) -> int:

    # Converting the 'Population (as of 2010)' column from string to numeric
    table['Population (as of 2010)'] = pd.to_numeric(table['Population (as of 2010)'])

    # Selecting only 'Population (as of 2010)' column
    population = table[['Population (as of 2010)']]

    # Calculating total population
    total_population = table['Population (as of 2010)'].sum()
    
    # Returning the total population
    return int(total_population)
# END""",
        }
    ]

    _llm: CodeLLM = None

    def __init__(self, llm) -> None:
        self._llm = llm

    def _check_dtype(result):
        valid_types = [pd.DataFrame, list, int, float, str]

        if not isinstance(result, tuple(valid_types)):
            raise ValueError(
                f"Invalid data type: {type(result)}, Please return one of [ pd.DataFrame, List[int], List[float], List[str], int, float, str ]"
            )

        if isinstance(result, list):
            if not all(
                isinstance(element, (int, float)) for element in result
            ) or not all(isinstance(element, str) for element in result):
                raise ValueError(
                    "Invalid list element type. Please return one of [ pd.DataFrame, List[int], List[float], List[str], int, float, str ]"
                )

    def _run_code(
        self,
        fn_name: str,
        function: str,
        table: pd.DataFrame,
        show_code: bool = False,
        return_code: bool = False,
    ):
        LIBVAR = locals()
        exec("\n".join(self._PYTHONLIB), globals(), LIBVAR)
        GLOBALVAR = globals()
        GLOBALVAR.update(LIBVAR)

        cnt = 0
        last_exception = None
        while cnt < self._max_retries:
            local_var = {"table": table}
            try:
                exec(
                    "\n".join([function, f"result = {fn_name}(table)"]),
                    GLOBALVAR,
                    local_var,
                )

                result = local_var["result"]
                # self._check_dtype(result)

                if return_code:
                    return function, result
                else:
                    return result
            except Exception as e:
                cnt += 1
                last_exception = e
                print("[Exception]", e.args[0])
                print("[Auto Debugging]")
                function = self._debug_code(fn_name, function, str(e), table)

                if show_code:
                    print("Corrected Code".center(64, "-"))
                    print(function)

        print(f"Auto debugging excel max_retries = {self._max_retries}.")
        return "Exception: " + str(last_exception)

    def _debug_code(
        self, fn_name: str, function: str, observation: str, table: pd.DataFrame
    ):
        debug_prompt = self._DEBUGGING_QUERY_TEMPLATE.format(
            num_rows=table.shape[0],
            num_columns=table.shape[1],
            rows_to_display=self._rows_to_display,
            table_head=self._table_head,
            table_columns=str(list(table.columns)),
            column_types=self._column_dtype,
            query=self._query,
            fn_name=fn_name,
            function=function,
            error_returned=observation,
        )

        return self._llm.generate_code(
            self._DEBUGGING_INSTRUCTION, self._DEBUGGING_EXAMPLES, debug_prompt, None
        )

    def query(
        self,
        table: pd.DataFrame,
        query: str,
        show_code: bool = False,
        return_code: bool = False,
        rows_to_display=5,
        code_process_before_1st_run=None,
    ):
        table = table.copy()
        self._table_head = table.head(rows_to_display).to_csv(index=False)
        self._rows_to_display = rows_to_display
        self._query = query

        column_dtype = table.dtypes.to_dict()
        column_dtype = {str(k): str(v) for k, v in column_dtype.items()}
        self._column_dtype = json.dumps(column_dtype)

        query_prompt = self._USER_QUERY_TEMPLATE.format(
            num_rows=table.shape[0],
            num_columns=table.shape[1],
            rows_to_display=rows_to_display,
            table_head=self._table_head,
            table_columns=str(list(table.columns)),
            column_types=self._column_dtype,
            query=query,
        )

        def extract_code_fn(code):
            splits = code.strip().strip("#").strip().split("\n")
            fn_name = json.loads(splits[0])["name"]
            fn = "\n".join(splits[1:])
            return fn_name, fn

        print("[Reqesting LLM]")
        fn_name, fn = self._llm.generate_code(
            self._INSTRUCTION, self._EXAMPLES, query_prompt, extract_code_fn
        )

        if code_process_before_1st_run is not None:
            fn_name, fn = code_process_before_1st_run(fn_name, fn)

        if show_code:
            print(fn)

        return self._run_code(fn_name, fn, table, show_code, return_code)


def program_synthesis(
    task_name: str,
    split: str,
    row_filter_type: str,
    k_shot: int,
):
    table_loader = TableLoader(task_name, split)
    row_filter = RowFilter(
        task_name=task_name,
        split=split,
        row_filter_type=row_filter_type,
    )

    aug_loop_key = json.loads("config.json")["augloop_key"]
    llm = CodeLLM(aug_loop_key, use_azure=True, model="gpt-35-turbo", stop="# END")
    auto_pd = ProgramGeneration(llm)

    call_llm = CallLLM("config.json")
    batch_size = call_llm.BATCH_SIZE

    # run the end2end with batch input
    num_samples = len(table_loader.dataset)
    num_batches = num_samples // batch_size
    remaining_samples = num_samples % batch_size

    # response & groundtruth
    grd, pred = [], []

    # generated codes
    codes = []

    batches = []
    for batch_num in range(num_batches):
        batch_prompt = []
        start_index = batch_num * batch_size
        end_index = start_index + batch_size
        batch = table_loader.dataset[start_index:end_index]
        for i in range(batch_size):
            parsed_sample = table_loader.parse_table(
                {key: value[i] for key, value in batch.items()}
            )
            query = parsed_sample["query"]
            grd.append(parsed_sample["label"])
            row_filter.run(query)
            df = row_filter.filtered_table[start_index + i]
            batch_prompt.append({"df": df, "query": query})
        batches.append(batch_prompt)
    if remaining_samples > 0:
        batch_prompt = []
        start_index = num_batches * batch_size
        end_index = start_index + remaining_samples
        batch = table_loader.dataset[start_index:end_index]
        for i in range(remaining_samples):
            parsed_sample = table_loader.parse_table(
                {key: value[i] for key, value in batch.items()}
            )
            query = parsed_sample["query"]
            grd.append(parsed_sample["label"])
            row_filter.run(query)
            df = row_filter.filtered_table[start_index + i]
            batch_prompt.append({"df": df, "query": query})
        batches.append(batch_prompt)
    for batch in batches:
        code, response = auto_pd.query(
            batch["df"],
            batch["query"],
            show_code=True,
            return_code=True,
        )
        codes.append(code)
        pred.append(response)

    # evaluation
    evaluation = Evaluator.run(grd, pred, task_name)
    print(evaluation)
    return evaluation

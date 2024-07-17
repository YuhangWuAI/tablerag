import pandas as pd
import re
import unicodedata
import recognizers_suite
from recognizers_suite import Culture
from typing import List, Dict

culture = Culture.English


def str_normalize(user_input, recognition_types=None):
    """A string normalizer which recognize and normalize value based on recognizers_suite"""
    user_input = str(user_input)
    user_input = user_input.replace("\\n", "; ")

    def replace_by_idx_pairs(orig_str, strs_to_replace, idx_pairs):
        assert len(strs_to_replace) == len(idx_pairs)
        last_end = 0
        to_concat = []
        for idx_pair, str_to_replace in zip(idx_pairs, strs_to_replace):
            to_concat.append(orig_str[last_end : idx_pair[0]])
            to_concat.append(str_to_replace)
            last_end = idx_pair[1]
        to_concat.append(orig_str[last_end:])
        return ''.join(to_concat)

    if recognition_types is None:
        recognition_types = [
            "datetime",
            "number",
            "ordinal",
            "percentage",
            "age",
            "currency",
            "dimension",
            "temperature",
        ]

    for recognition_type in recognition_types:
        if re.match("\d+/\d+", user_input):
            # avoid calculating str as 1991/92
            continue
        recognized_list = getattr(
            recognizers_suite, "recognize_{}".format(recognition_type)
        )(
            user_input, culture
        )  # may match multiple parts
        strs_to_replace = []
        idx_pairs = []
        for recognized in recognized_list:
            if not recognition_type == 'datetime':
                recognized_value = recognized.resolution['value']
                if str(recognized_value).startswith("P"):
                    # if the datetime is a period:
                    continue
                else:
                    strs_to_replace.append(recognized_value)
                    idx_pairs.append((recognized.start, recognized.end + 1))
            else:
                if recognized.resolution:  # in some cases, this variable could be none.
                    if len(recognized.resolution['values']) == 1:
                        strs_to_replace.append(
                            recognized.resolution['values'][0]['timex']
                        )  # We use timex as normalization
                        idx_pairs.append((recognized.start, recognized.end + 1))

        if len(strs_to_replace) > 0:
            user_input = replace_by_idx_pairs(user_input, strs_to_replace, idx_pairs)

    if re.match("(.*)-(.*)-(.*) 00:00:00", user_input):
        user_input = user_input[: -len("00:00:00") - 1]
        # '2008-04-13 00:00:00' -> '2008-04-13'
    return user_input


def generate_field_type(field_name, recognition_types=None) -> List[str]:
    if recognition_types is None:
        recognition_types = [
            "datetime",
            "number",
            "ordinal",
            "percentage",
            "age",
            "currency",
            "dimension",
            "temperature",
        ]

    recognition_list = []
    for recognition_type in recognition_types:
        try:
            # only get the first recognized result
            recognized_results = getattr(
                recognizers_suite, "recognize_{}".format(recognition_type)
            )(field_name, culture)
            if isinstance(recognized_results, List) and len(recognized_results) > 0:
                top_zero_recognition = recognized_results[0]
                if top_zero_recognition.type_name != "":
                    recognition_list.append(top_zero_recognition)
        except:
            continue
    if len(recognition_list) == 0:
        return ["Unknown"]
    return [recognition.type_name for recognition in recognition_list]


def convert_df_type(df: pd.DataFrame, lower_case=True):
    """
    A simple converter of dataframe data type from string to int/float/datetime.
    Leverage some efforts from https://github.com/HKUNLP/Binder/blob/main/utils/normalizer.py
    """

    def get_table_content_in_column(table):
        if isinstance(table, pd.DataFrame):
            header = table.columns.tolist()
            rows = table.values.tolist()
        else:
            # Standard table dict format
            header, rows = table['header'], table['rows']
        all_col_values = []
        for i in range(len(header)):
            one_col_values = []
            for _row in rows:
                one_col_values.append(_row[i])
            all_col_values.append(one_col_values)
        return all_col_values

    # Rename empty columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header == '':
            new_columns.append(
                'FilledColumnName'
            )  # Fixme: give it a better name when all finished!
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Rename duplicate columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header in new_columns:
            new_header, suffix = header, 2
            while new_header in new_columns:
                new_header = header + '_' + str(suffix)
                suffix += 1
            new_columns.append(new_header)
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Recognize null values like "-"
    null_tokens = ['', '-', '/']
    for header in df.columns:
        df[header] = df[header].map(lambda x: str(None) if x in null_tokens else x)

    # Convert the null values in digit column to "NaN"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_number_flag = True
        for row_i, cell_value in enumerate(one_col_values):
            try:
                float(cell_value)
            except Exception as e:
                if not cell_value in [str(None), str(None).lower()]:
                    # None or none
                    all_number_flag = False
        if all_number_flag:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(
                lambda x: "NaN" if x in [str(None), str(None).lower()] else x
            )

    # Normalize cell values.
    for header in df.columns:
        df[header] = df[header].map(lambda x: str_normalize(x))

    # Strip the mis-added "01-01 00:00:00"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_with_00_00_00 = True
        all_with_01_00_00_00 = True
        all_with_01_01_00_00_00 = True
        for row_i, cell_value in enumerate(one_col_values):
            if not str(cell_value).endswith(" 00:00:00"):
                all_with_00_00_00 = False
            if not str(cell_value).endswith("-01 00:00:00"):
                all_with_01_00_00_00 = False
            if not str(cell_value).endswith("-01-01 00:00:00"):
                all_with_01_01_00_00_00 = False
        if all_with_01_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[: -len("-01-01 00:00:00")])
            continue

        if all_with_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[: -len("-01 00:00:00")])
            continue

        if all_with_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[: -len(" 00:00:00")])
            continue

    # Do header and cell value lower case
    if lower_case:
        new_columns = []
        for header in df.columns:
            lower_header = str(header).lower()
            if lower_header in new_columns:
                new_header, suffix = lower_header, 2
                while new_header in new_columns:
                    new_header = lower_header + '-' + str(suffix)
                    suffix += 1
                new_columns.append(new_header)
            else:
                new_columns.append(lower_header)
        df.columns = new_columns
        for header in df.columns:
            # df[header] = df[header].map(lambda x: str(x).lower())
            df[header] = df[header].map(lambda x: str(x).lower().strip())

    # Recognize header type (for pandas dataframe)
    for header in df.columns:
        float_able = False
        int_able = False
        datetime_able = False

        # Recognize int & float type
        try:
            df[header].astype("float")
            float_able = True
        except:
            pass

        if float_able:
            try:
                if all(df[header].astype("float") == df[header].astype(int)):
                    int_able = True
            except:
                pass

        if float_able:
            if int_able:
                df[header] = df[header].astype(int)
            else:
                df[header] = df[header].astype(float)

        # Recognize datetime type
        try:
            df[header].astype("datetime64")
            datetime_able = True
        except:
            pass

        if datetime_able:
            df[header] = df[header].astype("datetime64")

    # Recognize header type (for normalization based on recognizers_suite)
    field_types = []
    for header in df.columns:
        list_1 = generate_field_type(header)
        list_2 = generate_field_type(df[header][:2].to_string())
        for item_1, item_2 in zip(list_1, list_2):
            if item_1 != "Unknown":
                field_types.append(item_1)
            elif item_2 != "Unknown":
                field_types.append(item_2)
            else:
                field_types.append("Unknown")
    return df, field_types


def normalize(x):
    """Normalize string."""
    # Copied from WikiTableQuestions dataset official evaluator.
    if x is None:
        return None
    # Remove diacritics
    x = ''.join(
        c for c in unicodedata.normalize('NFKD', x) if unicodedata.category(c) != 'Mn'
    )
    # Normalize quotes and dashes
    x = re.sub("[‘’´`]", "'", x)
    x = re.sub("[“”]", "\"", x)
    x = re.sub("[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub("((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub("(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub('^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub('\s+', ' ', x, flags=re.U).lower().strip()
    return x

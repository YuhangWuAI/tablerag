import unittest
import pandas as pd
from table_provider.agents import *


class TestAgents:
    def __init__(self) -> None:
        pass

    @staticmethod
    def run(self):
        self.test_str_normalize()
        self.test_field_type_generation()
        self.test_generate_numeric_range()
        self.test_generate_time_series_interval()
        self.test_metadata_api()

    @staticmethod
    def test_str_normalize(
        user_input: str = "I have two apples", expected_output: str = "2"
    ) -> None:
        """
        Test the str_normalize function.
        Example: TestUtils.test_str_normalize('2008-04-13 00:00:00', '2008-4-13 0:0:0')
        """
        result = str_normalize(user_input)
        assert (
            result == expected_output
        ), f"Expected '{expected_output}', but got '{result}'"
        print(f"{TestAgents.test_str_normalize.__name__} - Test passed!")

    @staticmethod
    def test_field_type_generation(
        data: pd.DataFrame = pd.DataFrame(), lower_case=True
    ) -> None:
        """
        Test the field_type_generation function.
        Example: TestUtils.test_field_type_generation(data)
        """
        data = pd.read_json("test/examples/test.json", orient="index")
        print(data)
        df, field_types = convert_df_type(data, lower_case)
        print(df)
        print(field_types)
        assert field_types is not None, f"Expected '{field_types}' to be not None"
        print(f"{TestAgents.test_field_type_generation.__name__} - Test passed!")

    @staticmethod
    def test_generate_numeric_range(
        data: pd.DataFrame = pd.DataFrame(), lower_case=True
    ) -> None:
        """
        Test the generate_numeric_range function.
        Example: TestUtils.test_generate_numeric_range(data)
        """
        data = pd.read_json("test/examples/test.json", orient="index")
        df, field_types = convert_df_type(data, lower_case)
        field_range = generate_numerical_range(df)

        print(field_range)

        assert field_types is not None, f"Expected '{field_range}' to be not None"
        print(f"{TestAgents.test_generate_numeric_range.__name__} - Test passed!")

    @staticmethod
    def test_generate_time_series_interval(
        data: pd.DataFrame = pd.DataFrame(), lower_case=True
    ) -> None:
        """
        Test the generate_time_series_interval function.
        Example: TestUtils.test_generate_time_series_interval(data)
        """
        data = pd.read_json("test/examples/test_time_series.json", orient="column")
        df, field_types = convert_df_type(data, lower_case)
        time_series = generate_time_series_intervals(df)

        print(time_series)

    @staticmethod
    def test_metadata_api():
        data = pd.read_json("test/examples/test_1.json")

        metadata_api = MetadataApi(
            'table_provider/agents/Metadata/model/model/metadata_tapas_202202_d0e0.pt'
        )
        emb = metadata_api.embedding(data)
        metadata_info = metadata_api.predict(data, emb)
        print(metadata_info)


class TestConvertDfType(unittest.TestCase):
    def setUp(self):
        # Set up some sample dataframes to use in our tests
        self.df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['4', '5', '6']})
        self.df2 = pd.DataFrame({'A': ['', '-', None], 'B': ['1.23', '4.56', '7.89']})

    def test_empty_column_names(self):
        # Test that empty column names are filled with a default value
        df = convert_df_type(pd.DataFrame({'': [1, 2, 3]}))
        self.assertListEqual(list(df.columns), ['FilledColumnName'])

    def test_duplicate_column_names(self):
        # Test that duplicate column names are renamed with a suffix
        df = convert_df_type(
            pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'A': [7, 8, 9]})
        )
        self.assertListEqual(list(df.columns), ['A', 'B', 'A_2'])

    def test_null_tokens(self):
        # Test that null tokens are recognized and replaced with None
        df = convert_df_type(self.df1.copy())
        self.assertListEqual(df.iloc[1].tolist(), [2, '5'])
        df = convert_df_type(self.df2.copy())
        self.assertListEqual(df.iloc[0].tolist(), [None, 1.23])

    def test_numeric_columns(self):
        # Test that numeric columns with null values are converted to NaN
        df = convert_df_type(self.df2.copy())
        self.assertTrue(pd.isna(df['A']).all())

    def test_string_normalization(self):
        # Test that cell values are normalized
        df = convert_df_type(pd.DataFrame({'A': [' AbCd  ', 'EFG']}))
        self.assertListEqual(df.iloc[:, 0].tolist(), ['abcd', 'efg'])

    def test_datetime_stripping(self):
        # Test that date/time strings are stripped if they end in a specific format
        df = convert_df_type(pd.DataFrame({'A': ['2021-01-01 00:00:00', '2021-01-01']}))
        self.assertListEqual(df.iloc[:, 0].tolist(), ['2021-01-01', '2021-01-01'])

    def test_lower_case(self):
        # Test that column headers and cell values are converted to lowercase
        df = convert_df_type(pd.DataFrame({'Abc': ['DeF ', 'Ghi'], 'JKl': [1, 2]}))
        self.assertListEqual(list(df.columns), ['abc', 'jkl'])
        self.assertListEqual(df.iloc[:, 0].tolist(), ['def', 'ghi'])

    def test_data_types(self):
        # Test that columns are correctly recognized as int, float, or datetime
        df = convert_df_type(self.df2.copy())
        self.assertTrue(df.dtypes['A'] == pd.Float64Dtype())
        self.assertTrue(df.dtypes['B'] == pd.Float64Dtype())

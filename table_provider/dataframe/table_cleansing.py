import pandas as pd
from ..agents import CallLLM
from ..contract.enum_type import TableCleansingType


class TableCleansing:
    """Class for cleaning dataframes."""

    @staticmethod
    def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with no text. Noted that this operation is very strict only all the cells in the row are empty,
        the row will be removed.
        """
        df = df.dropna(axis="index", how="all")
        return df

    @staticmethod
    def remove_empty_rows_with_threshold(
        df: pd.DataFrame, threshold: int
    ) -> pd.DataFrame:
        """Keep only the rows with at least {threshold} non-NA values."""
        df = df.dropna(axis="index", thresh=threshold)
        return df

    @staticmethod
    def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with no text. Noted that this operation is very strict only all the cells in the column are empty,
        the column will be removed.
        """
        df = df.dropna(axis="columns", how="all")
        return df

    @staticmethod
    def remove_empty_columns_with_threshold(
        df: pd.DataFrame, threshold: int
    ) -> pd.DataFrame:
        """Keep only the columns with at least {threshold} non-NA values."""
        df = df.dropna(axis="columns", thresh=threshold)
        return df

    @staticmethod
    def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        mask = df.duplicated(keep=False)
        df[mask].style.highlight_min(axis="index")  # highlight duplicate rows
        df = df.drop_duplicates()
        return df

    @staticmethod
    def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate columns."""
        mask = df.T.duplicated(keep=False)
        df.T[mask].style.highlight_min(axis="index")  # highlight duplicate columns
        df = df.T.drop_duplicates().T
        return df

    @staticmethod
    def fill_empty_cells(df: pd.DataFrame) -> pd.DataFrame:
        """Fill empty cells with column context."""
        call_llm = CallLLM("config.json")
        for col in df.columns:
            column_context = col + "|".join(df[col].dropna().tolist())
            df[col] = df[col].apply(
                lambda x: call_llm.fill_in_cell_with_context(column_context)
                if pd.isna(x) or x == "none" or pd.isnull(x)
                else x
            )
        return df

    @staticmethod
    def fill_empty_column_name(df: pd.DataFrame) -> pd.DataFrame:
        """Fill empty columns with empty strings."""
        call_llm = CallLLM("config.json")
        for col in df.columns:
            column_context = col + "|".join(df[col].dropna().tolist())
            if pd.isna(col) or col == "none" or pd.isnull(col):
                df = df.rename(
                    columns={col: call_llm.fill_in_column_with_context(column_context)}
                )
        return df

    @staticmethod
    def parse_text_into_structured_data(string: str) -> str:
        """Parse text into structured data. (Simplify the data/Use table to visualize the data)"""
        call_llm = CallLLM("config.json")
        return call_llm.parse_text_into_table(string)

    @staticmethod
    def remove_specific_string(df: pd.DataFrame, string: str) -> pd.DataFrame:
        """Remove specific string from dataframe."""
        df = df.replace(string, "")
        return df

    @staticmethod
    def replace_specific_pattern(
        df: pd.DataFrame, pattern: str, replacement: str
    ) -> pd.DataFrame:
        """Replace specific pattern in dataframe."""
        df = df.replace(pattern, replacement)
        return df

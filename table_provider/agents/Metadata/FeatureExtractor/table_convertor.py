import pandas as pd

def dataframe2table(df: pd.DataFrame):
    '''
    Convert dataframe to table
    '''
def get_field_type(col, formats, is_blank_dominated):
    strings_found = 0
    numbers_found = 0
    years_found = 0
    year_strings_found = 0
    dates_found = 0
    blanks_found = 0
    string_frequencies = {}
    sample_count = 0
    col_count = 0
    format_it = iter(formats) if formats else None
    for value in col:
        # Advance the format iterator if we have formats.
        # If there are fewer formats than values (should never happen), clear the iterator.
        if format_it is not None and not next(format_it, None):
            format_it = None

        sample_count += 1

        if value.is_number:
            if format_it is not None and format_it.current.is_date_time:
                dates_found += 1
            else:
                numbers_found += 1

            if is_numeric_year(value):
                years_found += 1
        else:
            str_value = value.string_value

            # Skip blank strings, and don't count them as samples (e.g. 10 numbers and 10 blank strings should still return a number type).
            if str_value and not str_value.isspace():
                strings_found += 1

                # If it looks like a year, bump the YearString count.
                if is_string_year(str_value):
                    year_strings_found += 1

                # Update the frequency table of non-blank strings.
                if str_value in string_frequencies:
                    string_frequencies[str_value] += 1
                else:
                    string_frequencies[str_value] = 1
            else:
                blanks_found += 1

        col_count += 1

    # For numeric prevalence calculations, we want to ignore blanks (if any) or ignore the most common string (to handle cases like '--' indicating a missing value).
    # Set sample_count_for_numbers to the appropriate adjustment of total sample count for numeric prevalence.
    # Don't bother with the most common nonblank string if all strings are unique.
    max_non_blank_string_count = max(string_frequencies.values()) if string_frequencies else 0
    if max_non_blank_string_count == 1:
        max_non_blank_string_count = 0

    sample_count_for_numbers = sample_count - (blanks_found if blanks_found > 0 else max_non_blank_string_count)

    # For string prevalence calculations, we only ignore true blanks.
    sample_count_for_strings = sample_count - blanks_found

    # The number of samples matching the same inferred type before we decide we know the column type from those values.
    # If the dataset is tiny, we want all of them to match; otherwise, use a percentage threshold.
    prevalence_threshold_percent = 0.9
    num_prevalence_threshold = sample_count_for_numbers if sample_count_for_numbers <= 2 else int(prevalence_threshold_percent * sample_count_for_numbers)
    string_prevalence_threshold = sample_count_for_strings if sample_count_for_strings <= 2 else int(prevalence_threshold_percent * sample_count_for_strings)

    # First check true blank domination with threshold rounded-down, regardless of the actual detection result.
    is_blank_dominated = blanks_found > 1 and blanks_found >= int(col_count * prevalence_threshold_percent)

    if num_prevalence_threshold > 0:
        # If the numeric threshold makes sense, check for numeric type prevalence.
        if dates_found >= num_prevalence_threshold:
            return 'DateTime'

        # Check for Year before Number, since Year is a subset of Number.
        if years_found >= num_prevalence_threshold:
            return 'Year'

        if numbers_found >= num_prevalence_threshold:
            return 'Decimal'

    if string_prevalence_threshold > 0:
        # Because YearString is a subset of String, check for YearString first.
        if year_strings_found >= string_prevalence_threshold:
            return 'YearString'

        if strings_found >= string_prevalence_threshold:
            return 'String'

    # If we didn't get a heavy prevalence of one type inferred from the values, Unknown is the right answer.
    return 'Unknown'


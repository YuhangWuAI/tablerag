# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from collections import Counter
from .feature_extractor_constants import (
    _DEFAULT_FORMAT_MAPPINGS,
    EPS,
    SAMPLED_LENGTH_LIMIT,
)


class SourceFeatures:
    def __init__(self, dt):
        self.dt = dt
        self.tUid = dt['tUid']
        self.pUids = dt['pUids']
        self.cUids = dt['cUids']
        self.isExternal = False
        self.nColumns = dt['nColumns']
        self.nRows = dt['nRows']
        self.fields = self.parse_fields()

    def parse_fields(self):
        parsed_fields = []
        fields = self.dt['fields']
        types = self.dt['fieldTypes']
        records = self.dt['records']
        shared_num_formats = self.dt['sharedNumFmts']
        spreadsheet_data = [[cell[1] for cell in row] for row in records]
        for current_field_idx, current_field in enumerate(fields):
            current_type = types[current_field_idx]
            current_data = [row[current_field_idx] for row in spreadsheet_data]
            current_num_format_id = str(current_field['numberFmtId'])
            current_parsed_field = {
                'name': current_field['name'],
                'index': current_field['index'],
                'type': current_type,
                'dataFeatures': self.get_data_features(
                    current_data, current_field_idx, current_type
                ),
                'inHeaderRegion': current_field['inHeaderRegion'],
            }

            current_num_format_string = shared_num_formats.get(current_num_format_id)
            if current_num_format_string == None:
                current_num_format_string = _DEFAULT_FORMAT_MAPPINGS.get(
                    current_num_format_id
                )
            if current_num_format_string != None:
                current_parsed_field['isPercent'] = (
                    '%' in current_num_format_string
                )  # If '%' in the format then True else False
                current_parsed_field['isCurrency'] = any(
                    _.startswith(currency_prefix)
                    for _ in current_data
                    for currency_prefix in ['$', '€', '¥', '£']
                )  # if any data starts with currency prefixes True else False
                current_parsed_field['hasYear'] = any(
                    year_string in current_num_format_string
                    for year_string in ['y', 'Y']
                )  # If 'y' or 'Y' in the format then True else False
                current_parsed_field['hasMonth'] = any(
                    month_string in current_num_format_string
                    for month_string in ['m', 'M']
                )  # If 'm' or 'M' in the format then True else False,
                current_parsed_field['hasDay'] = any(
                    day_string in current_num_format_string for day_string in ['d', 'D']
                )  # If 'd' or 'D' in the format then True else False
            else:
                current_parsed_field['isPercent'] = False
                current_parsed_field['isCurrency'] = False
                current_parsed_field['hasYear'] = False
                current_parsed_field['hasMonth'] = False
                current_parsed_field['hasDay'] = False

            parsed_fields.append(current_parsed_field)

        return parsed_fields

    def get_data_features(self, field_data, field_idx, field_type):
        data_features = {}
        if field_type in [0, 1, 3, 7]:  # 0(Unkown), 1(String), 3(DateTime), 7(Year)
            # string field
            (
                common_prefix_score,
                common_suffix_score,
                key_entropy_score,
                char_entropy_score,
            ) = self.analyze_string_and_entropy_feature(field_data)
            (
                change_rate,
                cardinality,
                absolute_cardinality,
                major,
            ) = self.analyze_string_feature(field_data)
            (
                median_length,
                length_variance,
                average_log_length,
            ) = self.analyze_string_length_feature(field_data)

            data_features['commonPrefix'] = common_prefix_score
            data_features['commonSuffix'] = common_suffix_score
            data_features['keyEntropy'] = key_entropy_score
            data_features['charEntropy'] = char_entropy_score
            data_features['changeRate'] = change_rate
            data_features['cardinality'] = cardinality
            data_features['major'] = major
            data_features['medianLength'] = median_length
            data_features['lengthVariance'] = length_variance
            data_features['averageLogLength'] = average_log_length
            data_features['absoluteCardinality'] = absolute_cardinality
            data_features['nRows'] = len(field_data)

        if field_type == 5:
            # numerics field
            (
                aggr_percent_formatted,
                aggr01_ranged,
                aggr0100_ranged,
                aggr_integers,
                aggr_negative,
            ) = self.get_aggr_features(field_data, field_idx)
            (
                common_prefix_score,
                common_suffix_score,
                key_entropy_score,
                char_entropy_score,
                benford,
            ) = self.analyze_string_and_entropy_feature_numerical(field_data)
            (
                data_range,
                change_rate,
                partial_ordered,
                variance,
                cov,
                cardinality,
                absolute_cardinality,
                spread,
                major,
            ) = self.analyze_numerical_feature(field_data)
            equal_progression_confidence = self.analyze_equal_progression(field_data)
            geometric_progression_confidence = self.analyze_geometric_progression(
                field_data
            )
            median_length, length_variance = self.analyze_string_length_feature_string(
                field_data
            )
            sum_in_01, sum_in_0100 = self.get_sum_features(field_data)
            skewness, kurtosis, gini = self.analyze_distribution_features(field_data)
            ordered_confidence = self.check_ordinal(field_data)

            data_features['aggrPercentFormatted'] = aggr_percent_formatted
            data_features['aggr01Ranged'] = aggr01_ranged
            data_features['aggr0100Ranged'] = aggr0100_ranged
            data_features['aggrIntegers'] = aggr_integers
            data_features['aggrNegative'] = aggr_negative
            data_features['commonPrefix'] = common_prefix_score
            data_features['commonSuffix'] = common_suffix_score
            data_features['keyEntropy'] = key_entropy_score
            data_features['charEntropy'] = char_entropy_score
            data_features['range'] = data_range
            data_features['changeRate'] = change_rate
            data_features['partialOrdered'] = partial_ordered
            data_features['variance'] = variance
            data_features['cov'] = cov
            data_features['cardinality'] = cardinality
            data_features['spread'] = spread
            data_features['major'] = major
            data_features['benford'] = benford
            data_features['orderedConfidence'] = ordered_confidence
            data_features['equalProgressionConfidence'] = equal_progression_confidence
            data_features[
                'geometricProgressionConfidence'
            ] = geometric_progression_confidence
            data_features['medianLength'] = median_length
            data_features['lengthVariance'] = length_variance
            data_features['sumIn01'] = sum_in_01
            data_features['sumIn0100'] = sum_in_0100
            data_features['absoluteCardinality'] = absolute_cardinality
            data_features['skewness'] = skewness
            data_features['kurtosis'] = kurtosis
            data_features['gini'] = gini
            data_features['nRows'] = len(field_data)
        return data_features

    def analyze_string_and_entropy_feature(self, field_data):
        if len(field_data) == 0:
            return 0, 0, 0, 0

        prefixes = {}
        suffixes = {}
        key_distribution = {}
        char_distribution = {}
        total_chars = 0
        sampled_count = 0

        # Build statistic dictionaries
        for cell in field_data:
            sampled_count += 1
            total_chars += len(cell)
            key_distribution[cell] = (
                key_distribution[cell] + 1 if cell in key_distribution else 1
            )
            for char in cell:
                char_distribution[char] = (
                    char_distribution[char] + 1 if char in char_distribution else 1
                )
            if len(cell) != 0:
                prefixes[cell[0]] = prefixes[cell[0]] + 1 if cell[0] in prefixes else 1
                suffixes[cell[-1]] = (
                    suffixes[cell[-1]] + 1 if cell[-1] in suffixes else 1
                )

        # Get the scores
        common_prefix_score = (
            max(prefixes.values()) / sampled_count if len(prefixes) != 0 else 0
        )
        common_suffix_score = (
            max(suffixes.values()) / sampled_count if len(suffixes) != 0 else 0
        )

        key_entropy_score = -sum(
            map(
                lambda x: x * np.log2(x),
                np.array(list(key_distribution.values())) / sampled_count,
            )
        )
        char_entropy_score = -sum(
            map(
                lambda x: x * np.log2(x),
                np.array(list(char_distribution.values())) / total_chars,
            )
        )

        return (
            common_prefix_score,
            common_suffix_score,
            key_entropy_score,
            char_entropy_score,
        )

    def analyze_string_feature(self, field_data):
        change_rate = 0

        # Ordinal Features
        if len(field_data) > 1:
            for cell_idx in range(1, len(field_data)):
                if field_data[cell_idx] != field_data[cell_idx - 1]:
                    change_rate += 1
            change_rate /= len(field_data) - 1
        else:
            change_rate = 0.5 * len(field_data)

        cnt = Counter(field_data)

        absolute_cardinality = len(cnt.keys())
        cardinality = (
            absolute_cardinality / len(field_data) if len(field_data) != 0 else 0
        )
        major = max(cnt.values()) / len(field_data) if len(field_data) != 0 else 1

        return change_rate, cardinality, absolute_cardinality, major

    def analyze_string_length_feature(self, field_data):
        if len(field_data) == 0:
            median_length = 0
            length_variance = 1

        lengths = sorted(list(map(len, field_data)))
        left_mid = lengths[(len(lengths) - 1) // 2]
        right_mid = lengths[len(lengths) // 2]

        median_length = left_mid + right_mid / 2
        length_variance = np.var(lengths)
        average_log_length = np.mean(np.log10(lengths))

        return median_length, length_variance, average_log_length

    def get_aggr_features(self, field_data, field_idx):
        shared_num_formats = self.dt['sharedNumFmts']
        num_format_ids = self.dt['numFmtIds']
        is_percent_count = 0
        is_01_count = 0
        is_0100_count = 0
        is_integer_count = 0
        is_negative_count = 0
        count = 0

        for row_idx, cell in enumerate(field_data):
            cell_num_format_id = str(num_format_ids[row_idx][field_idx])
            if cell != '':
                if cell_num_format_id in shared_num_formats:
                    if '%' in shared_num_formats[cell_num_format_id]:
                        is_percent_count += 1
                elif cell_num_format_id in _DEFAULT_FORMAT_MAPPINGS:
                    if '%' in _DEFAULT_FORMAT_MAPPINGS[cell_num_format_id]:
                        is_percent_count += 1
                cell_value = int(cell) if '.' not in cell else float(cell)
                if cell_value >= 0 and cell_value <= 1:
                    is_01_count += 1
                if cell_value >= 0 and cell_value <= 100:
                    is_0100_count += 1
                if type(cell_value) == int:
                    is_integer_count += 1
                if cell_value < 0:
                    is_negative_count += 1
                count += 1

        aggr_percent_formatted = is_percent_count / count
        aggr01_ranged = is_01_count / count
        aggr0100_ranged = is_0100_count / count
        aggr_integers = is_integer_count / count
        aggr_negative = is_negative_count / count

        return (
            aggr_percent_formatted,
            aggr01_ranged,
            aggr0100_ranged,
            aggr_integers,
            aggr_negative,
        )

    def analyze_string_and_entropy_feature_numerical(self, field_data):
        if len(field_data) == 0:
            return 0, 0, 0, 0, 0

        prefixes = {}
        suffixes = {}
        key_distribution = {}
        char_distribution = {}
        first_digit_count = {str(x): 0 for x in range(10)}
        total_chars = 0
        sampled_count = 0
        Benford_Std = [
            0.0,
            0.301,
            0.176,
            0.125,
            0.097,
            0.079,
            0.067,
            0.058,
            0.051,
            0.046,
        ]

        # Build statistic dictionaries
        for cell in field_data:
            if cell != '':
                sampled_count += 1
                total_chars += len(cell)
                key_distribution[cell] = (
                    key_distribution[cell] + 1 if cell in key_distribution else 1
                )
                for char in cell:
                    char_distribution[char] = (
                        char_distribution[char] + 1 if char in char_distribution else 1
                    )
                if len(cell) != 0:
                    prefixes[cell[0]] = (
                        prefixes[cell[0]] + 1 if cell[0] in prefixes else 1
                    )
                    suffixes[cell[-1]] = (
                        suffixes[cell[-1]] + 1 if cell[-1] in suffixes else 1
                    )
                    if cell[0] >= '0' and cell[0] <= '9':
                        first_digit_count[cell[0]] = (
                            first_digit_count[cell[0]] + 1
                            if cell[0] in first_digit_count
                            else 1
                        )
                    elif (
                        cell[0] == '-'
                        and len(cell) > 1
                        and cell[1] >= '0'
                        and cell[1] <= '9'
                    ):
                        first_digit_count[cell[1]] = (
                            first_digit_count[cell[1]] + 1
                            if cell[1] in first_digit_count
                            else 1
                        )

        # Get the scores
        common_prefix_score = (
            max(prefixes.values()) / sampled_count if len(prefixes) != 0 else 0
        )
        common_suffix_score = (
            max(suffixes.values()) / sampled_count if len(suffixes) != 0 else 0
        )

        key_entropy_score = -sum(
            map(
                lambda x: x * np.log2(x),
                np.array(list(key_distribution.values())) / sampled_count,
            )
        )
        char_entropy_score = -sum(
            map(
                lambda x: x * np.log2(x),
                np.array(list(char_distribution.values())) / total_chars,
            )
        )

        # get Benford feature
        skewing = np.sum(
            [
                (first_digit_count[str(i)] / sampled_count - Benford_Std[i]) ** 2
                for i in range(1, 10)
            ]
        )
        benford = np.sqrt(skewing)

        # TODO: Notice that the suffix score might not be accurate because the digit is largely affected by the precision of the data value
        return (
            common_prefix_score,
            common_suffix_score,
            key_entropy_score,
            char_entropy_score,
            benford,
        )

    def analyze_numerical_feature(self, field_data):
        data_range = (
            change_rate
        ) = (
            partial_ordered
        ) = variance = cov = cardinality = absolute_cardinality = spread = major = 0

        field_data = list(
            map(lambda x: float(x), filter(lambda x: x != '', field_data))
        )
        if len(field_data) == 0:
            data_range, change_rate, partial_ordered, variance, cov, cardinality, absolute_cardinality, spread, major

        # Calculate ordinal features
        if len(field_data) > 1:
            increase = decrease = 0
            for idx in range(1, len(field_data)):
                less_or_equal = field_data[idx - 1] - 1e-7 < field_data[idx]
                equal_or_greater = field_data[idx - 1] + 1e-7 > field_data[idx]
                if less_or_equal:
                    increase += 1
                if equal_or_greater:
                    decrease += 1
                if not less_or_equal or not equal_or_greater:
                    change_rate += 1
            change_rate /= len(field_data) - 1
            partial_ordered = max(increase, decrease) / (len(field_data) - 1)
        else:
            change_rate = partial_ordered = 0.5

        # Calculate satisitc features
        # variance = np.var(field_data)
        average = np.mean(field_data)
        variance = np.sqrt(np.mean((np.array(field_data) - average) ** 2))
        cov = variance / average if abs(average) > 1e-7 else 0

        # Sort numeric content array and calculate range, cardinality, major and spread
        field_data.sort()

        major_count = 1
        current_count = 1
        full_cardinality = 1

        for idx in range(1, len(field_data)):
            if field_data[idx] - field_data[idx - 1] <= 1e-7:
                if current_count + 1 > major_count:
                    major_count = current_count + 1
            else:
                full_cardinality += 1
                current_count = 1

        data_range = field_data[-1] - field_data[0]
        absolute_cardinality = full_cardinality
        cardinality = absolute_cardinality / len(field_data)
        major = major_count / len(field_data)
        spread = cardinality / (data_range + 1)

        return (
            data_range,
            change_rate,
            partial_ordered,
            variance,
            cov,
            cardinality,
            absolute_cardinality,
            spread,
            major,
        )

    def analyze_equal_progression(self, field_data):
        flag_has_inf_nan = False
        max_abs_value = 0
        field_data = list(filter(lambda x: x != '', field_data))
        for cell in field_data:
            if 'Nan' in cell or 'inf' in cell or 'nan' in cell or 'Inf' in cell:
                flag_has_inf_nan = True
                break
            else:
                max_abs_value = max(abs(float(cell)), max_abs_value)
        if flag_has_inf_nan:
            equal_progression_confidence = 0

        # less than 2 entries in the sequence must be equal-progression
        # If all numbers are very small, it should be considered as const sequence (then equal-progression)
        else:
            if len(field_data) <= 2 or max_abs_value < 1e-7:
                equal_progression_confidence = 1
            else:
                equal_progression_confidence = self.unified_diff_seq_var(field_data)
            equal_progression_confidence += EPS

        # Theoretically, equalProgressionConfidence should all lie in (0, 1], but if it has outlier, we set it to -1
        if equal_progression_confidence > 0:
            equal_progression_confidence = (
                -1 * np.log10(equal_progression_confidence) / 34.0
            )
        else:
            equal_progression_confidence = 0

        return equal_progression_confidence

    def analyze_geometric_progression(self, field_data):
        flag_has_zero_inf_nan = False
        flag_all_pos_neg = True
        is_positive = True
        min_abs_value = np.inf
        field_data = list(filter(lambda x: x != '', field_data))
        for idx, cell in enumerate(field_data):
            if self.is_outlier_double_string(cell):
                flag_has_zero_inf_nan = True
                break
            if idx == 0 and float(cell) < 0:
                is_positive = False
            else:
                if (is_positive and float(cell) < 0) or (
                    not is_positive and float(cell) > 0
                ):
                    flag_all_pos_neg = False
            min_abs_value = (
                abs(float(cell)) if min_abs_value > abs(float(cell)) else min_abs_value
            )

        if flag_has_zero_inf_nan or (not flag_all_pos_neg):
            geometric_progression_confidence = 0
        else:
            if len(field_data) <= 2:
                geometric_progression_confidence = 1
            else:
                if not is_positive:
                    transformed_sequence = [
                        np.log10(np.abs(float(x))) for x in field_data
                    ]
                else:
                    transformed_sequence = [np.log10(float(x)) for x in field_data]
                geometric_progression_confidence = self.unified_diff_seq_var(
                    transformed_sequence
                )

        geometric_progression_confidence += EPS

        if geometric_progression_confidence > 0:
            geometric_progression_confidence = (
                -1 * np.log10(geometric_progression_confidence) / 34
            )  # The original value lies in (0,34), divide by 34 to normalize
        else:
            geometric_progression_confidence = 0

        return geometric_progression_confidence

    def unified_diff_seq_var(self, sequence):
        if len(sequence) <= 2:
            return 0

        sequence = list(filter(lambda x: x != '', sequence))
        abs_values = list(map(lambda x: abs(float(x)), sequence))
        max_abs_value = max(abs_values)

        # Unify the value of sequence into [-1, 1]
        unified_sequence = list(map(lambda x: float(x) / max_abs_value, sequence))

        # Calculate difference between 2 neighboring elements, the range of diff_sequence is [-2, 2]
        diff_sequence = [
            unified_sequence[idx + 1] - unified_sequence[idx]
            for idx in range(0, len(sequence) - 1)
        ]
        average_diff = np.mean(diff_sequence)

        # Calulate the variance of the difference sequence
        # The maximum of this value is 16, so divide it by 16 to unify it.
        difference_var = (
            (1.0 / (len(sequence) - 1))
            * np.sum((np.array(diff_sequence) - average_diff) ** 2)
            * (1.0 / 16.0)
        )
        return difference_var

    def is_illegal_double_string(self, string):
        return 'Nan' in string or 'inf' in string or 'nan' in string or 'Inf' in string

    def is_outlier_double_string(self, string):
        return float(string) == 0 or self.is_illegal_double_string(string)

    def analyze_string_length_feature_string(self, field_data):
        field_data = [x for x in field_data if x != '']
        if len(field_data) == 0:
            median_length = 0
            length_variance = 1

        else:
            lengths = [len(x) for x in field_data]
            median_length = np.median(lengths)
            length_variance = np.var(lengths)

        return median_length, length_variance

    def get_sum_features(self, field_data):
        field_data = [float(x) for x in field_data if x != '']
        sum_value = sum(field_data)
        sum_in_01 = sum_value if sum_value >= 0 and sum_value <= 1 else 0
        sum_in_0100 = sum_value if sum_value >= 0 and sum_value <= 100 else 0
        return sum_in_01, sum_in_0100

    def analyze_distribution_features(self, field_data):
        field_data = [float(x) for x in field_data if x != '']
        if len(field_data) == 0:
            return 0, 0, 0

        sum_value = sum(field_data)
        mean = sum_value / len(field_data)
        bias_squared = [(x - mean) ** 2 for x in field_data]
        variance = np.sqrt(np.mean(bias_squared))
        skewness = np.sum([(x - mean) ** 3 for x in field_data]) / (
            (len(field_data) - 1) * variance**3
        )
        kurtosis = np.sum([bs**2 for bs in bias_squared]) / np.sum(bias_squared) ** 2

        sorted_data = sorted(field_data)
        gini = 0
        for i in range(0, len(sorted_data)):
            gini += ((4 * i) - (2 * len(field_data)) + 3) * sorted_data[i]
        gini /= 2 * mean * len(field_data) ** 2

        return skewness, kurtosis, gini

    def check_ordinal(self, field_data):
        if len(field_data) == 0:
            return 0
        column_count = len(field_data)
        sampled_length = min(column_count, SAMPLED_LENGTH_LIMIT)
        column_pieces = [None] * sampled_length
        ordinal_offset_1 = [None] * sampled_length
        ordinal_offset_4 = [None] * sampled_length
        score_1_sum = 0
        score_4_sum = 0

        for cell_idx, value in enumerate(field_data[:sampled_length]):
            if value != '':
                column_pieces[cell_idx] = float(value)

            if cell_idx < 1:
                score_1_sum += 0
            else:
                result, aux_value = self.compare_pieces(
                    column_pieces[cell_idx - 1],
                    column_pieces[cell_idx],
                    ordinal_offset_1[cell_idx - 1],
                )
                score_1_sum += result
                ordinal_offset_1[cell_idx] = aux_value

            if cell_idx < 4:
                score_4_sum += 0
            else:
                result, aux_value = self.compare_pieces(
                    column_pieces[cell_idx - 4],
                    column_pieces[cell_idx],
                    ordinal_offset_4[cell_idx - 4],
                )
                score_4_sum += result
                ordinal_offset_4[cell_idx] = aux_value

        if sampled_length > 1:
            score_1_sum = score_1_sum / (sampled_length - 1)
        else:
            score_1_sum = 0

        if sampled_length > 4:
            score_4_sum = score_4_sum / (sampled_length - 4)
        else:
            score_4_sum = 0

        return max(score_1_sum, score_4_sum)

    def compare_pieces(self, x_piece, y_piece, x_offset):
        result = -np.inf
        if x_piece is not None and y_piece is not None:
            y_piece_offset = y_piece - x_piece
            starting_new_seq = (x_offset is None) or np.isnan(x_offset)
            if (
                starting_new_seq or (y_piece_offset == x_offset)
            ) and y_piece_offset > 0:
                operand_1 = (
                    y_piece_offset if y_piece_offset <= 1 else 1 / y_piece_offset
                )
                operand_2 = 0.5 if starting_new_seq else 1
                result = operand_1 * operand_2
            y_offset = y_piece_offset
        else:
            y_offset = None
        if result > 0:
            return result, y_offset
        else:
            return 0, y_offset

    def delete_dt(self):
        del self.dt

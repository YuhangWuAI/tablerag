"""
This code file contains functions that borrow certain logic from an anonymous repository associated with the paper:
"TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning" (arXiv:2312.09039).
Original source: https://anonymous.4open.science/r/TableProvider-4CC3/README.md.
The repository does not list an author, but it is linked to the above paper.

Specifically, portions of the code related to data loading, data packing, and evaluation logic have been borrowed and integrated into this project.

Current author: Yuhang Wu
Contact: yuhang.wu-4 [at] postgrad.manchester.ac.uk
GitHub: https://github.com/YuhangWuAI/

If you believe that any content in this file infringes your rights or if you have any concerns,
please contact me at the email address above.
"""

import re
import recognizers_suite
import collections
import string
from recognizers_suite import Culture

def str_normalize(user_input, recognition_types=None):
    """
    Normalize a string by recognizing and standardizing values using the recognizers_suite.

    :param user_input: The input string to be normalized.
    :param recognition_types: A list of types to recognize and normalize (e.g., datetime, number).
    :return: A normalized string.
    """
    user_input = str(user_input)
    user_input = user_input.replace("\\n", "; ")

    def replace_by_idx_pairs(orig_str, strs_to_replace, idx_pairs):
        # Helper function to replace parts of the string based on index pairs
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
    culture = Culture.English
    for recognition_type in recognition_types:
        if re.match("\d+/\d+", user_input):
            # Avoid calculating strings like '1991/92' as fractions
            continue
        recognized_list = getattr(
            recognizers_suite, "recognize_{}".format(recognition_type)
        )(user_input, culture)
        strs_to_replace = []
        idx_pairs = []
        for recognized in recognized_list:
            if not recognition_type == 'datetime':
                recognized_value = recognized.resolution['value']
                if str(recognized_value).startswith("P"):
                    # Skip datetime periods
                    continue
                else:
                    strs_to_replace.append(recognized_value)
                    idx_pairs.append((recognized.start, recognized.end + 1))
            else:
                if recognized.resolution:  # Ensure resolution is not None
                    if len(recognized.resolution['values']) == 1:
                        strs_to_replace.append(
                            recognized.resolution['values'][0]['timex']
                        )
                        idx_pairs.append((recognized.start, recognized.end + 1))

        if len(strs_to_replace) > 0:
            user_input = replace_by_idx_pairs(user_input, strs_to_replace, idx_pairs)

    if re.match("(.*)-(.*)-(.*) 00:00:00", user_input):
        user_input = user_input[: -len("00:00:00") - 1]
        # Normalize datetime strings like '2008-04-13 00:00:00' to '2008-04-13'
    return user_input


def check_denotation(target_values, predicted_values):
    """
    Check if the predicted denotation matches the target values.

    :param target_values: List of target values.
    :param predicted_values: List of predicted values.
    :return: True if the predicted values match the target values, False otherwise.
    """
    if len(target_values) != len(predicted_values):
        return False
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


def extract_yes_no_and_map(text):
    """
    Extract yes/no from text and map to '1' for yes/true and '0' for no/false.

    :param text: Input text to analyze.
    :return: '1' for yes/true, '0' for no/false, '2' if neither is found.
    """
    text = text.lower()

    yes_patterns = [r'\byes\b', r'\btrue\b']
    no_patterns = [r'\bno\b', r'\bfalse\b']

    if text == "0":
        return "0"

    if text == "1":
        return "1"

    for pattern in yes_patterns:
        if re.search(pattern, text):
            return "1"

    for pattern in no_patterns:
        if re.search(pattern, text):
            return "0"

    return "2"


class Evaluator:
    def __init__(self):
        pass

    def flatten_iterative(self, lst):
        """
        Flatten a list of lists iteratively.
        
        :param lst: List of lists to be flattened.
        :return: Flattened list.
        """
        stack = lst[::-1]
        result = []
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item[::-1])
            else:
                result.append(item)
        return result

    def run(
        self,
        pred_answer: list,
        gold_answer: list,
        dataset: str,
        allow_semantic=False,
        question=str,
    ):
        """
        Run evaluation based on the dataset type.

        :param pred_answer: List of predicted answers.
        :param gold_answer: List of ground truth answers.
        :param dataset: Name of the dataset being evaluated.
        :param allow_semantic: Whether to allow semantic matching.
        :param question: The question being evaluated, required if allow_semantic is True.
        :return: Evaluation score.
        """
        pred_answer = (
            self.flatten_iterative(pred_answer)
            if isinstance(pred_answer, list)
            and all(isinstance(sub_list, list) for sub_list in pred_answer)
            else pred_answer
        )
        if dataset == 'hybridqa' or dataset == "sqa":
            return self.eval_ex_match(
                pred_answer, gold_answer, allow_semantic, dataset, question
            )
        elif dataset == 'tabfact' or dataset == "feverous":
            return self.eval_fv_match(pred_answer, gold_answer)
        else:
            raise ValueError(f'{dataset} evaluator is not supported.')

    def eval_ex_match(
        self,
        pred_list,
        gold_list,
        allow_semantic=False,
        task_name=None,
        question=None,
    ):
        """
        Evaluate exact match (and optionally F1 score) between predictions and ground truth.

        :param pred_list: List of predicted answers.
        :param gold_list: List of ground truth answers.
        :param allow_semantic: Whether to allow semantic matching.
        :param task_name: The task being evaluated, e.g., 'hybridqa'.
        :param question: The question being evaluated, required if allow_semantic is True.
        :return: Exact match score (and optionally F1 score).
        """
        def normalize_answer(s):
            # Normalize answer strings by removing articles, punctuation, and extra whitespace
            def remove_articles(text):
                return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", text)

            def whilt_space_fix(text):
                return " ".join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return whilt_space_fix(remove_articles(remove_punc(lower(s))))

        def get_tokens(s):
            # Tokenize the normalized answer string
            if not s:
                return []
            return normalize_answer(s).split()

        def compute_exact(a_gold, a_pred):
            # Compute exact match score
            return int(normalize_answer(a_gold) == normalize_answer(a_pred))

        def compute_f1(a_gold, a_pred):
            # Compute F1 score
            gold_toks = get_tokens(a_gold)
            pred_toks = get_tokens(a_pred)
            common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
            num_same = sum(common.values())
            if len(gold_toks) == 0 or len(pred_toks) == 0:
                return int(gold_toks == pred_toks)
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        if not allow_semantic:
            exact_scores = 0.0
            f1_scores = 0.0

            if task_name == "hybridqa":
                for pred, gold in zip(pred_list, gold_list):
                    exact_scores += compute_exact(gold, pred)
                    f1_scores += compute_f1(gold, pred)
            else:
                for pred, gold in zip(pred_list, gold_list):
                    exact_scores += max(compute_exact(g, pred) for g in gold)
                    f1_scores += max(compute_f1(g, pred) for g in gold)
            total = len(pred_list)
            exact_scores = exact_scores / total
            f1_scores = f1_scores / total
            return exact_scores

        else:
            assert isinstance(question, str)
            question = re.sub('\s+', ' ', question).strip().lower()
            pred_list = [str_normalize(span) for span in pred_list]
            gold_list = [str_normalize(span) for span in gold_list]
            pred_list = sorted(list(set(pred_list)))
            gold_list = sorted(list(set(gold_list)))
            if len(pred_list) == 1 and len(gold_list) == 1:
                if (pred_list[0] == '0' and gold_list[0] == 'no') or (
                    pred_list[0] == '1' and gold_list[0] == 'yes'
                ):
                    return True
                question_tokens = question.split()
                try:
                    pos_or = question_tokens.index('or')
                    token_before_or, token_after_or = (
                        question_tokens[pos_or - 1],
                        question_tokens[pos_or + 1],
                    )
                    if (pred_list[0] == '0' and gold_list[0] == token_after_or) or (
                        pred_list[0] == '1' and gold_list[0] == token_before_or
                    ):
                        return True
                except Exception as e:
                    pass
            if len(pred_list) == 1 and len(gold_list) == 1:
                NUMBER_UNITS_PATTERN = re.compile(
                    '^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$'
                )
                DATE_PATTERN = re.compile(
                    '[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?'
                )
                DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
                p, g = pred_list[0], gold_list[0]
                if re.match(DURATION_PATTERN, p):
                    p = re.match(DURATION_PATTERN, p).group(2)
                if re.match(DURATION_PATTERN, g):
                    g = re.match(DURATION_PATTERN, g).group(2)
                match = False
                num_flag, date_flag = False, False
                if re.match(NUMBER_UNITS_PATTERN, p) or re.match(
                    NUMBER_UNITS_PATTERN, g
                ):
                    num_flag = True
                if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                    date_flag = True
                if num_flag:
                    p_set, g_set = set(p.split()), set(g.split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if date_flag:
                    p_set, g_set = set(p.replace('-', ' ').split()), set(
                        g.replace('-', ' ').split()
                    )
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if match:
                    return True
            return check_denotation(pred_list, gold_list)

    def eval_fv_match(self, pred_list, gold_list):
        """
        Evaluate accuracy based on yes/no or true/false answers.

        :param pred_list: List of predicted answers.
        :param gold_list: List of ground truth answers.
        :return: Accuracy score.
        """
        acc = 0.0
        for pred, gold in zip(pred_list, gold_list):
            pred, gold = extract_yes_no_and_map(pred), extract_yes_no_and_map(gold)
            if pred == gold:
                acc += 1
        acc = acc / len(pred_list)
        return acc

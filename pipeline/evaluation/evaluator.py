import re
import recognizers_suite
import collections
import string
import numpy as np
from recognizers_suite import Culture
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from evaluate import load
from datasets import load_metric
from typing import Dict, Any


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
    culture = Culture.English
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


def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


def extract_yes_no_and_map(text):
    # Convert the input text to lowercase for case-insensitive matching
    text = text.lower()

    # Define regular expressions for yes/no matching
    yes_patterns = [r'\byes\b', r'\btrue\b']
    no_patterns = [r'\bno\b', r'\bfalse\b']

    # Check for "0"
    if text == "0":
        return "0"

    # Check for "1"
    if text == "1":
        return "1"

    # Check for yes
    for pattern in yes_patterns:
        if re.search(pattern, text):
            return "1"

    # Check for no
    for pattern in no_patterns:
        if re.search(pattern, text):
            return "0"

    # Return 2 if neither yes nor no is found
    return "2"


class Evaluator:
    def __init__(self):
        pass

    def flatten_iterative(self, lst):
        """
        Flatten a list of lists iteratively.
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
        elif dataset == 'totto':
            return self.eval_bleu_score(pred_answer, gold_answer)
        elif dataset == 'spider':
            return self.compute_exact_match_metric(pred_answer, gold_answer)
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
        def normalize_answer(s):
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
            if not s:
                return []
            return normalize_answer(s).split()

        def compute_exact(a_gold, a_pred):
            return int(normalize_answer(a_gold) == normalize_answer(a_pred))

        def compute_f1(a_gold, a_pred):
            gold_toks = get_tokens(a_gold)
            pred_toks = get_tokens(a_pred)
            common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
            num_same = sum(common.values())
            if len(gold_toks) == 0 or len(pred_toks) == 0:
                # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
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
            # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
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
            # (2) Number value (allow units) and Date substring match
            if len(pred_list) == 1 and len(gold_list) == 1:
                NUMBER_UNITS_PATTERN = re.compile(
                    '^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$'
                )
                DATE_PATTERN = re.compile(
                    '[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?'
                )
                DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
                p, g = pred_list[0], gold_list[0]
                # Restore `duration` type, e.g., from 'P3Y' -> '3'
                if re.match(DURATION_PATTERN, p):
                    p = re.match(DURATION_PATTERN, p).group(2)
                if re.match(DURATION_PATTERN, g):
                    g = re.match(DURATION_PATTERN, g).group(2)
                match = False
                num_flag, date_flag = False, False
                # Number w. unit match after string normalization.
                # Either pred or gold being number w. units suffices it.
                if re.match(NUMBER_UNITS_PATTERN, p) or re.match(
                    NUMBER_UNITS_PATTERN, g
                ):
                    num_flag = True
                # Date match after string normalization.
                # Either pred or gold being date suffices it.
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
        acc = 0.0
        for pred, gold in zip(pred_list, gold_list):
            pred, gold = extract_yes_no_and_map(pred), extract_yes_no_and_map(gold)
            if pred == gold:
                acc += 1
        acc = acc / len(pred_list)
        return acc

    def eval_bleu_score(
        self,
        pred_list: list,
        gold_list: list,
        metric_list: list = ['bleu'],
    ) -> float:
        def bleurt_to_scaled(score):
            return 50 * (score + 1)

        def postprocess_text(preds, references_s):
            preds = [pred.strip() for pred in preds]
            references_s = [
                [reference.strip() for reference in references]
                for references in references_s
            ]

            if metric_name in ["sacrebleu"]:
                # since hf sacrebleu only support references with same length, we have to pad them into the same length
                ref_max_len = max([len(ref) for ref in references_s])
                for ref in references_s:
                    for _ in range(ref_max_len - len(ref)):
                        ref.append(None)
            elif metric_name in ["bleurt"]:
                preds = [pred.strip().split(' ') for pred in preds]
                references_s = [
                    [reference.strip().split(' ') for reference in references]
                    for references in references_s
                ]
            else:
                pass
            return preds, references_s

        summary = {}
        for metric_name in metric_list:
            if metric_name == "bleu":
                bleu = 0.0
                weights_dict = {
                    "BLEU-1": (1, 0, 0, 0),
                    "BLEU-2": (0.5, 0.5, 0, 0),
                    "BLEU-3": (0.33, 0.33, 0.34, 0),
                    "BLEU-4": (0.25, 0.25, 0.25, 0.25),
                }
                weights = weights_dict["BLEU-4"]
                for i in range(len(pred_list)):
                    references = []
                    for j in range(len(gold_list[i])):
                        references.append(gold_list[i])
                    candidates = []
                    candidates.append(
                        pred_list[i]
                        .strip("\n")
                        .replace("Line 1: ", "")
                        .replace("1. ", "")
                        .replace("2. ", "")
                        .replace("Line 2: ", "")
                    )
                    candidate_score = 0
                    for candidate in candidates:
                        candidate_score += corpus_bleu(
                            [references], [candidate], weights=weights
                        )
                    bleu += candidate_score
                bleu /= len(pred_list)
                summary[metric_name] = bleu
            elif metric_name == "sacrebleu":
                metric = load_metric(metric_name)
                processed_preds, processed_references = postprocess_text(
                    pred_list, gold_list
                )
                res = metric.compute(
                    predictions=processed_preds, references=processed_references
                )
                summary[metric_name] = res["score"] * 0.01
            elif metric_name == "bleurt":
                metric = load_metric(metric_name)
                processed_preds, processed_references = postprocess_text(
                    pred_list, gold_list
                )
                # We refer to the realization in https://github.com/google-research/language/blob/13fd14e1b285002412252097586f8fe405ba8a24/language/totto/totto_bleurt_eval.py#L94-L131
                multi_references = [[], [], []]
                for references in processed_references:
                    # here "references" mean references for one prediction string.
                    if len(references) == 2:
                        multi_references[2].append('')
                    elif len(references) == 3:
                        multi_references[2].append(references[2])
                    else:
                        raise ValueError(
                            "The references num for each candidate should be 2 or 3 in ToTTo dataset."
                        )
                    multi_references[0].append(references[0])
                    multi_references[1].append(references[1])

                multi_bleurt_scores = []
                for references in multi_references:
                    multi_bleurt_scores.append(
                        metric.compute(
                            predictions=processed_preds, references=references
                        )['scores']
                    )
                    print("multi_bleurt_scores", multi_bleurt_scores)

                assert len(multi_references) == 3
                avg_bleurt_scores = []
                for i in range(len(processed_preds)):
                    # All examples have atleast two references but some do not have three.
                    assert multi_references[0][i] and multi_references[1][i]
                    r2 = multi_references[2][i]
                    if r2:
                        # Take average over 3 references.
                        score_i = (
                            multi_bleurt_scores[0][i]
                            + multi_bleurt_scores[1][i]
                            + multi_bleurt_scores[2][i]
                        ) / 3
                    else:
                        # print("only two refs")
                        # Take average over two references.
                        score_i = (
                            multi_bleurt_scores[0][i] + multi_bleurt_scores[1][i]
                        ) / 2
                    avg_bleurt_scores.append(score_i)
                summary["bleurt"] = bleurt_to_scaled(np.mean(avg_bleurt_scores))
            else:
                metric = load_metric(metric_name)
                processed_preds, processed_references = postprocess_text(
                    pred_list, gold_list
                )
                res = metric.compute(
                    predictions=processed_preds, references=processed_references
                )
                summary[metric_name] = res[metric_name]

            return summary[metric_name]
        
    def compute_exact_match_metric(self, predictions, references) -> Dict[str, Any]:
        foreign_key_maps = dict()
        for reference in references:
            if reference["db_id"] not in foreign_key_maps:
                foreign_key_maps[reference["db_id"]] = spider_evaluation.build_foreign_key_map(
                    {
                        "table_names_original": reference["db_table_names"],
                        "column_names_original": list(
                            zip(
                                reference["db_column_names"]["table_id"],
                                reference["db_column_names"]["column_name"],
                            )
                        ),
                        "foreign_keys": list(
                            zip(
                                reference["db_foreign_keys"]["column_id"],
                                reference["db_foreign_keys"]["other_column_id"],
                            )
                        ),
                    }
                )
        evaluator = spider_evaluation.Evaluator(references[0]["db_path"], foreign_key_maps, "match")
        for prediction, reference in zip(predictions, references):
            turn_idx = reference.get("turn_idx", 0)
            # skip final utterance-query pairs
            if turn_idx < 0:
                continue
            _ = evaluator.evaluate_one(reference["db_id"], reference["query"], prediction)
        evaluator.finalize()
        return evaluator.scores["all"]["exact"]

'''
Find evaluation metric that we need in log.
'''
import argparse
import csv
import re

from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--log_file", required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    file = args.log_file
    print("reading lines")
    with open(f"/home/exp/{file}", 'r') as f:
        lines = f.readlines()
    print("finding")
    evaluation = {
        r"Evaluation -   Measure:.*Precision=([0-9\.]*),Recall=([0-9\.]*),f1=([0-9\.]*), Acc=([0-9\.]*)": [],
        r"Evaluation -   Measure pair:.*Precision=([0-9\.]*),Recall=([0-9\.]*),f1=([0-9\.]*), Acc=([0-9\.]*)": [],
        r"Weighted prf -   Weighted measure type.*\(([0-9\.]*), ([0-9\.]*), ([0-9\.]*)": [],
        r"DimMsrRecall -   .* Scorer R@1.*\(([0-9\.\%]*)\)": [],
        r"DimMsrRecall -   .* Scorer R@3.*\(([0-9\.\%]*)\)": [],
        r"DimMsrRecall -   .* Scorer R@5.*\(([0-9\.\%]*)\)": [],
        r"DimMsrRecall -   .* Scorer MMR.*\(([0-9\.\%]*)\)": [],  # All MMR are in here
        r"DimMsrNDCG -   .* Scorer AverageNDCG@1.*\(([0-9\.\%]*)\)": [],
        r"DimMsrNDCG -   .* Scorer AverageNDCG@3.*\(([0-9\.\%]*)\)": [],
        r"DimMsrNDCG -   .* Scorer AverageNDCG@5.*\(([0-9\.\%]*)\)": []
    }
    type_start = "Evaluation -   Measure type(SubCategory): matrices"
    type_end = r"Acc=([0-9\.]*)"
    type_eval = []
    type_flag = False
    for line in tqdm(lines):
        if type_flag:
            type = re.findall(type_end, line)
            if len(type) > 0:
                type_eval.append(type[0])
                type_flag = False
        elif type_start in line:
            type_flag = True
        else:
            for key in evaluation.keys():
                value = re.findall(key, line)
                if len(value) > 0:
                    evaluation[key].append(value[0])
                    break

    exp_num = len(type_eval)
    for key in evaluation.keys():
        if len(evaluation[key]) // exp_num not in [1, 6] or len(evaluation[key]) % exp_num != 0:
            raise ValueError(f"{key} length is {len(evaluation[key])}, while exp number is {exp_num}")
    print(f"Experiment: {exp_num}")
    with open("/home/exp/result.csv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["", "msr_p", "msr_r", "msr_f1", "msr_acc",
                         "gby_R@1", "gby_R@3", "gby_R@5", "gby_MMR", "gby_NDCG@1", "gby_NDCG@3", "gby_NDCG@5",
                         "msr_R@1", "msr_R@3", "msr_R@5", "msr_MMR", "msr_NDCG@1", "msr_NDCG@3", "msr_NDCG@5",
                         "key_R@1", "key_R@3", "key_R@5", "key_MMR", "key_NDCG@1", "key_NDCG@3", "key_NDCG@5",
                         "agg_R@1", "agg_R@3", "agg_R@5", "agg_MMR", "agg_NDCG@1", "agg_NDCG@3", "key_NDCG@5",
                         "msr_pair_p", "msr_pair_r", "msr_pair_f1", "msr_pair_acc",
                         "msr_type_p", "msr_type_r", "msr_type_f1", "msr_type_acc",
                         ])
        for i in range(exp_num):
            evaluation_value = list(evaluation.values())
            result = []
            result.extend(list(evaluation_value[0][i]))
            for j in range(4):  # diffenert task
                for k in range(3, 10):  # 7 metric
                    result.append(evaluation_value[k][i * 6 + j])
            result.extend(list(evaluation_value[1][i]))
            result.extend(list(evaluation_value[2][i]))
            result.append(type_eval[i])

            writer.writerow([""] + result)
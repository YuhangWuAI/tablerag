import json
import os
from collections import defaultdict, Counter

from tqdm import tqdm

PATH = "/storage1/pivot-20201029/"
print(f"Corpus: {PATH}")
statistics = defaultdict(list)  # {field_type: ["type-property",]}


def load_json(file_path: str, encoding: str):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


FieldType = {
    0: "Unknown",
    1: "String",
    3: "DateTime",
    5: "Decimal",
    7: "Year"
}

for mtab_label_path in tqdm(os.listdir(os.path.join(PATH, "mtab/labels"))):
    tuid = mtab_label_path[:-5]
    labels = load_json(os.path.join(PATH, "mtab/labels", mtab_label_path), encoding='utf-8-sig')
    try:
        table = load_json(os.path.join(PATH, "data", f"{tuid}.DF.json"), encoding='utf-8-sig')
    except:
        print(f"No DF {tuid}")
        continue
    property = [0 for _ in range(table["nColumns"])]
    column_type = [0 for _ in range(table["nColumns"])]
    field_type = [i["type"] for i in table["fields"]]
    for i in labels["semantic"]["cpa"]:
        property[i["target"][1]] = 1
    for i in labels["semantic"]["cta"]:
        column_type[i["target"]] = 1
    for i in range(table["nColumns"]):
        statistics[field_type[i]].append(f"{column_type[i]}-{property[i]}")

for i in statistics.keys():
    total = len(statistics[i])
    print(
        f"Field type:{i}({FieldType[i]}), Statistics: {[f'{j}: {k}({k * 100 / total:.2f}%)' for j, k in Counter(statistics[i]).items()]}")

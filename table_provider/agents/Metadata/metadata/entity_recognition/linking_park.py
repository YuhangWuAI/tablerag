import csv
import json
import os

from tqdm import tqdm


def load_json(file_path: str, encoding: str):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


PATH = "/storage1/chart-202011"

os.makedirs(f"{PATH}/linkingpark/labels", exist_ok=True)

sUids = load_json(os.path.join(PATH, "index", "schema_ids.json"), encoding="utf-8-sig")
tUids = []
for sUid in sUids:
    info = load_json(os.path.join(PATH, "sample-new", f"{sUid}.sample.json"), encoding="utf-8-sig")
    if info['lang'] != 'en':
        continue
    if "tableAnalysisPairs" in info:
        tableAnalysisPairs = info["tableAnalysisPairs"]
        tUids.extend([f"{sUid}.t{key}" for key in tableAnalysisPairs.keys()])
    else:
        tUids.append(f"{sUid}.t0")
print(f"There are {len(tUids)} tables")

exist_tuid = ['.'.join(i.split('.')[:2]) for i in os.listdir(f"{PATH}/linkingpark/labels")]
exist_tuid.reverse()
tUids = list(set(tUids) - set(exist_tuid))
print(f"There are {len(tUids)} tables without entity.")

for file in tqdm(tUids):
    if not os.path.exists(f'{PATH}/csv/{file}.csv'):
        continue
    table = []
    with open(os.path.join(PATH, f'{PATH}/csv/{file}.csv'), 'r') as f_read:
        rows = csv.reader(f_read)
        row_id = 0
        for row in rows:
            if len(row) == 0:
                continue
            if row_id == 0:
                row_id += 1
                continue
            if 512 / 2 / len(row) < row_id or 31 < row_id:
                break
            if row_id > 10:
                break
            table.append(row)
            row_id += 1
    if row_id == 1:  # Only have header
        os.remove(os.path.join(PATH, f'{PATH}/linkingpark/tables/{file}.csv'))

    # entity recognition
    try:
        results = os.popen(
            f'curl -X POST -H "Content-Type: application/json" -d {json.dumps({"table": table})} https://linkingpark.eastus.cloudapp.azure.com:6009/autodetect/api/v1.0/detect?sid=true').readline()
        results = json.loads(results)
        with open(f"{PATH}/linkingpark/labels/{file[:-4]}.json", 'w', encoding='utf-8-sig') as f:
            json.dump(r, f)
    except:
        print(f"Error{file}")
        continue

import csv
import json
import math
import os
import shutil
import zipfile

from tqdm import tqdm


def load_json(file_path: str, encoding: str):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


PATH = "/storage/chart-mj-202211"
MAX_FILE = 90
if os.path.exists(f"{PATH}/mtab/tables"):
    shutil.rmtree(f"{PATH}/mtab/tables")
os.makedirs(f"{PATH}/mtab/tables", exist_ok=True)
os.makedirs(f"{PATH}/mtab/labels", exist_ok=True)

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

exist_tuid = ['.'.join(i.split('.')[:2]) for i in os.listdir(f"{PATH}/mtab/labels")]
exist_tuid.reverse()
tUids = list(set(tUids) - set(exist_tuid))
print(f"There are {len(tUids)} tables without entity.")

for i in tqdm(range(math.ceil(len(tUids) / MAX_FILE))):
    files_batch = tUids[i * MAX_FILE:(i + 1) * MAX_FILE]
    for file in files_batch:
        if not os.path.exists(f'{PATH}/csv/{file}.csv'):
            continue
        with open(os.path.join(PATH, f'{PATH}/csv/{file}.csv'), 'r') as f_read:
            with open(os.path.join(PATH, f'{PATH}/mtab/tables/{file}.csv'), 'w', newline="") as f_write:
                rows = csv.reader(f_read)
                spamwriter = csv.writer(f_write)
                row_id = 0
                for row in rows:
                    if len(row) == 0:
                        continue
                    if 512 / 2 / len(row) < row_id or 31 < row_id:
                        break
                    if row_id > 10:
                        break
                    spamwriter.writerow(row)
                    row_id += 1
        if row_id == 1:  # Only have header
            os.remove(os.path.join(PATH, f'{PATH}/mtab/tables/{file}.csv'))
    # zip
    f = zipfile.ZipFile(f'{PATH}/mtab/tables.zip', 'w', zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(f'{PATH}/mtab/tables/'):
        fpath = path.replace(f'{PATH}/mtab/tables/', 'tables/')
        for filename in filenames:
            f.write(os.path.join(path, filename), os.path.join(fpath, filename))
        f.close()

    # entity recognition
    try:
        results = os.popen(f'curl -X POST -F file=@"{PATH}/mtab/tables.zip" https://mtab.app/api/v1/mtab').readline()
        results = json.loads(results)
        for r in results["tables"]:
            if "name" in r:  # success inference
                name = r["name"]
                with open(f"{PATH}/mtab/labels/{name}.json", 'w', encoding='utf-8-sig') as f:
                    json.dump(r, f)
    except:
        print("Error")
        # Remove table.zip
        shutil.rmtree(f"{PATH}/mtab/tables")
        os.makedirs(f"{PATH}/mtab/tables", exist_ok=True)
        continue

    # Remove table.zip
    shutil.rmtree(f"{PATH}/mtab/tables")
    os.makedirs(f"{PATH}/mtab/tables", exist_ok=True)

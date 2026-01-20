from pathlib import Path
import json
import random
import shutil

ROOT = Path(__file__).resolve().parent
ds_root = ROOT / "datasets" / "ModelNet40"

train_dir = ds_root / "train_dataset"
eval_dir = ds_root / "eval_dataset"
test_dir = ds_root / "test_dataset"

for d in (train_dir, eval_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

train_json, eval_json, test_json = [], [], []

random.seed(1234)

# Find all class dirs that have 'train' and/or 'test' subdirs with .off files
class_dirs = set()

for train_sub in ds_root.rglob("train"):
    if not train_sub.is_dir():
        continue
    # parent dir should be the class name
    cls_dir = train_sub.parent
    # check if it has .off
    has_off = any(child.suffix.lower() == ".off" for child in train_sub.iterdir() if child.is_file())
    if has_off:
        class_dirs.add(cls_dir)

for test_sub in ds_root.rglob("test"):
    if not test_sub.is_dir():
        continue
    cls_dir = test_sub.parent
    has_off = any(child.suffix.lower() == ".off" for child in test_sub.iterdir() if child.is_file())
    if has_off:
        class_dirs.add(cls_dir)

class_dirs = sorted(class_dirs)
if not class_dirs:
    print("No ModelNet40 class dirs found under", ds_root)
else:
    print("Found", len(class_dirs), "class dirs. Example:", class_dirs[0])

for cls_dir in class_dirs:
    label = cls_dir.name
    train_src_dir = cls_dir / "train"
    test_src_dir = cls_dir / "test"

    # TRAIN/EVAL from train folder
    if train_src_dir.is_dir():
        train_files = sorted(p for p in train_src_dir.glob("*.off"))
        random.shuffle(train_files)
        n = len(train_files)
        if n > 0:
            n_train = max(1, int(0.9 * n))
            n_eval = n - n_train

            for src in train_files[:n_train]:
                dst_name = f"{label}__{src.name}"
                dst = train_dir / dst_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                train_json.append({
                    "data": dst_name,
                    "label": label
                })

            for src in train_files[n_train:]:
                dst_name = f"{label}__{src.name}"
                dst = eval_dir / dst_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                eval_json.append({
                    "data": dst_name,
                    "label": label
                })

    # TEST from test folder
    if test_src_dir.is_dir():
        test_files = sorted(p for p in test_src_dir.glob("*.off"))
        for src in test_files:
            dst_name = f"{label}__{src.name}"
            dst = test_dir / dst_name
            if not dst.exists():
                shutil.copy2(src, dst)
            test_json.append({
                "data": dst_name,
                "label": label
            })

for out_path, data in [
    (ds_root / "train_data.json", train_json),
    (ds_root / "eval_data.json", eval_json),
    (ds_root / "test_data.json", test_json),
]:
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

print("ModelNet40 v2 done.")
print("Train:", len(train_json), "Eval:", len(eval_json), "Test:", len(test_json))

from pathlib import Path
import json
import random
import shutil

ROOT = Path(__file__).resolve().parent
ds_root = ROOT / "datasets" / "caltech-101"

raw_root = ds_root / "101_ObjectCategories"


# Where the raw images live.
# If your data is in caltech-101/101_ObjectCategories, change this line accordingly.
raw_root = ds_root  # or ds_root / "101_ObjectCategories"

train_dir = ds_root / "train_dataset"
eval_dir = ds_root / "eval_dataset"
test_dir = ds_root / "test_dataset"

for d in (train_dir, eval_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

train_json, eval_json, test_json = [], [], []

random.seed(1234)

# Treat each subfolder as a class
for cls_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
    label = cls_dir.name
    images = sorted(
        p for p in cls_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        continue

    # Shuffle and split: 70% train, 10% eval, 20% test
    random.shuffle(images)
    n = len(images)
    n_train = int(0.7 * n)
    n_eval = int(0.1 * n)
    n_test = n - n_train - n_eval

    splits = [
        (images[:n_train], train_dir, train_json),
        (images[n_train:n_train + n_eval], eval_dir, eval_json),
        (images[n_train + n_eval:], test_dir, test_json),
    ]

    for imgs, dst_root, json_list in splits:
        for src in imgs:
            dst = dst_root / f"{cls_dir.name}__{src.name}"
            if not dst.exists():
                shutil.copy2(src, dst)
            json_list.append({
                "data": dst.name,
                "label": label
            })

for out_path, data in [
    (ds_root / "train_data.json", train_json),
    (ds_root / "eval_data.json", eval_json),
    (ds_root / "test_data.json", test_json),
]:
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

print("Caltech-101 done.")
print("Train:", len(train_json), "Eval:", len(eval_json), "Test:", len(test_json))

from pathlib import Path
import json
import random
import shutil

ROOT = Path(__file__).resolve().parent
ds_root = ROOT / "datasets" / "caltech-101"

image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

train_dir = ds_root / "train_dataset"
eval_dir = ds_root / "eval_dataset"
test_dir = ds_root / "test_dataset"

for d in (train_dir, eval_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

train_json, eval_json, test_json = [], [], []

random.seed(1234)

# 1) Find all class directories that actually contain images
class_dirs = []
for p in ds_root.rglob("*"):
    if not p.is_dir():
        continue
    # skip our own split dirs and macos junk
    if any(part in {"train_dataset", "eval_dataset", "test_dataset", "__MACOSX"} for part in p.parts):
        continue
    # check if this folder directly contains image files
    has_image = any(
        child.is_file() and child.suffix.lower() in image_exts
        for child in p.iterdir()
    )
    if has_image:
        class_dirs.append(p)

if not class_dirs:
    print("No class directories with images found under", ds_root)
else:
    print("Found", len(class_dirs), "class dirs. Example:", class_dirs[0])

# 2) For each class dir, collect images and split
for cls_dir in sorted(class_dirs):
    label = cls_dir.name
    images = sorted(
        p for p in cls_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )
    if not images:
        continue

    random.shuffle(images)
    n = len(images)
    if n == 0:
        continue

    # 70% train, 10% eval, 20% test
    n_train = max(1, int(0.7 * n))
    n_eval = max(1, int(0.1 * n))
    n_test = n - n_train - n_eval
    if n_test < 0:
        n_test = 0

    splits = [
        (images[:n_train], train_dir, train_json),
        (images[n_train:n_train + n_eval], eval_dir, eval_json),
        (images[n_train + n_eval:], test_dir, test_json),
    ]

    for imgs, dst_root, json_list in splits:
        for src in imgs:
            dst_name = f"{label}__{src.name}"
            dst = dst_root / dst_name
            if not dst.exists():
                shutil.copy2(src, dst)
            json_list.append({
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

print("Caltech-101 v2 done.")
print("Train:", len(train_json), "Eval:", len(eval_json), "Test:", len(test_json))


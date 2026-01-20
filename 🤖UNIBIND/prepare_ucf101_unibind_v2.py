from pathlib import Path
import json
import random
import shutil

ROOT = Path(__file__).resolve().parent
ds_root = ROOT / "datasets" / "UCF101"

video_exts = {".avi", ".mp4", ".mkv", ".mov"}

train_dir = ds_root / "train_dataset"
eval_dir = ds_root / "eval_dataset"
test_dir = ds_root / "test_dataset"

for d in (train_dir, eval_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

train_json, eval_json, test_json = [], [], []

random.seed(1234)

# 1) Find class directories that contain video files directly
class_dirs = []
for p in ds_root.rglob("*"):
    if not p.is_dir():
        continue
    if any(part in {"train_dataset", "eval_dataset", "test_dataset", "__MACOSX"} for part in p.parts):
        continue
    has_video = any(
        child.is_file() and child.suffix.lower() in video_exts
        for child in p.iterdir()
    )
    if has_video:
        class_dirs.append(p)

if not class_dirs:
    print("No class dirs with videos found under", ds_root)
else:
    print("Found", len(class_dirs), "class dirs. Example:", class_dirs[0])

# 2) Split videos per class
for cls_dir in sorted(class_dirs):
    label = cls_dir.name
    videos = sorted(
        p for p in cls_dir.iterdir()
        if p.is_file() and p.suffix.lower() in video_exts
    )
    if not videos:
        continue

    random.shuffle(videos)
    n = len(videos)
    if n == 0:
        continue

    n_train = max(1, int(0.7 * n))
    n_eval = max(1, int(0.1 * n))
    n_test = n - n_train - n_eval
    if n_test < 0:
        n_test = 0

    splits = [
        (videos[:n_train], train_dir, train_json),
        (videos[n_train:n_train + n_eval], eval_dir, eval_json),
        (videos[n_train + n_eval:], test_dir, test_json),
    ]

    for vids, dst_root, json_list in splits:
        for src in vids:
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

print("UCF101 v2 done.")
print("Train:", len(train_json), "Eval:", len(eval_json), "Test:", len(test_json))

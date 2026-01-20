from pathlib import Path
import csv
import json
import shutil

# UniBind repo root
ROOT = Path(__file__).resolve().parent
esc_root = ROOT / "datasets" / "ESC-50"

audio_dir = esc_root / "audio"
meta_csv = esc_root / "meta" / "esc50.csv"

train_dir = esc_root / "train_dataset"
eval_dir = esc_root / "eval_dataset"
test_dir = esc_root / "test_dataset"

for d in (train_dir, eval_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

train_json, eval_json, test_json = [], [], []

with open(meta_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row["filename"]          # e.g. "1-100032-A-0.wav"
        fold = int(row["fold"])          # 1..5
        label = row["category"]          # e.g. "dog"

        src = audio_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing audio file: {src}")

        # Folds 1–3 -> train, 4 -> eval, 5 -> test (common ESC-50 split)
        if fold in (1, 2, 3):
            dst_dir = train_dir
            json_list = train_json
        elif fold == 4:
            dst_dir = eval_dir
            json_list = eval_json
        else:  # fold == 5
            dst_dir = test_dir
            json_list = test_json

        dst = dst_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

        json_list.append({
            "data": fname,   # filename only; UniBind joins with *_dataset/
            "label": label   # text label
        })

for out_path, data in [
    (esc_root / "train_data.json", train_json),
    (esc_root / "eval_data.json", eval_json),
    (esc_root / "test_data.json", test_json),
]:
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

print("Done.")
print("Train examples:", len(train_json))
print("Eval examples :", len(eval_json))
print("Test examples :", len(test_json))

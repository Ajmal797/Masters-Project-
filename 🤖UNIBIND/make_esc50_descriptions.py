import os
import json
import csv

# Base paths
BASE_DIR = "datasets/ESC-50"
META_CSV = os.path.join(BASE_DIR, "meta", "esc50.csv")

SPLITS = ["train", "eval", "test"]  # we will process train_data.json, eval_data.json, test_data.json


def load_filename_to_category():
    """
    Read meta/esc50.csv and build a mapping:
    filename.wav -> category (string)
    """
    mapping = {}
    with open(META_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]          # e.g. "1-100032-A-0.wav"
            category = row["category"]       # e.g. "dog"
            mapping[fname] = category
    print(f"Loaded {len(mapping)} filename->category mappings from esc50.csv")
    return mapping


def add_descriptions_to_split(split, filename_to_category):
    in_path = os.path.join(BASE_DIR, f"{split}_data.json")
    out_path = os.path.join(BASE_DIR, f"{split}_data_desc.json")

    if not os.path.exists(in_path):
        print(f"[WARNING] {in_path} does not exist, skipping {split} split.")
        return

    with open(in_path, "r") as f:
        data = json.load(f)

    new_data = []
    missing = 0

    for item in data:
        rel_path = item["data"]          # e.g. "audio/1-100032-A-0.wav"
        fname = os.path.basename(rel_path)

        category = filename_to_category.get(fname)
        if category is None:
            # fallback if not found in CSV
            missing += 1
            category = "unknown sound"

        # Simple description template; you can tweak the wording if you like
        description = f"An environmental sound of {category}."

        # Copy original fields and add description
        new_item = dict(item)
        new_item["description"] = description
        new_data.append(new_item)

    with open(out_path, "w") as f:
        json.dump(new_data, f)

    print(f"[{split}] wrote {len(new_data)} samples with descriptions to {out_path}")
    if missing > 0:
        print(f"[{split}] WARNING: {missing} samples had no category in CSV and were marked as 'unknown sound'.")


def main():
    filename_to_category = load_filename_to_category()
    for split in SPLITS:
        add_descriptions_to_split(split, filename_to_category)


if __name__ == "__main__":
    main()

import os
import json
import csv
import random

BASE_DIR = "datasets/ESC-50"
META_CSV = os.path.join(BASE_DIR, "meta", "esc50.csv")
CAT_DESC_JSON = os.path.join(BASE_DIR, "esc50_gpt_category_descriptions.json")

SPLITS = ["train", "eval", "test"]


def load_filename_to_category():
    mapping = {}
    with open(META_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]        # e.g. "1-100032-A-0.wav"
            category = row["category"]     # e.g. "dog"
            mapping[fname] = category
    print(f"Loaded {len(mapping)} filename->category mappings from esc50.csv")
    return mapping


def load_category_descriptions():
    with open(CAT_DESC_JSON, "r") as f:
        cat_desc = json.load(f)
    # ensure lists
    for k, v in cat_desc.items():
        if not isinstance(v, list):
            cat_desc[k] = [str(v)]
    print(f"Loaded GPT descriptions for {len(cat_desc)} categories")
    return cat_desc


def add_descriptions_to_split(split, filename_to_category, cat_desc):
    in_path = os.path.join(BASE_DIR, f"{split}_data.json")
    out_path = os.path.join(BASE_DIR, f"{split}_data_gpt_desc.json")

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
            missing += 1
            category = "unknown"

        desc_list = cat_desc.get(category, [])
        if not desc_list:
            # fallback simple template if no GPT desc
            description = f"An environmental sound of {category}."
        else:
            description = random.choice(desc_list).strip()

        new_item = dict(item)
        new_item["description"] = description
        new_data.append(new_item)

    with open(out_path, "w") as f:
        json.dump(new_data, f)

    print(f"[{split}] wrote {len(new_data)} samples with GPT descriptions to {out_path}")
    if missing > 0:
        print(f"[{split}] WARNING: {missing} samples had no category in CSV.")


def main():
    random.seed(1234)
    filename_to_category = load_filename_to_category()
    cat_desc = load_category_descriptions()
    for split in SPLITS:
        add_descriptions_to_split(split, filename_to_category, cat_desc)


if __name__ == "__main__":
    main()

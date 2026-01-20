import os
import json
import pickle

# ---- CONFIG ----
# This is the INNER folder that directly contains all the class folders
ROOT = "datasets/Caltech101/Caltech101"

# Centre embeddings file for event Caltech (from centre_embs/)
CENTER_EMB_PATH = "centre_embs/event_caltech_center_embeddings.pkl"

# Where we will save the JSON file that infer.py will read
OUT_JSON = "datasets/Caltech101/test_data.json"
# ----------------


def main():
    # 1. Load centre embeddings (to know class names + label order)
    with open(CENTER_EMB_PATH, "rb") as f:
        centers = pickle.load(f)

    if not isinstance(centers, dict):
        raise TypeError(
            f"Expected dict from {CENTER_EMB_PATH}, got {type(centers)} instead."
        )

    class_names = list(centers.keys())
    name_to_label = {name: i for i, name in enumerate(class_names)}

    print("Number of classes in centre embeddings:", len(class_names))
    print("First 10 class names from center embeddings:", class_names[:10])

    # 2. List dataset folders under ROOT
    print("\nFolders in dataset root (Caltech101/Caltech101):")
    ds_classes = [
        d for d in sorted(os.listdir(ROOT))
        if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith(".")
    ]
    print(ds_classes[:30], "..." if len(ds_classes) > 30 else "")

    entries = []
    skipped_classes = []

    for cls in ds_classes:
        if cls not in name_to_label:
            print(f"[WARNING] Class folder '{cls}' not in centre embeddings. Skipping.")
            skipped_classes.append(cls)
            continue

        label = name_to_label[cls]
        cls_dir = os.path.join(ROOT, cls)

        for fname in sorted(os.listdir(cls_dir)):
            if fname.startswith("."):
                continue
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath):
                continue

            # Path RELATIVE to ROOT (what infer.py expects with test_dataset_dir)
            rel_path = os.path.join(cls, fname)
            entries.append({
                "data": rel_path,
                "label": label
            })

    print("\nTotal samples collected:", len(entries))
    if skipped_classes:
        print("Skipped classes (no match in centre embeddings):")
        print(skipped_classes)

    # 3. Save JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(entries, f)

    print("\nWrote JSON to:", OUT_JSON)


if __name__ == "__main__":
    main()

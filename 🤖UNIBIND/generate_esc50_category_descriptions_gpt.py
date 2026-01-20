import os
import csv
import json
from openai import OpenAI

# Create a client using the API key from environment
client = OpenAI()

BASE_DIR = "datasets/ESC-50"
META_CSV = os.path.join(BASE_DIR, "meta", "esc50.csv")
OUT_JSON = os.path.join(BASE_DIR, "esc50_gpt_category_descriptions.json")

# how many descriptions per class you want
NUM_DESCRIPTIONS_PER_CLASS = 30  # you can change this to 20–50 if needed


def load_categories():
    cats = set()
    with open(META_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cats.add(row["category"])
    cats = sorted(list(cats))
    print(f"Found {len(cats)} unique categories in ESC-50.")
    return cats


def ask_gpt_for_category(category: str, num_desc: int = 30):
    """
    Call GPT-4o to generate multiple descriptions for a class.
    """
    system_msg = (
        "You are an audio expert. "
        "You generate short, diverse natural language descriptions of environmental sounds."
    )

    user_msg = (
        f"Generate {num_desc} short, diverse sentences describing environmental sounds "
        f"that belong to the ESC-50 dataset category '{category}'. "
        f"Each sentence should describe a plausible audio clip of this class. "
        f"Return ONLY a valid JSON array of strings, with no extra text."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",   # you can also use gpt-4o-mini if you want cheaper
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.9,
    )

    content = resp.choices[0].message.content
    # content should be a JSON array; parse it
    try:
        desc_list = json.loads(content)
        if not isinstance(desc_list, list):
            raise ValueError("Parsed JSON is not a list")
        desc_list = [str(x).strip() for x in desc_list if str(x).strip()]
        return desc_list
    except Exception as e:
        print(f"[ERROR] Failed to parse GPT output for category '{category}': {e}")
        print("Raw content (first 300 chars):", content[:300])
        return []


def main():
    categories = load_categories()
    all_desc = {}

    for cat in categories:
        print(f"\n=== Generating descriptions for category: {cat} ===")
        descs = ask_gpt_for_category(cat, NUM_DESCRIPTIONS_PER_CLASS)
        print(f"Got {len(descs)} descriptions for '{cat}'.")
        all_desc[cat] = descs

    with open(OUT_JSON, "w") as f:
        json.dump(all_desc, f, indent=2)

    print(f"\nSaved GPT-4 descriptions to {OUT_JSON}")


if __name__ == "__main__":
    main()

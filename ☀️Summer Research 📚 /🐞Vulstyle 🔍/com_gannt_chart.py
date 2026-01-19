import os
import json
import pandas as pd
import numpy as np
import torch
import ast
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

# ========== Paths ==========
DEVIGN_PATH = r"/x1/vselvaraj/snap/summEr/devign.json"
MVD_PATH = r"/x1/vselvaraj/snap/summEr/mvd.csv"
REVEAL_PATH = r"/x1/vselvaraj/snap/summEr/reveal.csv"
FEATURES_PATH_CPU = "features_cpu.npz"
FEATURES_PATH_GPU = "features_gpu.npz"

# ========== Load Local Datasets ==========
def detect_column(df, options):
    for opt in options:
        if opt in df.columns:
            return opt
    raise KeyError(f"None of the columns {options} found.")

def load_local_datasets(devign_path, mvd_path, reveal_path):
    with open(devign_path, "r", encoding="utf-8") as f:
        devign_data = json.load(f)
    devign = [{"func": d["func"], "target": d["target"]} for d in devign_data]

    mvd_df = pd.read_csv(mvd_path)
    reveal_df = pd.read_csv(reveal_path)

    mvd_code_col = detect_column(mvd_df, ["code", "func", "function", "code_snippet", "functionSource"])
    mvd_label_col = detect_column(mvd_df, ["label", "target", "vul", "class"])
    reveal_code_col = detect_column(reveal_df, ["code", "func", "function", "code_snippet", "functionSource"])
    reveal_label_col = detect_column(reveal_df, ["label", "target", "vul", "class"])

    mvd = [{"func": row[mvd_code_col], "target": row[mvd_label_col]} for _, row in mvd_df.iterrows()]
    reveal = [{"func": row[reveal_code_col], "target": row[reveal_label_col]} for _, row in reveal_df.iterrows()]

    return devign + mvd + reveal

# ========== AST Stylometry ==========
def extract_ast_features(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return [0] * 5
    types = {"If": 0, "For": 0, "While": 0, "Try": 0, "FunctionDef": 0}
    for node in ast.walk(tree):
        for name in types:
            if isinstance(node, getattr(ast, name)):
                types[name] += 1
    return list(types.values())

# ========== Batched CodeBERT Embedding ==========
def get_codeberta_batch(codes, tokenizer, model, device, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(codes), batch_size), desc="CodeBERTa Batching"):
        batch = codes[i:i+batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", max_length=128, truncation=True,
                               padding="max_length").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        except Exception:
            emb = np.zeros((len(batch), model.config.hidden_size))
        embeddings.append(emb)
    return np.vstack(embeddings)

# ========== Feature Extraction ==========
def extract_features(dataset, device):
    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1").to(device)
    model.eval()

    codes = [sample["func"] for sample in dataset]
    labels = [sample["target"] for sample in dataset]

    print(f"[{device}] AST extraction...")
    start_ast = time.time()
    ast_features = [extract_ast_features(code) for code in tqdm(codes, desc="AST Extraction")]
    end_ast = time.time()

    print(f"[{device}] CodeBERT encoding...")
    start_enc = time.time()
    codeberta_features = get_codeberta_batch(codes, tokenizer, model, device)
    end_enc = time.time()

    X = np.hstack([ast_features, codeberta_features])
    y = np.array(labels)

    return X, y, end_ast - start_ast, end_enc - start_enc

# ========== Pipeline Execution ==========
print("Loading datasets...")
dataset = load_local_datasets(DEVIGN_PATH, MVD_PATH, REVEAL_PATH)
print(f"Total samples: {len(dataset)}")

stage_times = {}

for mode in ["cpu", "cuda"]:
    if mode == "cuda" and not torch.cuda.is_available():
        print("CUDA not available.")
        continue

    feature_path = FEATURES_PATH_GPU if mode == "CUDA" else FEATURES_PATH_CPU

    if os.path.exists(feature_path):
        data = np.load(feature_path)
        X, y = data["X"], data["y"]
        ast_t, enc_t = 0, 0
    else:
        X, y, ast_t, enc_t = extract_features(dataset, torch.device(mode))
        np.savez_compressed(feature_path, X=X, y=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[{mode.upper()}] Training MLP...")
    start_train = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(1024, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    end_train = time.time()

    y_pred = clf.predict(X_test)
    print(f"\n=== {mode.upper()} Evaluation ===")
    print(classification_report(y_test, y_pred, digits=4))

    stage_times[mode.upper()] = {
        "AST Extraction": round(ast_t, 2),
        "CodeBERT Encoding": round(enc_t, 2),
        "MLP Training": round(end_train - start_train, 2)
    }

# ========== Gantt Chart ==========
# ========== Gantt Chart: Enhanced View ==========
base_time = datetime(2025, 7, 23, 10, 0, 0)
records = []

# Create timeline records for CPU and GPU stages
for device, stages in stage_times.items():
    curr_time = base_time
    for stage, duration in stages.items():
        end = curr_time + timedelta(seconds=duration)
        records.append({
            "Device": device,
            "Stage": stage,
            "Start": curr_time,
            "End": end,
            "Duration": duration
        })
        # Add gap of 3s for better visual separation between stages
        curr_time = end + timedelta(seconds=3)

df = pd.DataFrame(records)

# Define consistent colors
colors = {
    "AST Extraction": "#87CEEB",     # Sky Blue
    "CodeBERT Encoding": "#FFA500",  # Orange
    "MLP Training": "#32CD32"        # Lime Green
}

# Plot Gantt chart
fig, ax = plt.subplots(figsize=(14, 7))

yticks = []
yticklabels = []

# Create bars with spacing and clean labels
for i, row in df.iterrows():
    y = i * 1.5  # dynamic spacing between bars
    ax.barh(y, (row['End'] - row['Start']).total_seconds(),
            left=row['Start'],
            height=1.1,
            color=colors.get(row['Stage'], 'gray'))

    # Add label in the middle of each bar
    label = f"{row['Stage']} ({row['Duration']}s)"
    ax.text(row['Start'] + timedelta(seconds=row['Duration']/2), y,
            label, ha='center', va='center', fontsize=9, color='black')

    yticks.append(y)
    yticklabels.append(f"{row['Device']}")

# Configure axes and labels
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.set_title("Pipeline Execution Timeline (CPU vs GPU)", fontsize=14)
ax.set_xlabel("Time")
ax.set_ylabel("Execution Device")
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

# Add legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
labels = list(colors.keys())
ax.legend(handles, labels, title="Pipeline Stage", loc="upper right")

plt.tight_layout()
plt.savefig("gantt_cpu_gpu_pipeline_cleaned.png")
plt.show()

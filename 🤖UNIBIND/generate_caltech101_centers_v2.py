import torch
import pickle
from pathlib import Path
from model import UniBind
from utils.logging import setup_logging
import argparse
import yaml

# -------------------------------------------------------
# STEP 1: Load UniBind config EXACTLY like infer.py does
# -------------------------------------------------------

def load_config(config_path="configs/infer.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)

# -------------------------------------------------------
# STEP 2: Initialize UniBind model with pretrained weights
# -------------------------------------------------------

def load_unibind_model():
    args = load_config()
    model = UniBind(args)
    ckpt = torch.load("ckpts/pretrained_weights.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

# -------------------------------------------------------
# STEP 3: Main script
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent
data_root = ROOT / "datasets" / "caltech-101" / "101_ObjectCategories"
save_path = ROOT / "centre_embs" / "caltech101_center_embeddings.pkl"

# 1. Collect class names
class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
class_names = [d.name for d in class_dirs]
print("Found classes:", len(class_names))

# 2. Prepare text prompts
prompts = [f"a photo of a {cls.replace('_', ' ')}" for cls in class_names]

# 3. Load UniBind model
print("Loading UniBind model...")
model = load_unibind_model()

# 4. Encode text prompts into embeddings
print("Encoding text prompts...")
with torch.no_grad():
    text_embs = model.encode_text(prompts).cpu().numpy()

# 5. Save embeddings
center_dict = {
    "labels": class_names,
    "embeddings": text_embs
}

with open(save_path, "wb") as f:
    pickle.dump(center_dict, f)

print("Saved Caltech-101 center embeddings →", save_path)

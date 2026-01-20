import torch
import pickle
from pathlib import Path
from model import UniBind
import argparse
import yaml

# -------------------------------------------------------
# Load config from configs/infer.yaml (same style as infer.py)
# -------------------------------------------------------

def load_config(config_path="configs/infer.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    # convert dict -> argparse.Namespace so it looks like args
    return argparse.Namespace(**cfg_dict)

def load_unibind_model():
    args = load_config()
    model = UniBind(args)
    ckpt_path = Path("ckpts/pretrained_weights.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

# -------------------------------------------------------
# Main: build Caltech-101 center embeddings
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent
data_root = ROOT / "datasets" / "caltech-101" / "101_ObjectCategories"
save_path = ROOT / "centre_embs" / "caltech101_center_embeddings.pkl"

if not data_root.exists():
    raise FileNotFoundError(f"Caltech-101 classes folder not found at: {data_root}")

class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
class_names = [d.name for d in class_dirs]
print("Found classes:", len(class_names))

# Build simple text prompts
prompts = [f"a photo of a {cls.replace('_', ' ')}" for cls in class_names]

print("Loading UniBind model...")
model = load_unibind_model()

print("Encoding text prompts...")
with torch.no_grad():
    text_embs = model.encode_text(prompts).cpu().numpy()

center_dict = {
    "labels": class_names,
    "embeddings": text_embs,
}

save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(center_dict, f)

print("Saved Caltech-101 center embeddings →", save_path)

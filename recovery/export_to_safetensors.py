"""
One-time recovery script.

Your checkpoint (vit_best.pth) was saved as a raw state_dict by a
pre-v5 `transformers` (the internal ViT attention module names changed
in v5: query/key/value -> q_proj/k_proj/v_proj, attention.output.dense ->
o_proj, intermediate/output.dense -> mlp.fc1/fc2). That's why loading it
in a v5 environment produced 192 missing + 192 unexpected keys and
silently served an untrained model.

Run this ONCE, in an environment pinned to the old version, to reload
the checkpoint correctly and re-export it as config.json + safetensors
(the portable HF format). That format is what api.py should load from
now on -- it doesn't depend on raw module names lining up, so it survives
future transformers upgrades.

Usage:
    python -m venv recovery_venv
    source recovery_venv/bin/activate            # or recovery_venv\\Scripts\\activate on Windows
    pip install "transformers==4.57.3" torch torchvision huggingface_hub --break-system-packages
    python export_to_safetensors.py --checkpoint vit_best.pth --out ./vit_export
"""
import argparse
import sys

import torch
from transformers import ViTForImageClassification, __version__ as tf_version

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def build_model():
    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True,
    )


def main(checkpoint_path: str, out_dir: str):
    print(f"transformers version: {tf_version}")
    if tf_version.split(".")[0] not in ("4",):
        print(
            "WARNING: this script expects a pre-v5 (4.x) transformers install "
            "to match the checkpoint's original key names. You are on "
            f"{tf_version}. Install transformers==4.57.3 and re-run."
        )

    model = build_model()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys or result.unexpected_keys:
        print(f"Missing keys:    {len(result.missing_keys)}")
        print(f"Unexpected keys: {len(result.unexpected_keys)}")
        print("\nCheckpoint did NOT load cleanly. Do not export -- this would")
        print("just re-package an untrained model. Fix the transformers")
        print("version mismatch first, then re-run this script.")
        sys.exit(1)

    print("Checkpoint loaded cleanly: 0 missing, 0 unexpected keys.")

    # Sanity check: a trained classifier head should have non-trivial,
    # non-zero weight spread. Catches the "looks loaded but is actually
    # still random" failure mode even when key names happen to match.
    std = model.classifier.weight.std().item()
    print(f"classifier.weight std = {std:.4f} (near-default-init values around "
          f"~0.02 for an untouched Linear(768,4) would be suspicious here)")

    model.eval()
    model.config.id2label = {i: c for i, c in enumerate(CLASSES)}
    model.config.label2id = {c: i for i, c in enumerate(CLASSES)}

    model.save_pretrained(out_dir, safe_serialization=True)
    print(f"\nExported portable model to: {out_dir}")
    print("Next: upload the contents of this folder to your HF Hub repo,")
    print("replacing vit_best.pth. Then point api.py at the repo with")
    print("ViTForImageClassification.from_pretrained(repo_id).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="vit_best.pth")
    parser.add_argument("--out", default="./vit_export")
    args = parser.parse_args()
    main(args.checkpoint, args.out)

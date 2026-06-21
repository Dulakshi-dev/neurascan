"""
Quick sanity-check: runs the exported model against a sample of the local
test set and prints per-class + overall accuracy. A single prediction can't
tell you whether the recovery worked -- this gives a real number to compare
against the test accuracy your training notebook reported.

Usage:
    python quick_eval.py --test_dir "C:\\Users\\Dulakshi\\Projects\\brain-tumor-classifier\\data\\processed\\test" --n_per_class 20
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main(test_dir: str, n_per_class: int, model_dir: str):
    model = ViTForImageClassification.from_pretrained(model_dir).eval()

    overall_correct, overall_total = 0, 0
    for cls in CLASSES:
        cls_dir = Path(test_dir) / cls
        if not cls_dir.exists():
            print(f"  (skipping {cls}: folder not found at {cls_dir})")
            continue

        files = sorted(cls_dir.glob("*.jpg"))[:n_per_class]
        correct = 0
        for f in files:
            img = Image.open(f).convert("RGB")
            x = tfm(img).unsqueeze(0)
            with torch.no_grad():
                pred_idx = model(pixel_values=x).logits.argmax(dim=1).item()
            if CLASSES[pred_idx] == cls:
                correct += 1

        total = len(files)
        overall_correct += correct
        overall_total += total
        acc = correct / total if total else 0.0
        print(f"  {cls:12s} {correct:>3}/{total:<3}  ({acc:.1%})")

    if overall_total:
        print(f"\nOverall: {overall_correct}/{overall_total} ({overall_correct/overall_total:.1%})")
    else:
        print("\nNo test images found -- check --test_dir.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, help="Folder containing glioma/ meningioma/ notumor/ pituitary/ subfolders")
    parser.add_argument("--n_per_class", type=int, default=20)
    parser.add_argument("--model_dir", default="vit_export")
    args = parser.parse_args()
    main(args.test_dir, args.n_per_class, args.model_dir)
import argparse
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

import pydicom
from pydicom import dcmread
from PIL import Image

CLASS_MAP = {
    "ausencia": 0,
    "ambos": 1,
    "pleura": 1,
    "intrapulmonar": 1
}

VALID_PROJECTIONS = {"VD", "LLD", "LLE"}

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_projection(filename: str, wanted: str) -> bool:
    base = Path(filename).stem
    return re.search(rf"_(?:{wanted})(?:_\d+)?$", base) is not None

def read_dicom_any(path: Path) -> np.ndarray:
    ds = dcmread(str(path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.float32)

    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    arr = arr * slope + intercept

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter)
        ww = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth)
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        arr = np.clip((arr - low) / (high - low + 1e-6), 0.0, 1.0)
    else:
        p1, p99 = np.percentile(arr, (1, 99))
        arr = np.clip((arr - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    return arr

def to_pil_3ch(img01: np.ndarray) -> Image.Image:
    img255 = (img01 * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img255, mode="L").convert("RGB")
    return pil

class CXRPairs(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img01 = read_dicom_any(path)
        pil = to_pil_3ch(img01)
        img = self.transform(pil)
        return img, torch.tensor(label, dtype=torch.long)

def gather_items(root: Path, wanted_projection: str):
    assert wanted_projection in VALID_PROJECTIONS, f"Projection must be one of {VALID_PROJECTIONS}"
    items_by_patient = defaultdict(list)
    labels_by_patient = dict()

    for class_name, y in CLASS_MAP.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for patient_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
            # Base patient id = '1234' from '1234' or '1234_2'
            base_id = patient_dir.name.split("_")[0]
            matched_any = False
            for f in patient_dir.iterdir():
                if f.is_file() and is_projection(f.name, wanted_projection):
                    matched_any = True
                    items_by_patient[base_id].append((f, y))
            if matched_any:
                labels_by_patient[base_id] = y

    patients = sorted(items_by_patient.keys())
    labels = [labels_by_patient[p] for p in patients]
    return patients, labels, items_by_patient

def split_patients(patients, labels, val_split, test_split, seed):
    assert 0.0 < val_split < 1.0 and 0.0 < test_split < 1.0 and val_split + test_split < 1.0
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(patients, labels))
    trainval_patients = [patients[i] for i in trainval_idx]
    trainval_labels = [labels[i] for i in trainval_idx]
    test_patients = [patients[i] for i in test_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_split / (1.0 - test_split), random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_patients, trainval_labels))
    train_patients = [trainval_patients[i] for i in train_idx]
    val_patients = [trainval_patients[i] for i in val_idx]

    return train_patients, val_patients, test_patients

def make_items_subset(patient_ids, items_by_patient):
    items = []
    for pid in patient_ids:
        items.extend(items_by_patient[pid])
    return items

def make_transforms(use_aug: bool, image_size: int = 224):
    train_aug = []
    if use_aug:
        train_aug += [
            T.RandomRotation(degrees=5, fill=0),
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        ]
    else:
        train_aug += [T.Resize((image_size, image_size))]

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    train_transform = T.Compose(train_aug + [T.ToTensor(), normalize])
    val_transform   = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), normalize])
    return train_transform, val_transform

class BioViLTFeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        use_cuda = (device.startswith("cuda") and torch.cuda.is_available())
        self.device = "cuda" if use_cuda else "cpu"
        self.out_dim = 128

        try:
            from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
            self.model = get_biovil_t_image_encoder()
            self.model.eval().to(self.device)
            self.mode = "himl_multimodal"
        except Exception as e:
            raise RuntimeError(
                "Failed to load BioViL‑T image encoder from hi-ml-multimodal. "
                "Make sure `pip install hi-ml-multimodal` succeeded."
            ) from e

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.model(images.to(self.device))
        return out.projected_global_embedding

class Head(nn.Module):
    def __init__(self, in_dim=128, dropout=False, p=0.3, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(p) if dropout else nn.Identity(),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_one_epoch(feat, head, loader, optimizer, criterion, device):
    feat.eval()
    head.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            feats = feat(imgs)
        logits = head(feats)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(feat, head, loader, criterion, device):
    feat.eval()
    head.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        feats = feat(imgs)
        logits = head(feats)
        probs = torch.softmax(logits, dim=1)[:, 1]
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, preds).tolist()
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    return (total_loss / len(loader.dataset)), acc, prec, rec, f1, cm, (fpr, tpr, roc_auc)

def plot_losses(train_losses, val_losses, out_path):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_roc(fpr, tpr, roc_auc, out_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--projection", type=str, default="VD", choices=list(VALID_PROJECTIONS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--finetune_encoder", action="store_true",
                        help="If set, will also train the BioViL-T image encoder (slower, more memory).")
    args = parser.parse_args()

    set_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("ssl_output") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    patients, labels, items_by_patient = gather_items(root, args.projection)
    if len(patients) == 0:
        raise RuntimeError(f"No patients found for projection {args.projection} under {root}")

    train_p, val_p, test_p = split_patients(patients, labels, args.val_split, args.test_split, args.seed)
    train_items = make_items_subset(train_p, items_by_patient)
    val_items   = make_items_subset(val_p, items_by_patient)
    test_items  = make_items_subset(test_p, items_by_patient)

    train_tf, val_tf = make_transforms(args.use_augmentation, args.image_size)

    train_ds = CXRPairs(train_items, train_tf)
    val_ds   = CXRPairs(val_items,   val_tf)
    test_ds  = CXRPairs(test_items,  val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    feat = BioViLTFeatureExtractor(device=args.device)
    if args.finetune_encoder:
        feat.train()
        for p in feat.parameters():
            p.requires_grad = True
    else:
        for p in feat.parameters():
            p.requires_grad = False

    head = Head(in_dim=getattr(feat, "out_dim", 128), dropout=args.dropout, p=args.dropout_rate, num_classes=2).to(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    params = [{"params": head.parameters(), "lr": args.lr}]
    if args.finetune_encoder:
        params.insert(0, {"params": feat.parameters(), "lr": args.lr * 0.1})

    optimizer = optim.AdamW(params, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    best_val_f1 = -1.0
    best_state = None

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    feat = feat.to(device)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(feat, head, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_prec, va_rec, va_f1, va_cm, va_roc = evaluate(feat, head, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {
                "feat": feat.state_dict(),
                "head": head.state_dict(),
                "epoch": epoch
            }

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} | val loss {va_loss:.4f} | val F1 {va_f1:.4f} | "
              f"time {time.time()-t0:.1f}s")

    if best_state is not None:
        feat.load_state_dict(best_state["feat"])
        head.load_state_dict(best_state["head"])

    te_loss, te_acc, te_prec, te_rec, te_f1, te_cm, te_roc = evaluate(feat, head, test_loader, criterion, device)

    config = {
        "data_root": args.data_root,
        "projection": args.projection,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "use_augmentation": bool(args.use_augmentation),
        "dropout": bool(args.dropout),
        "dropout_rate": args.dropout_rate,
        "seed": args.seed,
        "splits": {"val": args.val_split, "test": args.test_split},
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "lr": args.lr,
        "device": device,
        "finetune_encoder": bool(args.finetune_encoder),
        "class_map": CLASS_MAP,
        "patients": {
            "train": train_p,
            "val": val_p,
            "test": test_p
        }
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics = {
        "val": {
            "best_val_f1": best_val_f1,
            "loss": val_losses[-1] if len(val_losses) else None
        },
        "test": {
            "loss": te_loss,
            "accuracy": te_acc,
            "precision": te_prec,
            "recall": te_rec,
            "f1": te_f1,
            "confusion_matrix": te_cm,
            "roc_auc": te_roc[2]
        }
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_losses(train_losses, val_losses, out_dir / "loss_curve.png")
    fpr, tpr, roc_auc = te_roc
    plot_roc(fpr, tpr, roc_auc, out_dir / "roc_curve.png")

    torch.save({"feat": feat.state_dict(), "head": head.state_dict(), "config": config}, out_dir / "model.pt")

    print(f"\nSaved run to: {out_dir.resolve()}")
    print(f"Test — loss: {te_loss:.4f} | acc: {te_acc:.4f} | prec: {te_prec:.4f} | rec: {te_rec:.4f} | f1: {te_f1:.4f} | auc: {roc_auc:.4f}")

if __name__ == "__main__":
    main()

# exemplo execução:
# python3 main.py --data_root ../data --projection VD --epochs 20 --batch_size 16 --use_augmentation --dropout --dropout_rate 0.3 --seed 1337 --test_split 0.2 --val_split 0.1 --num_workers 4

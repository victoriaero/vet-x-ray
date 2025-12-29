import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

from src.models.model_factory import get_model
from src.utils.misc import load_dataset_split, set_seed
from src.datasets.vet_dataset import VetRadiographDataset
from src.utils.experiment_utils import create_output_dir, save_config, plot_loss_curve, plot_confusion_matrix, plot_roc_curve

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def build_transforms(config, is_train=True):
    grayscale = config.get("grayscale_input", False)
    use_aug = config.get("use_augmentation", False)

    use_xrv = config.get("use_torchxrayvision", False)
    xrv_name = config.get("xrv_model_name", "")
    use_evax = config.get("use_evax", False)

    if use_xrv and xrv_name.startswith("resnet"):
        image_size = 512
    else:
        image_size = 224

    transform_list = [transforms.Resize((image_size, image_size))]

    if is_train and use_aug:
        transform_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]

    transform_list += [transforms.ToTensor()]

    use_xrv = config.get("use_torchxrayvision", False)

    if not grayscale and not use_xrv:
        transform_list += [transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        transform_list += [transforms.Normalize([0.5] * 3, [0.5] * 3)]
    else:
        transform_list += [transforms.Normalize([0.5], [0.5])]


    return transforms.Compose(transform_list)


def run_training(config):
    set_seed(config["seed"])
    output_path = create_output_dir(config["output_dir"])
    # save_config(config, output_path)

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = load_dataset_split(
        config["data_dir"],
        config["projection"],
        config["val_split"],
        config["test_split"],
        seed=config["seed"],
        class_map=config["class_map"]
    )

    print("Treino:", Counter(train_labels))
    print("Validação:", Counter(val_labels))
    print("Teste:", Counter(test_labels))

    train_transform = build_transforms(config, is_train=True)
    eval_transform = build_transforms(config, is_train=False)

    use_evax = config.get("use_evax", False)
    use_xrv = config.get("use_torchxrayvision", False)
    xrv_name = config.get("xrv_model_name", "")

    image_size = 512 if use_evax or (use_xrv and xrv_name.startswith("resnet")) else 224

    train_ds = VetRadiographDataset(train_paths, train_labels, train_transform, config["projection"])
    val_ds = VetRadiographDataset(val_paths, val_labels, eval_transform, config["projection"])
    test_ds = VetRadiographDataset(test_paths, test_labels, eval_transform, config["projection"])

    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        config["model_name"], config["pretrained"],
        use_dropout=config.get("use_dropout", False),
        dropout_rate=config.get("dropout_rate", 0.5),
        grayscale_input=config.get("grayscale_input", False),
        use_torchxrayvision=config.get("use_torchxrayvision", False),
        xrv_model_name=config.get("xrv_model_name", ""),
        use_evax=config.get("use_evax", False),
        eva_x_size=config.get("eva_x_size", "small")
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))

    train_losses, val_losses = [], []

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = train_loss / total
        train_losses.append(epoch_loss)
        acc = correct / total
        print(f"\n[{epoch + 1}/{config['num_epochs']}] Loss: {epoch_loss:.4f}, Treinamento Acc: {acc:.4f}")

        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / len(val_ds)
        val_losses.append(val_epoch_loss)

        cm = confusion_matrix(val_targets, val_preds)
        acc = accuracy_score(val_targets, val_preds)
        prec = precision_score(val_targets, val_preds, zero_division=0)
        rec = recall_score(val_targets, val_preds, zero_division=0)
        f1 = f1_score(val_targets, val_preds, zero_division=0)

        print(f"\n[Validação] Loss: {val_epoch_loss:.4f}, Acc: {acc:.4f}")

    plot_loss_curve(train_losses, val_losses, output_path)

    print("\nTeste")
    model.eval()
    test_preds, test_targets, test_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    cm = confusion_matrix(test_targets, test_preds)
    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, zero_division=0)
    rec = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)

    config["final_test_metrics"] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

    print(f"→ Acurácia : {acc:.4f}")
    print(f"→ Precisão : {prec:.4f}")
    print(f"→ Recall   : {rec:.4f}")
    print(f"→ F1-Score : {f1:.4f}")
    print(f"→ Matriz de confusão:\n{cm}")

    plot_confusion_matrix(cm, output_path)
    plot_roc_curve(test_targets, test_probs, output_path)
    save_config(config, output_path)
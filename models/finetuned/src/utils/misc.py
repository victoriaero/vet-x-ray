import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset_split(root_dir, projection="VD", val_split=0.1, test_split=0.1, seed=42, class_map=None):
    if class_map is None:
        raise ValueError("class_map precisa ser definido no config.yaml e passado para load_dataset_split().")

    projection = projection.upper()
    sample_paths, labels = [], []

    for class_name, label in class_map.items():
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for exam_folder in os.listdir(class_path):
            folder_path = os.path.join(class_path, exam_folder)
            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                full_path = os.path.join(folder_path, file)

                if not os.path.isfile(full_path):
                    continue
                if file.startswith("."):
                    continue
                if projection not in file.upper():
                    continue

                sample_paths.append(full_path)
                labels.append(label)

    id_to_paths = {}
    for path, label in zip(sample_paths, labels):
        filename = os.path.basename(path)
        patient_id = filename.split("_")[0]
        id_to_paths.setdefault(patient_id, []).append((path, label))

    unique_ids = list(id_to_paths.keys())
    if len(unique_ids) == 0:
        raise ValueError(f"Nenhum arquivo encontrado com projeção '{projection}'.")

    random.seed(seed)
    random.shuffle(unique_ids)

    test_size = int(test_split * len(unique_ids))
    val_size = int(val_split * len(unique_ids))

    test_ids = unique_ids[:test_size]
    val_ids = unique_ids[test_size:test_size + val_size]
    train_ids = unique_ids[test_size + val_size:]

    def collect(ids):
        paths, lbls = [], []
        for pid in ids:
            for p, l in id_to_paths[pid]:
                paths.append(p)
                lbls.append(l)
        return paths, lbls

    train_paths, train_labels = collect(train_ids)
    val_paths, val_labels = collect(val_ids)
    test_paths, test_labels = collect(test_ids)

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
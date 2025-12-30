"""
PyTorch Dataset for veterinary thoracic radiographs stored as DICOM files.
"""

import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset

class VetRadiographDataset(Dataset):
    def __init__(self, sample_paths, labels, transform=None, projection="VD"):
        self.sample_paths = sample_paths
        self.labels = labels
        self.transform = transform
        self.projection = projection

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        dicom_path = self.sample_paths[idx]

        try:
            ds = pydicom.dcmread(dicom_path)
            img = ds.pixel_array.astype("float32")
        except Exception as e:
            print(f"DICOM inválido ignorado: {dicom_path} → {e}")
            return self.__getitem__((idx + 1) % len(self.sample_paths))

        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = Image.fromarray(img).convert("L")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)
    
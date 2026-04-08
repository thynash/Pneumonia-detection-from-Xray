import os
from PIL import Image
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = ["normal", "pneumonia"]
        self.data = []

        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                self.data.append((os.path.join(cls_path, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label

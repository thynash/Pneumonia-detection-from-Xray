import torch
import os
from torch.utils.data import DataLoader

from data.dataset import XRayDataset
from data.transforms import get_transforms
from training.engine import train_one_epoch, evaluate
from utils.metrics import compute_metrics

# import multiple models
from models import resnet, efficientnet, densenet


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------------------
    # DATA
    # -------------------
    train_tf, val_tf = get_transforms()

    train_ds = XRayDataset("dataset/train", transform=train_tf)
    val_ds = XRayDataset("dataset/val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # -------------------
    # MODEL REGISTRY
    # -------------------
    MODELS = {
        "resnet18": resnet.get_model,
        "efficientnet_b0": efficientnet.get_model,
        "densenet121": densenet.get_model
    }

    # -------------------
    # OUTPUT DIRS
    # -------------------
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # -------------------
    # TRAIN LOOP
    # -------------------
    for name, get_model_fn in MODELS.items():

        print(f"\n==============================")
        print(f"Training {name}")
        print(f"==============================")

        model = get_model_fn().to(device)

        # imbalance handling
        pos_weight = torch.tensor([3.0]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # ---- TRAIN ----
        for epoch in range(5):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"{name} | Epoch {epoch+1} | Loss: {loss:.4f}")

        # ---- EVAL ----
        y_true, y_pred = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred)

        print(f"{name} Metrics:", metrics)

        # ---- SAVE ----
        torch.save(model.state_dict(), f"outputs/checkpoints/{name}.pt")


if __name__ == "__main__":
    run()

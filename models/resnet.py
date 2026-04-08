import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet18(pretrained=True)

    # change input channel to 1 (grayscale)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # binary output
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model

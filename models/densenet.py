import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.densenet121(pretrained=True)

    # modify input for grayscale
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # binary output
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    return model

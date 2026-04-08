import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.efficientnet_b0(pretrained=True)

    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    return model

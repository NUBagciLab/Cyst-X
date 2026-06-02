import torch
import torch.nn as nn
from copy import deepcopy
import monai.networks.nets as nets

class get_model(nn.Module):
    def __init__(self, name="densenet121", num_classes=1):
        super(get_model, self).__init__()
        if 'densenet' in name:
            model = getattr(nets, name.replace('densenet', 'DenseNet'))(spatial_dims=3, in_channels=1,out_channels=num_classes)
            in_features = model.class_layers.out.in_features
            model.class_layers.out = nn.Identity()
        else:
            raise Exception("Model is not supported")
        self.backbone = model
        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x:list):
        f = torch.cat([self.backbone(x[0]), self.backbone(x[1])], dim=1)
        return self.classifier(f)
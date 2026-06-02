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
        self.backbones = nn.ModuleList([model] + [deepcopy(model)])
        self.classifier = nn.Linear(2 * in_features, num_classes)

    def forward(self, x:list):
        f = torch.cat([self.backbones[0](x[0]), self.backbones[1](x[1])], dim=1)
        return self.classifier(f)
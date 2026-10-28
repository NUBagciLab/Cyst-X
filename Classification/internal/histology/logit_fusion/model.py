import torch
import torch.nn as nn
import monai.networks.nets as nets

class get_model(nn.Module):
    def __init__(self, name="densenet121", num_classes=1):
        super(get_model, self).__init__()
        if 'densenet' in name:
            self.model1 = getattr(nets, name.replace('densenet', 'DenseNet'))(spatial_dims=3, in_channels=1,out_channels=num_classes)
            self.model2 = getattr(nets, name.replace('densenet', 'DenseNet'))(spatial_dims=3, in_channels=1,out_channels=num_classes)
        else:
            raise Exception("Model is not supported")
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x:list):
        weight = torch.sigmoid(self.fusion_weight)
        return weight*self.model1(x[0])+(1-weight)*self.model2(x[1])
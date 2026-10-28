import torch
import torch.nn as nn
import monai.networks.nets as nets

class get_model(nn.Module):
    def __init__(self, name='densenet121', num_classes=1):
        super().__init__()
        
        # Instantiate backbone with 2 input channels
        if 'resnet' in name:
            self.backbone = getattr(nets, name)(
                spatial_dims=3, 
                n_input_channels=2, 
                num_classes=num_classes
            )
        elif 'efficientnet' in name:
            self.backbone = nets.EfficientNetBN(
                model_name=name.replace('efficientnet_b', 'efficientnet-b'), 
                in_channels=2, 
                num_classes=num_classes, 
                spatial_dims=3
            )
        elif 'densenet' in name:
            self.backbone = getattr(nets, name.replace('densenet', 'DenseNet'))(
                spatial_dims=3, 
                in_channels=2, 
                out_channels=num_classes
            )
        else:
            raise Exception("Model is not supported")

    def forward(self, x:list):
        # Concatenate image and mask along the channel axis (dim=1)
        x = torch.cat([x[0], x[1]], dim=1)
        return self.backbone(x)
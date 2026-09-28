import monai.networks.nets as nets


def get_model(name = 'densenet121', num_classes=1):   
    if 'resnet' in name:
        return getattr(nets, name)(spatial_dims=3, n_input_channels=1, num_classes=num_classes)
    elif 'efficientnet' in name:
        return nets.EfficientNetBN(model_name=name.replace('efficientnet_b', 'efficientnet-b'), in_channels=1,num_classes=num_classes,spatial_dims=3)
    elif 'densenet' in name:
        return getattr(nets, name.replace('densenet', 'DenseNet'))(spatial_dims=3, in_channels=1,out_channels=num_classes)
    else:
        raise Exception("Model is not supported")

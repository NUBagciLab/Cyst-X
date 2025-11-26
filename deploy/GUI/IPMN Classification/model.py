from monai.networks.nets import DenseNet121

def get_model(in_channel = 1, out_channels = 2):
    return DenseNet121(
        spatial_dims=3,  # 3D input
        in_channels=in_channel,   # Typically for grayscale (e.g., MRI/CT scans), change to 3 for RGB
        out_channels=out_channels   # Adjust for binary or multi-class segmentation/classification
    )


import segmentation_models_pytorch as smp


def get_pretrained_backbone_unet(backbone_name='resnet18', in_channels=1, n_classes=1):
    model = smp.Unet(
        encoder_name=backbone_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=n_classes,  # model output channels (number of classes in your dataset)
    )

    return model
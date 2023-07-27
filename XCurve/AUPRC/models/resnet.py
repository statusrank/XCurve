from ._resnet import _resnet50


def resnet50(output_channels):
    model = _resnet50(
        num_classes=output_channels,
        pretrained=False
    )

    return model

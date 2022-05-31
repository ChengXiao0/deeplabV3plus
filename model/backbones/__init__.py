from model.backbones.resnet import resnet50, resnet101
from model.backbones.MobileNet import mobilenet_v2
import torch.nn as nn

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        model = resnet50()
        return model
    elif backbone == 'resnet101':
        model = resnet101()
        return model
    elif backbone == "mobilenet":
        model = mobilenet_v2(pretrained=True)
        return model


# if __name__ == '__main__':
#     model_net = build_backbone(backbone='mobilenet', output_stride=16, nn.BatchNorm2d)

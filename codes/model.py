""" The model of Digit Recognition """

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.detection as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class CBAM(nn.Module):
    """
    Define CBAM module
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention Module
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward Pass
        """

        # Channel Attention Module
        avg_pool = self.mlp(self.avg_pool(x))
        max_pool = self.mlp(self.max_pool(x))
        channdel_out = self.channel_sigmoid(avg_pool+max_pool)
        x = x * channdel_out

        # Spatial Attention Module
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        spatial_out = self.spatial_sigmoid(spatial_out)
        out = x * spatial_out

        return out

def get_model(num_classes):
    """
    Init the digit recognition model
    """

    model = models.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')

    model.backbone.body.layer2.add_module("cbam", CBAM(512))
    model.backbone.body.layer3.add_module("cbam", CBAM(1024))
    model.backbone.body.layer4.add_module("cbam", CBAM(2048))

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.fg_iou_thresh = 0.4
    model.roi_heads.bg_iou_thresh = 0.3
    model.roi_heads.nms_thresh    = 0.3

    return model

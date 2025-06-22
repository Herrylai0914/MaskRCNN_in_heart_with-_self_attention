import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
import numpy as np
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.resnet import resnet101
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
import torch.nn.functional as F_torch
from torch import nn
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection import MaskRCNN
from collections import OrderedDict
import torchvision.ops as ops
import torch.nn.functional as F_torch
import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
    RegionProposalNetwork
)
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import (
    TwoMLPHead,
    FastRCNNPredictor
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNHeads,
    MaskRCNNPredictor
)
from torchvision.models.detection.image_list import ImageList
class AttentionMaskRCNNHeads(nn.Module):
    def __init__(self, in_channels, layers=(256, 256, 256, 256), dilation=1):
        super().__init__()
        self.convs = nn.Sequential()
        next_channels = in_channels
        for idx, out_channels in enumerate(layers):
            self.convs.add_module(
                f"mask_fcn{idx + 1}",
                nn.Conv2d(next_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
            )
            self.convs.add_module(f"relu{idx + 1}", nn.ReLU(inplace=True))
            next_channels = out_channels

        self.attn = SimpleSelfAttention(in_channels=next_channels)

    def forward(self, x):
        x = self.convs(x)
        x = self.attn(x)  # 加在最後再丟去 predictor
        return x
class SimpleSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimpleSelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if x.shape[0] == 0:
            return x  # 空 batch 就直接跳過，不做 attention
        
        B, C, H, W = x.shape
        proj_q = self.query(x).reshape(B, -1, H * W).permute(0, 2, 1)  # [B, N, C']
        proj_k = self.key(x).reshape(B, -1, H * W)                     # [B, C', N]
        attn = torch.bmm(proj_q, proj_k) / (C ** 0.5)                  # [B, N, N]
        attn = self.softmax(attn)

        proj_v = self.value(x).reshape(B, -1, H * W)                   # [B, C, N]
        out = torch.bmm(proj_v, attn.permute(0, 2, 1))                 # [B, C, N]
        out = out.reshape(B, C, H, W)
        return x + out  # 殘差連接
    
class GrayEnhancementModule(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16):
        super(GrayEnhancementModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)

        self.final = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)    
    def forward(self, x):
        # x 的形狀 (B, 1, H, W)
        residual = x
        x = F_torch.relu(self.bn1(self.conv1(x)))
        x = F_torch.relu(self.bn2(self.conv2(x)))
        x = F_torch.relu(self.bn3(self.conv3(x)))
        x = F_torch.relu(self.bn4(self.conv4(x)))
        x = self.final(x)  # 回到 [B,1,H,W]
        # 加入殘差連結，讓原始資訊得以保留
        return residual + x

# 定義一個把預處理模組跟 backbone 串接起來的 wrapper
class EnhancedBackbone(nn.Module):
    def __init__(self, backbone, enhancement_module):
        super(EnhancedBackbone, self).__init__()
        self.enhancement_module = enhancement_module
        self.backbone = backbone
         # 將原 backbone 的 out_channels 屬性保存下來
        self.out_channels = backbone.out_channels
        
    def forward(self, x):
        # 先用增強模組處理輸入的灰階影像
        x = self.enhancement_module(x)
        # 再送入原本的 backbone (例如 FPN 提取多層特徵)
        return self.backbone(x)

def create_maskrcnn_resnet101_fpn_gray(num_classes, pretrained=True, trainable_layers=5):
    """
    建立一個「單通道輸入」的 MaskRCNN (ResNet101+FPN)，
    並用部分 ImageNet 預訓練權重初始。
    
    參數：
      - num_classes: 最後輸出分類 (含背景)
      - pretrained: 是否載入 ImageNet 預訓練
      - trainable_layers: 有幾層可學習(0~5)
    回傳：
      - MaskRCNN model，第一層 conv1 改為單通道 in_channels=1
        並把預訓練權重 (64,3,7,7) => (64,1,7,7)
    """
    if pretrained:
        weights = ResNet101_Weights.IMAGENET1K_V1
        base_model = resnet101(weights=weights)
    else:
        base_model = resnet101(weights=None)
    
    # --- 修改第一層 conv => in_channels=1 ---
    # 原本: base_model.conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 1) 先取原本 weights: shape=(64,3,7,7)
    old_weights = base_model.conv1.weight  # Tensor

    # 2) 建立新的 conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # 3) 將預訓練權重從 (64,3,7,7) => (64,1,7,7)
    #    常用方法: 將 3 通道的權重取平均 => 單通道
    with torch.no_grad():
        # old_weights.shape => (64,3,7,7)
        # mean over dim=1 => (64,1,7,7)
        mean_weights = old_weights.mean(dim=1, keepdim=True)  # shape=(64,1,7,7)
        new_conv.weight.copy_(mean_weights)

    # 4) 替換回 ResNet101
    base_model.conv1 = new_conv

    # --- 接下來把 ResNet101 + FPN 接起來 ---
    backbone = _resnet_fpn_extractor(base_model, trainable_layers=trainable_layers)

      # --- 增加前處理的增強模組 ---
    enhancement_module = GrayEnhancementModule(in_channels=1, hidden_channels=16)
    # 將增強模組與 backbone 串接起來
    enhanced_backbone = EnhancedBackbone(backbone, enhancement_module)

    # --- 建立 MaskRCNN ---
    model = MaskRCNN(backbone=enhanced_backbone, num_classes=num_classes)
    return model
def create_maskrcnn_resnet101_fpn(num_classes, pretrained=True, trainable_layers=5):
    """
    建立一個 MaskRCNN，其 backbone 是 ResNet101 + FPN (類似官方的 maskrcnn_resnet50_fpn)。
    參數：
      - num_classes: 分類數量 (含背景)
      - pretrained: 是否使用 ImageNet 預訓練 (True => ResNet101_Weights.IMAGENET1K_V1)
      - trainable_layers: 有幾層是可訓練 (0~5 之間) => 5 表示全部都可學。
    回傳：
      - 一個 MaskRCNN 模型物件，但你主要只會用到它的 .backbone。
    """
    if pretrained:
        weights = ResNet101_Weights.IMAGENET1K_V1
        base_model = resnet101(weights=weights)  # 下載 ImageNet 預訓練
    else:
        base_model = resnet101(weights=None)

    # 將 ResNet101 + FPN 接起來，成為可以輸出多尺度特徵的 backbone
    backbone = _resnet_fpn_extractor(base_model, trainable_layers=trainable_layers)
    # 這時 backbone.out_channels = 256 (FPN預設)

    # 用 backbone 建立一個 MaskRCNN，num_classes 由你指定
    # 這個 model 裡最終會有 .backbone = <ResNet101+FPN>
    model = MaskRCNN(backbone=backbone, num_classes=num_classes)
    return model
def create_official_rpn():
    """
    建立一個「與官方 fasterrcnn_resnet50_fpn 預設參數相似」的 RPN。
    """
    # (a) 建立 AnchorGenerator
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    # (b) RPNHead: in_channels=256, num_anchors=3
    in_channels = 256
    num_anchors = anchor_generator.num_anchors_per_location()[0]  # 3
    rpn_head = RPNHead(in_channels, num_anchors)

    # (c) RegionProposalNetwork
    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n=dict(training=2000, testing=1000),
        post_nms_top_n=dict(training=2000, testing=1000),
        nms_thresh=0.7,
        score_thresh=0.0  # 過濾提案的分數閾值
    )
    return rpn
# ------------------------------
# 2) 建立 Box 分支 (ROIHeads的 box_* 部分)
# ------------------------------
def create_official_box_head(num_classes=2):
    """
    建立 RoIHeads 需要的:
      - box_roi_pool (MultiScaleRoIAlign)
      - box_head (TwoMLPHead)
      - box_predictor (FastRCNNPredictor)
    """

    # (a) box_roi_pool: MultiScaleRoIAlign
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],  # 假設FPN四層
        output_size=7,
        sampling_ratio=2
    )
    # (b) TwoMLPHead
    resolution = box_roi_pool.output_size[0]  # 7
    in_channels = 256  # FPN 通道數
    mlp_in_feature = in_channels * (resolution ** 2)  # 256 * 7 * 7 = 12544
    representation_size = 1024
    box_head = TwoMLPHead(mlp_in_feature, representation_size)

    # (c) FastRCNNPredictor
    #    representation_size=1024, num_classes=2(含背景)
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    return box_roi_pool, box_head, box_predictor
# ------------------------------
# 3) 建立 Mask 分支 (ROIHeads的 mask_* 部分)
# ------------------------------
def create_official_mask_head(num_classes=2):
    """
    建立 RoIHeads 需要的:
      - mask_roi_pool (MultiScaleRoIAlign for masks)
      - mask_head (MaskRCNNHeads)
      - mask_predictor (MaskRCNNPredictor)
    """
    # (a) mask_roi_pool
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0','1','2','3'],
        output_size=14,   # 官方預設 mask分支輸出 14x14
        sampling_ratio=2
    )
    # (b) mask_head
    #   官方預設: 4個 256-ch conv
    #   dilation=1
    mask_head =  AttentionMaskRCNNHeads(
        in_channels=256,
        layers=(256, 256, 256, 256),  # 4個 layer
        dilation=1
    )
    # (c) mask_predictor
    #   mask_head 輸出 256 通道 => predictor => (num_classes) mask
    mask_predictor = MaskRCNNPredictor(
        in_channels=256, 
        dim_reduced=256,
        num_classes=num_classes
    )
    return mask_roi_pool, mask_head, mask_predictor
# ------------------------------
# 4) 建立最終 RoIHeads (含 box + mask 分支)
# ------------------------------
def create_official_roi_heads(num_classes=2, want_mask=True):
    """
    若 want_mask=True => 包含 mask分支
       want_mask=False => 不包含 mask分支
    """
    # (a) Box 分支
    box_roi_pool, box_head, box_predictor = create_official_box_head(num_classes)

    # (b) 如果要 mask 分支 => 取得 mask_roi_pool, mask_head, mask_predictor
    #    若不要 => 全設 None
    if want_mask:
        mask_roi_pool, mask_head, mask_predictor = create_official_mask_head(num_classes)
    else:
        mask_roi_pool, mask_head, mask_predictor = None, None, None

    # (c) 建立 RoIHeads
    roi_heads = RoIHeads(
        # --- Box 分支 ---
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,

        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,

        # --- Mask 分支 (可能為 None) ---
        mask_roi_pool=mask_roi_pool,
        mask_head=mask_head,
        mask_predictor=mask_predictor
    )

    return roi_heads
class MyCustomBackboneSingle(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 只要一個 model
        self.model_task = create_maskrcnn_resnet101_fpn_gray(pretrained=True, num_classes=num_classes)
        #self.model_task = create_maskrcnn_resnet101_fpn(pretrained=True, num_classes=num_classes)
        self.out_channels = 256  # FPN 輸出 channel

    def forward(self, images):
        # images: shape=(B,1,H,W) => 單通道
        device = next(self.parameters()).device
        
        images = images.to(device)
        #print(f"In Dataset __getitem__, img.shape = {images.shape}")    
        # 直接做 backbone 提取特徵 => dict
        features = self.model_task.backbone(images)
        return features

class CustomROIHead(nn.Module):
    def __init__(self, num_classes):
        super(CustomROIHead, self).__init__()
        # 分类器和回归器网络（通常是一些全连接层）
        self.cls_head = nn.Sequential(
            nn.Linear(1280 * 7 * 7, 1024),
            nn.ReLU(),
           nn.Dropout(p=0.3),  # 添加 Dropout
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 添加 Dropout
            nn.Linear(1024, num_classes + 1)
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(1280 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 添加 Dropout
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 添加 Dropout
            nn.Linear(1024, 4)
        )

        # 掩码头部分，可以根据需要调整层数和结构
        self.mask_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
)
        

    def forward(self, shared_features, proposals, image_shapes, targets=None):
        # Step 1: Apply ROI Align to extract fixed-size features from proposals
        #print(f"Targets: {targets}")
        roi_pooled_features_list = []
        spatial_scales = [1/8, 1/16, 1/32, 1/64, 1/128]  # 根據特徵圖的縮放比例
        for feature in shared_features.values():
    # 调试输出，确保 feature 是 Tensor
            #print(f"Feature type: {type(feature)}")
            if isinstance(feature, tuple):
                raise ValueError("Feature is of type tuple, but expected a Tensor.")
        for feature, scale in zip(shared_features.values(), spatial_scales):
    # 確保特徵圖是有效的 Tensor
            assert isinstance(feature, torch.Tensor), f"Expected feature to be a Tensor, but got {type(feature)}"
    
    # 執行 ROI Align
            roi_pooled = ops.roi_align(
            feature, proposals, output_size=(7, 7), spatial_scale=scale, sampling_ratio=2
    )
    
    # 將 ROI Align 的輸出加入列表
            roi_pooled_features_list.append(roi_pooled)
        # 将所有层级的结果合并，通常在通道维度上进行拼接
        roi_pooled_features = torch.cat(roi_pooled_features_list, dim=1)
        #print("ROI Pooled Features Shape:", roi_pooled_features.shape)
        #print(f"Is contiguous: {roi_pooled_features.is_contiguous()}")
        # Flatten the features for the classification and regression head
        flat_features = torch.flatten(roi_pooled_features, start_dim=1)

        # Step 2: Classify the proposals
        class_logits = self.cls_head(flat_features)
        bbox_deltas = self.bbox_regressor(flat_features)
        #print("Class logits:", class_logits)
        #print("Bounding box deltas:", bbox_deltas)
        # Step 3: Generate masks if there are any targets (for training)
        if targets is not None:
            mask_logits = self.mask_head(roi_pooled_features)

            mask_logits = F_torch.interpolate(mask_logits, size=(128, 128), mode='bilinear', align_corners=False)
            #print("label")
            return class_logits, bbox_deltas, mask_logits

        return class_logits, bbox_deltas
class MyCustomBackbone_feature_share(nn.Module):

    def __init__(self, num_classes_task1, num_classes_task2):
        super().__init__()
        # 這裡可以包含你原本的 2 個 ResNet-FPN, 或直接只定義transform conv
        #self.model_task1 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes_task1)
        #self.model_task2 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes_task2)
        self.model_task1 = create_maskrcnn_resnet101_fpn_gray(pretrained=True, num_classes=num_classes_task1)
        self.model_task2 = create_maskrcnn_resnet101_fpn_gray(pretrained=True, num_classes=num_classes_task2)
        self.out_channels = 256
        self.fuse_convs = nn.ModuleDict({
            "0": nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size=3, padding=1),
            "1": nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size=3, padding=1),
            "2": nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size=1),
            "3": nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size=1),
            "pool": nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size=1)
        })
    def forward(self, imagesA, imagesB):
        device = next(self.parameters()).device
        imagesA = imagesA.to(device)
        imagesB = imagesB.to(device)
        #print(f"In Dataset __getitem__, imgB.shape = {imagesB.shape}")
        #print(f"In Dataset __getitem__, imgA.shape = {imagesA.shape}")
        # 1) 分別跑 backbone
        features_task1 = self.model_task1.backbone(imagesA)
        features_task2 = self.model_task2.backbone(imagesB)
        # 2) 同層加起來除以 2
        # 共享特征 (FPN 各层特征相加取平均)
        shared_features = {}
        shared_features = self.compute_shared_features(features_task1, features_task2)
                
  
        return shared_features  # 這就作為 single FPN
    
    def compute_shared_features(self, features_task1, features_task2):
        """ 计算共享特征 """
        shared_features = {}
        device = next(self.model_task1.parameters()).device
        for layer_name in features_task1.keys():
            feature1 = features_task1[layer_name].to(device)
            feature2 = features_task2[layer_name].to(device)
            #print(f"Layer {layer_name} before fusion: ", feature1.shape, feature2.shape)
            # 拼接：沿 channel 維度拼接後通道數變為 2 * out_channels
            fused = torch.cat([feature1, feature2], dim=1)
            #print(f"Layer {layer_name} after concatenation: ", fused.shape)
            # 通過 1x1 卷積融合並降維，保留重要信息
            if layer_name in self.fuse_convs:
                fused = self.fuse_convs[layer_name](fused)
                #print(f"Layer {layer_name} after fuse conv: ", fused.shape)
            shared_features[layer_name] = fused
        return shared_features
    
class MultiTaskMaskRCNN(nn.Module):


    def __init__(self, num_classes_task1, 
                 num_classes_task2,
                ):
      
        super(MultiTaskMaskRCNN, self).__init__()
        
        # 加载两个 Mask R-CNN 模型
        self.model_task1 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes_task1)
        self.model_task2 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes_task2)

        self.model_task1.roi_heads = CustomROIHead(num_classes=num_classes_task1)
        self.model_task2.roi_heads = CustomROIHead(num_classes=num_classes_task2)

        # Define separate convolutional layers for each layer in both models
        # Define separate convolutional layers for each layer in the backbone (FPN output)
        # We use hard-coded layer names based on the output layers of FPN ('0', '1', etc., for P2, P3, etc.)
        self.feature_convs_task1 = nn.ModuleDict({
            layer_name: nn.Conv2d(256, 256, kernel_size=3, padding=1)
            for layer_name in ['0', '1', '2', '3' ,'pool']
        })

        self.feature_convs_task2 = nn.ModuleDict({
            layer_name: nn.Conv2d(256, 256, kernel_size=3, padding=1)
            for layer_name in ['0', '1', '2', '3' ,'pool']
        })
       

        
    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        
        device = next(self.model_task1.parameters()).device
        images = images.to(device)

        if not self.training:  # 推理模式 (评估)
            return self.inference(images,targets)

        else:  # 训练模式
            return self.training_forward(images, targets)
        
    def inference(self, images,targets=None):

            # 检查模型在哪个设备上
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_task1.to(device)
            self.model_task2.to(device)
            self.model_task1.eval()
            self.model_task2.eval()
            # 在推理模式下，不需要特征共享，只需要生成预测的输出
            image_shapes = [(128, 128) for _ in images]
            # 将特征图提取为一个列表           

                # 尝试执行前向传播ki
            with torch.no_grad():
                inference_features_task1 = self.model_task1.backbone(images)

                # 共享特征 (FPN 各层特征相加取平均)

                #print(f"Total number of shared features: {len(shared_features)}")
                    # 使用 RPN 生成候选框
                # 将特征图从 shared_features 取出并转换为 OrderedDict 格式
                feature_maps = OrderedDict(inference_features_task1)
                #for idx, (layer_name, feature_map) in enumerate(feature_maps.items()):
                    #print(f"Feature map {idx} ({layer_name}): shape {feature_map.shape}")

                # 为每个特征图生成 anchors，每层有一个对应的大小（可以根据需求调整）
                anchors = [((8, 8),), ((16, 16),), ((32, 32),), ((64, 64),), ((128, 128),)]

                # 确保特征图的数量与 anchors 的数量一致
                assert len(feature_maps) == len(anchors), "Feature map数量和anchors数量不匹配。"

                # 将图像转换为 ImageList 格式（假设所有图像的大小相同，这里设置为128x128）
                images_list = ImageList(images, [(128, 128) for _ in range(images.shape[0])])

                proposals, _ = self.model_task1.rpn(images_list, feature_maps)

                # 确保 RPN 提供了候选框
                if proposals is None or len(proposals) == 0:
                    print("Warning: No proposals were generated by RPN.")
                    return None, None  # 在推理模式下如果没有候选框，返回空值
                # 使用自定义的 roi_heads 进行前向传播
                task1_output = self.model_task1.roi_heads(inference_features_task1,proposals, image_shapes,targets=targets)

                return task1_output

    def training_forward(self, images, targets):  
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            self.model_task1.train()
            self.model_task2.train()
            
            features_task1 = self.model_task1.backbone(images)
            features_task2 = self.model_task2.backbone(images)

            # 共享特征 (FPN 各层特征相加取平均)
            shared_features = {}
            shared_features = self.compute_shared_features(features_task1, features_task2)
                

            # 确保 targets 不为空
            if targets is None:
                raise ValueError("Targets cannot be None during training.")

            # 使用自定义的 roi_heads 进行前向传播
            task1_output = self.model_task1.roi_heads(shared_features, [t['boxes'] for t in targets], [t['image_id'] for t in targets], targets=targets)
            task2_output = self.model_task2.roi_heads(shared_features, [t['boxes'] for t in targets], [t['image_id'] for t in targets], targets=targets)
            return task1_output, task2_output

class CustomMaskRCNN(MaskRCNN):


    def __init__(self, 
             backbone, 
             rpnA, roiHeadsA, 
             rpnB, roiHeadsB, 
             num_classes=2):
      
       # backbone: 你自訂的 "MyCustomBackbone" 或 其他
        # num_classes: total classes (background + X classes)
           super().__init__(backbone = backbone,
                            num_classes = num_classes)
           
           self.rpnA = rpnA
           self.rpnB = rpnB
           self.roiHeadsA = roiHeadsA
           self.roiHeadsB = roiHeadsB
           

        # 兩套 ROIHeads
           self.roiHeadsA = roiHeadsA
           self.roiHeadsB = roiHeadsB
            

        
    def forward(self, imagesA, targetsA, imagesB, targetsB):
        """
        imagesA, imagesB: Tensor shape=(B,3,128,128) (相同batch?)
        targets: list[Dict], optional
        """

        if self.training and targetsA is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training and targetsB is None:
            raise ValueError("In training mode, targets should be passed")

        device = next(self.parameters()).device

        # ========== (1) 共享backbone, fuse A,B 特徵 ==========
        #   e.g. features = self.backbone(imagesA, imagesB)
        features = self.backbone(imagesA, imagesB)  # dict[str, Tensor], shape=(B,256,H',W')

        # ========== (2) 建立 imageList for A & B ==========
        batchA = imagesA.shape[0]
        batchB = imagesB.shape[0]
        # e.g. 大小固定 => (H,W)，否則得自己算
        image_shapesA = [(imagesA.shape[-2], imagesA.shape[-1])] * batchA
        image_shapesB = [(imagesB.shape[-2], imagesB.shape[-1])] * batchB

        images_listA = ImageList(imagesA, image_shapesA)
        images_listB = ImageList(imagesB, image_shapesB)

        # ========== (3) RPN_A => proposalsA ==========
        proposalsA, rpn_lossesA = {}, {}
        if self.training and targetsA is not None:
            proposalsA, rpn_lossesA = self.rpnA(images_listA, features, targetsA)
        else:
            proposalsA, _ = self.rpnA(images_listA, features)  # eval mode

        # ========== (4) RPN_B => proposalsB ==========
        proposalsB, rpn_lossesB = {}, {}
        if self.training and targetsB is not None:
            proposalsB, rpn_lossesB = self.rpnB(images_listB, features, targetsB)
        else:
            proposalsB, _ = self.rpnB(images_listB, features)

        # ========== (5) ROIHeadsA => detectionsA ==========
        detectionsA, roi_lossesA = [], {}
        if self.training and targetsA is not None:
            detectionsA, roi_lossesA = self.roiHeadsA(features, proposalsA, image_shapesA, targetsA)
        else:
            detectionsA, _ = self.roiHeadsA(features, proposalsA, image_shapesA)

        # ========== (6) ROIHeadsB => detectionsB ==========
        detectionsB, roi_lossesB = [], {}
        if self.training and targetsB is not None:
            detectionsB, roi_lossesB = self.roiHeadsB(features, proposalsB, image_shapesB, targetsB)
        else:
            detectionsB, _ = self.roiHeadsB(features, proposalsB, image_shapesB)
        lossesA = {}
        lossesA.update(rpn_lossesA)
        lossesA.update(roi_lossesA)

    # 把 B 的 loss 整理成 lossesB
        lossesB = {}
        lossesB.update(rpn_lossesB)
        lossesB.update(roi_lossesB)
        if self.training:
             return lossesA, lossesB
        else:
            # 推論 => 回傳 (detectionsA, detectionsB)
            return detectionsA, detectionsB
        

class CustomMaskRCNN_single(MaskRCNN):
    def __init__(self, backbone, rpn, roiHeads, num_classes=2):
        """
        backbone: 你自訂的 backbone，例如 MyCustomBackboneSingle
        rpn: 區域提議網絡 (RPN)
        roiHeads: ROI Head
        num_classes: 分類數量
        """
        super().__init__(backbone=backbone, num_classes=num_classes)
        
        self.rpn = rpn  # 只保留一個 RPN
        self.roiHeads = roiHeads  # 只保留一個 ROI Head

    def forward(self, images, targets=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        device = next(self.parameters()).device
        '''if isinstance(images, list):
            image_shapes = [(img.shape[-2], img.shape[-1]) for img in images]  # 記錄每張圖像大小
            images = ImageList(images, image_shapes)  # ✅ 轉成 ImageList
        else:
        # **處理 Tensor，建立 ImageList**
            print("**處理 Tensor，建立 ImageList**")
            batch_size = images.shape[0]
            image_shapes = [(images.shape[-2], images.shape[-1])] * batch_size
            images = ImageList(images, image_shapes)
            images = images.to(device)

        if isinstance(images, ImageList):
            images = images.tensors 
            images = torch.stack(images)  # 產生的形狀將會是 [B, C, H, W]''' 
        # 1️⃣ 提取 backbone 特徵

        if isinstance(images, list):
    # 如果是普通的列表，先统一为 ImageList
            image_shapes = [(img.shape[-2], img.shape[-1]) for img in images]
            images = ImageList(images, image_shapes)
        if isinstance(images, ImageList):
# 此时，无论原来是 list 还是已经是 ImageList，`images` 一定是 ImageList
            images = images.tensors  # 直接使用 .tensors，得到了 [B, C, H, W] 的 Tensor
            images = torch.stack(images)  # 產生的形狀將會是 [B, C, H, W]''' 
        features = self.backbone(images)  # dict[str, Tensor], shape=(B, 256, H', W')

        # 2️⃣ 建立 ImageList
        batch_size = images.shape[0]
        image_shapes = [(images.shape[-2], images.shape[-1])] * batch_size
        images_list = ImageList(images, image_shapes)

        # 3️⃣ RPN (區域提議網絡)
        proposals, rpn_losses = {}, {}
        if self.training and targets is not None:
            proposals, rpn_losses = self.rpn(images_list, features, targets)
        else:
            proposals, _ = self.rpn(images_list, features)

        # 4️⃣ ROIHeads (目標檢測)
        detections, roi_losses = [], {}
        if self.training and targets is not None:
            detections, roi_losses = self.roiHeads(features, proposals, image_shapes, targets)
        else:
            detections, _ = self.roiHeads(features, proposals, image_shapes)

        # 5️⃣ 計算 Loss
        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)

        if self.training:
            return losses  # 訓練模式回傳 Loss
        else:
            return detections  # 推論模式回傳 Detections
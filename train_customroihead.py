r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
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
#from coco_utils import get_coco, get_coco_kp
from dataset import MYODataset,HeartDataset,RV_HeartDataset
from args import get_args_parser
from engine_custom import evaluate, train_one_epoch,calculate_iou
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from PIL import Image
from torch import nn
from torch.utils.data import random_split
from torchvision import transforms
import torch.nn.functional as F_torch
import torchvision.ops as ops
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
os.environ['CUDA_LAUNCH_BLOCKING']= '1'

def copypaste_collate_fn(batch):

    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    images, targets = utils.collate_fn(batch)
    # 调试信息
    print(f"Images type after collate_fn: {type(images)}, Targets type after collate_fn: {type(targets)}")

    # 修正 list 类型
    if isinstance(targets, list):
        print("Warning: Targets is of type list. Attempting to fix this.")
        targets_fixed = []
        for t in targets:
            if isinstance(t, dict):
                targets_fixed.append(t)
            elif isinstance(t, torch.Tensor):
                targets_fixed.append(t)
        targets = targets_fixed

    # 进行数据增强
    enhanced_images, enhanced_targets = copypaste(images, targets)
    return enhanced_images, enhanced_targets


def custom_collate_fn(batch1, batch2):
    images, targets = [], []

    # 遍歷兩個批次中的數據
    for batch in [batch1, batch2]:
        for item in batch:
            image, target = item
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image, dtype=torch.float32)
            images.append(image)

            # 如果 target 是字典，則檢查其中的值是否是張量，確保每個值都是張量
            if isinstance(target, dict):
                for key, value in target.items():
                    if not isinstance(value, torch.Tensor):
                        target[key] = torch.as_tensor(value)
            targets.append(target)
    
    # 合併圖像張量
    images = torch.stack(images, dim=0)

    return images, targets

def get_dataset(name, image_set, transform, data_path):
    if name == "heart":
        ds = HeartDataset(root=data_path, transforms=transform)
        num_classes = 2  # 假設只有一個前景類別（心臟）
    elif name == "Myo":
        ds = MYODataset(root=data_path, transforms=transform)
        num_classes = 2  # 假設只有一個前景類別（腦部）
    else:
        ds = RV_HeartDataset(root=data_path, transforms=transform)
        num_classes = 2  # 假設只有一個前景類別（腦部）
    return ds, num_classes

def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()
def check_model_on_device(model):
        for name, param in model.named_parameters():
            print(f"Parameter {name} is on device: {param.device}")

# Define the multitask model that uses feature sharing



class CustomROIHead(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(CustomROIHead, self).__init__()
        # 分类器和回归器网络（通常是一些全连接层）
        # 共享全连接层
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # 分类和回归头
        self.cls_head = nn.Linear(1024, num_classes)
        self.bbox_regressor = nn.Linear(1024, 4)

        # 掩码头部分，可以根据需要调整层数和结构
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)  # 单通道输出
)
        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # 降维
            nn.ReLU()
    
)

    def forward(self, shared_features, proposals, targets=None):
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
            feature, 
            proposals, 
            output_size=(7, 7), 
            spatial_scale=scale, 
            sampling_ratio=2
            )
            # 將 ROI Align 的輸出加入列表
            roi_pooled_features_list.append(roi_pooled)

        # 将所有层级的结果合并，通常在通道维度上进行拼接
        roi_pooled_features = torch.cat(roi_pooled_features_list, dim=1)
        #print(roi_pooled_features.shape)
        #print("ROI Pooled Features Shape:", roi_pooled_features.shape)
        #print(f"Is contiguous: {roi_pooled_features.is_contiguous()}")
        # Flatten the features for the classification and regression head
        flat_features = torch.flatten(roi_pooled_features, start_dim=1)
        # 前两层全连接层共享特征
        shared_fc = self.shared_fc(flat_features)
        # Step 2: Classify the proposals
        class_logits = self.cls_head(shared_fc)
        bbox_deltas = self.bbox_regressor(shared_fc)
        # 假設 proposals 是一個列表
        # Step 3: Generate masks if there are any targets (for training)
        if targets is not None:
            mask_logits = self.mask_head(roi_pooled_features)

            mask_logits = F_torch.interpolate(mask_logits, size=(128, 128), mode='bilinear', align_corners=False)
            #print("label")
            return class_logits, bbox_deltas, mask_logits

        return class_logits, bbox_deltas
class MultiTaskMaskRCNN(nn.Module):


    def __init__(self, num_classes_task1):
        super(MultiTaskMaskRCNN, self).__init__()
        
        # 加载两个 Mask R-CNN 模型
        self.model_task1 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes_task1)
        

        #self.model_task1.roi_heads = CustomROIHead(1280,num_classes=num_classes_task1)
    
        self.feature_convs_task1 = nn.ModuleDict({
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
            self.model_task1.eval()
            # 在推理模式下，不需要特征共享，只需要生成预测的输出
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
                anchors = [((32, 32),), ((64, 64),), ((128, 128),), ((256, 256),), ((512, 512),)]

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
                task1_output = self.model_task1.roi_heads(inference_features_task1,proposals, targets=targets)

                return task1_output,proposals

    def training_forward(self, images, targets):  
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            self.model_task1.train()
            
            features_task1 = self.model_task1.backbone(images)

            # Step 2: 调用 RPN 生成候选框
            # 将 images 转为 ImageList 格式
            image_shapes = [(img.shape[-2], img.shape[-1]) for img in images]
            images_list = ImageList(images, image_shapes)

    # 调用 RPN 模块，生成候选框
            proposals, rpn_losses = self.model_task1.rpn(images_list, features_task1, targets)
            if proposals is None or len(proposals) == 0:
                print("Warning: No proposals were generated by RPN.")
            # Step 2: 根据 IoU 提取正负样本

            # 因為你目前只處理單張圖，所以假設 proposals 是 [Tensor] (list長度=1)
            # 先把它取出來 (shape: [num_props, 4])
            proposals = proposals[0]

            matched_proposals, matched_gt_boxes, matched_ious, ious_matrix,matched_proposal_gt_indices = calculate_iou(proposals, targets[0]['boxes'])
           
            # 假設你想要選擇 IoU 在 0.1 到 0.5 區間內的所有 proposals

            # matched_ious 是一個與 matched_proposals 對應的一維張量，存放每個 proposal 與對應 GT box 的 IoU
            # 透過布林遮罩 (boolean mask) 過濾出符合條件的 proposals
            mask_any = (matched_ious >= 0.0)
            filtered_proposals = matched_proposals[mask_any]
            filtered_gt_boxes  = matched_gt_boxes[mask_any]
            filtered_ious      = matched_ious[mask_any]
            filtered_gt_indices= matched_proposal_gt_indices[mask_any]

            fg_iou_thresh = 0.3
            fg_mask = filtered_ious >= fg_iou_thresh
            bg_mask = filtered_ious < fg_iou_thresh

            #print(fg_mask.sum())
            #print(bg_mask.sum())
            # -------------------------------------------------------
    # 5) 在「做法 A」中，我們會在這裡抽樣 (sample) 前景/背景 proposals
    #    例如，設定 batch_size_per_image = 256, positive_fraction=0.25 => FG <= 64
    # -------------------------------------------------------
            batch_size_per_image = 512
            positive_fraction    = 0.25

    # 先找到前景/背景索引
            fg_inds = torch.where(fg_mask)[0]  # 前景 indices
            bg_inds = torch.where(bg_mask)[0]  # 背景 indices

            num_fg = int(batch_size_per_image * positive_fraction)  # e.g. 256 * 0.25 = 64
            num_bg = batch_size_per_image - num_fg                 # 256 - 64 = 192

    # 如果前景太多，就隨機抽 num_fg 個
            if len(fg_inds) > num_fg:
                fg_inds = fg_inds[torch.randperm(len(fg_inds), device=device)[:num_fg]]

    # 如果背景太多，就隨機抽 num_bg 個
            if len(bg_inds) > num_bg:
                bg_inds = bg_inds[torch.randperm(len(bg_inds), device=device)[:num_bg]]

    # 最終保留的索引
            keep_inds = torch.cat([fg_inds, bg_inds], dim=0)
            keep_inds = keep_inds.to(device)

            sampled_proposals   = filtered_proposals[keep_inds]   # shape [M,4], M <= 256
            sampled_gt_boxes    = filtered_gt_boxes[keep_inds]
            sampled_ious        = filtered_ious[keep_inds]
            sampled_gt_indices  = filtered_gt_indices[keep_inds]

    # 再更新 mask
            sampled_fg_mask = (sampled_ious >= fg_iou_thresh)

                   
             # -------------------------------------------------------
    # 6) 建立 proposal labels
    #    前景 => GT labels；背景 => 0
    # -------------------------------------------------------
            gt_labels = targets[0]['labels']   # e.g. [1,2,...]
            proposal_labels = torch.zeros_like(sampled_gt_indices, dtype=torch.int64)  # 先全部=0(背景)
            proposal_labels[sampled_fg_mask] = gt_labels[sampled_gt_indices[sampled_fg_mask]]

            #print("proposal_labels:", proposal_labels.unique(return_counts=True))
    # 這就是 (M,) shape

            # 7) 計算 bbox regression target (proposal_deltas)
    #    只有前景需要 bbox regression
    # -------------------------------------------------------
            proposal_deltas = torch.zeros_like(sampled_proposals)  # shape [M,4]
            fg_gt_boxes  = sampled_gt_boxes[sampled_fg_mask]
            fg_proposals = sampled_proposals[sampled_fg_mask]

            num_classes = 2  # 0=背景, 1=心臟
            M = sampled_proposals.shape[0]  # proposals數量

            # 先把單通道 proposal_deltas 改成 multi-channel
            proposal_deltas_multiclass = torch.zeros(
            (M, num_classes * 4), 
            dtype=torch.float32, 
            device=sampled_proposals.device
            )  # shape=(M,8)

# compute_deltas 仍只對前景 proposals 做 (dx,dy,dw,dh)
            fg_inds = torch.where(sampled_fg_mask)[0]  # 前景索引
            if len(fg_inds) > 0:
    # 原本 compute_deltas -> shape=(fg_inds數量,4)
                deltas_fg = compute_deltas(fg_proposals, fg_gt_boxes)  # shape=(len(fg_inds),4)

    # 將對應的回歸值填入 multi-channel
                for idx_in_fg, proposal_idx in enumerate(fg_inds):
                    cls = proposal_labels[proposal_idx].item()  # 0 or 1
                    if cls > 0:
            # 若 cls=1 => 填入 proposal_deltas_multiclass[proposal_idx, 4:8]
                        start = cls * 4
                        end   = start + 4
                        proposal_deltas_multiclass[proposal_idx, start:end] = deltas_fg[idx_in_fg]

            proposal_deltas[sampled_fg_mask] = compute_deltas(fg_proposals, fg_gt_boxes)



           # -------------------------------------------------------
    # 8) 對「前景 proposals」再做 mask 的 ROI Align
    #    (若你要自訂外部對前景 mask 做對齊)
    # -------------------------------------------------------
            sampled_fg_indices = torch.where(sampled_fg_mask)[0]
            fg_gt_masks_list = []



            for i in range(len(sampled_fg_indices)):
                idx    = sampled_fg_indices[i].item()
                gt_idx = sampled_gt_indices[idx].item()

                single_gt_mask = targets[0]['masks'][gt_idx]  # shape: [H, W]
                single_gt_mask = single_gt_mask.unsqueeze(0).unsqueeze(0).float().to(device)

        # sampled_proposals[idx]: shape [4]
        # 需組合成 [image_id, x1, y1, x2, y2]
                single_box = torch.cat([
                    torch.zeros((1,1), device=device),
                    sampled_proposals[idx].view(1,4)
                ], dim=1)

                aligned_gt_mask = ops.roi_align(
                    single_gt_mask,
                    single_box,
                    output_size=(128, 128),
                    spatial_scale=1.0,
                    sampling_ratio=2,
                    aligned=True
                )
                aligned_gt_mask = aligned_gt_mask.squeeze(0).squeeze(0)  # [128,128]
                fg_gt_masks_list.append(aligned_gt_mask)

            fg_gt_masks = None
            if len(fg_gt_masks_list) > 0:
                fg_gt_masks = torch.stack(fg_gt_masks_list, dim=0)  # [num_fg,128,128]
  
     # -------------------------------------------------------
    # 9) 把「抽樣後」的 proposals => (M,4) 丟給 ROIHeads
    #    這裡就包含前景&背景 (抽樣完)
    # -------------------------------------------------------
            all_proposals = [sampled_proposals]  # 需要裝成 list

            if targets is None:
                raise ValueError("Targets cannot be None during training.")
            #ious = calculate_iou(proposals, targets[0]['boxes'])
            #print(f"Max IoU with targets: {ious.max()}")

            # 使用自定义的 roi_heads 进行前向传播
            task1_output = self.model_task1.roi_heads(features_task1,
            all_proposals,
            targets=targets
            )


            ''' # 用官方 roi_heads => box_roi_pool, box_head, box_predictor
    # (1) multi-scale ROIAlign by official
            box_features = self.model_task1.roi_heads.box_roi_pool(features_task1, [all_proposals], image_shapes)

    # (2) 2層FC => box_head
            box_features = self.model_task1.roi_heads.box_head(box_features)  # shape(N,1024)

    # (3) predictor => (class_logits, box_regression)
            class_logits, box_regression = self.model_task1.roi_heads.box_predictor(box_features)

    # (4) compute loss: 
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, 
                                                          box_regression, 
                                                          proposal_labels,
                                                          proposal_deltas_multiclass)'''
            
            

    # or you can do manual CrossEntropyLoss + SmoothL1Loss if you prefer
            return task1_output,rpn_losses,proposal_labels, proposal_deltas,fg_gt_masks

##################################################################################################proposal_labels突然沒值 然後你gt box 就只有一個所以索引一定是0 不知道上面的求索引要幹嘛
from collections import OrderedDict
# 自定義 Backbone with FPN 類
def compute_deltas(proposals, gt_boxes):
        # proposals, gt_boxes shape: [M, 4]
        # 假設座標為 (x1, y1, x2, y2)
        px1, py1, px2, py2 = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]
        gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

        pw = px2 - px1
        ph = py2 - py1
        px = px1 + 0.5 * pw
        py = py1 + 0.5 * ph

        gw = gx2 - gx1
        gh = gy2 - gy1
        gx = gx1 + 0.5 * gw
        gy = gy1 + 0.5 * gh

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

        return torch.stack((dx, dy, dw, dh), dim=1)

def load_datasets(args, use_second_dataset=False):
    # 加载第一个数据集
    dataset1, num_classes1 = get_dataset(args.dataset1, "train", get_transform(True, args), args.data_path1)
    dataset_test1, _ = get_dataset(args.dataset1, "val", get_transform(False, args), args.data_path1)

    # 数据集划分
    train_size1 = int(0.8 * len(dataset1))
    val_size1 = len(dataset1) - train_size1
    train_dataset1, val_dataset1 = random_split(dataset1, [train_size1, val_size1])

    return train_dataset1, val_dataset1,  num_classes1,

def main(args):
    args.feature_share = False
    # Initialize model output directory
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Load datasets
    train_dataset1, val_dataset1, num_classes1 = load_datasets(args)

    # Create data loaders for task 1
    # Task 1 data loader
    if args.distributed:
        train_sampler1 = torch.utils.data.distributed.DistributedSampler(train_dataset1)
        test_sampler1 = torch.utils.data.distributed.DistributedSampler(val_dataset1, shuffle=False)
    else:
        train_sampler1 = torch.utils.data.RandomSampler(train_dataset1)
        test_sampler1 = torch.utils.data.SequentialSampler(val_dataset1)

    if args.aspect_ratio_group_factor >= 0:
        group_ids1 = create_aspect_ratio_groups(train_dataset1, k=args.aspect_ratio_group_factor)
        train_batch_sampler1 = GroupedBatchSampler(train_sampler1, group_ids1, args.batch_size)
    else:
        train_batch_sampler1 = torch.utils.data.BatchSampler(train_sampler1, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn  # 使用通用的 collate_fn 处理单个任务的数据集
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")
        train_collate_fn = copypaste_collate_fn

# Data loader for task 1
    data_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_sampler=train_batch_sampler1, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test1 = torch.utils.data.DataLoader(
        val_dataset1, batch_size=1, sampler=test_sampler1, num_workers=args.workers, collate_fn=utils.collate_fn
    )


    combined_data_loader = [(batch,) for batch in data_loader1]
    data_loader_length = len(data_loader1)

    # Creating MultiTask Model
    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    # 使用自定义的单任务模型
    multi_task_model = MultiTaskMaskRCNN(num_classes_task1=num_classes1).to(device)

    if args.distributed and args.sync_bn:
        multi_task_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(multi_task_model)

    model_without_ddp = multi_task_model

    if args.distributed:
        multi_task_model = torch.nn.parallel.DistributedDataParallel(multi_task_model, device_ids=[args.gpu])
        model_without_ddp = multi_task_model.module

# 准备优化器
    if args.norm_weight_decay is None:
        parameters = [p for p in multi_task_model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(multi_task_model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Scheduler
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

     # Fine-tuning loop
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler1.set_epoch(epoch)
            
        metric_logger = train_one_epoch(multi_task_model, optimizer, combined_data_loader, device, epoch, args.print_freq,data_loader_length, scaler)

        train_loss = metric_logger.meters['loss'].global_avg  # Record average loss

        # Adjust learning rate
        lr_scheduler.step()
        
        # Save model checkpoint
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\Pytorch-Mask-RCNN-master\\checkpoint\\checkpoint.pth"))
 
        # Evaluate after every epoch
        mean_average_precision,iou_mean = evaluate(multi_task_model, data_loader_test1, device)
        # 假设 mean_average_precision 是 (mAP_task1, mAP_task2)
        if isinstance(mean_average_precision, tuple):
            mAP_task1 = mean_average_precision
            iou_mean_task1 = iou_mean
            print(f"Epoch {epoch} completed. train_loss: {train_loss:.4f}, test_Task 1 mAP: {mAP_task1:.4f},test_ Mean IoU Task 1: {iou_mean_task1:.4f} ")
            
        elif isinstance(mean_average_precision, dict):
    # 如果返回的是字典类型
            mAP_task1 = mean_average_precision.get('task1', 0)
            iou_mean_task1 = iou_mean.get('task1', 0)

            print(f"Epoch {epoch} completed.train_loss: {train_loss:.4f}, test_Task 1 mAP: {mAP_task1:.4f}, test_Mean IoU Task 1: {iou_mean_task1:.4f}")     

        else:
    # 如果是其他类型 (例如 float 或直接是单个任务的返回值)
            mAP_task1 = mean_average_precision
            iou_mean_task1 = iou_mean
            print(f"Epoch {epoch} completed.train_loss: {train_loss:.4f},test_Task 1 mAP: {mAP_task1:.4f}, test_Mean IoU Task 1: {iou_mean_task1:.4f}")
            #print(f"Validation Loss: {val_loss:.4f}")
              
            # Record both Task 1 and Task 2 results to CSV file
        with open("C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\Pytorch-Mask-RCNN-master\\loss\\testing_log.csv", "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{mAP_task1},{iou_mean_task1}\n")

        


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

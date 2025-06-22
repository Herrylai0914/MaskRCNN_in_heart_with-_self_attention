import math
import sys
import time
import numpy as np
import torch
import torchvision.models.detection.mask_rcnn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
sys.path.append(r'C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\Pytorch-Mask-RCNN-master\\detection')
import utils  # 直接使用 utils 模組中的功能
from collections import defaultdict
#import detection.utils as utils
#from coco_eval import CocoEvaluator
#from coco_utils import get_coco_api_from_dataset
def compute_target_deltas(anchors, gt_boxes):
                
     # 解包列表中的 Tensor
    if isinstance(anchors, list) and len(anchors) == 1:
        anchors = anchors[0]  # 解包，取出 Tensor

    # 確保 anchors 是 Tensor
    if not isinstance(anchors, torch.Tensor):
        raise TypeError(f"Expected anchors to be a Tensor, but got {type(anchors)}")            
    anchors_x_center = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchors_y_center = (anchors[:, 1] + anchors[:, 3]) / 2.0
    anchors_width = anchors[:, 2] - anchors[:, 0]
    anchors_height = anchors[:, 3] - anchors[:, 1]

    gt_x_center = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    gt_y_center = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    gt_width = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_height = gt_boxes[:, 3] - gt_boxes[:, 1]

    target_deltas = torch.zeros_like(anchors)
    target_deltas[:, 0] = (gt_x_center - anchors_x_center) / anchors_width  # dx
    target_deltas[:, 1] = (gt_y_center - anchors_y_center) / anchors_height  # dy
    target_deltas[:, 2] = torch.log(gt_width / anchors_width)  # dw
    target_deltas[:, 3] = torch.log(gt_height / anchors_height)  # dh
    return target_deltas

def train_one_epoch(model, optimizer, combined_data_loader, device, epoch, print_freq,data_loader_length, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    lr_scheduler = None

    # 初始化学习率调度器
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, data_loader_length - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_bbox = torch.nn.SmoothL1Loss()
    criterion_mask = torch.nn.BCEWithLogitsLoss()    

    # 遍历数据集
    for idx, (batch,) in enumerate(metric_logger.log_every(combined_data_loader, print_freq, header)):

        if isinstance(batch, tuple) and len(batch) == 2:
            images, targets = batch
        else:
            raise ValueError(f"Unexpected batch format: {batch}")
        
        
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播并计算损失
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            task1_output, rpn_losses,proposal_labels, proposal_deltas,fg_gt_masks = model(images, targets)  
            class_logits, bbox_deltas, mask_logits = task1_output
             # 解包 RPN 损失
            loss_objectness = rpn_losses["loss_objectness"]
            loss_rpn_box_reg = rpn_losses["loss_rpn_box_reg"]

            # 确保 targets 格式正确
            for i in range(len(targets)):
                if targets[i]['masks'].dim() == 3:
                    targets[i]['masks'] = targets[i]['masks'].unsqueeze(1)  # 从 (batch_size, H, W) -> (batch_size, 1, H, W)
                if targets[i]['masks'].dtype == torch.bool:
                    targets[i]['masks'] = targets[i]['masks'].float()

            #target_deltas = compute_target_deltas(proposal_labels, filtered_gt_boxes)
            fg_mask = proposal_labels > 0
            fg_mask_logits = mask_logits[fg_mask]
            fg_pred_masks = fg_mask_logits.squeeze(1)  # -> shape: [N, H, W]
            

            # 计算分类损失
            clas_loss = criterion_cls(class_logits,proposal_labels)
            if torch.isnan(clas_loss):
                print("Class Logits:", class_logits)
                raise ValueError("Classification Loss is NaN!")

            # 计算回归损失
            box_loss = criterion_bbox(bbox_deltas, proposal_deltas)
            if torch.isnan(box_loss):
                print("bbox Logits:", bbox_deltas)
                raise ValueError("Bounding Box Loss is NaN!")

            # 计算掩码损失
            if fg_pred_masks is None or fg_gt_masks is None or fg_gt_masks.size(0) == 0:
    # 表示沒有前景可以做 mask
                mask_loss = torch.tensor(0.0, device=device)  
            else:
                mask_loss = criterion_mask(fg_pred_masks, fg_gt_masks)

            if torch.isnan(mask_loss):
                print("mask Logits:", mask_logits)
                raise ValueError("Mask Loss is NaN!")

            # 总损失
            total_loss = clas_loss + box_loss + mask_loss + loss_objectness + loss_rpn_box_reg
        # 聚合总损失
        loss_dict = {
    'clas_loss': clas_loss,
    'box_loss': box_loss,
    'mask_loss': mask_loss,
    'loss_objectness': loss_objectness,
    'loss_rpn_box_reg':loss_rpn_box_reg,
        }

# 减少字典（适用于多设备的分布式训练）
        loss_dict_reduced = utils.reduce_dict(loss_dict)

# 计算总损失
        total_loss = sum(loss for loss in loss_dict_reduced.values())
        # 确保损失是标量
        total_loss_value = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # 更新学习率
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 更新日志记录器
        for loss_name, loss_value in loss_dict_reduced.items():
            loss_value_item = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
            metric_logger.update(**{loss_name: loss_value_item})

        metric_logger.update(loss=total_loss_value)

# 记录学习率
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

@torch.inference_mode()
def evaluate(model, data_loader, device, iou_threshold=0.3):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(cpu_device)
    # 确认模型已经在 CPU 上
    model_device = next(model.parameters()).device
    print(f"Model is on device after moving to CPU: {model_device}")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    # 保存预测和真实值以计算 mAP
    all_detections_task1 = defaultdict(list)
    all_annotations_task1 = defaultdict(list)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [img.to(cpu_device) for img in images]
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
        #for idx, img in enumerate(images):
            #print(f"Image {idx} is on device: {img.device}")


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if isinstance(images, list):
                images = torch.stack(images, dim=0)

        # 如果图像是单张图片的情况，确保添加批次维度
        if len(images.shape) == 3:
            images = images.unsqueeze(0)  # 变成 [1, 3, H, W]

    # 检查图片和模型是否在同一设备上
        device = next(model.parameters()).device
        images = images.to(device)  
              
        model_time = time.time()
        task1_output = None
        
        task1_output = model(images,targets=targets)   
        model_time = time.time() - model_time
         # 將模型輸出移動到 CPU 以便進行後續處理
        #task1_output = [{k: v.to(cpu_device) for k, v in t.items()} for t in task1_output]
        #if task2_output:
            #task2_output = [{k: v.to(cpu_device) for k, v in t.items()} for t in task2_output]
        proposals = task1_output[-1]  # 提取最後一個元素，即 proposals 列表

# 確保 proposals 是列表，並將其轉為 Tensor
        if isinstance(proposals, list):
            proposals = torch.stack(proposals).to(device)  # 合併為單一 Tensor並移到設備 
        task1_output = task1_output[:-1] + (proposals,)   
        task1_output= tuple(
                tuple(v.to(cpu_device) for v in t) if isinstance(t, tuple) else t.to(cpu_device)
                for t in task1_output
            )        
        # 保存任务1的预测和真实值
        for target, t1_output in zip(targets, [task1_output]):
                image_id = target["image_id"].item()
                all_annotations_task1[image_id].append(target)

    # 解包 tuple，並將每個元素單獨保存
        (triplet, proposals) = t1_output
        (class_logits, bbox_deltas, labels) = triplet
       #class_logits, bbox_deltas,labels,proposals = t1_output  # 解包

                # 创建一个字典来存储解包后的内容
        detection_dict = {
                    "class_logits": class_logits,
                    "bbox_deltas": bbox_deltas,
                    "labels": labels,
                    "proposals":proposals
                    
                }
                  # 确保 image_id 存在于 all_detections_task1 中
        
        if image_id not in all_detections_task1:
                    all_detections_task1[image_id] = []
                
                
                 # 将字典加入到对应的 image_id 的列表中
        all_detections_task1[image_id].append(detection_dict)
        
            
         
    mean_average_precision_task1,iou_mean_task1 = calculate_map(images,all_detections_task1, all_annotations_task1, iou_threshold)
    metric_logger.update(model_time=model_time)

    return mean_average_precision_task1, iou_mean_task1

def calculate_map(images,detections, annotations, iou_threshold=0.3,nms_threshold=0.3, 
                           score_threshold=0.2):

    """计算 mean Average Precision (mAP)."""
    device = torch.device("cuda")
    average_precisions = []
    iou_results = []
    iou_mean = []  # 初始化為空列表
    for image_id in detections.keys():
        if image_id not in detections or len(detections[image_id]) == 0:
            #print(f"No detections found for image_id {image_id}, skipping.")
            continue  
    for image_id in detections.keys():
        pred_deltas = detections[image_id][0]["bbox_deltas"].detach()
        class_logits = detections[image_id][0]["class_logits"].detach()
        gt_boxes = annotations[image_id][0]["boxes"].cpu().detach().numpy()
        proposals = detections[image_id][0]["proposals"].detach()
        gt_classes = annotations[image_id][0]["labels"].cpu().detach().numpy()  # 真值框类别
    
# 1) logits → probability
        pred_probs = torch.softmax(class_logits, dim=1)  # shape: [num_proposals, num_classes]
        pred_probs = pred_probs.cpu()

        # 2) 取「最高分」分數和類別
        pred_scores, pred_classes = torch.max(pred_probs, dim=1)   # [num_proposals], [num_proposals]
        #print(pred_classes )
        # 3) 解碼 proposals + bbox_deltas → final boxes
        decoded_boxes = decode_bbox_deltas(proposals, pred_deltas) # Tensor [num_proposals, 4]
        decoded_boxes = decoded_boxes.cpu()

        # 4) 先過濾score < score_threshold的
        keep_mask = pred_scores >= score_threshold
        decoded_boxes = decoded_boxes[keep_mask]
        pred_scores   = pred_scores[keep_mask]
        pred_classes  = pred_classes[keep_mask]

        if len(decoded_boxes) == 0:
            # 沒有任何有效預測，這張圖直接略過或計為AP=0
            continue

        # 5) NMS (PyTorch原生API: torchvision.ops.nms)
        #   注意：nms的boxes格式是 [x1, y1, x2, y2], shape=(N,4)
        nms_indices = torchvision.ops.nms(decoded_boxes, pred_scores, nms_threshold)
        decoded_boxes = decoded_boxes[nms_indices]
        pred_scores   = pred_scores[nms_indices]
        pred_classes  = pred_classes[nms_indices]

        # numpy化
        pred_boxes   = decoded_boxes.numpy()
        pred_scores  = pred_scores.numpy()
        pred_classes = pred_classes.numpy()
        image_tensor=images[0]  # shape=(3,H,W)
        '''visualize_boxes(
        image_tensor=images,
        pred_boxes= pred_boxes,
        gt_boxes=gt_boxes,
        pred_color=(255,0,0),
        gt_color=(0,255,0)
        )'''
        # 6) 依照分數降序排序(非必需，NMS後也常會維持大致順序，但保險起見可再排一次)
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes   = pred_boxes[sorted_indices]
        pred_scores  = pred_scores[sorted_indices]
        pred_classes = pred_classes[sorted_indices]




        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        detected = []
        
        for i, pred_box in enumerate(pred_boxes):
            # 使用新的 IoU 计算逻辑
            x_min = np.maximum(pred_box[0], gt_boxes[:, 0])
            y_min = np.maximum(pred_box[1], gt_boxes[:, 1])
            x_max = np.minimum(pred_box[2], gt_boxes[:, 2])
            y_max = np.minimum(pred_box[3], gt_boxes[:, 3])

            # 计算交集面积
            intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)

            # 计算预测框和真实框的面积
            box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_boxes_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

            # 计算并集面积
            union = box_area + gt_boxes_area - intersection

            # 避免除以零并计算 IoU
            ious = intersection / (union + 1e-6)
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            #print("Intersection:", intersection)
            #print("Union:", union)
            #print("IoUs:", ious)
            iou_results.append(max_iou)

             # 类别和 IoU 匹配条件
            if max_iou >= iou_threshold and pred_classes[i] == gt_classes[max_iou_idx] and max_iou_idx not in detected:
                tp[i] = 1  # True Positive
                detected.append(max_iou_idx)
            else:
                fp[i] = 1  # False Positive

        # 计算 Precision 和 Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (len(gt_boxes) + 1e-6)
        #print("precisions is",precisions)
        #print("recalls is",recalls)
        # 计算 AP
        ap = compute_ap(precisions, recalls)
        #print("ap is",ap)
        average_precisions.append(ap)
    average_precisions = np.array(average_precisions)


    iou_mean= np.array(iou_mean)
    mean_average_precision = np.mean(average_precisions[average_precisions > 0]) if np.any(average_precisions > 0) else 0
    iou_results = np.array(iou_results)  # 確保轉換為 numpy 數組
    iou_mean = np.mean(iou_results[iou_results > 0]) if np.any(iou_results > 0) else 0

    return mean_average_precision, iou_mean


def compute_ap(precisions, recalls):
    """通过插值计算 AP。"""
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def calculate_map_single_task(detections, annotations, iou_threshold=0.5):
    """计算单任务的 mean Average Precision (mAP) 和平均 IoU。"""
    average_precisions = []
    iou_results = [] 

    for image_id in detections.keys():
        if image_id not in detections or len(detections[image_id]) == 0:
           #print(f"No detections found for image_id {image_id}, skipping.")
            continue   
        # 单任务情况下提取数据
        pred_boxes = detections[image_id][0]["boxes"].cpu().detach().numpy()  # 使用预测的 "boxes"
        pred_scores = detections[image_id][0]["labels"].cpu().detach().numpy()  # 使用预测的 "labels"
        gt_boxes = annotations[image_id][0]["boxes"].cpu().detach().numpy()  # 使用真实的 "boxes"

        # 按照预测分数降序排列预测框
        sorted_indices = pred_scores.argsort()[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        detected = []

        for i, pred_box in enumerate(pred_boxes):
            # 使用 IoU 计算逻辑
            x_min = np.maximum(pred_box[0], gt_boxes[:, 0])
            y_min = np.maximum(pred_box[1], gt_boxes[:, 1])
            x_max = np.minimum(pred_box[2], gt_boxes[:, 2])
            y_max = np.minimum(pred_box[3], gt_boxes[:, 3])

            # 计算交集面积
            intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)

            # 计算预测框和真实框的面积
            box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_boxes_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

            # 计算并集面积
            union = box_area + gt_boxes_area - intersection

            # 避免除以零并计算 IoU
            ious = intersection / (union + 1e-6)
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            iou_results.append(max_iou)

            # 判断是否为真阳性或假阳性
            if max_iou >= iou_threshold and max_iou_idx not in detected:
                tp[i] = 1  # True positive
                detected.append(max_iou_idx)
            else:
                fp[i] = 1  # False positive

        # 计算 Precision 和 Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (len(gt_boxes) + 1e-6)

        # 计算 AP
        ap = compute_ap(precisions, recalls)
        average_precisions.append(ap)

    mean_average_precision = np.mean(average_precisions) if average_precisions else 0
    iou_mean = np.mean(iou_results) if iou_results else 0

    return mean_average_precision, iou_mean



def decode_bbox_deltas(anchors, bbox_deltas):
    """
    解碼 bbox_deltas 為絕對框
    """
     # 解包列表中的 Tensor
    if isinstance(anchors, list) and len(anchors) == 1:
        anchors = anchors[0]  # 解包，取出 Tensor

    # 確保 anchors 是 Tensor
    if not isinstance(anchors, torch.Tensor):
        raise TypeError(f"Expected anchors to be a Tensor, but got {type(anchors)}")  
    # 檢查形狀是否匹配
    if anchors.dim() == 3 and anchors.shape[0] == 1:
        anchors = anchors.squeeze(0)  # 去掉第一個維度
    if anchors.shape[0] != bbox_deltas.shape[0]:
        raise ValueError(
            f"Shape mismatch: anchors has {anchors.shape[0]} rows, "
            f"but bbox_deltas has {bbox_deltas.shape[0]} rows."
        )

    # 確保數據在相同設備上
    if anchors.device != bbox_deltas.device:
        raise ValueError(
            f"Device mismatch: anchors on {anchors.device}, "
            f"but bbox_deltas on {bbox_deltas.device}."
        )

    anchors_x_center = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchors_y_center = (anchors[:, 1] + anchors[:, 3]) / 2.0
    anchors_width = anchors[:, 2] - anchors[:, 0]
    anchors_height = anchors[:, 3] - anchors[:, 1]

    pred_x_center = bbox_deltas[:, 0] * anchors_width + anchors_x_center
    pred_y_center = bbox_deltas[:, 1] * anchors_height + anchors_y_center
    pred_width = torch.exp(bbox_deltas[:, 2]) * anchors_width
    pred_height = torch.exp(bbox_deltas[:, 3]) * anchors_height

    pred_boxes = torch.zeros_like(bbox_deltas)
    pred_boxes[:, 0] = pred_x_center - 0.5 * pred_width  # x_min
    pred_boxes[:, 1] = pred_y_center - 0.5 * pred_height  # y_min
    pred_boxes[:, 2] = pred_x_center + 0.5 * pred_width  # x_max
    pred_boxes[:, 3] = pred_y_center + 0.5 * pred_height  # y_max

    return pred_boxes


def calculate_iou(proposals, gt_boxes):
    """
    计算 proposals 和 gt_boxes 之间的 IoU。
    
    参数:
    - proposals: Tensor, shape [N, 4], 候选框 (x_min, y_min, x_max, y_max)
    - gt_boxes: Tensor, shape [M, 4], 真实框 (x_min, y_min, x_max, y_max)
    
    返回:
    - ious: Tensor, shape [N, M], proposals 和 gt_boxes 的 IoU 矩阵
    
    """
    # 计算 proposals 和 gt_boxes 的面积
     # 如果 proposals 是列表，先将其转换为 Tensor
    if isinstance(proposals, list):
        proposals = torch.cat(proposals, dim=0)

    # 如果 gt_boxes 是列表，也需要合并
    if isinstance(gt_boxes, list):
        gt_boxes = torch.cat(gt_boxes, dim=0)


    # 检查输入是否合法
    if proposals.numel() == 0 or gt_boxes.numel() == 0:
        raise ValueError("Proposals or ground truth boxes cannot be empty.")

    # 计算候选框和目标框的面积

    proposal_areas = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # 初始化 IoU 矩阵
    ious_matrix = torch.zeros((proposals.size(0), gt_boxes.size(0)), device=proposals.device)

    # 遍历每个 gt_box，计算与所有 proposals 的 IoU
    for i, gt_box in enumerate(gt_boxes):
        x_min = torch.max(proposals[:, 0], gt_box[0])  # 交集的左上角 x
        y_min = torch.max(proposals[:, 1], gt_box[1])  # 交集的左上角 y
        x_max = torch.min(proposals[:, 2], gt_box[2])  # 交集的右下角 x
        y_max = torch.min(proposals[:, 3], gt_box[3])  # 交集的右下角 y

        # 计算交集面积
        intersection = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)

        # 计算并集面积
        union = proposal_areas + gt_areas[i] - intersection

        # 计算 IoU
        ious_matrix[:, i] = intersection / union

    # 找到每个候选框对应的最高 IoU 目标框
    max_ious, max_indices = ious_matrix.max(dim=1)  # [N]

    # 筛选 IoU > 0 的候选框和目标框
    valid_indices = max_ious > 0  # 避免选择 IoU 为 0 的框
    matched_proposals = proposals[valid_indices]
    matched_gt_boxes = gt_boxes[max_indices[valid_indices]]
    matched_ious = max_ious[valid_indices]
    matched_proposal_gt_indices = max_indices[valid_indices]

    return matched_proposals, matched_gt_boxes, matched_ious, ious_matrix,matched_proposal_gt_indices
def visualize_boxes(image_tensor, pred_boxes, gt_boxes=None, pred_color=(255,0,0), gt_color=(0,255,0)):
    """
    將預測框 pred_boxes 與（可選的）GT框 gt_boxes，畫到影像上並顯示。
    Args:
        image_tensor: 形狀 (C,H,W) 的張量，數值範圍通常是 [0,1] 或 [0,255] (視你的前處理而定)。
        pred_boxes: shape=(N,4)，每個box=[x_min, y_min, x_max, y_max] (float或int皆可)。
        gt_boxes (optional): shape=(M,4)，同上格式的 GT框。
        pred_color: 畫預測框的顏色 (RGB)。預設紅色 (255,0,0)。
        gt_color: 畫 GT框的顏色 (RGB)。預設綠色 (0,255,0)。
    """

    # 將張量轉為 PIL Image 能處理的格式
    # 假設 image_tensor 在 [0,1]，要乘回 255；若已經是[0,255]，則不用
    image_np = image_tensor.cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    # 維度 (C,H,W) => (H,W,C)
    image_np = np.transpose(image_np, (1,2,0))
    
    # 建立 PIL Image
    pil_image = Image.fromarray(image_np)

    # 在 PIL image 上畫框
    draw = ImageDraw.Draw(pil_image)

    # 畫預測框
    if pred_boxes is not None:
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=2)

    # 畫 GT框（若有提供）
    if gt_boxes is not None:
        for gt_box in gt_boxes:
            gx1, gy1, gx2, gy2 = gt_box
            draw.rectangle([gx1, gy1, gx2, gy2], outline=gt_color, width=2)

    # 顯示圖像
    plt.figure(figsize=(8,8))
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()
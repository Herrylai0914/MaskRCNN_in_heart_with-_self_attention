import math
import sys
import time
import numpy as np
import torch
import torchvision.models.detection.mask_rcnn
import sys
sys.path.append(r'C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\Pytorch-Mask-RCNN-master\\detection')
import utils  # 直接使用 utils 模組中的功能
from collections import defaultdict
#import detection.utils as utils
#from coco_eval import CocoEvaluator
#from coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, combined_data_loader, device, epoch, print_freq,data_loader_length, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    lr_scheduler = None

    # 初始化学习率调度器
    if epoch == 0:
        warmup_factor = 1.0 / 100
        warmup_iters = min(100, data_loader_length - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # 遍历数据集
    for idx, batches in enumerate(metric_logger.log_every(combined_data_loader, print_freq, header)):
        if isinstance(batches, tuple) and len(batches) == 2:
        # 多任务情况：batches 是包含两个任务数据的元组
            batch1, batch2 = batches
        # 确保 batch1 和 batch2 都是包含图像和目标的元组
            if isinstance(batch1, tuple) and len(batch1) == 2:
                images1, targets1 = batch1
            else:
                raise ValueError(f"Unexpected format for batch1: {batch1}")

            if isinstance(batch2, tuple) and len(batch2) == 2:
                images2, targets2 = batch2
            else:
                raise ValueError(f"Unexpected format for batch2: {batch2}")


            # 将图片和标签移至指定设备（如 GPU）
            images1 = [image.to(device) for image in images1]
            targets1 = [{k: v.to(device) for k, v in t.items()} for t in targets1]
            images2 = [image.to(device) for image in images2]
            targets2 = [{k: v.to(device) for k, v in t.items()} for t in targets2]

        elif isinstance(batches, tuple) and len(batches) == 1:
            # 单任务模式
            batch1 = batches[0]
            if isinstance(batch1, tuple):
                images1, targets1 = batch1
            else:
                raise ValueError("Expected batch1 to be a tuple containing images and targets")

            # 将图片和标签移至指定设备（如 GPU）
            images1 = [image.to(device) for image in images1]
            targets1 = [{k: v.to(device) for k, v in t.items()} for t in targets1]

        # 前向传播并计算损失
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                if hasattr(model, 'model_task2') and model.model_task2 is not None:
                # 多任务训练，计算两个任务的输出
         
                 # 多任务训练，计算两个任务的输出
                    task1_loss_dict = model(images1, targets1)
                    task2_loss_dict = model(images2, targets2)
                
                # 假设 CustomROIHead 返回 class_logits, bbox_deltas, mask_logits
                    (class_logits1, bbox_deltas1, mask_logits1), _ = task1_loss_dict
                    (class_logits2, bbox_deltas2, mask_logits2), _ = task2_loss_dict
            # 定义损失函数
                    criterion_cls = torch.nn.CrossEntropyLoss()
                    criterion_bbox = torch.nn.SmoothL1Loss()
                    criterion_mask = torch.nn.BCEWithLogitsLoss()


                    for targets in [targets1, targets2]:
                        for i in range(len(targets)):
                            if targets[i]['masks'].dim() == 3:
                                targets[i]['masks'] = targets[i]['masks'].unsqueeze(1)  # 从 (batch_size, height, width) -> (batch_size, 1, height, width)
                            if targets[i]['masks'].dtype == torch.bool:
                                targets[i]['masks'] = targets[i]['masks'].float()

            # 分别计算任务 1 和任务 2 的损失
                    clas_loss = criterion_cls(class_logits1, targets1[0]['labels'])
                    if torch.isnan(clas_loss):
                            print(class_logits1)
                            print(targets1[0]['labels'])
                            raise ValueError("Classification Loss is NaN!")
                    box_loss =  criterion_bbox(bbox_deltas1, targets1[0]['boxes'])
                    if torch.isnan(box_loss):
                            raise ValueError("box_loss is NaN!")
                    mask_loss = criterion_mask(mask_logits1, targets1[0]['masks'])
                    if torch.isnan(mask_loss):
                            raise ValueError("mask_loss is NaN!")
                    loss_task1 = clas_loss + box_loss +mask_loss

                    clas_loss_2 = criterion_cls(class_logits2, targets2[0]['labels'])
                    if torch.isnan(clas_loss_2):
                            raise ValueError("Classification Loss is NaN!")
                    box_loss_2 =  criterion_bbox(bbox_deltas2, targets2[0]['boxes'])
                    if torch.isnan(box_loss_2):
                            raise ValueError("box_loss is NaN!")
                    mask_loss_2 = criterion_mask(mask_logits2, targets2[0]['masks'])
                    if torch.isnan(mask_loss_2):
                            raise ValueError("mask_loss is NaN!")
                    loss_task2 = clas_loss_2 + box_loss_2 +mask_loss_2

            # 联合损失
                    losses = loss_task1 + loss_task2

                else:
                    #print("Running single-task training")
                    loss_dict = model(images1, targets1)
                    losses = sum(loss for loss in loss_dict.values())


        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # 减少所有 GPU 的损失以进行日志记录
        if hasattr(model, 'model_task2') and model.model_task2 is not None:
            # 特征共享情况下
            loss_dict_reduced_task1 = utils.reduce_dict({'loss_task1': loss_task1})
            loss_dict_reduced_task2 = utils.reduce_dict({'loss_task2': loss_task2})
            losses_reduced = sum(loss for loss in loss_dict_reduced_task1.values()) + \
                             sum(loss for loss in loss_dict_reduced_task2.values())
        else:
            # 单任务模式下
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    # 记录单任务的损失值
        loss_value = losses_reduced.item() if isinstance(losses_reduced, torch.Tensor) else losses_reduced
        metric_logger.update(loss=loss_value)
# 更新學習率
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger
def train_feature_share_epoch(model, optimizer, combined_data_loader, device, epoch, print_freq,data_loader_length, scaler=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    lr_scheduler = None
    model.train()


    # 初始化学习率调度器
    if epoch == 0:
        warmup_factor = 1.0 / 10
        warmup_iters = min(100, data_loader_length - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for idx, batches in enumerate(metric_logger.log_every(combined_data_loader, print_freq, header)):
        if isinstance(batches, tuple) and len(batches) == 2:
        # 多任务情况：batches 是包含两个任务数据的元组
            batch1, batch2 = batches
        # 确保 batch1 和 batch2 都是包含图像和目标的元组
            if isinstance(batch1, tuple) and len(batch1) == 2:
                imagesA, targetsA = batch1
            else:
                raise ValueError(f"Unexpected format for batch1: {batch1}")

            if isinstance(batch2, tuple) and len(batch2) == 2:
                imagesB, targetsB = batch2
            else:
                raise ValueError(f"Unexpected format for batch2: {batch2}")


            # 将图片和标签移至指定设备（如 GPU）
        imagesA = [image.to(device) for image in imagesA]
        targetsA = [{k: v.to(device) for k, v in t.items()} for t in targetsA]
        imagesB = [image.to(device) for image in imagesB]
        targetsB = [{k: v.to(device) for k, v in t.items()} for t in targetsB]

        # imagesA, imagesB => Tensors shape=(B,3,128,128)
        # targetsA, targetsB => list[Dict], each have boxes, labels, etc.

        # 移到GPU
        imagesA = torch.stack(imagesA, dim=0)   # shape=(B,3,H,W)
        imagesB = torch.stack(imagesB, dim=0)   # 同理
        imagesA = imagesA.to(device)
        imagesB = imagesB.to(device)

        # (可考慮把 targetsA, targetsB merge? or only use one?)
        # 這裡視你如何定義 "多任務" => 可能你要合併?
        # Example: 只用 targetsA => 
        loss_dictA, loss_dictB = model(imagesA, targetsA, imagesB, targetsB)  # 回傳 (dictA, dictB)
        # 也可能合併 targets, 依你多任務設計

        lossesA = sum(loss for loss in loss_dictA.values())
        lossesB = sum(loss for loss in loss_dictB.values())
        losses = lossesA + lossesB


        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if idx % print_freq == 0:
            print(f"Epoch {epoch}, iter {idx}, loss={losses.item()}")

        loss_dict_reducedA = utils.reduce_dict(loss_dictA)
        loss_dict_reducedB = utils.reduce_dict(loss_dictB)

# 將每個 dict 裡的 loss tensor 加總
        lossA_reduced = sum(v for v in loss_dict_reducedA.values())
        lossB_reduced = sum(v for v in loss_dict_reducedB.values())

# 轉為 float
        lossA_val = lossA_reduced.item()
        lossB_val = lossB_reduced.item()

# 也可以紀錄 total
        lossTotal_val = (lossA_reduced + lossB_reduced).item()

# 在 metric_logger 裡，把它們 update 成獨立欄位
        metric_logger.update(lossA=lossA_val,
                     lossB=lossB_val,
                     loss=lossTotal_val,
                     lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def print_nested_tuple(nested_tuple, indent=0):
    if isinstance(nested_tuple, tuple):
        for idx, item in enumerate(nested_tuple):
            print(" " * indent + f"Element {idx}:")
            print_nested_tuple(item, indent + 4)  # 增加縮進層次
    elif isinstance(nested_tuple, torch.Tensor):
        print(" " * indent + f"Tensor shape: {nested_tuple.shape}, Device: {nested_tuple.device}")
        print(" " * indent + f"Tensor content: {nested_tuple.cpu().detach().numpy()}")  # 打印內容
    else:
        print(" " * indent + f"Value: {nested_tuple}")

@torch.inference_mode()
def evaluate(model, data_loader, device, iou_threshold=0.5):
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
        task1_output, task2_output = None, None
        

        if hasattr(model, 'model_task2') and model.model_task2 is not None:

            task1_output = model(images,targets=targets)   
            model_time = time.time() - model_time
         # 將模型輸出移動到 CPU 以便進行後續處理
        #task1_output = [{k: v.to(cpu_device) for k, v in t.items()} for t in task1_output]
        #if task2_output:
            #task2_output = [{k: v.to(cpu_device) for k, v in t.items()} for t in task2_output]
            task1_output= tuple(
                tuple(v.to(cpu_device) for v in t) if isinstance(t, tuple) else t.to(cpu_device)
                for t in task1_output
            )        
        # 保存任务1的预测和真实值
            for target, t1_output in zip(targets, [task1_output]):
                image_id = target["image_id"].item()
                all_annotations_task1[image_id].append(target)

    # 解包 tuple，並將每個元素單獨保存
                class_logits, bbox_deltas,labels = t1_output  # 解包

                # 创建一个字典来存储解包后的内容
                detection_dict = {
                    "class_logits": class_logits,
                    "bbox_deltas": bbox_deltas,
                    "labels": labels
                }
                  # 确保 image_id 存在于 all_detections_task1 中
                if image_id not in all_detections_task1:
                    all_detections_task1[image_id] = []
                
                
                 # 将字典加入到对应的 image_id 的列表中
                all_detections_task1[image_id].append(detection_dict)
            
         
            mean_average_precision_task1,iou_mean_task1 = calculate_map(all_detections_task1, all_annotations_task1, iou_threshold)
        else:
            # 单任务情况下，将输出移动到 CPU
            task1_output = model(images)   
            model_time = time.time() - model_time
            if isinstance(task1_output, dict):
                task1_output = {k: v.to(cpu_device) for k, v in task1_output.items()}

            elif isinstance(task1_output, list):
                task1_output = [{k: v.to(cpu_device) for k, v in t.items()} for t in task1_output]

            # 保存任务1的预测和真实值
            for target, t1_output in zip(targets, [task1_output]):
                image_id = target["image_id"].item()
                all_annotations_task1[image_id].append(target)

    # 确保 image_id 存在于 all_detections_task1 中
            if image_id not in all_detections_task1:
                all_detections_task1[image_id] = []
            if isinstance(task1_output, list):
                for t1_output in task1_output:
                    if isinstance(t1_output, dict):
            # 假设在单任务情况下，模型输出中的键是 'boxes', 'labels', 'scores', 'masks'。
                        pred_boxes = t1_output.get("boxes")
                        pred_labels = t1_output.get("labels")
                        pred_scores = t1_output.get("scores")
                        pred_masks = t1_output.get("masks")  # 如果有mask信息

            # 确保有有效的预测框和标签
                        if pred_boxes is not None and pred_boxes.numel() > 0:
                # 确保 image_id 存在于 all_detections_task1 中
                            if image_id not in all_detections_task1:
                                all_detections_task1[image_id] = []

                # 将预测结果存储到 all_detections_task1 中
                            all_detections_task1[image_id].append({
                                "boxes": pred_boxes,
                                "labels": pred_labels,
                                "scores": pred_scores,
                                "masks": pred_masks if pred_masks is not None else None
                            })
            else:
                # 输出警告信息，如果没有有效的预测框
                print(f"Warning: No valid prediction boxes for image ID: {image_id}")


            mean_average_precision_task1,iou_mean_task1, dice_mean =calculate_map_single_task(all_detections_task1, all_annotations_task1,iou_thresholds=np.arange(0.5, 1.0, 0.05))
        metric_logger.update(model_time=model_time)

    return mean_average_precision_task1, iou_mean_task1,dice_mean

def evaluate_feature(model, combined_data_loader, device, iou_threshold=0.5):
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
    all_detectionsA = defaultdict(list)
    all_annotationsA = defaultdict(list)

    all_detectionsB = defaultdict(list)
    all_annotationsB = defaultdict(list)

    for idx, batches in enumerate(metric_logger.log_every(combined_data_loader, 100, header)):
        # batches => ( (imagesA, targetsA), (imagesB, targetsB) )
        (imagesA, targetsA), (imagesB, targetsB) = batches

        # 搬到 GPU
        imagesA = [img.to(device) for img in imagesA]
        imagesB = [img.to(device) for img in imagesB]

        for t in targetsA:
            for k, v in t.items():
                t[k] = v.to(device)
        for t in targetsB:
            for k, v in t.items():
                t[k] = v.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if isinstance(imagesA, list):
                imagesA = torch.stack(imagesA, dim=0)
        if isinstance(imagesB, list):
                imagesB = torch.stack(imagesB, dim=0)

        # 如果图像是单张图片的情况，确保添加批次维度
        if len(imagesA.shape) == 3:
            imagesA = imagesA.unsqueeze(0)  # 变成 [1, 3, H, W]
        if len(imagesB.shape) == 3:
            imagesB = imagesB.unsqueeze(0)  # 变成 [1, 3, H, W]

     # forward => 推理 => no targets
        start_time = time.time()
        with torch.no_grad():
            # 使用位置參數:
            detectionsA, detectionsB = model(imagesA, None,imagesB, None)

        model_time = time.time() - start_time
         # 分別收集 A 的 GT / 預測
        for i, (t_gt, t_pred) in enumerate(zip(targetsA, detectionsA)):

            if "image_id" not in t_gt:
                print(f"[WARNING] 'image_id' not found in targetsA index={i}. t_gt={t_gt}")
                continue
            image_id = t_gt["image_id"].item()
            all_annotationsA[image_id].append(t_gt)
            all_detectionsA[image_id].append(t_pred)

        # 分別收集 B 的 GT / 預測
        for i, (t_gt, t_pred) in enumerate(zip(targetsB, detectionsB)):

            if "image_id" not in t_gt:
                print(f"[WARNING] 'image_id' not found in targetsB index={i}. t_gt={t_gt}")
                continue
            image_id = t_gt["image_id"].item()
            all_annotationsB[image_id].append(t_gt)
            all_detectionsB[image_id].append(t_pred)

        metric_logger.update(model_time=model_time)

                # 输出警告信息，如果没有有效的预测框
        
    mapA, iouA = calculate_map_single_task(all_detectionsA, all_annotationsA)
    mapB, iouB = calculate_map_single_task(all_detectionsB, all_annotationsB)

    return (mapA, iouA), (mapB, iouB)

def calculate_map(detections, annotations, iou_threshold=0.5):
    """计算 mean Average Precision (mAP)."""
    average_precisions = []
    iou_results = []
    for image_id in detections.keys():
        if image_id not in detections or len(detections[image_id]) == 0:
            #print(f"No detections found for image_id {image_id}, skipping.")
            continue  
    for image_id in detections.keys():
        pred_boxes = detections[image_id][0]["bbox_deltas"].cpu().detach().numpy()
        pred_scores = detections[image_id][0]["class_logits"].cpu().detach().numpy()
        gt_boxes = annotations[image_id][0]["boxes"].cpu().detach().numpy()
        #print("Class logits:", pred_scores)
        #print("Bounding box deltas:", pred_boxes)
        #print("boxes:", gt_boxes)

        # 按照预测分数降序排列预测框
        sorted_indices = pred_scores.argsort()[::-1]
        pred_boxes = np.max(pred_boxes, axis=0, keepdims=True)
        pred_scores = pred_scores[sorted_indices]
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
            #print(f"Predicted Box: {pred_box}")
            #print(f"Intersection: {intersection}")
            #print(f"Box Area: {box_area}, Ground Truth Box Area: {gt_boxes_area}")
            #print(f"Union: {union}")
            #print(f"IoUs: {ious}")
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            #print(f"Image ID: {image_id},Max IoU: {max_iou:.4f}")
            iou_results.append(max_iou)
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


def compute_ap(precisions, recalls):
    """通过插值计算 AP。"""
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def calculate_map_single_task(detections, annotations, iou_thresholds=np.arange(0.5, 1.0, 0.1),image_size=(128, 128)):
    """
    计算单任务的 mean Average Precision (mAP) 和平均 IoU，
    使用多个 IoU 阈值进行评估。

    参数:
        detections: 检测结果字典，每个 image_id 对应预测结果
        annotations: 标注字典，每个 image_id 对应真实标注
        iou_thresholds: IoU 阈值数组，默认为 np.arange(0.5, 1.0, 0.1)
    
    返回:
        mean_average_precision: 多阈值下 AP 的均值（mAP）
        iou_mean: 所有预测框计算得到的 IoU 平均值
    """
    ap_thresholds = []      # 存放每个阈值下所有图像 AP 的均值
    all_iou_results = []    # 存放所有预测框的 IoU（全局计算，每个预测框只计一次）
    dice_list = []
    # 1. 先计算每个图像所有预测框的最大 IoU（全局，不依赖阈值）
    for image_id in detections.keys():
        if len(detections[image_id]) == 0:
            all_iou_results.append(0)
            dice_list.append(0)
            continue

        pred_data = detections[image_id][0]
        # 优先使用 scores 作为得分
        if "scores" in pred_data:
            pred_scores = pred_data["scores"].cpu().detach().numpy()
        else:
            pred_scores = pred_data["labels"].cpu().detach().numpy()
        pred_boxes = pred_data["boxes"].cpu().detach().numpy()
        gt_boxes = annotations[image_id][0]["boxes"].cpu().detach().numpy()
         # === IOU ===
        # 对每个预测框计算与所有真实框的最大 IoU
        for pred_box in pred_boxes:
            x_min = np.maximum(pred_box[0], gt_boxes[:, 0])
            y_min = np.maximum(pred_box[1], gt_boxes[:, 1])
            x_max = np.minimum(pred_box[2], gt_boxes[:, 2])
            y_max = np.minimum(pred_box[3], gt_boxes[:, 3])
            intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
            box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_boxes_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            union = box_area + gt_boxes_area - intersection
            ious = intersection / (union + 1e-6)
            avg_iou = np.max(ious)  # 取平均而不是最大值
            all_iou_results.append(avg_iou)
        # === Dice ===
        pred_mask = np.zeros(image_size, dtype=np.uint8)
        gt_mask = np.zeros(image_size, dtype=np.uint8)

        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            pred_mask[y1:y2, x1:x2] = 1

        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            gt_mask[y1:y2, x1:x2] = 1

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice_score = 2.0 * intersection / (union + 1e-6)
        dice_list.append(dice_score)


    # 2. 针对每个 IoU 阈值，计算对应的 AP
    for thresh in iou_thresholds:
        average_precisions = []  # 存放当前阈值下每张图像的 AP

        for image_id in detections.keys():
            if len(detections[image_id]) == 0:
                all_iou_results.append(0)
                continue

            pred_data = detections[image_id][0]
            if "scores" in pred_data:
                pred_scores = pred_data["scores"].cpu().detach().numpy()
            else:
                pred_scores = pred_data["labels"].cpu().detach().numpy()
            pred_boxes = pred_data["boxes"].cpu().detach().numpy()
            gt_boxes = annotations[image_id][0]["boxes"].cpu().detach().numpy()

            # 按预测得分降序排序预测框
            sorted_indices = pred_scores.argsort()[::-1]
            pred_boxes = pred_boxes[sorted_indices]
            pred_scores = pred_scores[sorted_indices]

            tp = np.zeros(len(pred_boxes))
            fp = np.zeros(len(pred_boxes))
            detected = []  # 用于记录已匹配的 gt_box

            for i, pred_box in enumerate(pred_boxes):
                x_min = np.maximum(pred_box[0], gt_boxes[:, 0])
                y_min = np.maximum(pred_box[1], gt_boxes[:, 1])
                x_max = np.minimum(pred_box[2], gt_boxes[:, 2])
                y_max = np.minimum(pred_box[3], gt_boxes[:, 3])
                intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
                box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                gt_boxes_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                union = box_area + gt_boxes_area - intersection
                ious = intersection / (union + 1e-6)
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]

                # 根据当前阈值判断 TP / FP，确保同一个 gt_box 只匹配一次
                if max_iou >= thresh and max_iou_idx not in detected:
                    tp[i] = 1
                    detected.append(max_iou_idx)
                else:
                    fp[i] = 1

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recalls = tp_cumsum / (len(gt_boxes) + 1e-6)

            ap = compute_ap(precisions, recalls)
            average_precisions.append(ap)

        if average_precisions:
            ap_thresholds.append(np.mean(average_precisions))
        else:
            ap_thresholds.append(0)
    
    mean_average_precision = np.mean(ap_thresholds)
    iou_mean = np.mean(all_iou_results) if all_iou_results else 0
    dice_mean = np.mean(dice_list) if dice_list else 0
    return mean_average_precision, iou_mean

def evaluate_single_task(model, data_loader, device):
    model.eval()  # 设置模型为评估模式，不会执行 dropout 等正则化
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation:"

    val_loss = 0.0  # 初始化验证损失值

    with torch.no_grad():  # 在验证阶段不需要计算梯度
        for images1, targets1 in metric_logger.log_every(data_loader, 100, header):  # 遍历验证数据集
            images1 = list(image.to(device) for image in images1)
            targets1 = [{k: v.to(device) for k, v in t.items()} for t in targets1]

            # 计算模型的输出（类似于训练阶段）
            loss_dict = model(images1, targets1)
            losses = sum(loss for loss in loss_dict.values())

            # 合并所有 GPU 的损失（如果是多 GPU 环境）
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # 将当前批次的损失累加到总的验证损失中
            val_loss += losses_reduced.item() if isinstance(losses_reduced, torch.Tensor) else losses_reduced

    # 计算验证集的平均损失
    val_loss /= len(data_loader)

    # 打印验证损失
    print(f'Validation Loss: {val_loss:.4f}')
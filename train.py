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
from featureshare import CustomMaskRCNN,MyCustomBackbone_feature_share,create_official_rpn,create_official_roi_heads,MyCustomBackboneSingle,CustomMaskRCNN_single
from dataset import load_datasets
from args import get_args_parser
from engine import evaluate, train_one_epoch,train_feature_share_epoch,evaluate_feature
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste
from torchvision.models.detection import MaskRCNN
import copy
'''def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))'''
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


def check_model_on_device(model):
        for name, param in model.named_parameters():
            print(f"Parameter {name} is on device: {param.device}")


def main(args):
    args.feature_share = True # 強制開啟特徵共享
    args.gray = True  # 強制開啟灰階模式

    # Initialize model output directory
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    use_feature_sharing = args.feature_share  # Check if feature sharing is enabled

    # Load datasets
    train_dataset1, val_dataset1, train_dataset2, val_dataset2, num_classes1, num_classes2 = load_datasets(args, use_second_dataset=use_feature_sharing)

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

# Task 2 data loader (for feature sharing)
    if use_feature_sharing:
        if args.distributed:
            train_sampler2 = torch.utils.data.distributed.DistributedSampler(train_dataset2)
            test_sampler2 = torch.utils.data.distributed.DistributedSampler(val_dataset2, shuffle=False)
        else:
            train_sampler2 = torch.utils.data.RandomSampler(train_dataset2)
            test_sampler2 = torch.utils.data.SequentialSampler(val_dataset2)

        if args.aspect_ratio_group_factor >= 0:
            group_ids2 = create_aspect_ratio_groups(train_dataset2, k=args.aspect_ratio_group_factor)
            train_batch_sampler2 = GroupedBatchSampler(train_sampler2, group_ids2, args.batch_size)
        else:
            train_batch_sampler2 = torch.utils.data.BatchSampler(train_sampler2, args.batch_size, drop_last=True)

        data_loader2 = torch.utils.data.DataLoader(
            train_dataset2, batch_sampler=train_batch_sampler2, num_workers=args.workers, collate_fn=utils.collate_fn
    )

        data_loader_test2 = torch.utils.data.DataLoader(
            val_dataset2, batch_size=1, sampler=test_sampler2, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    # Combine data loaders for feature sharing
        combined_data_loader = list(zip(data_loader1, data_loader2))
        data_loader_length = min(len(data_loader1), len(data_loader2))

        combined_data_loader_test = list(zip( data_loader_test1,  data_loader_test2))
        data_loader_length_test = min(len(data_loader_test1), len(data_loader_test2))

    else:
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
    some_rpn = create_official_rpn()

    # 2套 RPN
    rpnA = copy.deepcopy(some_rpn)
    rpnB = copy.deepcopy(some_rpn)

    # 建立 RoIHeads(含mask)
    roi_heads = create_official_roi_heads(num_classes=2, want_mask=True)  

    if use_feature_sharing:
        print("maskrcnn_resnet101_fpn_gray_feature_share")
        mybackbone = MyCustomBackbone_feature_share(num_classes1, num_classes2)
        multi_task_model =  CustomMaskRCNN(
                                            backbone=mybackbone, 
                                            rpnA=rpnA,
                                            roiHeadsA=roi_heads,
                                            rpnB=rpnB,
                                            roiHeadsB=roi_heads,
                                           num_classes=num_classes1+1)
        multi_task_model.to(device)
    else:
        # 使用现成的单任务模型
        if args.model == "maskrcnn_resnet101_fpn_gray":
            print("maskrcnn_resnet101_fpn_gray")
            mybackbone = MyCustomBackboneSingle(num_classes1)
    # 單通道
            multi_task_model =  CustomMaskRCNN_single(backbone=mybackbone,
                                                      rpn=rpnA,
                                            roiHeads=roi_heads,
                                num_classes=num_classes1+1)
            multi_task_model.to(device)
        else:   
            print("base_model")
            base_model = torchvision.models.detection.__dict__[args.model](
            weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes1, **kwargs
    )
            base_model = torchvision.models.detection.__dict__[args.model](
                weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes1, **kwargs
        )
            base_model.to(device)
    
     # **最後將模型搬到 `device`**


    
    if args.distributed and args.sync_bn:
        multi_task_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(multi_task_model)

    model_without_ddp = multi_task_model
    if args.distributed:
        multi_task_model = torch.nn.parallel.DistributedDataParallel(multi_task_model, device_ids=[args.gpu])
        model_without_ddp = multi_task_model.module

    # Prepare optimizer
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
    if args.fine_tune:
        print("Starting Fine-Tuning Process")
        for epoch in range(args.start_epoch, args.fine_tune_epochs):
            if args.distributed:
                train_sampler1.set_epoch(epoch)
                if use_feature_sharing:
                    train_sampler2.set_epoch(epoch)

             # Use combined data loader if feature sharing is enabled
            if use_feature_sharing:
                metric_logger = train_one_epoch(multi_task_model, optimizer, combined_data_loader, device, epoch, args.print_freq,data_loader_length, scaler)
            else:
                metric_logger = train_one_epoch(multi_task_model, optimizer, data_loader1, device, epoch, args.print_freq, data_loader_length,scaler)

            train_loss = metric_logger.meters['loss'].global_avg

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
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"fine_tune_model_{epoch}.pth"))

            # Evaluate after fine-tuning epoch
            mean_average_precision, accuracy = evaluate(multi_task_model, data_loader_test1, device)
            print(f"Epoch {epoch} completed. mAP: {mean_average_precision:.4f}, Loss: {train_loss:.4f}")
    # Training loop
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler1.set_epoch(epoch)
            if use_feature_sharing:
                train_sampler2.set_epoch(epoch)

        if use_feature_sharing:
                metric_logger = train_feature_share_epoch(multi_task_model, optimizer, combined_data_loader, device, epoch, args.print_freq,data_loader_length, scaler=None)
        else:
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
        if use_feature_sharing:

            mean_average_precision,iou_mean = evaluate_feature(multi_task_model, combined_data_loader_test, device)
            if isinstance(mean_average_precision, tuple) and len(mean_average_precision) == 2:
    # 代表多任務 => mean_average_precision = (mapA, iouA), iou_mean = (mapB, iouB)
                mapA, iouA = mean_average_precision
                mapB, iouB = iou_mean

                print(f"Epoch {epoch} completed. train_loss: {train_loss:.4f}, "
                    f"TaskA => mAP: {mapA:.4f}, IoU: {iouA:.4f} "
                    f"TaskB => mAP: {mapB:.4f}, IoU: {iouB:.4f}"
                    )

    # 寫入 CSV
                with open("C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\Pytorch-Mask-RCNN-master\\loss\\testing_log.csv", "a") as f:
                    f.write(f"{epoch},{train_loss:.4f},{mapA},{iouA},{mapB},{iouB}\n")
        else:
        # Evaluate after every epoch
            mean_average_precision,iou_mean,dice_mean = evaluate(multi_task_model, data_loader_test1, device)
        # 假设 mean_average_precision 是 (mAP_task1, mAP_task2)
            if isinstance(mean_average_precision, tuple):
                mAP_task1 = mean_average_precision
                iou_mean_task1 = iou_mean
                dice_mean_task1 = dice_mean
                print(f"Epoch {epoch} completed. train_loss: {train_loss:.4f}, test_Task 1 mAP: {mAP_task1:.4f},test_ Mean IoU Task 1: {iou_mean_task1:.4f},test_ Mean dice Task 1: {dice_mean_task1:.4f} ")
            
            elif isinstance(mean_average_precision, dict):
    # 如果返回的是字典类型
                mAP_task1 = mean_average_precision.get('task1', 0)
                iou_mean_task1 = iou_mean.get('task1', 0)
                dice_mean_task1 = dice_mean.get('task1', 0)
                print(f"Epoch {epoch} completed.train_loss: {train_loss:.4f}, test_Task 1 mAP: {mAP_task1:.4f}, test_Mean IoU Task 1: {iou_mean_task1:.4f},test_ Mean dice Task 1: {dice_mean_task1:.4f} ")     

            else:
    # 如果是其他类型 (例如 float 或直接是单个任务的返回值)
                mAP_task1 = mean_average_precision
                iou_mean_task1 = iou_mean
                dice_mean_task1 = dice_mean
                print(f"Epoch {epoch} completed.train_loss: {train_loss:.4f},test_Task 1 mAP: {mAP_task1:.4f}, test_Mean IoU Task 1: {iou_mean_task1:.4f},test_ Mean dice Task 1: {dice_mean_task1:.4f} ")
            #print(f"Validation Loss: {val_loss:.4f}")
              
            # Record both Task 1 and Task 2 results to CSV file
            with open("C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\Pytorch-Mask-RCNN-master\\loss\\testing_log.csv", "a") as f:
                    f.write(f"{epoch},{train_loss:.4f},{mAP_task1},{iou_mean_task1},{dice_mean_task1}\n")

        


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

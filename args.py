def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("--feature-share", action="store_true", help="Enable feature sharing between tasks")
    parser.add_argument("--gray", action="store_true", help="If set, use grayscale input instead of RGB.")
    parser.add_argument("--data-path1", default="C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\ACDC&MMs\\LV", type=str, help="dataset path1")
    #parser.add_argument("--data-path1", default="C:\\ACDC\\Mask_RCNN_master\\testimg\\ROI\\LV_test", type=str, help="dataset path1")
    #parser.add_argument("--data-path1", default="C:\\MMWHS\\Train\\class\\LV_test", type=str, help="dataset path1")
    parser.add_argument("--dataset1", default="LV", type=str, help="dataset name")
    parser.add_argument("--data-path2", default="C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\ACDC&MMs\\Myo", type=str, help="dataset path2")
    parser.add_argument("--dataset2", default="Myo", type=str, help="dataset name")
    #parser.add_argument("--data-path2", default="C:\\ACDC\\Mask_RCNN_master\\testimg\\ROI\\Myo_test", type=str, help="dataset path1")
    parser.add_argument("--model", default="maskrcnn_resnet101_fpn_gray", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    #LR
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    #checkpoint
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="C:\\maskrcnn_pytorch\\Pytorch-Mask-RCNN-master\\Pytorch-Mask-RCNN-master\\output_modle\\", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    #test
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )
    #fine-tune
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the model from pre-trained weights")
    parser.add_argument("--fine-tune-lr", default=None, type=float, help="Learning rate for fine-tuning")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze the backbone layers during fine-tuning")
    parser.add_argument("--freeze-until", default=None, type=int, help="Number of layers to freeze in the backbone (0 means no freeze)")
    parser.add_argument("--fine-tune-epochs", default=10, type=int, help="Number of epochs for fine-tuning")
    parser.add_argument("--fine-tune-target-map", default=0.9, type=float, help="Target mAP for fine-tuning")

    return parser
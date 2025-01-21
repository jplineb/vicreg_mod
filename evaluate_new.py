import argparse
from pathlib import Path
import gc

import torch
import torch.optim as optim
import wandb

from utils.logging import configure_logging
from utils.construct_model import LoadVICRegModel
from utils.training_loop import TrainingLoop
from custom_datasets import DATASETS

logger = configure_logging()


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    parser.add_argument(
        "--task_ds",
        type=str,
        choices=["chexpert", "vinddrcxr"],
        required=True,
    )
    # Checkpoint
    parser.add_argument(
        "--pretrained_path", default=None, type=Path, help="path to pretrained model"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from previous epoch if crash occurs",
    )
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/unfrozen_chexpert_AP_PA/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )
    parser.add_argument("--run-id", type=str, required=False)

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--slice", type=int, default=8)
    parser.add_argument(
        "--pretrained-how",
        type=str,
        choices=["VICReg", "Supervised", "Frank"],
        required=True,
    )
    parser.add_argument(
        "--pretrained-dataset",
        type=str,
        choices=["ImageNet", "RadImageNet"],
        required=True,
    )

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.03,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )

    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser


def environment_setup():
    gc.collect(True)
    gpu = torch.cuda.current_device()
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    args = get_arguments().parse_args()
    return args, gpu


def wandb_init(args):
    wandb.init(
        project="Cleaned_VICReg_Experiments",
        config={
            "backbone_learning_rate": args.lr_backbone,
            "head_learning_rate": args.lr_head,
            "weight_decay": args.weight_decay,
            "args.batch_size": args.batch_size,
            "epochs": args.epochs,
            "backbone_weights": args.weights,
            "pretrained_how": args.pretrained_how,
            "pretrained_dataset": args.pretrained_dataset,
            "backbone": args.weights,
            "pretraining_path": args.pretrained_path,
            "task_ds": args.task_ds,
        },
        resume=args.resume,
    )


def main():
    # Environment setup
    args, gpu = environment_setup()
    wandb_init(args)

    # Load dataset and dataloader
    dataset = DATASETS[args.task_ds](
        batch_size=args.batch_size,
        num_workers=args.workers,
        gpu=gpu,
        # transforms_pytorch="RGB",
    )
    num_classes = dataset.num_classes
    train_loader = dataset.get_dataloader(split="train")
    val_loader = dataset.get_dataloader(split="valid")

    # Construct model
    model = LoadVICRegModel(args.arch)
    model.load_pretrained_weights(args.pretrained_path)
    model.modify_head(num_classes=num_classes)
    model = model.produce_model()
    model.cuda(gpu)

    # Train/Eval loop
    ## Load up param groups
    param_groups = [dict(params=model[-1].parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=model[:-1].parameters(), lr=args.lr_backbone))
    ## Setup optimizer and scheduler
    optimizer = optim.Adam(param_groups, 0, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    training_loop = TrainingLoop(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        dataset,
        args,
        stats_file=None,
        gpu=gpu,
    )
    training_loop.train(start_epoch=0, num_epochs=args.epochs)


if __name__ == "__main__":
    main()

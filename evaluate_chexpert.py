# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib
from datetime import datetime
from collections import OrderedDict
import gc
import wandb

from torch import nn, optim
from torchvision import datasets, transforms
import torch
import numpy as np

import resnet

from custom_datasets import Chexpert


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
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

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--slice", type=int, default=8)
    parser.add_argument(
        "--pretrained-how",
        type=str,
        choices=["VICReg", "Shallow", "Frank"],
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


def main():
    # Enable garbage collector
    gc.collect(True)
    print("YOU ARE TRAINING ON CHEXPERT USING THE CHEXPERT DATASET CLASS")
    parser = get_arguments()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node

    main_worker(torch.cuda.current_device(), args)


def main_worker(gpu, args):
    print(f"Begining main worker on gpu: {gpu}")
    args.rank += gpu
    # Run WANDB Init
    wandb.init(
        project="Cleaned_VICReg_Experiments",
        config={
            "backbone_learning_rate": args.lr_backbone,
            "head_learning_rate": args.lr_head,
            "weight_decay": args.weight_decay,
            "dataset": "chexpert",
            "epochs": args.epochs,
            "backbone_weights": args.weights,
            "pretrained_how": args.pretrained_how,
            "pretrained_dataset": args.pretrained_dataset,
            "backbone": args.weights,
            "pretraining_path": args.pretrained_path,
            "resume": args.resume,
        },
        resume=args.resume,
    )

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # Decide whether to load ImageNet pretrained using pytorch models or load local resnet arch
    if args.pretrained_how == "Shallow":
        from torchvision.models import resnet50

        print("Loading Pretrained ResNet50 Model (Shallow Learned)")

        if args.pretrained_dataset == "ImageNet":
            # Load ImageNet Pretrained model
            from torchvision.models import ResNet50_Weights

            # Use built-in methods for loading model
            print("Initializing ResNet50 Model with ImageNet Weights")
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            backbone.fc = nn.Identity()

        elif args.pretrained_dataset == "RadImageNet":
            print("Initializing ResNet50 Model with RadImageNet Weights")
            # Raise an error if user doesn't specify the path
            if args.pretrained_path is None:
                raise (f"Must specify pretrained path for {args.pretrained_dataset=}")
            # Load model weights from checkpoing
            pretrained_state_dict = torch.load(args.pretrained_path, map_location="cpu")
            # Load weights into model
            backbone = resnet50(weights=pretrained_state_dict["model"])
            backbone.fc = nn.Identity()

        # Freeze ResNet backbone weights if specified
        # backbone = nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))
        if args.weights == "freeze":
            print("Freezing ResNet50 pretrained backbone weights")
            print("Are you sure you should be doing this??")
            backbone.requires_grad_(False)

        # Define head classifier for task
        head = nn.Linear(2048, 13)  # modify to number of labels (here its 13)
        head.weight.data.normal_(mean=0.0, std=0.01)
        head.bias.data.zero_()
        head.requires_grad_(True)

        # Swap out the linear classifier on the pretrained ImageNet Model with ours
        model = nn.Sequential(backbone, head)
        model.cuda(gpu)

    elif args.pretrained_path is not None and args.pretrained_how == "Frank":
        print("Loading Frankenstein Model using pretrained ImageNet and RadImageNet")
        wandb.config["slice_location"] = args.slice

        # Load ImageNet Pretrained model
        from torchvision.models import resnet50, ResNet50_Weights

        imagenet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        ## Slice it up
        imagenet_slice_location = args.slice
        imagenet_slice = nn.Sequential(
            OrderedDict(
                [*(list(imagenet_model.named_children())[:imagenet_slice_location])]
            )
        )

        # Load up VICReg model code
        radimagenet_model, embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        state_dict = torch.load(args.pretrained_path, map_location="cpu")
        if "model" in state_dict:
            print("Loading model from state_dict")
            state_dict = state_dict["model"]
            state_dict = {
                key.replace("module.backbone.", ""): value
                for (key, value) in state_dict.items()
            }
            radimagenet_model.load_state_dict(state_dict, strict=False)

        ## Slice it up
        radimagenet_slice_location = args.slice + 1
        radimagenet_slice = nn.Sequential(
            OrderedDict(
                [
                    *(
                        list(radimagenet_model.named_children())[
                            radimagenet_slice_location:
                        ]
                    )
                ]
            )
        )

        # Put two models together
        backbone = nn.Sequential(imagenet_slice, radimagenet_slice)

        # Define head classifier for task
        head = nn.Linear(2048, 13)  # modify to number of labels (here its 13)
        head.weight.data.normal_(mean=0.0, std=0.01)
        head.bias.data.zero_()
        head.requires_grad_(True)

        # Swap out the linear classifier on the pretrained ImageNet Model with ours
        model = nn.Sequential(backbone, nn.Flatten(), head)
        model.cuda(gpu)

        if args.weights == "freeze":
            backbone.requires_grad_(False)
            head.requires_grad_(True)

    elif args.pretrained_how == "VICReg":
        print("Loading local VICReg ResNet50 arch Model")
        # Load VICReg paper version of ResNet50
        backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)

        # state_dict = torch.load(args.pretrained, map_location="cpu")

        # If pretrained model is provided, load the weights
        if args.pretrained_path is not None:
            wandb.config["pretraining"] = "Pretrained RadImageNet"
            print(f"loading pretrained model from {args.pretrained_path}")
            state_dict = torch.load(args.pretrained_path, map_location="cpu")
            if "model" in state_dict:
                print("Loading model from state_dict")
                state_dict = state_dict["model"]
                state_dict = {
                    key.replace("module.backbone.", ""): value
                    for (key, value) in state_dict.items()
                }
            backbone.load_state_dict(state_dict, strict=False)
        else:
            wandb.config["pretraining"] = "Randomly Initialized"

        print("Modifying model with linear layer")
        head = nn.Linear(embedding, 13)  # modify to number of labels (here its 13)
        head.weight.data.normal_(mean=0.0, std=0.01)
        head.bias.data.zero_()
        model = nn.Sequential(backbone, head)
        model.cuda(gpu)

        if args.weights == "freeze":
            backbone.requires_grad_(False)
            head.requires_grad_(True)

    else:
        raise (f"{args.pretrained_how} is not correctly selected")

    # Watch the model with wandb
    wandb.watch(model)

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        print("Resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        wandb.config["start_epoch"] = start_epoch  # log start epoch
        print(f"Continuing from epoch {start_epoch}")
        best_auc = ckpt["best_auc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        print("Starting fresh from epoch 0")
        start_epoch = 0
        best_auc = None
    print(f"Creating dataset with batchsize {args.batch_size}")
    chexpert_ds = Chexpert(
        batch_size=args.batch_size,
        num_workers=args.workers,
        transforms_pytorch="RGB",
        gpu=gpu,
    )
    train_loader = chexpert_ds.get_dataloader(split="train")
    val_loader = chexpert_ds.get_dataloader(split="valid")
    """
    NOTE: here we are going to define the pathologies we care about for benchmark comparions
    - The model is not training on these, just reporting results
    """
    pathologies_of_interest = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
    ]
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        print("Begining training")

        # train
        # TODO: CHECK HERE! Should weights be frozen??
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()
        else:
            assert False

        for step, data in enumerate(train_loader, start=epoch * len(train_loader)):
            # print(f"data len: {len(data)}")
            # print(f"data type: {type(data)}")
            # print(data)
            images = data["img"]
            # print(images.size())
            target = data["lab"]

            output = model(images.cuda(gpu, non_blocking=True))
            # loss = criterion(output, target.cuda(gpu, non_blocking=True))
            loss = chexpert_ds.calculate_loss(predictions=output, targets=target)
            # print(f"Loss for this run {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss.item(),
                        time=int(time.time() - start_time),
                    )
                    wandb.log(stats)
                    # Find garbage
                    print(gc.get_stats())
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        # evaluate
        model.eval()
        if args.rank == 0:
            # Keep track of total accuracy
            print(f"Starting Eval {datetime.now()}")

            all_outputs = []
            all_targets = []
            all_patient_ids = []
            all_views = []

            with torch.no_grad():
                for data in val_loader:
                    images = data["img"]
                    target = data["lab"]
                    patient_ids = data["patientid"]
                    views = data["view"]
                    output = model(images.cuda(gpu, non_blocking=True))
                    # Map outputs to range of 0-1
                    outputs = torch.sigmoid(output).cpu()
                    # Convert Nan targest to none
                    target = target.nan_to_num(0)
                    # Append to list of all outputs
                    all_outputs += outputs
                    all_targets += target.cpu()
                    all_views += views
                    all_patient_ids += patient_ids

            (
                all_auc,
                avg_auc_all,
                avg_auc_of_interest,
                auc_dict,
            ) = chexpert_ds.calculate_auc(all_outputs, all_targets)
            # all_results = chexpert_ds.store_round_results(all_outputs, all_targets, all_views, all_patient_ids)
            stats = dict(
                epoch=epoch,
                all_auc=auc_dict,
                avg_auc=avg_auc_all.tolist(),
                avg_auc_of_interest=avg_auc_of_interest,
            )
            wandb.log(
                {
                    "epoch": epoch,
                    **auc_dict,
                    "avg_auc": stats["avg_auc"],
                    "avg_auc_of_interest": stats["avg_auc_of_interest"],
                }
            )
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

            # Clean up. Maybe memory problem here?
            del all_outputs
            del all_targets
            del all_patient_ids
            del all_views

        scheduler.step()
        if args.rank == 0:
            # best_auc = max(all_auc).to_list()
            state = dict(
                epoch=epoch + 1,
                all_auc=all_auc.tolist(),
                best_auc=avg_auc_all.tolist(),
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            torch.save(state, args.exp_dir / "checkpoint.pth")
            # wandb.log_artifact(model)


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#         lab_values = target

#         # _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


if __name__ == "__main__":
    main()

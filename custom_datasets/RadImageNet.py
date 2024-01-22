import torchxrayvision as xrv
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


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


class RadImageNet:
    """
    TODO: provide in depth description for dataset (i.e label information, labels, classes, source)
    """

    path_to_dataset = "/zfs/wficai/radimagenet/imagenet_fmt"

    def __init__(
        self,
        percent=100,
        transforms_pytorch="default",
        batch_size=64,
        num_workers=0,
        gpu=None,
    ):
        self.percent = percent
        self.transforms = transforms_pytorch
        self.data_loader_spec = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        self.gpu = gpu

    @property
    def transforms_pytorch(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        if self.transforms == "default":
            tfsms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            tfsms = self.transforms

        return tfsms

    def get_dataset(self, split="train") -> torch.utils.data.Dataset:
        traindir = self.path_to_dataset + "/train"
        valdir = self.path_to_dataset + "/valid"
        testdir = self.path_to_dataset + "/test"

        if split == "train":
            dataset = datasets.ImageFolder(traindir, self.transforms_pytorch)

        if split == "valid":
            dataset = datasets.ImageFolder(valdir, self.transforms_pytorch)

        if split == "test":
            dataset = datasets.ImageFolder(testdir, self.transforms_pytorch)

        return dataset

    def check_dataset(self, idx=1, split="train") -> dict:
        # Get idx
        instance = self.get_dataset(split=split)[idx]
        # Print labels
        print(f"labels: {instance[1]}")
        # Show image
        plt.imshow(instance[0].permute(1, 2, 0))

        return instance

    def get_dataloader(self, split="train") -> torch.utils.data.DataLoader:
        """
        Return dataloader given a certain split
        """
        print(f"Fetching dataloader for {split} split")
        dataloader = torch.utils.data.DataLoader(
            self.get_dataset(split=split), **self.data_loader_spec
        )

        return dataloader

    def check_dataloader(self, split="train"):
        print(next(iter(self.get_dataloader(split=split))))

    def calculate_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        best_acc,
        top1,
        top5,
        epoch,
    ) -> (float, float):
        """
        use accuracy
        TODO: move over AverageMeter functions
        """
        acc1, acc5 = self.accuracy(
            predictions, targets.cuda(self.gpu, non_blocking=True), topk=(1, 5)
        )

        return acc1, acc5

    def evaluate(self, model, loader, best_acc, epoch) -> dict:
        """
        Evaluate model performance
        """
        top1 = self.top1()
        top5 = self.top5()
        with torch.no_grad():
            for images, target in loader:
                output = model(images.cuda(self.gpu, non_blocking=True))
                acc1, acc5 = self.accuracy(
                    output, target.cuda(self.gpu, non_blocking=True), topk=(1, 5)
                )
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
        best_acc.top1 = max(best_acc.top1, top1.avg)
        best_acc.top5 = max(best_acc.top5, top5.avg)
        stats = dict(
            epoch=epoch,
            acc1=top1.avg,
            acc5=top5.avg,
            best_acc1=best_acc.top1,
            best_acc5=best_acc.top5,
        )

        return stats

    def top1(self):
        return AverageMeter("Acc@1")

    def top5(self):
        return AverageMeter("Acc@5")

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

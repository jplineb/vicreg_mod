import pytest
import torch
from utils.construct_model import LoadVICRegModel
from utils.training_loop import TrainingLoop
from custom_datasets import DATASETS
import torch.optim as optim

@pytest.fixture
def setup_environment():
    # Simulate environment setup
    gpu = torch.cuda.current_device()
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    return gpu

@pytest.fixture
def setup_args():
    # Simulate argument parsing
    class Args:
        task_ds = "chexpert"
        pretrained_path = "../VICReg_ImageNet/resnet50.pth"
        arch = "resnet50"
        epochs = 2
        batch_size = 16
        lr_backbone = 0.0
        lr_head = 0.03
        weight_decay = 1e-6
        weights = "finetune"
        workers = 4
    return Args()

def test_training_loop(setup_environment, setup_args):
    gpu = setup_environment
    args = setup_args

    # Load dataset and dataloader
    dataset = DATASETS[args.task_ds](
        batch_size=args.batch_size,
        num_workers=args.workers,
        gpu=gpu,
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

    # Setup optimizer and scheduler
    param_groups = [dict(params=model[-1].parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=model[:-1].parameters(), lr=args.lr_backbone))
    optimizer = optim.Adam(param_groups, 0, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Initialize and run training loop
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

    # Add assertions to verify expected outcomes
    assert model is not None
    assert optimizer is not None
    assert scheduler is not None
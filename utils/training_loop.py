import os
import time
import json
import numpy as np
import torch
import torch.optim as optim

from datetime import datetime

from utils.logging import configure_logging
from utils.patches import log_stats

logger = configure_logging()

class TrainingLoop:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        dataset_handler,
        args,
        stats_file=None,
        gpu=0,
        multi_label: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_handler = dataset_handler
        self.args = args
        self.stats_file = stats_file
        self.gpu = gpu
        self.multi_label = multi_label
        self.start_time = time.time()

        if not os.path.exists(self.args.exp_dir):
            os.makedirs(self.args.exp_dir)
        
    def train_epoch(self, epoch):
        """Run one epoch of training"""
        self.model.train()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"Gradients computed for {name}")

        for step, data in enumerate(
            self.train_loader, start=epoch * len(self.train_loader)
        ):
            images = data["img"]
            target = data["lab"]

            output = self.model(images.cuda(self.gpu, non_blocking=True))
            loss = self.dataset_handler.calculate_loss(
                predictions=output, targets=target
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.args.print_freq == 0:
                pg = self.optimizer.param_groups
                lr_head = pg[0]["lr"]
                lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                stats = dict(
                    epoch=epoch,
                    step=step,
                    lr_backbone=lr_backbone,
                    lr_head=lr_head,
                    loss=loss.item(),
                    time=int(time.time() - self.start_time),
                )
                log_stats(stats)
                logger.info(json.dumps(stats))
                if self.stats_file:
                    logger.info(json.dumps(stats), file=self.stats_file)

            del images
            del target

    def evaluate(self, epoch: int):
        """Run evaluation on validation set"""
        self.model.eval()
        logger.info(f"Starting Eval {datetime.now()}")

        all_outputs = []
        all_targets = []
        all_valid_loss = []

        with torch.no_grad():
            for data in self.val_loader:
                images = data["img"]
                target = data["lab"]
                output = self.model(images.cuda(self.gpu, non_blocking=True))
                outputs = torch.sigmoid(output)
                valid_loss = self.dataset_handler.calculate_loss(
                    predictions=outputs, targets=target
                )
                target = target.nan_to_num(0)
                all_outputs += outputs.cpu()
                all_targets += target.cpu()
                all_valid_loss.append(valid_loss.item())

        if self.multi_label:
            all_auc, avg_auc_all = (
                self.calculate_multi_label_validation_stats(
                    epoch, all_outputs, all_targets, all_valid_loss
                )
            )
        else:
            all_auc, avg_auc_all = (
                self.calculate_single_label_validation_stats(
                    epoch, all_outputs, all_targets, all_valid_loss
                )
            )
        del all_outputs
        del all_targets

        return all_auc, avg_auc_all
    
    def calculate_single_label_validation_stats(
        self, epoch: int, all_outputs, all_targets, all_valid_loss
    ):
        """Calculate and log validation statistics"""
        all_auc, avg_auc_all = (
            self.dataset_handler.calculate_auc(all_outputs, all_targets)
        )
        avg_valid_loss = np.average(all_valid_loss)

        stats = dict(
            epoch=epoch,
            all_auc=all_auc.tolist(),
            avg_auc=avg_auc_all.tolist(),
            validation_loss=avg_valid_loss,
        )

        log_stats(
            {
                "epoch": epoch,
                "avg_auc": stats["avg_auc"],
                "all_auc": stats["all_auc"],
                "validation_loss": stats["validation_loss"],
            }
        )

        logger.info(json.dumps(stats))
        if self.stats_file:
            logger.info(json.dumps(stats), file=self.stats_file)

        return all_auc, avg_auc_all
        

    def calculate_multi_label_validation_stats(
        self, epoch: int, all_outputs, all_targets, all_valid_loss
    ):
        """Calculate and log validation statistics"""
        all_auc, avg_auc_all, avg_auc_of_interest, auc_dict = (
            self.dataset_handler.calculate_auc(all_outputs, all_targets)
        )
        avg_valid_loss = np.average(all_valid_loss)
        stats = dict(
            epoch=epoch,
            all_auc=auc_dict,
            avg_auc=float(avg_auc_all),
            avg_auc_of_interest=float(avg_auc_of_interest),
            validation_loss=avg_valid_loss,
        )

        log_stats(
            {
                "epoch": epoch,
                **auc_dict,
                "avg_auc": stats["avg_auc"],
                "avg_auc_of_interest": stats["avg_auc_of_interest"],
                "validation_loss": stats["validation_loss"],
            }
        )

        logger.info(json.dumps(stats))
        if self.stats_file:
            logger.info(json.dumps(stats), file=self.stats_file)

        return all_auc, avg_auc_all

    def train(self, start_epoch: int, num_epochs: int):
        """Main training loop"""
        for epoch in range(start_epoch, num_epochs):
            logger.info("Beginning training")
            self.train_epoch(epoch)
            all_auc, avg_auc_all = self.evaluate(epoch)
            self.scheduler.step()

            state = dict(
                epoch=epoch + 1,
                all_auc=all_auc.tolist(),
                best_auc=avg_auc_all.tolist(),
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
            )
            torch.save(state, os.path.join(self.args.exp_dir, "checkpoint.pth"))

def create_scheduler(optimizer, total_epochs, warmup_epochs) -> optim.lr_scheduler.SequentialLR:
    """
    Create a scheduler that warms up the learning rate linearly for a few epochs,
    then decays it using a cosine schedule.
    """
    warmup_epochs = warmup_epochs
    
    # Create warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4,  # Start at 0.01% of base lr
        end_factor=1.0,     # End at 100% of base lr
        total_iters=warmup_epochs
    )
    
    # Create cosine scheduler for after warmup
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6 # minimum learning rate
    )
    
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

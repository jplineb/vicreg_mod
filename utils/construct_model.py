from typing import OrderedDict
import torch
import torch.nn as nn
import resnet
from torchvision.models import ResNet50_Weights, resnet50
import resnet

from utils.log_config import configure_logging

logger = configure_logging()

class ConstructModel:
    def __init__(self, pretrained_weights=None):
        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)
        else:
            self.backbone = None
            self.head = None

    def modify_head(self, num_classes):
        if self.head is not None and self.backbone is not None:
            self.head.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def modify_backbone(self, pretrained_weights):
        if self.backbone is not None:
            self.backbone.load_state_dict(torch.load(pretrained_weights))

    def frankenstien_model(self, late_model, cut_off_layer) -> torch.nn.Module:
        """
        Cut off early model at a certain layer and add late model on top.
        Similar to evaluate_vindrcxr.py implementation.

        Args:
            early_model: First model to take early layers from
            late_model: Second model to take later layers from
            cut_off_layer: Which layer number to cut at

        Returns:
            torch.nn.Module: Combined model
        """
        # Slice early model up to cut_off_layer
        early_slice = nn.Sequential(
            OrderedDict([*(list(self.backbone.named_children())[:cut_off_layer])])
        )

        # Slice late model after cut_off_layer
        late_slice = nn.Sequential(
            OrderedDict([*(list(late_model.named_children())[cut_off_layer + 1 :])])
        )

        # Combine the two slices
        self.backbone = nn.Sequential(early_slice, late_slice)

        return self.backbone

    def load_pretrained_weights(self, pretrained_weights, device=None):
        """Load pretrained weights into the backbone model.

        Args:
            pretrained_weights (str): Path to pretrained weights file
            device (torch.device, optional): Device to load weights to. Defaults to None.

        Returns:
            torch.nn.Module: Model with loaded weights
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        state_dict = torch.load(pretrained_weights, map_location=device)
        self.backbone.load_state_dict(state_dict)
        return self.backbone


class LoadVICRegModel:
    def __init__(self, arch):
        print("Loading local VICReg ResNet50 arch Model")
        self.backbone, self.embedding = resnet.__dict__[arch](zero_init_residual=True)
        logger.info("Training VICReg Model")

    def load_pretrained_weights(self, pretrained_path: str):
        # Get state dict
        state_dict = torch.load(pretrained_path, map_location="cpu")
        # Handle weights from distributed training
        if "model" in state_dict:
            print("Loading model from state_dict")
            state_dict = state_dict["model"]
            state_dict = {
                key.replace("module.backbone.", ""): value
                for (key, value) in state_dict.items()
            }
        # Finally load the weights
        self.backbone.load_state_dict(state_dict, strict=False)

    def modify_head(self, num_classes: int):
        self.head = nn.Linear(self.embedding, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()

    def freeze_backbone(self):
        """
        Freezes the backbone parameters to prevent them from being updated during training.
        """
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(True)

    def produce_model(self):
        return nn.Sequential(self.backbone, self.head)

class LoadSupervisedModel:
    def __init__(self, arch='resnet50'):
        """
        Initializes the supervised pretrained model.

        Args:
            arch (str): Architecture name. Default is 'resnet50'
        """
        self.arch = arch
        self.backbone = None
        self.head = None
        logger.info("Training Supervised Model")

    def load_pretrained_weights(self, pretrained_dataset: str, pretrained_path: str | None = None):
        if pretrained_dataset == "ImageNet":
            print("Initializing ResNet50 Model with ImageNet Weights")
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
            self.backbone.fc = nn.Identity()
        elif pretrained_dataset == "RadImageNet":
            if pretrained_path is None:
                raise ValueError(f"Must specify pretrained path for {pretrained_dataset}")
            print("Initializing ResNet50 Model with RadImageNet Weights")
            pretrained_state_dict = torch.load(pretrained_path, map_location="cpu")
            self.backbone = resnet50(weights=pretrained_state_dict["state_dict"])
            # self.backbone.load_state_dict(pretrained_state_dict["state_dict"])
            self.backbone.fc = nn.Identity()
        # else:
        #     raise ValueError(f"Unsupported pretrained_dataset: {pretrained_dataset}")

    def modify_head(self, num_classes: int):
        """
        Modifies the head of the model for the desired number of classes.

        Args:
            num_classes (int): Number of output classes.
        """
        self.head = nn.Linear(2048, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()

    def freeze_backbone(self):
        """
        Freezes the backbone parameters to prevent them from being updated during training.
        """
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(True)
        # TODO: Fix this check
        # Set assert to check that the backbone is frozen
        # assert self.backbone.requires_grad_ == False
        # assert self.head.requires_grad_ == True
        
    def produce_model(self) -> nn.Module:
        """
        Combines the backbone and head into a single model.

        Returns:
            torch.nn.Module: The combined model.
        """
        if self.head is None:
            raise ValueError("Head is not initialized. Call modify_head() first.")
        model = nn.Sequential(self.backbone, self.head)
        return model
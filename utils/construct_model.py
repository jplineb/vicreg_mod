from typing import OrderedDict
import torch
import torch.nn as nn


class ConstructModel:
    def __init__(self, backbone, head):
        self.backbone = backbone
        self.head = head

    def modify_head(self, num_classes):
        self.head.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def modify_backbone(self, pretrained_weights):
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

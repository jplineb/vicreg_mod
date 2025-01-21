

from typing import Tuple, Union


class Input2dSpec(object):
    """Defines the specs for 2d inputs."""

    input_type = "2d"

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        in_channels: int,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels

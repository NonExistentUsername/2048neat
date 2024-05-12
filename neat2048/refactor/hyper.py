import itertools
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


def scale_value(value: float, max_value: float) -> float:
    return 2 * (value / (max_value - (max_value % 2 == 0))) - 1


def scale_position(position: list[float], layers: List[Tuple[int, int]]) -> list[float]:
    new_position: list[float] = []

    for index in range(0, len(position), 2):
        size_x, size_y = layers[index // 2]
        new_position.extend(
            (
                position[index] / (size_x - size_x % 2),
                position[index + 1] / (size_y - size_y % 2),
            )
        )

    return new_position


def create_sequential(
    layers: List[Tuple[int, int]],
    cppn,
    activation_threshold: Optional[float] = None,
    activation_function: Callable = nn.ReLU,
) -> nn.Sequential:
    # Create linear layers
    linear_layers: List[nn.Linear] = []
    for index in range(len(layers) - 1):
        input_size = layers[index][0] * layers[index][1]
        output_size = layers[index + 1][0] * layers[index + 1][1]

        linear_layers.append(nn.Linear(input_size, output_size))

    # Init weights using cppn
    with torch.no_grad():
        for layer_bottom_index in range(len(layers) - 1):
            layer_top_index = layer_bottom_index + 1
            bottom_layer, top_layer = (
                layers[layer_bottom_index],
                layers[layer_top_index],
            )
            bottom_layer_size_x, bottom_layer_size_y = bottom_layer
            top_layer_size_x, top_layer_size_y = top_layer

            for bottom_position in itertools.product(
                range(bottom_layer[0]), range(bottom_layer[1])
            ):
                bottom_position_x, bottom_position_y = bottom_position

                for top_position in itertools.product(
                    range(top_layer[0]), range(top_layer[1])
                ):
                    top_position_x, top_position_y = top_position

                    position = [
                        scale_value(bottom_position_x, bottom_layer_size_x),
                        scale_value(bottom_position_y, bottom_layer_size_y),
                        scale_value(top_position_x, top_layer_size_x),
                        scale_value(top_position_y, top_layer_size_y),
                    ]
                    weights = cppn(position)
                    weight = weights[layer_bottom_index]

                    pos_a = bottom_position_x + bottom_position_y * bottom_layer_size_x
                    pos_b = top_position_x + top_position_y * top_layer_size_x
                    linear_layer = linear_layers[layer_bottom_index]
                    linear_layer.weight[pos_b, pos_a] = (
                        weight
                        if activation_threshold is None
                        or weight >= activation_threshold
                        else 0.0
                    )

    s_layers = []
    for linear_layer in linear_layers:
        s_layers.extend([linear_layer, activation_function()])

    return nn.Sequential(*s_layers)


class HyperNetwork(nn.Module):
    def __init__(
        self,
        layers: List[Tuple[int, int]],
        cppn: Callable[[list[float]], list[float]],
    ):
        super().__init__()
        self.net = create_sequential(
            layers, cppn, activation_threshold=None, activation_function=nn.SELU
        )

    def forward(self, input: list[list[float]]):
        np_input = np.array(input, dtype=np.float32)  # type: ignore
        tensor_input = torch.from_numpy(np_input)
        result: torch.Tensor = self.net(tensor_input)
        np_result = result.detach().numpy()
        np_result = np_result.reshape(np.shape(input)[0], -1)
        return np_result

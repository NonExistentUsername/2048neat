from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


class HyperNetwork:
    def __init__(
        self,
        first_layer: list[int],  # (size_x, size_y, size_z)
        hidden_layer: list[int],  # (size_x, size_y, size_z) or (size_x, size_y)
        output_layer: list[int],  # (size_x)
        cppn: Callable[[list[float]], list[float]],
    ):
        # write sigmoid function using numpy
        # self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        # Relu
        self.activation_function = lambda x: np.maximum(x, 0)

        self.first_layer_mat = np.zeros(
            (np.prod(hidden_layer), np.prod(first_layer)), dtype=np.float32
        )
        self.hidden_layer_mat = np.zeros(
            (np.prod(output_layer), np.prod(hidden_layer)), dtype=np.float32
        )

        self.init_weights(first_layer, hidden_layer, output_layer, cppn)

    # def _rec_init_weights(
    #     self,
    #     layers: list[list[int]],
    #     cppn: Callable[[list[float]], list[float]],
    #     index: int = 0,
    #     inputs: Optional[list[int]] = None,
    # ):
    #     if inputs is None:
    #         inputs = []

    #     layer = layers[index]  # (size_x, size_y, size_z)

    #     for x in range(layer[0]):
    #         for y in range(layer[1]):
    #             if len(layer) == 3:
    #                 for z in range(layer[2]):
    #                     inputs.append(x)
    #                     inputs.append(y)
    #                     inputs.append(z)

    #                     if index < len(layers) - 1:
    #                         self._rec_init_weights(layers, cppn, index + 1, inputs)
    #                     else:
    #                         weights = cppn(**inputs)

    #                         for layer_id in range(len(layers) - 1):
    #                             self.first_layer_mat[
    #                                 x * layer[1] + y,
    #                                 x * layer[1] + y,
    #                             ] = weights[layer_id]

    #                     inputs.pop()
    #                     inputs.pop()
    #                     inputs.pop()
    #             else:
    #                 inputs.append(x)
    #                 inputs.append(y)

    #                 if index < len(layers) - 1:
    #                     self._rec_init_weights(layers, cppn, index + 1, inputs)
    #                 else:
    #                     weights = cppn(inputs)

    #                     self.first_layer_mat[
    #                         x * layer[1] + y,
    #                         x * layer[1] + y,
    #                     ] = weights[0]

    #                     self.hidden_layer_mat[
    #                         x * layer[1] + y,
    #                         x * layer[1] + y,
    #                     ] = weights[1]

    #                 inputs.pop()
    #                 inputs.pop()

    def init_weights(
        self,
        first_layer: list[int],
        hidden_layer: list[int],
        output_layer: list[int],
        cppn: Callable[[list[float]], list[float]],
    ):
        first_layer_size = np.prod(first_layer)
        hidden_layer_size = np.prod(hidden_layer)
        output_layer_size = np.prod(output_layer)

        ### FIRST LAYER ###
        for fl_index in range(first_layer_size):
            inputs_fl = []
            _tmp = fl_index

            for dimension_size in first_layer:
                _tmp, index = divmod(_tmp, dimension_size)
                inputs_fl.append(index)
            ### FIRST LAYER ###

            ### HIDDEN LAYER ###
            for hl_index in range(hidden_layer_size):
                inputs_hl = []
                _tmp = hl_index

                for dimension_size in hidden_layer:
                    _tmp, index = divmod(_tmp, dimension_size)
                    inputs_hl.append(index)
                ### HIDDEN LAYER ###

                ### OUTPUT LAYER ###
                for ol_index in range(output_layer_size):
                    inputs_ol = []
                    _tmp = ol_index

                    for dimension_size in output_layer:
                        _tmp, index = divmod(_tmp, dimension_size)
                        inputs_ol.append(index)
                    ### OUTPUT LAYER ###

                    weights = cppn(
                        [
                            *inputs_fl,
                            *inputs_hl,
                            *inputs_ol,
                        ]
                    )

                    self.first_layer_mat[
                        hl_index,
                        fl_index,
                    ] = weights[0]

                    self.hidden_layer_mat[
                        ol_index,
                        hl_index,
                    ] = weights[1]

    def forward(self, input: list[float]):
        input = np.array(input, dtype=np.float32)
        hidden_layer_input = self.first_layer_mat @ input
        hidden_layer_output = self.activation_function(hidden_layer_input)
        output = self.hidden_layer_mat @ hidden_layer_output
        output = self.activation_function(output)
        return output

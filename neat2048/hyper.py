from dataclasses import dataclass

import neat
import torch


@dataclass
class LayerDescriptor:
    size_x: int
    size_y: int
    size_z: int

    @property
    def size(self) -> int:
        return self.size_x * self.size_y * self.size_z

    @property
    def sizes(self) -> tuple[int, int]:
        return (self.size_x, self.size_y, self.size_z)

    def id_to_coordinates(self, id: int) -> tuple[int, int, int]:
        if id >= self.size:
            raise ValueError("id is too big")
        if id < 0:
            raise ValueError("id is too small")

        z = id // (self.size_x * self.size_y)
        id -= z * self.size_x * self.size_y
        y = id // self.size_x
        x = id % self.size_x

        return (x, y, z)

    def normalize_coordinates(
        self, x: int, y: int, z: int
    ) -> tuple[float, float, float]:
        return (
            x / (self.size_x - 1),
            y / (self.size_y - 1),
            z / (self.size_z - 1),
        )

    def coordinates_to_id(self, x: int, y: int, z: int) -> int:
        return z * self.size_x * self.size_y + y * self.size_x + x


class Game2048Network(torch.nn.Module):
    def __init__(
        self,
        first_layer: LayerDescriptor,
        hidden_layer: LayerDescriptor,
        output_layer: LayerDescriptor,
        cppn: neat.nn.FeedForwardNetwork,
    ):
        super(Game2048Network, self).__init__()

        self.cppn = cppn  # It for weights initialization only
        # Input is coordinates of first square, second square and output
        # output have two squares of 64 neurons each
        # 7 inputs in total
        # (2 for first square, 2 for second square, 3 for output squares)
        # Output is the 4 weights for the connection between the 1 and 2 layer and
        # the 1 weight for the connection between the 2 and 3 layer
        # 5 outputs in total

        # self.first_l = LayerDescriptor(4, 4, 5)
        # self.hidden_l = LayerDescriptor(4, 4, 1)
        # self.output_l = LayerDescriptor(4, 1, 1)
        self.first_l = first_layer
        self.hidden_l = hidden_layer
        self.output_l = output_layer

        # First layer is 64 * 4 (piece type and color) = 256 neurons

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.first_l.size, self.hidden_l.size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_l.size, self.output_l.size),
            torch.nn.Sigmoid(),
        )

        self.init_weights()

    def _get_weights(
        self,
        inputs: list[float],
    ):
        return self.cppn.activate(inputs)

    def init_weights(self):
        ### FIRST LAYER ###
        for f_l_x in range(self.first_l.size_x):
            for f_l_y in range(self.first_l.size_y):
                for f_l_z in range(self.first_l.size_z):
                    ### FIRST LAYER ###

                    ### HIDDEN LAYER ###
                    for h_l_x in range(self.hidden_l.size_x):
                        for h_l_y in range(self.hidden_l.size_y):
                            for h_l_z in range(self.hidden_l.size_z):
                                ### HIDDEN LAYER ###

                                ### OUTPUT LAYER ###
                                for o_l_x in range(self.output_l.size_x):
                                    for o_l_y in range(self.output_l.size_y):
                                        o_l_z = 0

                                        ### OUTPUT LAYER ###

                                        # create inputs
                                        inputs = [
                                            (f_l_x - 1) / (self.first_l.size_x),
                                            (f_l_y - 1) / (self.first_l.size_y),
                                            (f_l_z - 1) / (self.first_l.size_z),
                                            (h_l_x - 1) / (self.hidden_l.size_x),
                                            (h_l_y - 1) / (self.hidden_l.size_y),
                                            (h_l_z - 1) / (self.hidden_l.size_z),
                                            (o_l_x - 1) / (self.output_l.size_x),
                                            (o_l_y - 1) / (self.output_l.size_y),
                                        ]

                                        outputs = self._get_weights(inputs)

                                        self.net[0].weight.data[
                                            self.hidden_l.coordinates_to_id(
                                                h_l_x, h_l_y, h_l_z
                                            ),
                                            self.first_l.coordinates_to_id(
                                                f_l_x, f_l_y, f_l_z
                                            ),
                                        ] = outputs[0]

                                        self.net[2].weight.data[
                                            self.output_l.coordinates_to_id(
                                                o_l_x, o_l_y, o_l_z
                                            ),
                                            self.hidden_l.coordinates_to_id(
                                                h_l_x, h_l_y, h_l_z
                                            ),
                                        ] = outputs[1]

    def forward(self, input):
        return self.net(input)

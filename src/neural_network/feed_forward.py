import random
from ai_math import RandomMatrix
from ai_math.matrix import Matrix


class FeedForwardNeuralNetworkLayer(object):
    def __init__(self, entry_size: int, output_size: int, spectre: float = 1) -> None:
        self.entry_size = entry_size
        self.output_size = output_size
        self.spectre = spectre
        self.bias = random.uniform(-1, 1)
        self.weigths = RandomMatrix(self.entry_size, self.output_size, spectre=spectre)

    def feed_forward(self, entry_matrix: Matrix) -> Matrix:
        return (entry_matrix * self.weigths) + self.bias


class FeedForwardNeuralNetwork(object):
    def __init__(self, topology: tuple) -> None:
        self.topology = topology
        self.layers = [
                FeedForwardNeuralNetworkLayer(
                    self.topology[i - 1],
                    self.topology[i]
                    ) 
                for i in range(len(self.topology)) if i != 0
                ]
        
    
    def feed_forward(self, entry_matrix: Matrix) -> Matrix:
        current_input  = entry_matrix

        for layer in self.layers:
            current_input = layer.feed_forward(current_input)

        return current_input


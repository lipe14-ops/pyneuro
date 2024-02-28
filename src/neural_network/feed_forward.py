import random
from ai_math import RandomMatrix
from ai_math.matrix import Matrix
from .activation_functions import ActivationFunction


class FeedForwardNeuralNetworkLayer(object):
    def __init__(self, entry_size: int, output_size: int, activation_function: ActivationFunction, spectre: float = 1) -> None:
        self.entry_size = entry_size
        self.output_size = output_size
        self.spectre = spectre
        self.bias = random.uniform(-1, 1)
        self.weigths = RandomMatrix(self.entry_size, self.output_size, spectre=spectre)
        self.activation_function = activation_function

    def feed_forward(self, entry_matrix: Matrix) -> Matrix:
        return self.activation_function.matrix_apply((entry_matrix * self.weigths) + self.bias)


class FeedForwardNeuralNetwork(object):
    def __init__(self, topology: tuple, activation_functions: list[ActivationFunction]) -> None:
        if len(topology) != len(activation_functions) + 1:
            raise ValueError("activations must contain n - 1 elements.")
        self.topology = topology
        self.activation_functions = activation_functions
        self.layers = [
                FeedForwardNeuralNetworkLayer(
                    self.topology[i - 1],
                    self.topology[i],
                    self.activation_functions[i - 1]
                    ) 
                for i in range(len(self.topology)) if i != 0
                ]
        
    
    def feed_forward(self, entry_matrix: Matrix) -> Matrix:
        current_input  = entry_matrix

        for layer in self.layers:
            current_input = layer.feed_forward(current_input)

        return current_input


    def cost(self, data: list[list[Matrix]]) -> float:
        return sum(
                pow(sum(round(value, 3) for value in (train_out - self.feed_forward(train_in)).values), 2) for (train_in, train_out) in data
                ) / len(data)


    def back_propagate(self, batch_size: int, data: list[list[Matrix]], learning_rate: float = 0.0001, log: bool = False) -> None:
        eps = 0.01  

        for _ in range(batch_size):
            for layer in self.layers:
                cost = self.cost(data)

                layer.bias += eps
                layer.bias -= learning_rate * ((self.cost(data) - cost) / eps)
                layer.bias -= eps

                for value in layer.weigths.values :
                    value += eps
                    value -= learning_rate * ((self.cost(data) - cost) / eps)
                    value -= eps

                print(f"batch: {_} cost: {cost}")


from ..ai_math import Matrix, RandomMatrix
from .activation_functions import ActivationFunction

class ConvolutionalNeuralNetworkLayer(object):
    def __init__(self, entry: tuple[int, int, int], output: tuple[int, int, int], activation_function: ActivationFunction, pool: tuple[int, int] = (1, 1)) -> None:
        self.neurons_amount_per_entry = output[2] / entry[2]

        if self.neurons_amount_per_entry * entry[2] != output[2]:
            raise ValueError("the output and input layer number must be multiples.")

        neuron_i = entry[0] - output[0] - pool[0] + 2
        neuron_j = entry[1] - output[1] - pool[1] + 2

        if neuron_j <= 0 and neuron_i <= 0:
            raise ValueError("the input dimensions must be smaller than the output dimensions.")

        self.kernels = [RandomMatrix(neuron_i, neuron_j) for _ in range(output[2])]
        self.pool = pool
        self.activation_function = activation_function

    def feed_forward(self, entry: list[Matrix]) -> list[Matrix]:
        output_layer = []

        for i, neuron in enumerate(self.kernels):
            entry_index = int(i % self.neurons_amount_per_entry)
            entry_matrix = entry[entry_index]
            
            out_matrix = entry_matrix.convolve(neuron)
            out_matrix = out_matrix.max_pooling(self.pool)
            out_matrix = self.activation_function.matrix_apply(out_matrix)

            output_layer.append(out_matrix)

        return output_layer


class ConvolutionalNeuralNetwork(object):
    ...


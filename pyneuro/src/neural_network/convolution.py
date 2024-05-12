from ..ai_math import Matrix, RandomMatrix
from .activation_functions import ActivationFunction
from .feed_forward import FeedForwardNeuralNetwork


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
        output_layers = []

        for neuron in self.kernels:
            for out_layer in entry:
                out_matrix = out_layer.convolve(neuron)
                out_matrix = out_matrix.max_pooling(self.pool)
                out_matrix = self.activation_function.matrix_apply(out_matrix)

                output_layers.append(out_matrix)

        return output_layers


class ConvolutionalNeuralNetwork(object):
    def __init__(self, entries_size: list[tuple[int, int, int]], pools: list[tuple[int, int]], activation_functions: list[ActivationFunction], feed_forward_nn: FeedForwardNeuralNetwork) -> None:
        self.entries_size = entries_size
        self.activation_functions = activation_functions
        self.pools = pools
        self.feed_forward_nn = feed_forward_nn
        self.layers = [
            ConvolutionalNeuralNetworkLayer(
                entry=entries_size[i - 1],
                output=entries_size[i],
                pool=pools[i - 1],
                activation_function=activation_functions[i - 1]
                )
            for i in range(1, len(self.entries_size))
                ]

    def feed_forward(self, entry: list[Matrix]) -> Matrix:
        flatten_layer = self.flatten_layer(entry)
        output_matrix = self.feed_forward_nn.feed_forward(flatten_layer)

        return output_matrix

    def flatten_layer(self, entry: list[Matrix]) -> Matrix:
        entry_matrix = entry

        for layer in self.layers:
            entry_matrix = layer.feed_forward(entry_matrix)

        output_matrix = Matrix(1, len(entry_matrix) * entry_matrix[0].rows * entry_matrix[0].cols)
        
        for i_matrix, matrix in enumerate(entry_matrix):
            for i_value, value in enumerate(matrix.values):
                output_matrix.set_value(0, i_matrix * matrix.rows * matrix.cols + i_value, value)

        return output_matrix 

    def backpropagate(self, batch_size: int, data: list[tuple[list[Matrix], Matrix]], learning_rate: float = 0.001, log: bool = False) -> None:

        feed_forward_dataset = []
        for entry, output in data:
            feed_forward_dataset.append([
                self.flatten_layer(entry), 
                output
            ])

        self.feed_forward_nn.back_propagate(
            batch_size=batch_size,
            data=feed_forward_dataset,
            learning_rate=learning_rate,
            log=log
        )

        offset = 10
        cost = 0
        for _ in range(batch_size):
            for layer_index in range(len(self.layers) - 1, -1, -1):
                for neuron_index in range(len(self.layers[layer_index].kernels)):
                    for value_index in range(len(self.layers[layer_index].kernels[neuron_index].values)):
                        cost = self.feed_forward_nn.cost(feed_forward_dataset)

                        self.layers[layer_index].kernels[neuron_index].values[value_index] += offset

                        feed_forward_dataset.clear()
                        for entry, output in data:
                            feed_forward_dataset.append([
                                self.flatten_layer(entry), 
                                output
                            ])

                        self.layers[layer_index].kernels[neuron_index].values[value_index] -= layer_index * (
                                self.feed_forward_nn.cost(feed_forward_dataset) - cost
                                ) 

                        self.layers[layer_index].kernels[neuron_index].values[value_index] -= offset

                        print(f"convolution batch: {_} cost:", cost)



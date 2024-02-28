from ai_math import Polynomial, Matrix, RandomMatrix
from neural_network import FeedForwardNeuralNetwork, LeakyReluActivationFunction, ReluActivationFunction, IdentityActivationFunction


data = [
    [Matrix(1, 2), Matrix(1, 1)],
    [Matrix(1, 2), Matrix(1, 1)],
    [Matrix(1, 2), Matrix(1, 1)],
    [Matrix(1, 2), Matrix(1, 1)],
        ]

data[0][0].set_value(0, 0, 0)
data[0][0].set_value(0, 1, 0)
data[0][1].set_value(0, 0, 0)

data[1][0].set_value(0, 0, 1)
data[1][0].set_value(0, 1, 0)
data[1][1].set_value(0, 0, 1)

data[2][0].set_value(0, 0, 0)
data[2][0].set_value(0, 1, 1)
data[2][1].set_value(0, 0, 0)

data[3][0].set_value(0, 0, 1)
data[3][0].set_value(0, 1, 1)
data[3][1].set_value(0, 0, 1)

def main() -> None:
    neural_network = FeedForwardNeuralNetwork(
            topology=(2, 10, 10, 1),
            activation_functions=[
                LeakyReluActivationFunction(0.01),
                LeakyReluActivationFunction(0.01),
                IdentityActivationFunction(),
                ]
            )

    neural_network.back_propagate(10000, data)

    while True:
        values = eval(input(">> "))
        entry = Matrix(1, len(values))
        for i, value in enumerate(values):
            entry.set_value(0, i, value)

        output = neural_network.feed_forward(entry)
        print(output.values)
            

if __name__ == "__main__":
    main()


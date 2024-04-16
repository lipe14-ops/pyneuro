from ai_math import Matrix
from neural_network import FeedForwardNeuralNetwork, LeakyReluActivationFunction, SoftMaxActivationFunction, ReluActivationFunction, IdentityActivationFunction


data = [
    [Matrix(1, 2), Matrix(1, 2)],
    [Matrix(1, 2), Matrix(1, 2)],
    [Matrix(1, 2), Matrix(1, 2)],
    [Matrix(1, 2), Matrix(1, 2)],
        ]

data[0][0].set_value(0, 0, 0)
data[0][0].set_value(0, 1, 0)
data[0][1].set_value(0, 0, 0)
data[0][1].set_value(0, 1, 1)

data[1][0].set_value(0, 0, 1)
data[1][0].set_value(0, 1, 0)
data[1][1].set_value(0, 0, 1)
data[1][1].set_value(0, 1, 0)

data[2][0].set_value(0, 0, 0)
data[2][0].set_value(0, 1, 1)
data[2][1].set_value(0, 0, 0)
data[2][1].set_value(0, 1, 1)

data[3][0].set_value(0, 0, 1)
data[3][0].set_value(0, 1, 1)
data[3][1].set_value(0, 0, 1)
data[3][1].set_value(0, 1, 0)

def main() -> None:
    neural_network = FeedForwardNeuralNetwork(
            topology=(2, 2, 2),
            activation_functions=[
                LeakyReluActivationFunction(0.1),
                SoftMaxActivationFunction()
                ]
            )

    neural_network.back_propagate(1000000, data, log=True)

    while True:
        values = eval(input(">> "))
        entry = Matrix(1, len(values))

        for i, value in enumerate(values):
            entry.set_value(0, i, value)

        output = neural_network.feed_forward(entry)
        print(output.values, sum(output.values))
            

if __name__ == "__main__":
    main()


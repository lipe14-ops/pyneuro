from ai_math import Polynomial, Matrix, RandomMatrix
from neural_network import FeedForwardNeuralNetwork, LeakyReluActivationFunction, ReluActivationFunction, IdentityActivationFunction

def main() -> None:
    matrix = RandomMatrix(4, 4, 31.5)

    matrix.set_value(0, 0, 6)
    matrix.set_value(0, 1, 1)
    matrix.set_value(0, 2, 1)
    matrix.set_value(1, 0, 4)
    matrix.set_value(1, 1, -2)
    matrix.set_value(1, 2, 5)
    matrix.set_value(2, 0, 2)
    matrix.set_value(2, 1, 8)
    matrix.set_value(2, 2, 7)

    print(matrix.trace())

    polynomial= Polynomial(
        degree=20,
    )

    value = 1
    output = polynomial.get_image_of(value);
    print(f"the image of {value} is: {output}")

    polynomial.set_coeficient(2, 487);
    output = polynomial.get_image_of(value);
    print(f"the image of {value} is: {output}")


    neural_network = FeedForwardNeuralNetwork(
            topology=(1, 110, 1000, 3),
            activation_functions=[
                ReluActivationFunction(),
                IdentityActivationFunction(),
                LeakyReluActivationFunction(0.01),
                ]
            )

    print(neural_network.layers)

    entry_matrix = Matrix(1, 1)

    output = neural_network.feed_forward(
                entry_matrix
            )

    print(output.values)


if __name__ == "__main__":
    main()


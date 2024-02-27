from abc import ABC, abstractmethod
from ai_math import Matrix

class ActivationFunction(ABC):

    @abstractmethod
    def get_image_of(self, x: int | float) -> int | float:
        ...

    @abstractmethod
    def matrix_apply(self, matrix: Matrix) -> Matrix:
        ...


class ReluActivationFunction(ActivationFunction):

    def get_image_of(self, x: int | float) -> int | float:
        return max(x, 0)
    
    def matrix_apply(self, matrix: Matrix) -> Matrix:
        tmp_matrix = Matrix(matrix.rows, matrix.cols)

        for i, value in enumerate(matrix.values):
            tmp_matrix.values[i] = self.get_image_of(value)

        return tmp_matrix

class LeakyReluActivationFunction(ActivationFunction):
    def __init__(self, coeficient: float) -> None:
        self.coeficient = coeficient

    def get_image_of(self, x: int | float) -> int | float:
        return max(x, x * self.coeficient)
    
    def matrix_apply(self, matrix: Matrix) -> Matrix:
        tmp_matrix = Matrix(matrix.rows, matrix.cols)

        for i, value in enumerate(matrix.values):
            tmp_matrix.values[i] = self.get_image_of(value)

        return tmp_matrix

 
class IdentityActivationFunction(ActivationFunction):

    def get_image_of(self, x: int | float) -> int | float:
        return x
    
    def matrix_apply(self, matrix: Matrix) -> Matrix:
        return matrix
 

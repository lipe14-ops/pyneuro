from __future__ import annotations
import random
 
class Matrix(object):
    def __init__(self, rows: int, cols: int) -> None:
        if (rows >= 0 or cols >= 0) and not isinstance(rows + cols, int):
            raise ValueError(f"matrix order must be composed by natural numbers.")

        self.rows = rows
        self.cols = cols
        self.values = [ 0.0 ] * (self.rows * self.cols)

    def get_value(self, row: int, col: int) -> float:
        if row >= self.rows or col >= self.cols:
            raise ValueError(f"the matrix order must be {self.rows}x{self.cols}, the element {row}x{col} is out of bound.")

        return self.values[row * self.cols + col]

    def set_value(self, row: int, col: int, value: float) -> None:
        if not isinstance(value, float | int):
            raise ValueError("the set value must be a number.")

        if row >= self.rows or col >= self.cols:
            raise ValueError(f"the matrix order must be {self.rows}x{self.cols}, the element {row}x{col} is out of bound.")

        self.values[row * self.cols + col] = value

    def transpose(self) -> Matrix:
        matrix = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                matrix.set_value(j, i, self.get_value(i, j))

        return matrix;

    def determinant(self) -> int | float:
        if self.rows != self.cols:
            raise ValueError("the matrix must be squared.")

        if self.rows == 1:
            return self.get_value(0, 0)

        if self.rows == 2:
            return self.get_value(0, 0) * self.get_value(1, 1) - self.get_value(0, 1) * self.get_value(1, 0)

        total = 0
        for j in range(self.cols):
            sub_matrix = Matrix(self.rows - 1, self.cols - 1)

            for row in range(1, self.rows):
                for col in range(self.cols):
                    if col == j: continue
                    value = self.get_value(row, col)

                    offset = 0 if col < j else + 1 
                    sub_matrix.set_value(row - 1, col - offset, value)

            total += self.get_value(0, j) * sub_matrix.determinant() * pow(-1, j)

        return total

    def trace(self) -> int | float:
        if self.rows != self.cols:
            raise ValueError("Matrix must be squared.")

        return sum(
                self.get_value(i, i) for i in range(self.rows)
                )

    def __radd__(self, other: Matrix | int | float) -> Matrix:
        matrix = Matrix(self.rows, self.cols)

        if isinstance(other, int | float): 
            for i, self_value in enumerate(self.values):
                matrix.values[i] =  other + self_value
            return matrix

        if not isinstance(other, Matrix):
            raise ValueError(f"invalid sum type.")

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"the matrices order must be equal.")

        for i, (self_value, other_value) in enumerate(zip(self.values, other.values)):
            matrix.values[i] = other_value + self_value 

        return matrix

    def __rsub__(self, other: Matrix | int | float) -> Matrix:
        matrix = Matrix(self.rows, self.cols)

        if isinstance(other, int | float): 
            for i, self_value in enumerate(self.values):
                matrix.values[i] = other - self_value
            return matrix

        if not isinstance(other, Matrix):
            raise ValueError(f"invalid sum type.")

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"the matrices order must be equal.")

        for i, (self_value, other_value) in enumerate(zip(self.values, other.values)):
            matrix.values[i] = other_value - self_value

        return matrix

    def __rmul__(self, other: Matrix | int | float) -> Matrix:
        if isinstance(other, int | float): 
            matrix = Matrix(self.rows, self.cols)
            for i, self_value in enumerate(self.values):
                matrix.values[i] = self_value * other

            return matrix

        if not isinstance(other, Matrix):
            raise ValueError(f"invalid sum type.")
        
        if self.rows != other.cols:
            raise ValueError(f"incompatible matrices dimentions.")

        matrix = Matrix(other.cols, self.rows)

        for i in range(matrix.rows):
            for j in range(matrix.cols):
                for k in range(other.rows):
                    v = matrix.get_value(i, j) + self.get_value(i, k) * other.get_value(k, j)
                    matrix.set_value(i, j, v)

        return matrix

    def __add__(self, other: Matrix | int | float) -> Matrix:
        matrix = Matrix(self.rows, self.cols)

        if isinstance(other, int | float): 
            for i, self_value in enumerate(self.values):
                matrix.values[i] = self_value + other
            return matrix

        if not isinstance(other, Matrix):
            raise ValueError(f"invalid sum type.")

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"the matrices order must be equal.")

        for i, (self_value, other_value) in enumerate(zip(self.values, other.values)):
            matrix.values[i] = self_value + other_value

        return matrix

    def __sub__(self, other: Matrix | int | float) -> Matrix:
        matrix = Matrix(self.rows, self.cols)

        if isinstance(other, int | float): 
            for i, self_value in enumerate(self.values):
                matrix.values[i] = self_value - other
            return matrix

        if not isinstance(other, Matrix):
            raise ValueError(f"invalid sum type.")

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"the matrices order must be equal.")

        for i, (self_value, other_value) in enumerate(zip(self.values, other.values)):
            matrix.values[i] = self_value - other_value

        return matrix

    def __mul__(self, other: Matrix | int | float) -> Matrix:
        if isinstance(other, int | float): 
            matrix = Matrix(self.rows, self.cols)
            for i, self_value in enumerate(self.values):
                matrix.values[i] = self_value * other

            return matrix

        if not isinstance(other, Matrix):
            raise ValueError(f"invalid sum type.")
        
        if self.cols != other.rows:
            raise ValueError(f"incompatible matrices dimentions.")

        matrix = Matrix(self.rows, other.cols)

        for i in range(matrix.rows):
            for j in range(matrix.cols):
                for k in range(other.rows):
                    v = matrix.get_value(i, j) + self.get_value(i, k) * other.get_value(k, j)
                    matrix.set_value(i, j, v)

        return matrix

    def __div__(self, other: int | float) -> Matrix:
        matrix = Matrix(self.rows, self.cols)
        for i, self_value in enumerate(self.values):
            matrix.values[i] = self_value / other

        return matrix

    def __iadd__(self, other: Matrix | int | float) -> Matrix:
        return self.__add__(other)

    def __isub__(self, other: Matrix | int | float) -> Matrix:
        return self.__sub__(other)

    def __imul__(self, other: Matrix | int | float) -> Matrix:
        return self.__mul__(other)

    def __idiv__(self, other: int | float) -> Matrix:
        return self.__div__(other)

    def convolve(self, matrix: Matrix) -> Matrix:
        if not (self.rows >= matrix.rows and self.cols >= matrix.cols):
            raise ValueError(f"the input matrix must have the max order {self.rows}x{self.cols}.")

        tmp_matrix = Matrix(self.rows - matrix.rows + 1, self.cols - matrix.cols + 1)

        for conv_i in range(tmp_matrix.rows):
            for conv_j in range(tmp_matrix.cols):
                accumulator = 0

                for matrix_i in range(matrix.rows):
                    for matrix_j in range(matrix.cols):
                        x = matrix_i + conv_i
                        y = matrix_j + conv_j
                        accumulator += self.get_value(x, y) * matrix.get_value(matrix_i, matrix_j)

                tmp_matrix.set_value(conv_i, conv_j, accumulator)
            
        return tmp_matrix


class RandomMatrix(Matrix):
    def __init__(self, rows: int, cols: int, spectre: int | float = 1) -> None:
        super().__init__(rows, cols)
        self.values = [ random.uniform(-spectre, spectre) for _ in range(self.rows * self.cols)]


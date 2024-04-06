from __future__ import annotations

class Vector(object):
    def __init__(self, values: list[float | int]) -> None:
        self.values = values

    def module(self) -> float | int:
        return pow(
                sum(
                    pow(value, 2) for value in self.values
                    ), 0.5)

    def normalize(self) -> Vector:
        module = self.module()
        return Vector(
                values=[ value / module for value in self.values]
                )

    def dot_product(self, vector: Vector) -> float | int:
        if len(self.values) != len(vector.values):
            raise ValueError("vectors must have the same dimensionality.")

        return sum(
            v1 * v2 for v1, v2 in zip(self.values, vector.values)
                )

    def cross_product(self, other: Vector) -> Vector:
        if len(self.values) != len(other.values):
            raise ValueError("vectors must have the same dimensionality.")

        dimension = len(self.values)
        out_vec = Vector([])

        for i in range(dimension):

            j, k = 0, 1

            if i == 0:
                j, k = 1, 2

            elif i == 1:
                j, k = 2, 0
                    
            out_vec.values.append(
                    self.values[j] * other.values[k] - self.values[k] * other.values[j]
                    )

        return out_vec

    def __add__(self, other:  Vector) -> Vector: 
        if not isinstance(other, Vector):
            raise ValueError("the value must be vector type.")

        out_vec = Vector([])
        for v1, v2 in zip(self.values, other.values):
            out_vec.values.append(v1 + v2)

        return out_vec


    def __sub__(self, other:  Vector) -> Vector: 
        if not isinstance(other, Vector):
            raise ValueError("the value must be vector type.")

        out_vec = Vector([])
        for v1, v2 in zip(self.values, other.values):
            out_vec.values.append(v1 - v2)

        return out_vec

    def __mul__(self, other: int | float) -> Vector:
        if not isinstance(other, int) or not isinstance(other, float):
            raise ValueError("the vector must be multiplied by a scalar value.")

        return Vector(
                values = [ value * other for value  in self.values]
                )


    def __isub__(self, other:  Vector) -> Vector: 
        return self.__sub__(other)

    def __isum__(self, other:  Vector) -> Vector: 
        return self.__add__(other)

    def __imul__(self, other: int | float) -> Vector:
        return self.__mul__(other)


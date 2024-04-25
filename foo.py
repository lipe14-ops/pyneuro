from pyneuro import Matrix, RandomMatrix

matrix_1 = Matrix(4, 4)
matrix_2 = RandomMatrix(3, 3)

matrix_1.set_value(0, 0, 1)
matrix_1.set_value(0, 1, 2)
matrix_1.set_value(0, 2, 3)
matrix_1.set_value(0, 3, 4)

matrix_1.set_value(1, 0, 5)
matrix_1.set_value(1, 1, 6)
matrix_1.set_value(1, 2, 7)
matrix_1.set_value(1, 3, 8)

matrix_1.set_value(2, 0, 9)
matrix_1.set_value(2, 1, 10)
matrix_1.set_value(2, 2, 11)
matrix_1.set_value(2, 3, 12)

matrix_1.set_value(3, 0, 13)
matrix_1.set_value(3, 1, 14)
matrix_1.set_value(3, 2, 15)
matrix_1.set_value(3, 3, 16)

matrix_3 = matrix_1.convolve(matrix_2)
print(matrix_3.values)

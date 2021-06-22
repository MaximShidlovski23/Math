import numpy as np
import copy

def nummatrix(num, matrix):
    line = len(matrix)
    column = len(matrix[0])
    for i in range(0, line):
        for j in range(0, column):
            matrix[i][j] *= num
    return matrix

def matrixmult(matrix1, matrix2):
    sum = 0
    time_matrix = []
    res = []
    if len(matrix2) != len(matrix1[0]):
        print("Матрицы не могут быть перемножены")
    else:
        line1 = len(matrix1)
        column1 = len(matrix1[0])
        line2 = column1
        column2 = len(matrix2[0])
        for i in range(0, line1):
            for j in range(0, column2):
                for k in range(0, column1):
                    sum = round(sum + matrix1[i][k] * matrix2[k][j], 2)
                time_matrix.append(sum)
                sum = 0
            res.append(time_matrix)
            time_matrix = []
        return res


def invertible_matrix(array, array_1, x, i):
    n = len(x)
    #l = matrixmult(array_1, x)
    l = np.matmul(array_1, x)
    print("l = \n", l)
    if l[i][0] == 0:
        print("Матрица необратима l[i] = 0.")
        sys.exit()
    #l_v - l с волной
    l_v = l.copy()
    l_v[i] = -1
    #l_sh - l с шапкой
    l_sh = (-1 / l[i][0]) * l_v
    P = np.eye(n)
    for k in range(0, n):
        P[k][i] = l_sh[k]
    res_array_1 = np.matmul(P, array_1)
    #res_array_1 = matrixmult(P, array_1)
    print("res_array_1 = \n", res_array_1)
    #res_array = np.array([[2., 1., 4.], [1., 3., 5.], [5., 5., 4.]])
    #res_array = np.array([[1., 1., 7.], [2., 3., 8.], [3., 5., 10.]])
    return res_array_1

def check(array, x, i, res_array_1):
    n = len(x)
    res_array = array.copy()
    for k in range(0, n):
        res_array[k][i] = x[k][0]
    print("res_array = \n", res_array)
    print("Проверка на правильность нахождения обратной матрицы: res_array * res_array_1 = E")
    print(matrixmult(res_array, res_array_1))

def main():
    #array = np.array([[2., 3., 4.], [1., 3., 5.], [5., 6., 4.]])
    #array_1 = np.linalg.inv(array)
    #x = np.array([[1.], [3.], [5.]])
    array = np.array([[1., 4., 7.], [2., 5., 8.], [3., 6., 10.]])
    array_1 = np.linalg.inv(array)
    x = np.array([[1.], [3.], [5.]])
    i = 1
    res_array_1 = invertible_matrix(array, array_1, x, i)
    check(array, x, i, res_array_1)


if __name__ == "__main__":
    main()

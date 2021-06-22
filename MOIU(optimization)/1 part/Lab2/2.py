import numpy as np
import copy
import sys
from Lab1 import *

def find_first(pred, iterable):
    return next(filter(pred, iterable), None)

def get_matrix_and_invetible_simple(A_dop, j_b):
    len_j = len(j_b)
    j_b_list = list(j_b)
    A_b = np.zeros((len_j, len_j))
    for i in range(len_j):
        for j in j_b:
            A_b[i][j_b_list.index(j)] = A_dop[i][j]
    if np.linalg.det(A_b) == 0:
        print('Нет оптимального плана')
        sys.exit()
    A_b_1 = np.linalg.inv(A_b)
    return A_b, A_b_1

def get_matrix_and_invertible(j_b, JB, m, A_dop, A_b, A_b_1):
    #i - позиция для вставки
    j_i = list(set(j_b) - set(JB))
    new_column = np.zeros((m, 1))
    for i in range(m):
        new_column[i][list(j_b).index(j_i)] = A_dop[i][list(j_b).index(j_i)]
    A_b_1 = invertible_matrix(A_b, A_b_1, new_column, j_i)
    A_b = matrix(A_b, new_column, j_i, A_b_1)
    return A_b, A_b_1


def initial_simplex_method(A, b, c):
    m = len(A)
    n = len(A[0])
    c_dop = c.tolist() + [-1] * m
    c_dop = np.array(c_dop)
    A_dop = np.zeros((m, m + n))
    A_dop[:, :n] = A[:, :]
    for i in range(m):
        A_dop[i, i + n] = 1
    x = np.zeros(n + len(b))
    for i in range(len(b)):
        x[i + n] = b[i]
    j_b = np.zeros(m, dtype=int)
    for i in range(m):
        j_b[i] = i + n
    A_b, A_b_1 = get_matrix_and_invetible_simple(A_dop, j_b)
    J_B = copy.copy(j_b)
    x, j_b, theta_0_index = search_for_one_iteration(A_b, A_b_1, x, j_b, A_dop, c_dop, b)
    #x, j_b = search_optimal_plan(x, j_b, A_b, A_b_1, theta_0_index, A_dop, c_dop, b)
    for i in range(n, n + m):
        if x[i] != 0:
            print('ЗАДАЧА НЕ СОВМЕСТНА!!!')
            sys.exit()
    print('ЗАДАЧА СОВМЕСТНА!!!')
    while True:
        j_k = None
        for j in j_b:
            if j > n - 1:
                j_k = j
                break
        if j_k is None:
            break
        j_b_list = list(j_b)
        j_k_index = j_b_list.index(j_k)
        j_not_b = set(j_b) - set(range(n, n + m))
        j_not_b = list(set(range(n)) - j_not_b)
        L = []
        A_b, A_b_1 = get_matrix_and_invertible(j_b, J_B, m, A_dop, A_b, A_b_1)
        for j in j_not_b:
            L.append(np.matmul(A_b_1, A_dop[:, j]))
        if all(l[j_k_index] == 0 for l in L):
            A_dop = np.delete(A_dop, j_k_index, 0)
            j_b = np.delete(j_b, j_b_list.index(j_k))
        else:
            j_b[j_k_index] = j_not_b[0]
    return x[:n], j_b

def main():
    A = np.array([[1.0, 2.0, 1.0, 1.0],
                  [2.0, 3.0, 1.0, 1.0]])
    b = np.array([[1.0], [2.0]])
    c = np.array([1.0, 1.0, 0.0])
    x, j = initial_simplex_method(A, b, c)
    print('='*100)
    print('x = ', x)
    print('j = ', j)
if __name__ == '__main__':
    main()
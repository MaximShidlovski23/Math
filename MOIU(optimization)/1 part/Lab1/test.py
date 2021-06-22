import numpy as np
import copy
import sys


c = np.array([1, 2, 0, 0, 0, 0])
A = np.array([[-1, 2, 1, 0, 0, 0],
           [1, 1, 0, -4, 0, 0],
           [1, -1, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 1]])
b = np.array([[2], [4], [2],[6]])


#где x - новый столбец
def invertible_matrix(array, array_1, x, i):
    n = len(x)
    l = np.matmul(array_1, x)
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
    return res_array_1


def matrix(array, x, i, res_array_1):
    n = len(x)
    res_array = array.copy()
    for k in range(0, n):
        res_array[k][i] = x[k][0]
    return res_array


def swap(x1, x2):
    temp = x1
    x1 = x2
    x2 = temp
    return x1, x2


def search_optimal_plan(x, j_b, A_b, A_b_1, theta_0_index):
    len_j = len(j_b)
    while True:
        len_A = len(A)
        len_j = len(j_b)
        diff = len_A - len_j
        A_b = np.zeros((len_j, len_j))
        j_b_list = list(j_b)
        for i in range(len_j):
            for j in j_b:
                A_b[i][j_b_list.index(j)] = A[i][j - 1]
        det_A_b = np.linalg.det(A_b)
        print(det_A_b)
        if det_A_b == 0:
            print('STOP')
            sys.exit()
        A_b_1 = np.linalg.inv(A_b)
        print(A_b_1)
        print(A_b_1, 'A_b_1')
        x, j_b, theta_0_index = search_for_one_iteration(A_b, A_b_1, x, j_b)


def search_for_one_iteration(A_b, A_b_1, x, j_b):
    j_b_list = list(j_b)
    len_j = len(j_b)
    c_b_ch = np.zeros(len_j)
    for i in j_b:
        c_b_ch[j_b_list.index(i)] = c[i - 1]
    u_ch = np.matmul(c_b_ch, A_b_1)
    delta_ch = (np.matmul(u_ch, A)) - c
    print('delta = ',delta_ch)
    check = 0
    for i in range(len(c)):
        if delta_ch[i] >= 0:
            check += 1
    if check == len(c):
        print('OPTIMAL PLAN')
        sys.exit()
    z0 = list(delta_ch).index(min(delta_ch))
    A_z0 = np.zeros((len(b), 1))
    for i in range(len(b)):
        A_z0[i][0] = A[i][z0]
    z = np.matmul(A_b_1, A_z0)
    theta_i = []
    for i in range(len(z)):
        if z[i] > 0:
            theta_i.append(x[j_b[i] - 1] / z[i])
        else:
            theta_i.append(np.inf)
    print('theta_i = ' ,theta_i)
    theta_0 = min(theta_i)
    theta_0_index = theta_i.index(min(theta_i))
    index = j_b[theta_0_index] - 1
    j_b[theta_0_index] = z0 + 1
    x[theta_0_index - 1], x[index] = swap(x[theta_0_index - 1], x[index])
    for i in range(len(j_b)):
        if j_b[i] != j_b[theta_0_index]:
            x[j_b[i] - 1] = x[j_b[i] - 1] - theta_0 * z[i]
    print('x = ', x)
    print('j_b = ', j_b)
    return x, j_b, theta_0_index


def main():
    x = np.array([2,2,0,0,2,4])
    j_b = np.array([1,2,5,6])
    len_j = len(j_b)
    A_b = np.zeros((len_j, len_j))
    j_b_list = list(j_b)
    for i in range(len_j):
        for j in j_b:
            A_b[i][j_b_list.index(j)] = A[i][j-1]
    print(A_b, 'A_b')
    if np.linalg.det(A_b) == 0:
        print('NOT OPTIMAL PLAN')
        sys.exit()
    A_b_1 = np.linalg.inv(A_b)
    print(A_b_1, 'A_b_1')
    x, j_b, theta_0_index = search_for_one_iteration(A_b, A_b_1, x, j_b)
    search_optimal_plan(x, j_b, A_b, A_b_1, theta_0_index)


if __name__ == '__main__':
    main()
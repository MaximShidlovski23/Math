import numpy as np
import copy
import sys


c = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
A = np.array([[-1.0, 1.0, 1.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 1.0]])
b = np.array([[1.0], [3.0], [2.0]])


#где x - новый столбец
def invertible_matrix(array, array_1, x, i):
    n = len(x)
    print(array_1, 'lol'*10, x)
    l = np.matmul(array_1, x)
    print(l, 'dkdlkk', l[i], 'cjicj', l[i][0])
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


def search_optimal_plan(x, j_b, A_b, A_b_1, theta_0_index, A, c, b):
    len_j = len(j_b)
    while True:
        #lol - x
        lol = np.zeros((len_j, 1))
        for i in range(len_j):
            lol[i][0] = A[i][theta_0_index-1]
        print(A, 'xxxx'*100, theta_0_index)
        print('as', lol, 'dcdcd', i, 'vrv', theta_0_index)
        A_b_1 = invertible_matrix(A_b, A_b_1, lol, theta_0_index)
        A_b = matrix(A_b, lol, theta_0_index, A_b_1)

        if np.linalg.det(A_b) == 0:
            print('Определитель = 0')
            sys.exit()
        x, j_b, theta_0_index = search_for_one_iteration(A_b, A_b_1, x, j_b, A, c, b)
        if theta_0_index == 'stop':
            return x, j_b


def search_for_one_iteration(A_b, A_b_1, x, j_b, A, c, b):
    j_b_list = list(j_b)
    len_j = len(j_b)
    c_b_ch = np.zeros(len_j)
    for i in j_b:
        c_b_ch[j_b_list.index(i)] = c[i]
    u_ch = np.matmul(c_b_ch, A_b_1)
    delta_ch = (np.matmul(u_ch, A)) - c
    check = 0
    for i in range(len(c)):
        if delta_ch[i] >= 0:
            check += 1
    #print(check, 'ffff', delta_ch,'dede', len(c), 'ffff', x)
    if check == len(c):
        print('Оптимальный план')
        print('x = ', x, '\nj_b = ', j_b)
        return x, j_b, 'stop'
    """if all(delta_ch[i] >= 0 for i in range(len(c))):
        print('Оптимальный план')
        print('x = ', x, '\nj_b = ', j_b)
        return x, j_b, 'stop'"""
    z0 = list(delta_ch).index(min(delta_ch))
    A_z0 = np.zeros((len(b), 1))
    for i in range(len(b)):
        A_z0[i][0] = A[i][z0]
    print(A_b, 'lol', A_b_1, 'dkdoko', A_z0)
    z = np.matmul(A_b_1, A_z0)
    print('z = ', z)
    theta_i = []
    for i in range(len(z)):
        if z[i] > 0:
            theta_i.append(x[j_b[i]] / z[i])
        else:
            theta_i.append(np.inf)
    theta_0 = min(theta_i)
    theta_0_index = theta_i.index(min(theta_i))
    print(theta_i, 'lol'*10, theta_0, 'kek'*10, theta_0_index)
    index = j_b[theta_0_index]
    j_b[theta_0_index] = z0
    print(x, 'XXX')
    x[theta_0_index-1], x[index] = swap(x[theta_0_index-1], x[index])
    print(x, 'XXX')
    for i in range(len(j_b)):
        if j_b[i] != j_b[theta_0_index]:
            x[j_b[i]] = x[j_b[i]] - theta_0 * z[i]
    print(x, 'XXX')
    print('x = ', x)
    print('j_b = ', j_b)
    print('theta_0_index = ', theta_0_index)
    return x, j_b, theta_0_index


def main():
    x = np.array([0.0, 0.0, 1.0, 3.0, 2.0])
    j_b = np.array([2, 3, 4])
    len_j = len(j_b)
    A_b = np.zeros((len_j, len_j))

    j_b_list = list(j_b)
    for i in range(len_j):
        for j in j_b:
            A_b[i][j_b_list.index(j)] = A[i][j]
    print(A_b, 'kke')
    if np.linalg.det(A_b) == 0:
        print('NOT OPTIMAL PLAN')
        sys.exit()
    A_b_1 = np.linalg.inv(A_b)
    x, j_b, theta_0_index = search_for_one_iteration(A_b, A_b_1, x, j_b, A, c, b)
    search_optimal_plan(x, j_b, A_b, A_b_1, theta_0_index, A, c, b)


if __name__ == '__main__':
    main()

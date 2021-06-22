import math
import numpy as np


def checkDeterminant(a):
    if np.linalg.det(a) == 0:
        raise print("Определитель равен 0")
        quit()


def Search_max_elem(a, len1):
    max_elem = 0
    check_i = 0
    check_j = 0
    for i in range(len1):
        for j in range(len1):
            if j > i and abs(max_elem) < abs(a[i][j]):
                max_elem = a[i][j]
                check_i = i
                check_j = j
    return check_i, check_j


def Eigenvalues(a, len1):
    k = 0
    eps = 0.0001
    h_check = np.eye(len1)
    check_i, check_j = Search_max_elem(a, len1)  # получаем максимальный элемент и его индексы
    while abs(a[check_i][check_j]) > eps:  # условия выполнения итераций
        fi = 0.5 * math.atan(
            2 * a[check_i][check_j] / (a[check_i][check_i] - a[check_j][check_j]))  # находим угол поворота фи
        print("max = %.4f" % (a[check_i][check_j]))
        print("fi = %.4f" % (fi))
        sin_1 = math.sin(fi)
        # print("sin_1 = %.4f" % (sin_1))
        cos_1 = math.cos(fi)
        # print("cos_1 = %.4f" % (cos_1))

        h = a.copy()
        for i in range(len1):
            for j in range(len1):
                if a[check_i][check_i] == a[i][j] or a[check_j][check_j] == a[i][j]:
                    # если индексы элемента равны индексам макс.
                    h[i][j] = cos_1  # то косинус
                elif j == i:
                    h[i][j] = 1  # если нет, то 1
                else:
                    h[i][j] = 0
        h[check_i][check_j] = -sin_1  # if index i & j, else -sin
        h[check_j][check_i] = sin_1  # if index j & i, else sin
        # print("H = ")
        # print(h)

        h_t = a.copy()
        for i in range(len1):
            for j in range(len1):
                h_t[i][j] = h[j][i]

        a = np.dot(h_t, a)  # итерация H^T * A * H
        a = np.dot(a, h)
        check_i, check_j = Search_max_elem(a, len1)
        h_check = h_check @ h  # нахождение вектора
        k += 1
    print("\nk = %d" % (k))
    return a, h_check

def main():
    a = np.array([[3.452, 0.458, 0.125, 0.236],
              [0.254, 2.458, 0.325, 0.126],
              [0.305, 0.125, 3.869, 0.458],
              [0.423, 0.452, 0.248, 3.896]])

    """a = np.array([[5.0, 1, 2],
              [1, 4.0, 1],
              [2, 1., 3]])"""

    eps = 0.001
    checkDeterminant(a)
    len1 = len(a)
    print(a)
    arr = a.copy()
    a_1, h_check = Eigenvalues(a, len1)
    print("\nСобственные значения матрицы без симметризации:\n")
    for i in range(len1):
        print("lymda%.d = %.4f" % (i + 1, a_1[i][i]))
    print("\nСобственные векторы матрицы без симметризации:\n")
    print(h_check)
    print("=" * 50)
    a_t = a.copy()
    for i in range(len1):
        for j in range(len1):
            a_t[i][j] = a[j][i]
    a = np.dot(a_t, a)
    print("\nМатрица после симметризации A = A*A^T:\n")
    print(a)
    a, h_check = Eigenvalues(a, len1)
    print("\nСобственные значения матрицы при A = A*A^T:\n")
    for i in range(len1):
        print("lymda%.d = %.4f" % (i + 1, math.sqrt(a[i][i])))
    print("\nСобственные векторы матрицы при A = A*A^T:\n")
    print(h_check)
    print("=" * 50)


if __name__ == "__main__":
    main()



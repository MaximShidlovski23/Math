import copy
import numpy as np


def checkDeterminant(a):
    if np.linalg.det(a) == 0:
        raise print("Определитель равен 0")
        quit()


def GaussFunc(a):
    c = np.array(a)
    a = np.array(a)

    len1 = len(a[:, 0])  # размер матрицы A, то есть n
    len2 = len(a[0, :])  # n+1

    for g in range(len1):
        max1 = abs(a[g][g])  # переменная для сохранения максимума в текущем столбце
        t0 = g
        t1 = g
        while t1 < len1:  # цикл поиска максимума в столбце g
            if abs(a[t1][g]) > max1:
                max1 = abs(a[t1][g])
                t0 = t1
            t1 += 1

        if t0 != g:
            buf = copy.deepcopy(a[g])
            a[g] = copy.deepcopy(a[t0])
            a[t0] = copy.deepcopy(buf)
            # swap() для строки с максимумом

    for g in range(len1):
        max_sum = 0
        for j in range(len1):
            if g != j:
                max_sum += abs(a[g][j])
        if abs(a[g][g]) > max_sum:
            print("OK!!! {0} > {1}".format(a[g][g], max_sum))
        else:
            print("BAD!!! {0} <= {1}".format(a[g][g], max_sum))
    array_norma1 = []
    array_norma2 = []
    for g in range(len1):
        coef = a[g][g]
        for j in range(len2):
            a[g][j] /= coef
        for j in range(len1):
            a[g][j] = -a[g][j]
        a[g][g] = 0
        check1 = 0
        for j in range(len1):
            check1 += abs(a[g][j])
        array_norma1.append(check1)
    norma1 = max(array_norma1)
    for j in range(len1):
        check2 = 0
        for g in range(len1):
            check2 += abs(a[g][j])
        array_norma2.append(check2)
    norma2 = max(array_norma2)
    print("{0} _____ {1}".format(norma1, norma2))
    if (norma1 < 1) or (norma2 < 1):
        print("OK!!!!")
    else:
        print("BAD!!!!")
    print(a)

    x0 = []
    for g in range(len1):
        x0.append(a[g][len1])
    # print(x_array)
    i = len1
    k = 0
    eps = 0.01
    x = x0.copy()
    x_reserve = x.copy()
    x1 = x.copy()
    for g in range(len1):
        x[g] = 0
        x1[g] = 0

    if abs(x[0] - x0[0]) < eps:
        print("true")
    else:
        print("false")
        print(abs(x[0] - x0[0]))
    while abs(x[0] - x0[0]) > eps:
        for g in range(len1):
            x1[g] = 0  # обнуляем x1
        for g in range(len1):
            for j in range(len2):
                if j == len1:
                    x1[g] += a[g][j]  # добавляем свободный элемент
                else:
                    x1[g] += a[g][j] * x_reserve[j]  # суммируем коэффициент * на значение х
        print("x1 = ", x1)
        x0 = x_reserve.copy()
        x_reserve = x1.copy()
        x = x1.copy()
        k += 1


def main():
    a = np.array([[3.452, 0.458, 0.125, 0.236],
                  [0.254, 2.458, 0.325, 0.126],
                  [0.305, 0.125, 3.869, 0.458],
                  [0.423, 0.452, 0.248, 3.896]])
    """a = np.array([[2.0, 2, 10],
              [10, 1.0, 1],
              [2, 10., 1]])"""

    checkDeterminant(a)

    b = np.array([0.745, 0.789, 0.654, 0.405])
    # b = np.array([14., 12, 13])
    c = np.column_stack((a, b))  # объединение матриц а и b

    print("Исходная матрица:")
    print(c)

    GaussFunc(c)


if __name__ == '__main__':
    main()

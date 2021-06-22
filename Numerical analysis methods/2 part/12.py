import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from mpl_toolkits.mplot3d import Axes3D


def print_table(columns_titles, ziped_rows):
    table = PrettyTable()
    table.field_names = columns_titles
    for row in ziped_rows:
        table.add_row(row)
    print(table)


def g1(x):
    return 1


def g2(x):
    return 0


def fi(x):
    return 1 - x


def f(x):
    return 2


def grid(M, N, h, tau):
    x_array = []
    t_array = []
    for i in range(N + 1):
        x_array.append(i * h)
    for j in range(M + 1):
        t_array.append(j * tau)
    return x_array, t_array


def explicit_difference_scheme(M, N, h, tau):
    x_array, t_array = grid(M, N, h, tau)

    y_array = np.zeros((N + 1, M + 1))
    for j in range(1, M + 1):
        y_array[0][j] = g1(t_array[j])
        y_array[N][j] = g2(t_array[j])
    for i in range(N + 1):
        y_array[i][0] = fi(x_array[i])

    for j in range(0, M):
        for i in range(1, N):
            y_array[i][j + 1] = ((y_array[i - 1][j] - 2 * y_array[i][j] + y_array[i + 1][j]) * tau +
                                 f(t_array[j]) * 2 * tau * h ** 2 + y_array[i][j] * 2 * h ** 2) / h ** 2 / 2
    return x_array, t_array, y_array


def y_for_shedule_when(N, coef, y_array):
    cur_y_array = []
    for i in range(N + 1):
        cur_y_array.append(y_array[i][coef])
    return cur_y_array


def main():
    a = 0
    b = 1
    l = b - a
    T = 0.1
    k = 0.4
    while True:
        h = l / 10
        # h = float(input("h = "))
        tau = float(input("tau = "))
        if tau <= 0.5 * (h ** 2 / k):
            break
        else:
            print("Условие устойчивости не выполнено!")
    N = int(l / h)
    M = int(T / tau)
    print("N = ", N)
    print("M = ", M)
    x_array, t_array, y_array = explicit_difference_scheme(M, N, h, tau)
    print(y_array)
    y_coef_1 = y_for_shedule_when(N, 10, y_array)
    y_coef_2 = y_for_shedule_when(N, 20, y_array)
    y_coef_3 = y_for_shedule_when(N, 40, y_array)
    print_table(['xi', 'yi_1', 'yi_2', 'yi_3'], zip(x_array, y_coef_1, y_coef_2, y_coef_3))
    plt.plot(x_array, y_coef_1)
    plt.plot(x_array, y_coef_2)
    plt.plot(x_array, y_coef_3)
    plt.grid()
    plt.show()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$X$ spatial axis')
    ax.set_ylabel('$T$ time axis')
    ax.set_zlabel('$Y$ axis')

    for k in range(0, int(T / tau) + 1, 2):
        y_coef = y_for_shedule_when(N, k, y_array)
        # plt.plot(x_array, y_coef)
        ax.plot3D(x_array, np.array([t_array[k]] * (N + 1)), y_coef)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

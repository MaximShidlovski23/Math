import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import math

x = sp.symbols('x')
expression = x**2
y_function = sp.lambdify(x, expression, 'numpy')
a = 0
b = 6
n = 200
h = []
A = []
B = []
v = []
f = []
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

def elements_in_interval(start, end, temp):
    count = 0
    for i in temp:
        if i > end:
            break
        if start <= i:
            count += 1
    return count


def uniform_distribution(n, a, b):
    x = []
    y = []
    x_random = np.random.uniform(0, 1, n)
    for i in range(n):
        x.append(round(x_random[i] * (b - a) + a, 3))
        y.append(round(y_function(x[i]), 3))
    return x, y

#характеристики равноинтервального метода
def equal_interval(M,temp):
    for i in range(0, M):
        h.append(round((temp[-1] - temp[0]) / M, 5))
        A.append(round(temp[0] + i * h[i], 5))
        B.append(round(temp[0] + (i + 1) * h[i], 5))
        v.append(elements_in_interval(A[i], B[i], temp))
        f.append(round(v[i] / (n * h[i]), 10))

#характеристики равновероятностного метода
def equal_probability(M, temp):
    for i in range(0, M):
        v.append(int(n / M))
    A.append(round(temp[0], 5))
    for i in range(1, M):
        A.append(round((temp[i * v[i] - 1] + temp[i * v[i]]) / 2, 5))
    B.extend(A[1:])
    B.append(round(temp[-1], 5))

    for i in range(0, M):
        h.append(round(B[i] - A[i], 5))
        f.append(round(v[i] / (n * h[i]), 10))


def create_table(M):
    table = PrettyTable()
    table.field_names = ['A', 'B', 'h', 'v', 'f']
    for i in range(0, M):
        table.add_row([A[i], B[i], h[i], v[i], f[i]])
    return table

def hist(mes, M, check):
    if check == 1:
        ax1.grid(True)
        ax1.set_title(mes)
        for i in range(0, M):
            ax1.bar(A[i], f[i], width=h[i], align='edge')
    elif check == 3:
        ax3.grid(True)
        ax3.set_title(mes)
        for i in range(0, M):
            ax3.bar(A[i], f[i], width=h[i], align='edge')

def polygon(M, check):
    if check == 1:
        for i in range(0, M - 1):
            ax1.plot([(B[i] + A[i]) / 2, (B[i + 1] + A[i + 1]) / 2], [f[i], f[i + 1]], color='r', marker='.')
    elif check == 3:
        for i in range(0, M - 1):
            ax3.plot([(B[i] + A[i]) / 2, (B[i + 1] + A[i + 1]) / 2], [f[i], f[i + 1]], color='r', marker='.')



#плотность
def theoretical_density(check):
    if check == 1:
        x_local = np.arange(0.1, 36.0, 0.1)
        y_local = 1 / (12 * x_local ** 1 / 2)
        ax1.plot(x_local, y_local)
    elif check == 3:
        x_local = np.arange(0.1, 36.0, 0.1)
        y_local = 1 / (12 * x_local ** 1 / 2)
        ax3.plot(x_local, y_local)

def calc_V(M):
    ans = 0.0
    for i in range(0, M):
        ans += f[i] * h[i]
    return ans


def empirical_func(Y, N, n):
    # group =  [[key, len(list(group))] for key, group in groupby(Y)]
    N = list(map(int, N))
    group = [[yi, ni] for yi, ni in zip(Y, N)]
    group.sort(key=lambda x: x[0])
    group = [[group[0][0] - 0.5, 0]] + group

    XX = [group[0][0]]
    YY = [0]
    for i in range(1, len(group)):
        XX.append(group[i][0])
        XX.append(group[i][0])
        YY.append(YY[2 * (i - 1)])
        YY.append(YY[2 * (i - 1)] + group[i][1] / n)
    XX.append(group[-1][0] + 0.5)
    YY.append(YY[-1])
    return XX, YY


def theoretical_func(check):
    if check == 2:
        x_local = np.arange(0.1, 36.0, 0.1)
        y_local = np.arange(0.1, 36.0, 0.1)
        for i in range(len(x_local)):
            y_local[i] = (x_local[i] ** (1 / 2)) / 6
        ax2.plot(x_local, y_local, color='r')
    elif check == 4:
        x_local = np.arange(0.1, 36.0, 0.1)
        y_local = np.arange(0.1, 36.0, 0.1)
        for i in range(len(x_local)):
            y_local[i] = (x_local[i] ** (1 / 2)) / 6
        ax4.plot(x_local, y_local, color='r')


def clear_arg():
    h.clear()
    A.clear()
    B.clear()
    v.clear()
    f.clear()

def main():
    X, temp = uniform_distribution(n, a, b)
    temp = sorted(temp)

    if n <= 100:
        M = int(np.sqrt(n))
    else:
        M = int(3 * math.log10(n))

    equal_interval(M, temp)
    print("Равноинтервальный метод")
    print(create_table(M))

    hist("Гистограмма равноинтервальным методам", M, 1)
    polygon(M, 1)
    theoretical_density(1)
    ylimit = max(f) + max(f) * 0.10
    ax1.set_ylim(top=ylimit)
    print(calc_V(M))

    pol_x1 = [(a + b) / 2 for a, b in zip(A, B)]
    XX, YY = empirical_func(pol_x1, v, n)
    ax2.plot(XX, YY)
    theoretical_func(2)

    clear_arg()

    equal_probability(M, temp)

    hist("Гистограмма равновероятностным методам", M, 3)
    polygon(M, 3)
    theoretical_density(3)
    ax3.set_ylim(top=ylimit)

    print("Равновероятностный метод")
    print(create_table(M))
    print(calc_V(M))

    pol_x2 = [(a + b) / 2 for a, b in zip(A, B)]
    XX, YY = empirical_func(pol_x2, v, n)
    ax4.plot(XX, YY)
    theoretical_func(4)
    plt.show()

if __name__ == '__main__':
    main()
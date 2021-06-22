import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import math

x = sp.symbols('x')
expression = x ** 2
#expression = sp.exp(x)
y_function = sp.lambdify(x, expression, 'numpy')
a = 0
b = 6
n_xi = 200
n_kolmo = 200
n_mizes = 200
h = []
A = []
B = []
v = []
f = []
p = []
p_v = []
xi_kvad = []

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

#характеристики равновероятностного метода
def equal_probability(M, temp, n):
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


def create_table(M, check):
    table = PrettyTable()
    if check == 1:
        table.field_names = ['A', 'B', 'h', 'v', 'f']
        for i in range(0, M):
            table.add_row([A[i], B[i], h[i], v[i], f[i]])
    elif check == 2:
        table.field_names = ['F(A)', 'F(B)', 'p', 'p*', 'X^2']
        for i in range(0, M):
            table.add_row([y_function(A[i]),y_function(B[i]), p[i], p_v[i], xi_kvad[i]])
    return table

def hist(mes, M, check):
        plt.grid(True)
        plt.title(mes)
        for i in range(0, M):
            plt.bar(A[i], f[i], width=h[i], align='edge')

def polygon(M, check):
        for i in range(0, M - 1):
            plt.plot([(B[i] + A[i]) / 2, (B[i + 1] + A[i + 1]) / 2], [f[i], f[i + 1]], color='r', marker='.')

#плотность
def theoretical_density(check):
        x_local = np.arange(0.1, 36.0, 0.1)
        y_local = 1 / (12 * x_local ** 1 / 2)
        plt.plot(x_local, y_local)

def calc_V(M):
    ans = 0.0
    for i in range(0, M):
        ans += f[i] * h[i]
    return ans

def theoretical_func(x_local):
    if x_local >= 0:
        return (x_local ** (1 / 2)) / 6
    else:
        return 0


def theoretical_probability(M):
    for i in range(M):
        #p.append(y_function(B[i]) - y_function(A[i]))
        p.append(theoretical_func(B[i]) - theoretical_func(A[i]))
        p_v.append(v[i] / n_xi)
        xi_kvad.append((n_xi * (p[i] - p_v[i])**2) / p[i])
    return sum(xi_kvad), sum(p)

def clear_arg():
    h.clear()
    A.clear()
    B.clear()
    v.clear()
    f.clear()

def main_xi_kvadrat():
    X, temp = uniform_distribution(n_xi, a, b)
    temp = sorted(temp)

    if n_xi <= 100:
        M = int(np.sqrt(n_xi))
    else:
        M = int(3 * math.log10(n_xi))
    M = 10

    equal_probability(M, temp, n_xi)

    hist("Гистограмма равновероятностным методам", M, 3)
    polygon(M, 3)
    theoretical_density(3)
    ylimit = max(f) + max(f) * 0.10
    plt.ylim(top=ylimit)
    plt.show()

    print("Равновероятностный метод")
    print(create_table(M, 1))
    print(calc_V(M))
    sum_xi_kvad, lol_p = theoretical_probability(M)
    print("Промежуточный результат метода Хи квадрат:")
    print(create_table(M, 2))
    print("Хи квадрат = {}".format(sum_xi_kvad))
    print(lol_p)

def empirical_func(x, y):
    cur = -10e9
    ind = 0
    for i in range(len(y)):
        if cur >= x:
            break
        cur = y[i]
        ind += 1
    return (ind - 1) / len(y)

def kolmogorov(y):
    cur = 0.0
    for i in range(0, len(y)):
        cur = max(abs(empirical_func(y[i], y) - theoretical_func(y[i])), cur)
    print(np.sqrt(len(y)) * cur)

def main_kolmogorov():
    x, y = uniform_distribution(n_kolmo, a, b)
    y.sort()
    print("Критерий Колмогорова")
    kolmogorov(y)

def main_mises():
    x, y = uniform_distribution(n_mizes, a, b)
    t_y = []
    e_y = []
    for i in range(len(y)):
        t_y.append(theoretical_func(y[i]))
        e_y.append(empirical_func(y[i], y))
    lol = []
    for i in range(1, len(y)):
        lol.append((theoretical_func(y[i]) - (i - 0.5) / len(y)) ** 2)
    C = 1 / (12 * len(y))
    for tmp in lol:
        C += tmp
    print("Критерий Мизеса:")
    print(C)


def main():
    main_xi_kvadrat()
    main_kolmogorov()
    main_mises()
if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# параметры равномерного распределения
a = 0
b = 6

def y_func(x):
    return x**2

def uniform_distribution(n):
    x = []
    y = []
    for i in range(n):
        random_num = np.random.uniform()
        check_x = random_num * (b - a) + a
        x.append(round(check_x, 3))
        #x.append(random_num * (b - a) + a)
        check_y = y_func(random_num * (b - a) + a)
        y.append(round(check_y, 3))
        #y.append(y_func(random_num * (b - a) + a))
    return x, y

def empirical_func(t, y):
    n = len(y)
    temp = 0
    y = list(set(y))
    y.sort()
    x_cur = [y[0] - abs(y[0]), y[0]]
    y_cur = [0, 0]
    temp += t[y[0]] / n
    plt.plot(x_cur, y_cur, marker='o')
    for i in range(1, len(y)):
        x_cur = [y[i - 1], y[i]]
        y_cur = [temp, temp]
        temp += t[y[i]] / n
        plt.plot(x_cur, y_cur, marker='o')
    plt.show()

def theoretical_func():
    x_local = np.arange(0.1, 36.0, 0.1)
    y_local = (x_local ** (1 / 2)) / 6
    plt.plot(x_local, y_local)
    plt.show()

def main():
    n = int(input())
    x, y = uniform_distribution(n)
    table = PrettyTable()
    table.field_names = ['x', 'y']
    for i in range(len(x)):
        table.add_row([x[i], y[i]])
    print(table)
    y.sort()
    t ={}
    for i in range(len(x)):
        if y[i] in t:
            t[y[i]] += 1
        else:
            t[y[i]] = 1
    print(t)
    print("Вариационный ряд:")
    print(y)
    empirical_func(t, y)
    theoretical_func()

if __name__ == '__main__':
    main()


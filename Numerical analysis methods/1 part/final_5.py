import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import sys


def f(x):
    return 4 * (1 + math.sqrt(x)) * math.log(x, math.e) - 10

def f11(x):
    return math.atan(1 - 1.5 * x) - x

def f21(x):
    return math.sqrt(1 - x ** 2)

def f22(x):
    return -math.sqrt(1 - x ** 2)


def plots(y, y1, y21, y22):
    figure, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(8, 4)
    )

    X = np.arange(0.001, 5, 0.001)
    Y = [y(np.real(x)) for x in X]
    ax1.plot(X, Y, 'g')

    X = np.arange(-2, 2, 0.01)
    Y = [y1(np.real(x)) for x in X]
    ax2.plot(X, Y, 'g')

    X = np.append(np.arange(-1, 1, 0.01), 1)
    Y = [y21(np.real(x)) for x in X]
    ax2.plot(X, Y, 'b')

    X = np.append(np.arange(-1, 1, 0.01), 1)
    Y = [y22(np.real(x)) for x in X]
    ax2.plot(X, Y, 'b')

    ax1.axis([0, 5, -12, 6])
    ax1.grid(True, which='both')
    ax1.axhline(y=0, color='k')

    ax2.axis([-1.1, 1.1, -1.1, 1.1])
    ax2.grid(True, which='both')
    ax2.axhline(y=0, color='k')
    ax2.axvline(x=0, color='k')

    plt.show()


def ff(x):
    return 4 * (1 + math.sqrt(x)) * math.log(x) - 10


def Method_Hord(a=0.1, b=5, eps=0.001):
    x0 = 0
    x1 = a
    count = 0
    while abs(x1 - x0) > eps:
        x0 = x1
        x1 = x0 - (ff(x0) / (ff(b) - ff(x0))) * (b - x0)
        count += 1
    return x1, count


def Method_Tangent(a=0.1, eps=0.001):
    x0 = a
    count = 0
    x = sp.symbols('x')
    f_x = 4 * (1 + sp.sqrt(x)) * sp.log(x) - 10
    diff_f = f_x.diff(x)
    diff_f2 = diff_f.diff(x)
    f_x_value = f_x.subs(x, x0)
    diff_f2_value = diff_f2.subs(x, x0)
    if f_x_value * diff_f2_value <= 0:
        print("Начальное приближение х0 не удовлетворяет неравенству")
        sys.exit()
    diff_f_value = diff_f.subs(x, x0)
    x1 = x0 - ff(x0) / diff_f_value
    while abs(x1 - x0) > eps:
        x0 = x1
        diff_f_value = diff_f.subs(x, x0)
        x1 = x0 - ff(x0) / diff_f_value
        count += 1
    return x1, count


def SimpleIteration(x0=0.001, y0=0.5, eps=0.001):
    x1 = (math.sin(x0 + y0) - 1) / 1.5
    y1 = math.sqrt(1 - x0 ** 2)
    count = 0
    while abs(y1 - y0) > eps or abs(x1 - x0) > eps:
        x0 = x1
        y0 = y1
        x1 = (math.sin(x0 + y0) - 1) / 1.5
        y1 = math.sqrt(1 - x0 ** 2)
        count += 1
    return x1, y1, count


def Jakobian(x0, y0, n=2):
    W = np.zeros([n, n])
    x = sp.symbols('x')
    y = sp.symbols('y')
    f1 = sp.sin(x + y) + 1.5 * x - 1
    f2 = x ** 2 + y ** 2 - 1
    diff_f1_x = f1.diff(x)
    diff_f1_y = f1.diff(y)
    diff_f2_x = f2.diff(x)
    diff_f2_y = f2.diff(y)
    W[0][0] = diff_f1_x.subs({x: x0, y: y0})
    W[0][1] = diff_f1_y.subs({x: x0, y: y0})
    W[1][0] = diff_f2_x.subs({x: x0, y: y0})
    W[1][1] = diff_f2_y.subs({x: x0, y: y0})
    return W


def f_s(x, y):
    A = np.zeros(2)
    A[0] = np.sin(x + y) + 1.5 * x - 1
    A[1] = x ** 2 + y ** 2 - 1
    return A


def Method_Newton(flag, x, y, eps=0.01):
    v0 = np.array([x, y])
    # v0 = np.array([0.5, -0.5])
    v1 = v0 - np.linalg.inv(Jakobian(v0[0], v0[1])) @ f_s(v0[0], v0[1])
    checkX = v0[0]
    checkY = v0[1]
    count = 0
    while abs(np.linalg.norm(v1 - v0)) > eps:
        v0 = v1
        if flag == 0:
            checkX = v0[0]
            checkY = v0[1]
        count += 1
        v1 = v0 - np.linalg.inv(Jakobian(checkX, checkY)) @ f_s(v0[0], v0[1])
    return v1, count


def main():
    plots(f, f11, f21, f22)
    a = 0.1
    b = 5.0
    eps = 0.001
    x_Horda, count_Horda = Method_Hord(a, b, eps)
    print("Метод хорд:")
    print(count_Horda, x_Horda)
    x = 0.1
    x_Tangent, count_Tangent = Method_Tangent(x, eps)
    print("Метод касательных:")
    print(count_Tangent, x_Tangent)
    x_Iteration, y_Iteration, count_Iteration = SimpleIteration(eps)
    print("Метод простых итераций:")
    print(count_Iteration, x_Iteration, y_Iteration)
    vect_Newton, count_Newton = Method_Newton(0, 0.5, -0.5)
    print("Метод Ньютона:")
    print(count_Newton, vect_Newton)
    vect_Newton, count_Newton = Method_Newton(0, 0.1, 0.1)
    print("Метод Ньютона:")
    print(count_Newton, vect_Newton)
    vect_ModNewton, count_ModNewton = Method_Newton(1, 0.5, -0.5)
    print("Модифицированный метод Ньютона:")
    print(count_ModNewton, vect_ModNewton)
    vect_ModNewton, count_ModNewton = Method_Newton(1, 0.9, 0.9)
    print("Модифицированный метод Ньютона:")
    print(count_ModNewton, vect_ModNewton)


if __name__ == '__main__':
    main()

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.stats import t
import math
from scipy.stats import chi2
from scipy.special import erf
from scipy.stats import norm

x = sp.symbols('x')
expression = x**2
y_function = sp.lambdify(x, expression, 'numpy')
a = 0
b = 6
n = 200
t95 = 1.972
t99 = 2.601
t999 = 3.340
izv_mo = 12
izv_disp1 = 115.2
izv_disp = izv_disp1 ** 0.5



def uniform_distribution(n, a, b):
    x = []
    y = []
    x_random = np.random.uniform(0, 1, n)
    for i in range(n):
        x.append(round(x_random[i] * (b - a) + a, 3))
        y.append(round(y_function(x[i]), 3))
    return y

def Mat_ojidanie(temp, n = 200):
    mo = 0
    for i in range(n):
        mo += temp[i] / n
    return mo

def dispersia(temp, mo, n = 200):
    disp = 0
    for i in range(n):
        disp += (temp[i] - mo) ** 2
    disp = disp / (n - 1)
    return disp

def doveritelniy_interval_mo(mo, disp, t, p):
    l_interval = mo - (disp * t) / math.sqrt(n)
    r_interval = mo + (disp * t) / math.sqrt(n)
    print("{}: {} <= {} <= {}".format(p, round(l_interval, 3), round(mo, 3), round(r_interval, 3)))

def doveritelniy_interval_disp(disp, p):
    l_interval = (n * disp ** 2) / chi2.isf((1 - p) / 2, n - 1)
    r_interval = (n * disp ** 2) / chi2.isf((1 + p) / 2, n - 1)
    print("{}: {} <= {} <= {}".format(p, round(l_interval, 3), round(disp ** 2, 3), round(r_interval, 3)))

def confidence_interval_dependency(n, disp):
    a_list = np.arange(0, 1, 0.001)
    interval_list = [2 * disp * t.ppf(1 - a / 2, n - 1) / n ** 0.5 for a in a_list]
    interval_list_with_dispersion = [2 * izv_disp / n ** 0.5 * norm.ppf(1 - a / 2.0) for a in a_list]
    plt.plot(1 - a_list, interval_list, label='Без известной дисперсии')
    plt.plot(1 - a_list, interval_list_with_dispersion, label='С известной дисперсией')
    plt.title("Зависимость величины доверительного интервала\n матожидания от уровня значимости")
    plt.legend()
    plt.show()

def confidence_interval_volume_dependency():
    alp = 0.01
    #n_list = np.arange(10, 251, 1)
    n_list = np.arange(10, 250, 2)
    interval_list = []
    interval_list_with_dispersion = []
    for nl in n_list:
        y = uniform_distribution(nl, a, b)
        y = sorted(y)
        mo = Mat_ojidanie(y, nl)
        disp = dispersia(y, mo, nl)
        interval_list.append(2 * disp ** 0.5 * t.ppf(1 - alp / 2, nl - 1) / (nl) ** 0.5)
        interval_list_with_dispersion.append(2 * izv_disp / nl**0.5 * norm.ppf(1 - alp / 2.0))
    plt.plot(n_list, interval_list, label='Без известной дисперсии')
    plt.plot(n_list, interval_list_with_dispersion, label='С известной дисперсией')
    plt.title("Зависимость величины доверительного интервала матожидания\n"
              " от объёма выборки с доверительным значением " + str(1 - alp))
    plt.legend()
    plt.show()


def confidence_interval_dependency_for_dispersion(y, disp2):
    a_list = np.arange(0.001, 1, 0.001)
    interval_list = [(n - 1) * disp2 / chi2.isf((1 + (1 - a)) / 2, n - 1) -
                     (n - 1) * disp2 / chi2.isf((1 - (1 - a)) / 2, n - 1) for a in a_list]
    disp_with_mo = dispersia(y, izv_mo)
    interval_list_with_mo = [n * disp_with_mo / chi2.isf((1 + (1 - a)) / 2, n) -
                     n * disp_with_mo / chi2.isf((1 - (1 - a)) / 2, n) for a in a_list]
    plt.plot(1 - a_list, interval_list, label='Без известного матожидания')
    plt.plot(1 - a_list, interval_list_with_mo, label='С известным МО')
    plt.title("Зависимость величины доверительного интервала\n дисперсии от уровня значимости")
    plt.legend()
    plt.show()

def confidence_interval_volume_dependency_for_dispersion():
    a = 0.01
    n_list = np.arange(10, 200, 2)
    interval_list = []
    interval_list_with_mo = []
    for nl in n_list:
        y = uniform_distribution(nl, a, b)
        y = sorted(y)
        mo = Mat_ojidanie(y, nl)
        disp = dispersia(y, mo, nl)
        interval_list.append((nl - 1) * disp / chi2.isf((1 + (1 - a)) / 2, nl - 1) -
                             (nl - 1) * disp / chi2.isf((1 - (1 - a)) / 2, nl - 1))
        disp_with_mo = dispersia(y, izv_mo, nl)
        interval_list_with_mo.append(nl * disp_with_mo / chi2.isf((1 + (1 - a)) / 2, nl) -
                     nl * disp_with_mo / chi2.isf((1 - (1 - a)) / 2, nl))
    plt.plot(n_list, interval_list, label='Без известного матожидания')
    plt.plot(n_list, interval_list_with_mo, label='С известным МО')
    plt.title(
        "Зависимость величины доверительного интервала дисперсии\n от объёма выборки с доверительным значением " + str(1 - a))
    plt.legend()
    plt.show()

def main():
    y = uniform_distribution(n, a, b)
    y = sorted(y)
    mo = Mat_ojidanie(y)
    disp2 = dispersia(y, mo)
    disp = sp.sqrt(disp2)
    print("Точечная оценка матожидания: {}".format(round(mo, 4)))
    print("Матожидания: {}".format(round(izv_mo, 4)))
    print("Точечная оценка дисперсии: {}".format(round(disp2, 4)))
    print("Дисперсия: {}".format(round(izv_disp1, 4)))
    print("Доверительный интервал с неизвестной дисперсией:")

    doveritelniy_interval_mo(mo, disp, t999, 0.999)
    doveritelniy_interval_mo(mo, disp, t99, 0.99)
    doveritelniy_interval_mo(mo, disp, t95, 0.95)
    print("Доверительный интервал с известной дисперсией:")
    doveritelniy_interval_mo(izv_mo, izv_disp, t999, 0.999)
    doveritelniy_interval_mo(izv_mo, izv_disp, t99, 0.99)
    doveritelniy_interval_mo(izv_mo, izv_disp, t95, 0.95)
    print("Доверительный интервал с неизвестным матожиданием: ")
    doveritelniy_interval_disp(disp, 0.999)
    doveritelniy_interval_disp(disp, 0.99)
    doveritelniy_interval_disp(disp, 0.95)
    print("Доверительный интервал с известным матожиданием: ")
    doveritelniy_interval_disp(izv_disp, 0.999)
    doveritelniy_interval_disp(izv_disp, 0.99)
    doveritelniy_interval_disp(izv_disp, 0.95)
    confidence_interval_dependency(n, disp)
    confidence_interval_volume_dependency()
    confidence_interval_dependency_for_dispersion(y, disp2)
    confidence_interval_volume_dependency_for_dispersion()

if __name__ == "__main__":
    main()
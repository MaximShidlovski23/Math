import numpy as np
import random
import math
from scipy import optimize
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import sympy as sp
import statsmodels.api as stm


def MO(arr, n):
    mo = 0
    for z in arr:
        mo += z
    return mo / n


def DI(arr, n, mo):
    di = 0
    for z in arr:
        di += z ** 2
    return (di / n) - mo ** 2


def central_moment(arr, n, mo):
    cent_mo = 0
    for z in arr:
        cent_mo = z - mo
    return cent_mo / n


def MO_xy(arr, s, n):
    mo_xy = 0
    for i in range(0, n - s):
        mo_xy += arr[i] * arr[s + i]
    return mo_xy / (n - s)


def cor_coef(arr, s, n):
    mo_x = MO(arr[s:], n - s)
    mo_y = MO(arr[:(n - s)], n - s)
    di_x = DI(arr[s:], n - s, mo_x)
    di_y = DI(arr[:(n - s)], n - s, mo_y)
    mo_xy = MO_xy(arr, s, n)
    return (mo_xy - mo_x * mo_y) / math.sqrt(di_x * di_y)


def get_t_mas(t, n):
    t_mas = []
    for i in range(n):
        t_mas.append(i * t)
    return t_mas


def cor_func(t_mas, D, a):
    R_mas = []
    for t in t_mas:
        R_mas.append(D * math.exp(-a * abs(t)) * (1 + a * abs(t)))
    return R_mas


def generate_system(n, c_0_mas):
    matrix = np.zeros((n, n))
    check_n = n - 1
    kek = 0
    for i in range(n):
        for j in range(n):
            if j == check_n:
                check_n -= 1
                matrix[i][j] = c_0_mas[j + kek]
                kek += 1
                break
            matrix[i][j] = c_0_mas[j + kek]
    return matrix


def fun(c_0_mas, r_mas):
    n = len(c_0_mas)
    matrix = generate_system(n, c_0_mas)
    f = np.zeros(n)
    mul_vector = np.dot(matrix, c_0_mas)
    for i in range(n):
        f[i] = mul_vector[i] - r_mas[i]
    return f


def generate_x(n):
    x_mas = []
    # n *= 2
    for i in range(n):
        x_mas.append(random.normalvariate(0.0, 1.0))
    return x_mas


def generate_random_process(x_mas, c_mas):
    N = len(c_mas)
    n = len(x_mas)
    rp_mas = []
    for i in range(n - N):
        rp = 0
        for j in range(N):
            rp += c_mas[j] * x_mas[i + j]
        rp_mas.append(rp)
    return rp_mas


def plot_rp_x2(t_mas, rand_proc_mas, rand_proc_mas_1):
    plt.grid(axis='both')
    plt.title("1 и 2 случайные сигналы СП")
    plt.plot(t_mas, rand_proc_mas, t_mas, rand_proc_mas_1)
    plt.show()


def plot(trp_mas, rp_mas, title_1, tr_mas, r_mas, title_2, title_3, colors='b'):
    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_2 = fig.add_subplot(1, 2, 2)
    ax_1.plot(trp_mas, rp_mas, color=colors)
    ax_1.grid(axis='both')
    ax_1.set_title(title_1)
    ax_2.plot(tr_mas, r_mas, color=colors)
    ax_2.grid(axis='both')
    ax_2.set_title(title_2)
    fig.suptitle(title_3)
    plt.show()


def Print_table(field_name, mas1, mas2, tr_mas):
    table = PrettyTable()
    table.field_names = field_name
    for i in range(len(mas1)):
        table.add_row([tr_mas[i], mas1[i], mas2[i], mas1[i] - mas2[i]])
    print(table)


def numerical_char(rp_mas_1, rp_mas_2, n):
    mo_1 = MO(rp_mas_1, n)
    mo_2 = MO(rp_mas_2, n)
    di_1 = DI(rp_mas_1, n, mo_1)
    di_2 = DI(rp_mas_2, n, mo_2)
    sco_1 = math.sqrt(di_1)
    sco_2 = math.sqrt(di_2)
    rp_max_1 = abs(max(rp_mas_1, key=abs))
    rp_max_2 = abs(max(rp_mas_2, key=abs))
    coef_form_1 = central_moment(rp_mas_1, n, mo_1) / sco_1
    coef_form_2 = central_moment(rp_mas_2, n, mo_2) / sco_2
    peak_factor_1 = rp_max_1 / sco_1
    peak_factor_2 = rp_max_2 / sco_2
    print("Численный характеристики:")
    table = PrettyTable()
    table.field_names = ['Характеристика', '1ый СП', '2ой СП']
    table.add_row(['Мат ожидание', mo_1, mo_2])
    table.add_row(['Дисперсия', di_1, di_2])
    table.add_row(['СКО', sco_1, sco_2])
    table.add_row(['Амплитуда', rp_max_1, rp_max_2])
    #table.add_row(['коэф. формы', coef_form_1, coef_form_2])
    #table.add_row(['Пикфактор', peak_factor_1, peak_factor_2])
    print(table)


def cor_char(D, a, rp_mas_1, N, n):
    teta = sp.symbols('teta')
    func = D * sp.exp(-a * abs(teta)) * (1 + a * abs(teta))
    f_i2 = func
    i2 = sp.integrate(f_i2, (teta, 0, sp.oo))
    f_i3 = abs(func)
    i3 = sp.integrate(f_i3, (teta, 0, sp.oo))
    f_i4 = func ** 2
    i4 = sp.integrate(f_i4, (teta, 0, sp.oo))
    print()
    print('Корреляционные характеристки:')
    print('Интервал корреляции равен = ', i2)
    print('Абсолютный интервал корреляции равен = ', i3)
    print('Квадратичный интервал корреляции равен = ', i4)
    check_i = 0
    for i in range(5, n, 5):
        if i < N:
            continue
        if abs(cor_coef(rp_mas_1[:i], N, i)) < 0.2:
            check_i = i - 5
            break
    print('Максимальный интервал корреляции равен = ', check_i)


# Спектральная плотность мошности
def spectral_ro(D, a):
    teta = sp.symbols('teta')
    w = sp.symbols('w')
    func = (D * sp.exp(-a * abs(teta)) * (1 + a * abs(teta)) * sp.cos(w) * teta) / sp.pi
    f_i2 = func
    i2 = sp.integrate(f_i2, (teta, 0, sp.oo))
    print('Спектральная плотность мощности: ', i2)
    '''kek = []
    for tr in trp_mas:
        if tr == 0:
            continue
        kek.append(i2.evalf(subs={'w': (2*sp.pi / tr)}))'''


def gen_rp_for_DI(n, N, c_mas, D):
    lol = []
    nn = 100
    lol_1 = []
    lol_2 = []
    for i in range(nn):
        x_mas = generate_x(n + N)
        rp_mas = generate_random_process(x_mas, c_mas)
        lol.append(rp_mas[234])
        lol_1.append(rp_mas[457])
        lol_2.append(rp_mas[26])
    mo = MO(lol, nn)
    di = DI(lol, nn, mo)
    mo_1 = MO(lol_1, nn)
    di_1 = DI(lol_1, nn, mo_1)
    mo_2 = MO(lol_2, nn)
    di_2 = DI(lol_2, nn, mo_2)
    print("Проверка на стационарность")
    print("mo = ", mo, " = ", mo_1, " = ", mo_2, "\nРазность между теоретическим мат ожиданием "
                       "и мат ожиданием нескольких реализаций: ", mo - 0.0)
    print("di = ", di, " = ", di_1, " = ", di_2, "\nРазность между теоретической дисперсией "
                       "и дисперсией нескольких реализаций: ", di - D)


def plot_autocorrelation(rp_mas, N, r_mas, t):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    stm.graphics.tsa.plot_acf(rp_mas, lags=N, ax=ax)
    ax.plot(get_t_mas(1, N), r_mas)
    title = 'Коррелограмма и функция АКФ при dt = ' + str(t)
    fig.suptitle(title)
    plt.show()
    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_2 = fig.add_subplot(1, 2, 2)
    stm.graphics.tsa.plot_acf(rp_mas, lags=N, ax=ax_1)
    ax_1.plot(get_t_mas(1, N), r_mas)
    stm.graphics.tsa.plot_pacf(rp_mas, lags=N, ax=ax_2)
    fig.suptitle('Коррелограммы')
    plt.show()


def main():
    N = 13
    n = 5000
    t = 0.4
    D = 1.0
    a = 1.0
    tr_mas = get_t_mas(t, N)
    r_mas = cor_func(tr_mas, D, a)
    if N < 16:
        c_0_mas = np.ones([N])
    else:
        c_0_mas = np.zeros([N])
    #print(generate_system(N, c_0_mas))
    c_mas = optimize.fsolve(fun, c_0_mas, r_mas)
    print(c_mas)
    if len(list(set(c_mas))) == 1:
        print('Измените число N, или значения D и а. \nЗатем запустите еще раз')
        exit(1)
    # sol = optimize.root(fun, c_0_mas, method='krylov')
    matrix_c_mas = generate_system(N, c_mas)
    print()
    #print(matrix_c_mas)
    Print_table(["Значение дискретизации для iой позиции", "Действительное значение АКФ", "Полученное значение АКФ",
                 "Разница между значениями"], r_mas, np.dot(matrix_c_mas, c_mas), tr_mas)
    x_mas_1 = generate_x(n + N)
    rp_mas_1 = generate_random_process(x_mas_1, c_mas)
    x_mas_2 = generate_x(n + N)
    rp_mas_2 = generate_random_process(x_mas_2, c_mas)
    numerical_char(rp_mas_1, rp_mas_2, n)
    trp_mas = get_t_mas(t, n)
    cor_char(D, a, rp_mas_1, N, n)
    spectral_ro(D, a)
    plot(trp_mas, rp_mas_1, "Случайный сигнал", tr_mas,
         r_mas, "Автокорреляционная функция", "1ый процесс")
    plot(trp_mas, rp_mas_2, "Случайный сигнал", tr_mas,
         r_mas, "Автокорреляционная функция", "2ой процесс", 'orange')
    plot_rp_x2(trp_mas, rp_mas_1, rp_mas_2)
    gen_rp_for_DI(n, N, c_mas, D)
    plot_autocorrelation(rp_mas_1, N, r_mas, t)

if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
from collections import Counter

def congruent_method(x_0, k, m, n, norm=1):
    i = 1
    x = 0
    x_arr = []
    while n >= i:
        x = (k * x_0) % m
        x_arr.append(x)
        x_0 = x
        i += 1
    arr = []
    for i in x_arr:
        arr.append(i / (norm * m))
    return arr

def one_dimen_neyman_method(arr_x_1, arr_x_2):
    res_x = []
    i = 0
    a = -20
    b = 20
    for x in arr_x_1:
        x_zv_1 = a + x * (b - a)
        x_zv_2 = arr_x_2[i] * (1 / math.pi**2)
        f_x = 1 / ((x_zv_1 ** 2) + math.pi ** 2)
        if f_x > x_zv_2:
            res_x.append(x_zv_1)
        i += 1
    return res_x

def one_dimen_neyman_method_for_y(arr_x_1, arr_y_2, arr_x_3):
    res_y = []
    i = 0
    a = -10
    b = 10
    for y in arr_y_2:
        y_zv_1 = a + y * (b - a)
        y_zv_2 = arr_x_3[i] * (1 / (2 * math.pi))
        f_y =  ((arr_x_1[i] ** 2) + math.pi**2) / (2 * math.sqrt(((arr_x_1[i] ** 2) + (y ** 2) + (math.pi ** 2)) ** 3))
        if f_y > y_zv_2:
            res_y.append(y_zv_1)
        i += 1
    return res_y

def two_dimen_neyman_method(arr_x, arr_y, arr_z):
    res_x = []
    res_y = []
    i = 0
    a = -20
    b = 20
    for x in arr_x:
        x_zv_1 = a + x * (b - a)
        x_zv_2 = a + arr_y[i] * (b - a)
        x_zv_3 = arr_z[i] * 1 / (2 * math.pi ** 3)
        f_x_y = 1 / (2 * math.sqrt((x_zv_1 ** 2 + x_zv_2 ** 2 + math.pi ** 2) ** 3))
        if f_x_y > x_zv_3:
            res_x.append(x_zv_1)
            res_y.append(x_zv_2)
        i += 1
    return res_x, res_y

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

def MO_xy(arr, s, n):
    mo_xy = 0
    for i in range(0, n - s):
        mo_xy += arr[i] * arr[s + i]
    return mo_xy / (n - s)

def MO_two(x_arr, y_arr, n):
    mo_xy = 0
    for i in range(0, n):
        mo_xy += x_arr[i] * y_arr[i]
    return mo_xy / n

def cor_coef(arr, s, n):
    mo_x = MO(arr[s:], n - s)
    mo_y = MO(arr[:(n - s)], n - s)
    di_x = DI(arr[s:], n - s, mo_x)
    di_y = DI(arr[:(n - s)], n - s, mo_y)
    mo_xy = MO_xy(arr, s, n)
    if math.sqrt(di_x * di_y) == 0:
        return 0.0
    return ((mo_xy - mo_x * mo_y) / math.sqrt(di_x * di_y))

def cor_coef_for_two(x_arr, y_arr):
    len_x = len(x_arr)
    len_y = len(y_arr)
    mo_x = MO(x_arr, len_x)
    print('Mo_x: ', mo_x)
    mo_y = MO(y_arr, len_y)
    print('Mo_y: ', mo_y)
    di_x = DI(x_arr, len_x, mo_x)
    print('Di_x: ', di_x)
    di_y = DI(y_arr, len_y, mo_y)
    print('Di_y: ', di_y)
    if len_x > len_y:
        mo_xy = MO_two(x_arr[:len_y], y_arr, len_y)
    else:
        mo_xy = MO_two(x_arr, y_arr[:len_x], len_x)
    print('MO_xy: ', mo_xy)
    return ((mo_xy - mo_x * mo_y) / math.sqrt(di_x * di_y))

def elements_in_interval(start, end, arr):
    count = 0
    for i in arr:
        if i > end:
            break
        if start <= i:
            count += 1
    return count

def equal_interval(M, arr, n, a = 0, b = 1):
    h = []
    A = []
    B = []
    v = []
    f = []
    for i in range(0, M):
        h.append((b - a) / M)
        A.append(i * h[i])
        B.append((i + 1) * h[i])
        v.append(elements_in_interval(A[i], B[i], arr))
        f.append(v[i] / n)
    print(create_table(M, A, B, h, v, f))
    return f, B

def equal_interval_for_xi(M, arr, n, a = 0, b = 1,):
    h = []
    A = []
    B = []
    v = []
    f = []
    for i in range(0, M):
        h.append((b - a) / M)
        A.append(a + i * h[i])
        B.append(a + (i + 1) * h[i])
        v.append(elements_in_interval(A[i], B[i], arr))
        f.append(v[i] / n)
    print(create_table(M, A, B, h, v, f))
    return v

def simple_histogram(arr,title, a=-20, b=20):
    plt.hist(arr, 10, (a, b))
    plt.title(title)
    plt.show()

def f_X_Y(arr_x, arr_y, n):
    f_x_y = []
    for i in range(0, n):
        f_x_y.append(1 / (2 * math.sqrt((math.pi ** 2 * arr_x[i] ** 2 * arr_y[i] **2) ** 3)))
    return f_x_y

def f_X(arr_x, n):
    f_x = []
    for i in range(0, n):
        f_x.append(1 / (math.pi ** 2 + arr_x[i] ** 2))
    return f_x

def create_area_into_squares(x, y, title, a=-20, b=20):
    h = 4
    n = (b - a) // h
    matrix_z = np.zeros((n, n))
    if len(x) > len(y):
        n = len(y)
    else:
        n = len(x)
    sum_m = 0
    for i in range(0, n):
        coor_x = 0
        coor_y = 0
        for h_i in range(a + h, b + h, h):
            if h_i > x[i]:
                break
            else:
                coor_x += 1
        for h_i in range(a + h, b + h, h):
            if h_i > y[i]:
                break
            else:
                coor_y += 1
        matrix_z[coor_x][coor_y] += 1
        sum_m += 1
    for i in range(0, len(matrix_z)):
        for j in range(0, len(matrix_z)):
            matrix_z[i][j] = matrix_z[i][j] / n
    hist_3d(matrix_z, n, a, b, h, title)

def hist_3d(matrix_z, n, a, b, h, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    lx = len(matrix_z)
    ly = lx
    xpos = np.arange(a, b, h)
    ypos = np.arange(a, b, h)
    xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)
    x_size = np.ones_like(zpos)
    y_size = x_size.copy()
    z_size = matrix_z.flatten()
    ax.set_title(title)
    ax.bar3d(xpos, ypos, zpos, x_size, y_size, z_size, color='b')
    plt.show()

def plt_3dfig(x, y, l):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    def xy_func(x, y):
        return 1 / (2 * math.sqrt((x ** 2 + y ** 2 + math.pi ** 2) ** 3))

    X, Y = np.meshgrid(x[:l], y[:l])
    xy_func_v = np.vectorize(xy_func)
    Z = xy_func_v(X, Y)
    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    plt.show()

def confidense_interval_mo(mo, sco, n):
    t_arr = [3.340, 2.601, 1.972]
    p_arr = [0.999, 0.99, 0.95]
    i = 0
    for t in t_arr:
        left_interval = mo - (sco * t) / math.sqrt(n - 1)
        right_interval = mo + (sco * t) / math.sqrt(n - 1)
        print(p_arr[i],': ', left_interval, ' <= ', mo, ' <= ', right_interval)
        i += 1

def confidense_interval_di(di, n):
    p_arr = [0.999, 0.99, 0.95]
    for p in p_arr:
        left_interval = (n * di) / chi2.isf((1 - p) / 2, n - 1)
        right_interval = (n * di) / chi2.isf((1 + p) / 2, n - 1)
        print(p, ': ', left_interval, ' <= ', di, ' <= ',  right_interval)

def F_X(x):
    return (1 / 2) + math.atan(x / math.pi) / math.pi

def gen_DSV(z_arr):
    a = -20.0
    b = 20.0
    n = len(z_arr)
    h = (b - a) / (n-3)
    x_arr = []
    p_arr = []
    f_arr = []
    f_x_1 = 0
    x_arr.append(-24)
    p_arr.append(0)
    f_arr.append(0)
    for i in range(0, n-3):
        x_arr.append(round(a + i * h, 2))
        f_x = F_X(a + i * h)
        f_arr.append(f_x)
        p_arr.append(f_x - f_x_1)
        f_x_1 = f_x
    x_arr.append(b)
    p_arr.append(F_X(b) - f_x_1)
    x_arr.append(24)
    p_arr.append(1 - F_X(b))
    f_arr.append(1)
    index_arr = []
    for z in z_arr:
        index_arr.append(f_arr.index(sorted(f_arr, key=lambda p: abs(p - z))[0]))
    xz_arr = []
    for i in index_arr:
        xz_arr.append(x_arr[i])
    return xz_arr, x_arr, p_arr

def theor_mo_dsv(x_arr, p_arr):
    mo = 0
    n = len(x_arr)
    x_arr = x_arr[2 : n - 1]
    p_arr = p_arr[2 : n - 1]
    for i in range(len(x_arr)):
        mo += x_arr[i] * p_arr[i]
    return mo

def theor_di_dsv(x_arr, p_arr, mo):
    di = 0
    n = len(x_arr)
    x_arr = x_arr[2 : n - 1]
    p_arr = p_arr[2 : n - 1]
    for i in range(len(x_arr)):
        di += x_arr[i] ** 2 * p_arr[i]
    di -= mo ** 2
    return di

def get_series(x_arr, N):
    count_mas = Counter(x_arr)
    v_mas = []
    mas = []
    for xz in sorted(x_arr):
        if xz not in mas:
            mas.append(xz)
            v_mas.append(count_mas[xz] / N)
    return v_mas

def get_empirical_matrix(x_arr, y_arr, Nx, Ny):
    sum_P = 0
    matrix_P = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            sum_P += x_arr[i] * y_arr[j]
            matrix_P[i][j] = x_arr[i] * y_arr[j]
    print('Эмпирическая матрица распределения: ')
    print(matrix_P)
    return matrix_P

def print_eval(arr, theor_mo, theor_di, text):
    mo = MO(arr, len(arr))
    di = DI(arr, len(arr), mo)
    print('точечное мат ожидание для', text ,' = ', mo)
    print('точечная дисперсия для', text ,' = ', di)
    print('теоретическое мат ожидание для ', text ,' = ', theor_mo)
    print('теоретическая дисперсия для ', text ,' = ', theor_di)
    print('Доверительный интервал мат ожидания с точечной оценкой дисперсии')
    confidense_interval_mo(mo, math.sqrt(di), 200)
    print('Доверительный интервал мат ожидания с теоретической оценкой дисперсии')
    confidense_interval_mo(theor_mo, math.sqrt(theor_di), 200)
    print('Доверительный интервал дисперсии с точечной оценкой мат ожидания')
    confidense_interval_di(di, 200)
    print('Доверительный интервал дисперсии с теоретической оценкой мат ожидания')
    confidense_interval_di(theor_di, 200)

def get_theor_matrix(p_x, p_y):
    theor_matrix = np.zeros((len(p_x), len(p_y)))
    sum_matrix = 0
    for i in range(0, len(p_x)):
        for j in range(len(p_y)):
            theor_matrix[i][j] = p_x[i] * p_y[j]
            sum_matrix += theor_matrix[i][j]
    return theor_matrix

def get_func(p):
    f = []
    p_sum = 0
    for i in p:
        p_sum += i
        f.append(p_sum)
    return f

def gen_DSV_matrix(z_arr, f_arr, x_series, p):
    index_arr = []
    for z in z_arr:
        index_arr.append(f_arr.index(sorted(f_arr, key=lambda p: abs(p - z))[0]))
    xz_arr = []
    fz_arr = []
    for i in index_arr:
        xz_arr.append(x_series[i])
        pz_arr.append(f_arr[i])
    return xz_arr, pz_arr

def main():
    n = 40000
    x_0 = [232323, 435503, 674943, 125583]
    k = [10838519, 59969537, 39916801, 29986577]
    m = [155555551, 562448657, 126247697, 406586897]
    arr_x = congruent_method(x_0[0], k[0], m[0], n)
    arr_y = congruent_method(x_0[1], k[1], m[1], n)
    arr_z = congruent_method(x_0[2], k[2], m[2], n)
    arr_f = congruent_method(x_0[3], k[3], m[3], n)
    print('=' * 30, '  НСВ  ', '=' * 30)
    res_x_one = one_dimen_neyman_method(arr_x, arr_y)
    res_y_one = one_dimen_neyman_method(arr_z, arr_f)
    res_y_one_2 = one_dimen_neyman_method_for_y(arr_x, arr_z, arr_f)
    res_x_two, res_y_two = two_dimen_neyman_method(arr_x, arr_y, arr_z)
    print('Метод Неймана')
    print('Числовые характеристики для одномерного метода')
    print('корреляция Х:', cor_coef(res_x_one, 10, len(res_x_one)))
    print('корреляция Y:', cor_coef(res_y_one, 10, len(res_y_one)))
    print('корреляция XY: ', cor_coef_for_two(res_x_one, res_y_one))
    print('='*20)
    print('Числовые характеристики для одномерного метода с зависмым Y')
    print('корреляция Х:', cor_coef(res_x_one, 10, len(res_x_one)))
    print('корреляция Y:', cor_coef(res_y_one_2, 10, len(res_y_one_2)))
    print('корреляция XY: ', cor_coef_for_two(res_x_one, res_y_one_2))
    print('='*20)
    print('Числовые характеристики для двумерного метода')
    print('корреляция Х:', cor_coef(res_x_two, 10, len(res_x_two)))
    print('корреляция Y:', cor_coef(res_y_two, 10, len(res_y_two)))
    print('корреляция XY: ', cor_coef_for_two(res_x_two, res_y_two))

    print('=' * 50)
    create_area_into_squares(res_x_one, res_y_one, 'Одномерный метод Неймана')
    create_area_into_squares(res_x_one, res_y_one_2, 'Одномерный метод Неймана с зависимым Y')
    create_area_into_squares(res_x_two, res_y_two, 'Двумерный метод Неймана')
    theor_mo = 0
    theor_di = 31.12
    print_eval(res_x_one, theor_mo, theor_di, 'X в НСВ')
    print('=' * 50)
    print_eval(res_y_one, theor_mo, theor_di, 'Y в НСВ')

    print('='*30, '  ДСВ  ', '='*30)
    N = 1000
    arr_z_1 = congruent_method(x_0[0], k[0], m[0], N)
    arr_f_1 = congruent_method(x_0[3], k[3], m[3], N)
    xz_dsv_arr, x_dsv_arr, px_dsv_arr = gen_DSV(arr_z_1)
    yz_dsv_arr, y_dsv_arr, py_dsv_arr = gen_DSV(arr_f_1)
    print('корреляция Х:', cor_coef(xz_dsv_arr, 10, len(xz_dsv_arr)))
    print('корреляция Y:', cor_coef(yz_dsv_arr, 10, len(yz_dsv_arr)))
    print('корреляция XY: ', cor_coef_for_two(xz_dsv_arr, yz_dsv_arr))
    create_area_into_squares(xz_dsv_arr, yz_dsv_arr, 'Распределение ДСВ', -24, 24)
    theor_mo_x = theor_mo_dsv(x_dsv_arr, px_dsv_arr)
    theor_mo_y = theor_mo_dsv(y_dsv_arr, py_dsv_arr)
    theor_di_x = theor_di_dsv(x_dsv_arr, px_dsv_arr, theor_mo_x)
    theor_di_y = theor_di_dsv(y_dsv_arr, py_dsv_arr, theor_mo_y)
    print_eval(xz_dsv_arr, theor_mo_x, theor_di_x, ' X в ДСВ: ')
    print('=' * 50)
    print_eval(yz_dsv_arr, theor_mo_y, theor_di_y, ' Y в ДСВ: ')
    print('=' * 50)
    vx_mas = get_series(xz_dsv_arr, N)
    vy_mas = get_series(yz_dsv_arr, N)
    matrix_emp = get_empirical_matrix(vx_mas, vy_mas, len(vx_mas), len(vy_mas))
    '''print('='*50)

    x_series = [2, 3, 6, 8, 11, 12]
    y_series = [1, 4, 5, 7, 9, 10]
    p_x = [0.2, 0.1, 0.1, 0.05, 0.5, 0.05]
    p_y = [0.3, 0.2, 0.1, 0.1, 0.1, 0.2]

    theor_matrix = get_theor_matrix(p_x, p_y)
    f_x = get_func(p_x)
    arr_z_1 = congruent_method(x_0[0], k[0], m[0], 100)
    arr_f_1 = congruent_method(x_0[3], k[3], m[3], 100)

    xz_arr, fz_arr = gen_DSV_matrix(arr_z_1, f_x, x_series)
    print(xz_arr, fz_arr)'''




if __name__ == '__main__':
    main()
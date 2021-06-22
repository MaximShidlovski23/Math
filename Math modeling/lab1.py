import matplotlib.pyplot as plt
import math

h = []
A = []
B = []
v = []
p = []

def len_int(x):
    if x == 0:
        return 1
    lenn = 0
    while x:
        x //= 10
        lenn += 1
    return lenn

def middle_square_method(x_0, n):
    k = len_int(x_0)
    i = 0
    x_arr = []
    while n > i:
        y = x_0 ** 2
        x = 0
        x = y // 10 ** (k / 2)
        x %= 10 ** k
        if x == 0:
            x_arr.append(x)
            break
        x_0 = x
        i += 1
        x_arr.append(x_0 / 10 ** k)
    return x_arr

def congruent_method(x_0, k, m, n):
    i = 1
    x = 0
    x_arr = []
    x_arr.append(x_0)
    while n > i:
        x = (k * x_0) % m
        if x in x_arr:
            break
        x_arr.append(x)
        x_0 = x
        i += 1
    arr = []
    for i in x_arr:
        arr.append(i / m)
    return arr

def elements_in_interval(start, end, arr):
    count = 0
    for i in arr:
        if i > end:
            break
        if start <= i:
            count += 1
    return count

def equal_interval(M, arr, n):
    for i in range(0, M):
        h.append(1 / M)
        A.append(i * h[i])
        B.append((i + 1) * h[i])
        v.append(elements_in_interval(A[i], B[i], arr))
        p.append(v[i] / n)

def hist(ax, M, color):
    t = 1 / (2 * M)
    ax.grid()
    for i in range(0, M):
        ax.bar(A[i] + t, p[i], width=h[i], color=color)
        ax.hlines(1/M, 0, 1, color='r', linewidth=2)

def clear_arg():
    h.clear()
    A.clear()
    B.clear()
    v.clear()
    p.clear()

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

def cor_coef(arr, s, n):
    mo_x = MO(arr[s:], n - s)
    mo_y = MO(arr[:(n - s)], n - s)
    di_x = DI(arr[s:], n - s, mo_x)
    di_y = DI(arr[:(n - s)], n - s, mo_y)
    mo_xy = MO_xy(arr, s, n)
    return ((mo_xy - mo_x * mo_y) / math.sqrt(di_x * di_y))

def print_estimates(arr, n, s):
    mo = MO(arr, n)
    print('n = ', n, 'MO = ', mo, 'DI = ', DI(arr, n, mo), 'cor_coef = ', cor_coef(arr, s, n))

def main():
    n_1 = 100
    n_2 = 10000
    M = 10
    s = 10
    x_0 = 10838519
    msm_arr = middle_square_method(x_0, n_2)
    print(len(msm_arr))
    print_estimates(msm_arr[:n_1], n_1, s)
    print_estimates(msm_arr, n_2, s)
    equal_interval(M, sorted(msm_arr[:n_1]), n_1)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    hist(ax1, M, 'b')
    clear_arg()
    equal_interval(M, sorted(msm_arr), n_2)
    ax2 = fig.add_subplot(222)
    hist(ax2, M, 'g')
    clear_arg()
    x_m0 = 232323
    k = 10838519
    m = 155555551
    cm_arr = congruent_method(x_m0, k, m, n_2)
    print(len(cm_arr))
    print_estimates(cm_arr[:n_1], n_1, s)
    print_estimates(cm_arr, n_2, s)
    equal_interval(M, sorted(cm_arr[:n_1]), n_1)
    ax3 = fig.add_subplot(223)
    hist(ax3, M, 'b')
    clear_arg()
    equal_interval(M, sorted(cm_arr), n_2)
    ax4 = fig.add_subplot(224)
    hist(ax4, M, 'g')
    plt.show()

if __name__ == '__main__':
    main()

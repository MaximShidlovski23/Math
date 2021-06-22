import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

def solve2(a, b, p, q, N_X, N_Y, tau, T):
    x = np.array([np.linspace(-a / 2, a / 2, N_X)]).T
    y = np.array([np.linspace(-b / 2, b / 2, N_Y)])

    h_x = x[1, 0] - x[0, 0]
    h_y = y[0, 1] - y[0, 0]

    n = 0
    # y - на  n+1 слое
    # y_1 - на n слое
    # y_2 - на n-1 слое
    y_2 = np.zeros((N_X, N_Y))
    # Задаем начальное условие
    y_2[1:-1, :] = p(x[1:-1], y)
    yield y_2.T

    n += 1
    y_1 = np.zeros((N_X, N_Y))
    #Вычисление на первом временном слое
    y_1[1:-1, 1:-1] = p(x[1:-1], y[:, 1:-1]) + tau * q(x[1:-1], y[:, 1:-1]) + \
                                (tau / h_x) ** 2 / 2 * (
                                            p(x[2:], y[:, 1:-1]) - 2 * p(x[1:-1], y[:, 1:-1])
                                            + p(x[:-2], y[:, 1:-1])) + \
                                (tau / h_y) ** 2 / 2 * (
                                            p(x[1:-1], y[:, 2:]) - 2 * p(x[1:-1], y[:, 1:-1])
                                            + p(x[1:-1], y[:, :-2]))
    # Изменяем переменные перед переходом на следующий временный слой
    y_1[1:-1, 0] = y_1[1:-1, 1]
    y_1[1:-1, -1] = y_1[1:-1, -2]
    yield y_1.T

    n += 1
    while n * tau <= T:
        # Вычисляем значение во внутренних узлах
        y = np.zeros((N_X, N_Y))
        y[1:-1, 1:-1] = (tau / h_x) ** 2 * (
                    y_1[2:, 1:-1] - 2 * y_1[1:-1, 1:-1] + y_1[:-2, 1:-1]) + \
                               (tau / h_y) ** 2 * \
                        (y_1[1:-1, 2:] - 2 * y_1[1:-1, 1:-1] + y_1[1:-1,:-2]) + \
                               2 * y_1[1:-1, 1:-1] - y_2[1:-1, 1:-1]
        y[1:-1, 0] = y[1:-1, 1]
        y[1:-1, -1] = y[1:-1, -2]
        yield y.T

        n += 1
        # Изменяем переменные перед переходом на следующий временный слой
        y_2 = y_1
        y_1 = y

def create_animation(n_x, n_y, a, b, p, q):
    fig = plt.figure()

    x = np.linspace(-a / 2, a / 2, n_x)
    y = np.linspace(-b / 2, b / 2, n_y).reshape(-1, 1)

    extent = -a/2, a/2, -b/2, b/2
    ims = []
    for i in solve2(a, b, p, q, n_x, n_y, 0.001, 4):
        ims.append([plt.imshow(i, animated=True, cmap=plt.cm.coolwarm, vmin=-.84, vmax=.84, extent=extent)])

    ani = ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat_delay=100)

    ani.save("final_lab16_2.mp4", writer='ffmpeg')

def main():
    a = 1
    b = 2
    p = lambda x, y: np.arctan(np.cos(np.pi*x/a))
    q = lambda x, y: np.sin(2*np.pi*x/a)*np.sin(2*np.pi*y/b)

    n_x = 70
    n_y = 70

    maximum = 0
    minimum = 0
    for i in solve2(a, b, p, q, n_x, n_y, 0.001, 4):
        maximum = max(maximum, i.max())
        minimum = min(minimum, i.min())
    print("Максимальное отклонение выше точки крепления и ниже точки крепления: ")
    print(maximum, minimum)

    #create_animation(n_x, n_y, a, b, p, q)

if __name__ == '__main__':
    main()
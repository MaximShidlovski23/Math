import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def solve(p, q, C, L, N_X, tau, T):
    x = np.linspace(0, L, N_X)
    h_x = x[1] - x[0]
    num_curanta = C * tau / h_x
    n = 0
    # y - на  n+1 слое
    # y_1 - на n слое
    # y_2 - на n-1 слое
    y_2 = np.zeros(N_X)
    # Задаем начальное условие
    y_2[1:-1] = p(x[1:-1])
    yield y_2

    n += 1
    y_1 = np.zeros(N_X)
    # Вычисление на первом временном слое
    y_1[1:-1] = tau * q(x) + p(x[1:-1]) + tau ** 2 * C / (2 * h_x ** 2) * \
                (p(x[2:]) - 2 * p(x[1:-1]) + p(x[:-2]))
    yield y_1

    n += 1
    while n * tau <= T:
        # Вычисляем значение во внутренних узлах
        y = np.zeros(N_X)
        #y[1:-1] = ((num_curanta) ** 2) * (y_1[2:] - 2 * y_1[1:-1] + y_1[:-2]) + \
                 # 2 * y_1[1:-1] - y_2[1:-1]
        y[1:-1] = C * ((tau / h_x) ** 2) * (y_1[2:] - 2 * y_1[1:-1] + y_1[:-2]) +\
                  2 * y_1[1:-1] - y_2[1:-1]
        yield y

        n += 1
        # Изменяем переменные перед переходом на следующий временный слой
        y_2 = y_1
        y_1 = y

def create_animation(x, y):
    plt.style.use('seaborn-pastel')
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 10), ylim=(-0.1, 0.1))
    line, = ax.plot([], [], lw=3, color = 'r')

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x, y[i % len(y)])
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=800, interval=20, blit=True)
    anim.save('final_lab16_1.mp4', writer='ffmpeg')



def main():
    L = 10
    delta_u = 0.1
    ro = 4.3 * 10**5
    E = 120 * 10**9
    C = E / ro
    p = lambda x: -4 * delta_u / (L ** 2) * x ** 2  + 4*delta_u / L * x
    #p = lambda x: -2 * ro * delta_u / (E * L ** 2) * x ** 2
    q = lambda x: 0
    x = np.linspace(0, L, 100)
    y = list(solve(p, q, C, L, 100, 1e-4, 1e-1))
    #create_animation(x, y)
    for i in range(10):
        plt.plot(x, y[i * 10], color = 'r')
        plt.show()
    for i in range(10):
        plt.plot(x, y[100 + i * 10], color = 'g')
        plt.show()

if __name__ == '__main__':
    main()
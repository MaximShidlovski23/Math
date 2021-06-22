import numpy as np
import sys

def new_matrix(A, j_b):
    A_b = np.zeros((len(j_b), len(j_b)))
    for i in range(len(j_b)):
        for j in j_b:
            A_b[i][j_b.index(j)] = A[i][j]
    return A_b


def dual_plan(A, c, b, j_b):
    len_c = len(c)
    for i in range(len(j_b)):
        j_b[i] -= 1
    c_b = np.zeros(len(j_b))
    for i in j_b:
        c_b[j_b.index(i)] = c[i]
    A_b = new_matrix(A, j_b)
    A_b_1 = np.linalg.inv(A_b)
    y = np.matmul(c_b, A_b_1).T
    while True:
        j_not_b = []
        for i in range(len_c):
            if i not in j_b:
                j_not_b.append(i)
        kappa_b = np.matmul(A_b_1, b).T[0]
        kappa_not_b = [0] * len_c
        for i, k in zip(j_b, kappa_b):
            kappa_not_b[i] = k
        if all(k >= 0 for k in kappa_not_b):
            return kappa_not_b
        kappa_b_list = kappa_b.tolist()
        kappa_min = min(kappa_b)
        kappa_min_index = kappa_b_list.index(kappa_min)
        y_ch = A_b_1[kappa_min_index]
        Mu = []
        for j in j_not_b:
            Mu.append(np.matmul(y_ch, A[:, j].T))
        if all(mu >= 0 for mu in Mu):
            print('Задача несовместна')
            sys.exit()
        sigma = []
        i = 0
        for j in j_not_b:
            if Mu[i] < 0:
                sigma.append((c[j] - np.matmul(A[:, j], y)) / Mu[i])
            i += 1
        sigma_min = min(sigma)
        sigma_min_index = sigma.index(sigma_min)
        j_b[kappa_min_index] = sigma_min_index
        y = y + sigma_min * y_ch.T
        A_b = new_matrix(A, j_b)
        A_b_1 = np.linalg.inv(A_b)


def main():
    A = np.array([[-2.0, -1.0, -4.0, 1.0, 0.0],
                  [-2.0, -2.0, -2.0, 0.0, 1.0]])
    c = np.array([-4.0, -3.0, -7.0, 0.0, 0.0])
    b = np.array([[-1.0], [-1.5]])
    j_b = [4, 5]
    kappa = dual_plan(A, c, b, j_b)
    print('Kappa = ', kappa)


if __name__ == '__main__':
    main()

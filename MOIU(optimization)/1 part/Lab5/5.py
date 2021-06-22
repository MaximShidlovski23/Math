import numpy as np
import sys

def print_info_iter(delta_sh, j0, H, b_sh, l_sh, theta, sigma, x_sh, j_b, j_b_zv):
    print('delta_sh = ', delta_sh, '\nj0 = ', j0, '\nH = \n', H, '\nb_sh = ', b_sh)
    print('l_sh = ', l_sh, '\ntheta = ', theta, '\nsigma = ', sigma)
    print('x_sh = ', x_sh, '\nj_b = ', j_b, '\nj_b_zv = ', j_b_zv)


def condition_of_existence(A, j_b, j_b_zv, c, D, x_sh):
    if len(j_b) == np.linalg.matrix_rank(A):
        A_b = np.zeros((len(j_b), len(j_b)))
        for i in j_b:
            for j in j_b:
                A_b[j_b.index(i)][j_b.index(j)] = A[i][j]
        if np.linalg.det(A_b) != 0:
            if set(j_b_zv).issuperset(set(j_b)):
                c_sh = c + np.matmul(D, x_sh)
                c_sh_b = np.zeros(len(j_b))
                for i in j_b:
                    c_sh_b[j_b.index(i)] = c_sh[i]
                u_sh = -np.matmul(c_sh_b, np.linalg.inv(A_b))
                delta_sh = np.matmul(u_sh, A) + c_sh
                for i in j_b_zv:
                    if delta_sh[i] < 0:
                        sys.exit()
                H = np.zeros((len(j_b_zv) + len(A), len(j_b_zv) + len(A)))
                A_T = A.T
                for i in j_b_zv:
                    for j in j_b_zv:
                        H[j_b_zv.index(i)][j_b_zv.index(j)] = D[i][j]
                for i in range(len(A)):
                    for j in j_b_zv:
                        H[len(j_b_zv) + i][j_b_zv.index(j)] = A[i][j]
                for i in j_b_zv:
                    for j in range(len(A)):
                        H[j_b_zv.index(i)][len(j_b_zv) + j] = A_T[i][j]
                if np.linalg.det(H) == 0:
                    print('stop')
                    sys.exit()
                return delta_sh, H, A_b

def quadratic_task(A, D, c, x_sh, j_b, j_b_zv, delta_sh, H, A_b):
    count = False
    while True:
        if count:
            delta_sh, H, A_b = condition_of_existence(A, j_b, j_b_zv, c, D, x_sh)
        if all(delta >= 0 for delta in delta_sh):
            print('Текущий правильный опорный план является оптимальным')
            return x_sh
        delta_min = 0
        for i in delta_sh:
            if i < 0:
                delta_min = i
                break
        j0 = delta_sh.tolist().index(delta_min)
        l_sh = np.zeros(len(delta_sh))
        l_sh[j0] = 1
        H_1 = np.linalg.inv(H)
        b_sh = np.zeros(len(H))
        for j in j_b_zv:
            b_sh[j_b_zv.index(j)] = D[j0][j]
        for i in range(len(A)):
            b_sh[len(j_b_zv) + i] = A[i][j0]
        x = -np.matmul(H_1, b_sh.T)
        for i in j_b_zv:
            l_sh[i] = x[j_b_zv.index(i)]
        #+1 запишем тету j0
        theta = np.zeros(len(j_b_zv) + 1)
        sigma = np.matmul((np.matmul(l_sh, D)), l_sh)
        if sigma == 0:
            theta[-1] = np.inf
        elif sigma > 0:
            theta[-1] = abs(delta_sh[j0]) / sigma
        for j in j_b_zv:
            if l_sh[j] >= 0:
                theta[j] = np.inf
            else:
                theta[j] = - x_sh[j] / l_sh[j]
        j_zv = min(theta)
        j_zv_index = theta.tolist().index(j_zv)
        if j_zv == np.inf:
            print('Целевая функция задачи не ограничена снизу на множестве допустимых планом')
            sys.exit()
        x_sh = x_sh + j_zv * l_sh
        j_b_set = set(j_b)
        j_b_zv_set = set(j_b_zv)
        if j_zv_index == j0:
            j_b_zv.append(j_zv_index)
        elif j_zv_index in (j_b_zv_set - j_b_set):
            del j_b_zv[j_zv_index]
        elif j_zv_index in j_b:
            #индекс j_zv на s-ой позиции = j_s
            j_s = j_b.index(j_zv)
            diff_set = j_b_zv_set - j_b_set
            if diff_set:
                for j_plus in diff_set:
                    j_plus_index = j_b_zv.index(j_plus)
                    A_j_plus = np.zeros(len(A_b))
                    for j in range(len(A_b)):
                        A_j_plus[j] = A[j][j_plus]
                    kek_lol = np.matmul(np.linalg.inv(A_b), A_j_plus)
                    if kek_lol[j_s] != 0:
                        j_b[j_s] = j_plus_index
                        del j_b_zv[j_b_zv.index(j_zv)]
        elif j_zv_index in j_b:
            #индекс j_zv на s-ой позиции = j_s
            j_s = j_b.index(j_zv)
            diff_set = j_b_zv_set - j_b_set
            if j_b == j_b_zv:
                j_b[j_s] = j0
                j_b_zv[j_b_zv.index(j_zv)] = j0
            elif diff_set:
                for j_plus in diff_set:
                    j_plus_index = j_b_zv.index(j_plus)
                    A_j_plus = np.zeros(len(A_b))
                    for j in range(len(A_b)):
                        A_j_plus[j] = A[j][j_plus]
                    kek_lol = np.matmul(np.linalg.inv(A_b), A_j_plus)
                    if kek_lol[j_s] == 0:
                        j_b[j_s] = j0
                        j_b_zv[j_b_zv.index(j_zv)] = j0
        count = True


def main():
    c = np.array([-8., -6., -4., -6])
    D = np.array([[2., 1., 1., 0.],
                  [1., 1., 0., 0.],
                  [1., 0., 1., 0.],
                  [0., 0., 0., 0.]])
    A = np.array([[1., 0., 2., 1.],
                  [0., 1., -1., 2.]])
    x_sh = [2, 3, 0, 0]
    j_b = [0, 1]
    j_b_zv = [0, 1]
    delta_sh, H, A_b = condition_of_existence(A, j_b, j_b_zv, c, D, x_sh)
    x_sh = quadratic_task(A, D, c, x_sh, j_b, j_b_zv, delta_sh, H, A_b)
    print(x_sh)

if __name__ == '__main__':
    main()
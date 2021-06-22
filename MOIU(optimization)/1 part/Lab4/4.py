import numpy as np
import sys

#Метод северно-западного узла
def initial_phasa(m, n, a, b):
    i = 0
    j = 0
    x = np.zeros((m, n))
    U_b = []
    check = 0
    iter = 1
    while i < m and j < n:
        pos = (i, j)
        U_b.append(pos)
        if a[i] > b[j]:
            delta = b[j]
            a[i] -= delta
            b[j] = 0
            j += 1
        elif a[i] < b[j]:
            delta = a[i]
            b[j] -= delta
            a[i] = 0
            i += 1
        else:
            if check % 2 == 0:
                delta = a[i]
                b[j] -= delta
                a[i] = 0
                i += 1
            else:
                delta = b[j]
                a[i] -= delta
                b[j] = 0
                j += 1
            check += 1
        print('iter: ', iter)
        x[pos] = delta
        iter += 1
        print('x = \n')
        print(x)
    return U_b, x

def decart(m,n):
    s1 = ''
    s2 = ''
    for s in range(m):
        s1 += str(s)
    for s in range(n):
        s2 += str(s)
    return [(int(a),int(b)) for a in s1 for b in s2]

def system_solution(m, n, U_b, c):
    A_for_system = np.zeros((m + n, m + n))
    A_for_system[0][0] = 1
    b_for_system = np.zeros(m + n)
    i = 1
    for ij in U_b:
        b_for_system[i] = c[ij]
        i += 1
    i = 1
    for pos in U_b:
        A_for_system[i][pos[0]] = 1
        A_for_system[i][m + pos[1]] = 1
        i += 1
    solve = np.linalg.solve(A_for_system, b_for_system)
    u_list, v_list = solve[:m], solve[m:]
    return u_list, v_list

def search_fail_pos(m, n, u_list, v_list, c):
    for j in range(n):
        for i in range(m):
            if u_list[i] + v_list[j] > c[(i, j)]:
                return (i, j)
    return None

def find_pos(prev_pos, start_pos, move_pos, limit, valid_array):
    pos = start_pos
    for rofl in range(limit):
        if valid_array[pos] and pos != prev_pos:
            return pos
        else:
            pos = (pos[0] + move_pos[0], pos[1] + move_pos[1])
            if valid_array[pos] and pos != prev_pos:
                return pos

def solve_transport_task(m, n, a, b, c):
    a = a.copy()
    b = b.copy()
    U_b, x = initial_phasa(m, n, a, b)
    print('Метод северно-западного узла:')
    print('U_b = ', U_b)
    print('x = \n', x)
    print('='*100)
    while True:
        U_not_b = list(set(decart(m, n)) - set(U_b))
        u_list, v_list = system_solution(m, n, U_b, c)
        fail_pos = search_fail_pos (m, n, u_list, v_list, c)
        if fail_pos is None:
            return x

        U_b.append(fail_pos)
        type_array = np.full((m, n), False)
        for pos in U_b:
            type_array[pos] = True
        valid_array = type_array.copy()
        is_removed = True
        while is_removed:
            is_removed = False
            for i in range(m):
                if sum(valid_array[i, :]) == 1:
                    is_removed = True
                    valid_array[i, :] = [False] * n
            for j in range(n):
                if sum(valid_array[:, j]) == 1:
                    is_removed = True
                    valid_array[:, j] = [False] * m

        pos_plus1, pos_plus2, pos_minus1, pos_minus2 = fail_pos, None, None, None
        pos_minus1 = find_pos(pos_plus1, (0, pos_plus1[1]), (1, 0), m, valid_array)
        pos_plus2 = find_pos(pos_minus1, (pos_minus1[0], 0), (0, 1), n, valid_array)
        pos_minus2 = find_pos(pos_plus2, (0, pos_plus2[1]), (1, 0), m, valid_array)

        basis_minus_min = min(x[pos_minus1], x[pos_minus2])
        x[pos_plus1] += basis_minus_min
        x[pos_plus2] += basis_minus_min
        x[pos_minus1] -= basis_minus_min
        x[pos_minus2] -= basis_minus_min

        if x[pos_minus1] == 0:
            U_b.remove(pos_minus1)
        else:
            U_b.remove(pos_minus2)

def main():
    m = 3
    n = 3
    """a = [100, 300, 300]
    b = [300, 200, 200]
    c = np.array([[8, 4, 1],
         [8, 4, 3],
         [9, 7, 5]])"""
    a = [50, 70, 80]
    b = [20, 90, 90]
    c = np.array([[5, 6, 11],
                  [3, 7, 8],
                  [10, 4, 9]])
    """a = [0, 0, 0]
    b = [0, 0, 0]
    c = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])"""
    X = solve_transport_task(m, n, a, b, c)
    print('Оптимальный план перевозок:')
    print(X)

if __name__ == '__main__':
    main()
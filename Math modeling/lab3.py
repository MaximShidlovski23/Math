from bs4 import BeautifulSoup
import numpy as np


def get_marking(file):
    f = open(file, 'r')
    html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    matrix = []
    for lol in soup.find_all('td'):
        matrix.append(lol.next_element)
    del matrix[0:4]
    del matrix[3::4]
    matrix_marking = []
    matrix_transition = []
    for elem in matrix:
        if not elem.isdigit():
            if elem[0] == '[':
                elem = elem.replace('[', '')
                elem = elem.replace(']', '')
                matrix_marking.append(elem.replace(' ', ''))
            else:
                matrix_transition.append(elem)
    return matrix_marking, matrix_transition


def create_diagram(matrix_marking, matrix_transition):
    matrix_diagram = []
    last_elem = matrix_marking[-1]
    del matrix_marking[1::2]
    del matrix_transition[1::2]
    matrix_transition_sym = []
    for elem in matrix_transition:
        matrix_transition_sym.append(elem.replace(' on', ''))
    matrix_marking.append(last_elem)
    count = 1
    matrix_diagram.append((count, matrix_marking[0]))
    for i in range(1, len(matrix_marking)):
        if matrix_marking[i] == matrix_marking[0]:
            count = 1
        else:
            count += 1
            matrix_diagram.append((count, matrix_marking[i]))
    sort_matrix_diagram = list(set(matrix_diagram))
    sort_matrix_diagram = sorted(sort_matrix_diagram, key=lambda x: x[0])
    max_count = max(matrix_diagram, key=lambda x: x[0])[0]
    print("Диаграмма маркировок: ")
    max_mat = list(range(1, max_count + 1))
    matrix_count = []
    for i in max_mat:
        count = 0
        for elem in sort_matrix_diagram:
            if elem[0] == i:
                count += 1
        matrix_count.append(count)
    tree = []
    for elem in sort_matrix_diagram:
        tree.append(elem[1])
    print(sort_matrix_diagram[0][0], ': ', 10 * ' ', tree[0])
    count = 0
    check = -1
    for i in matrix_count:
        count += 1
        if count > 1:
            print(sort_matrix_diagram[check + i][0], ': ', tree[check + 1:i + check + 1])
            check += i
        else:
            check += 1
    print('Все пути диаграммы маркировок:')
    prev_elem = 0
    count = 0
    for elem in matrix_diagram:
        if elem[0] == 1:
            print(elem[1])
            continue
        if elem[0] > prev_elem:
            print(round(len(elem[1]) / 2) * ' ', '|')
            print(round(len(elem[1]) / 2) * ' ', matrix_transition_sym[count])
            print(round(len(elem[1]) / 2) * ' ', '|')
            print(elem[1])
            prev_elem += 1
            count += 1
        else:
            print()
            count += 1
            print(matrix_diagram[0][1])
            print(round(len(elem[1]) / 2) * ' ', '|')
            print(round(len(elem[1]) / 2) * ' ', matrix_transition_sym[count])
            print(round(len(elem[1]) / 2) * ' ', '|')
            print(elem[1])
            count += 1
            prev_elem = elem[0]
    return tree, matrix_diagram


# k-ограниченность
def k_boundedness(tree):
    num_tree = []
    for elem in tree:
        num_tree.append(list(elem))
    sum_mas = []
    max_elem = 0
    for mas in tree:
        sum_m = 0
        for i in mas:
            max_elem = max(max_elem, int(i))
            sum_m += int(i)
        sum_mas.append(sum_m)
    return max_elem, sum_mas, num_tree


def safety(max_elem):
    if max_elem == 1:
        print('сеть Петри является безопасной')
    else:
        print('сеть Петри не является безопасной')


def boundedness(sum_mas, max_elem):
    rise = 0
    not_rise = 0
    for i in range(len(sum_mas) - 1):
        if sum_mas[i] < sum_mas[i + 1]:
            rise += 1
        else:
            not_rise += 1
    if rise < not_rise:
        print('сеть Петри ограничена')
        print('сеть Петри является', max_elem, '- ограниченая')
    else:
        print('сеть Петри неограничена')


def conservative_and_stability(num_tree, matrix_diagram, sum_mas):
    tree_all = []
    for elem in matrix_diagram:
        tree_all.append(elem[1])
    num_tree_all = []
    for elem in tree_all:
        num_tree_all.append(list(elem))
    num_tree_all_int = []
    for i in range(len(num_tree_all)):
        check_mas = []
        for j in range(len(num_tree_all[i])):
            check_mas.append(int(num_tree_all[i][j]))
        num_tree_all_int.append(check_mas)
    num_tree_all_bool = []
    for i in range(len(num_tree_all_int)):
        check_mas = []
        for j in range(len(num_tree_all_int[i])):
            if num_tree_all_int[i][j] == 0:
                check_mas.append(0)
            else:
                check_mas.append(1)
        num_tree_all_bool.append(check_mas)
    check_conservative = 0
    check_stability = 0
    for i in range(len(num_tree_all_bool) - 1):
        for j in range(i + 1, len(num_tree_all_bool)):
            if num_tree_all_bool[i] == num_tree_all_bool[j]:
                if sum(num_tree_all_int[i]) <= sum(num_tree_all_int[j]):
                    check_conservative += 1
                else:
                    check_conservative += -10000
                if j + 1 != len(num_tree_all_bool):
                    if num_tree_all_bool[i + 1] != num_tree_all_bool[j + 1]:
                        check_stability += 1
    if check_conservative > 0:
        print('сеть Петри является консервативной')
        if len(set(sum_mas)) == 1:
            print('сеть Петри является 1-консервативная')
    else:
        print('сеть Петри не является консервативной')
    if check_stability == 0:
        print('сеть Петри является устойчивой')
    else:
        print('сеть Петри не является устойчивой')


# mas_pos = np.zeros(len(num_tree_all_bool[1]))

def free_choice_net_and_marked_graph(D_input):
    mas_check = []
    for j in range(len(D_input[0])):
        check = 0
        for i in range(len(D_input)):
            check += D_input[i][j]
        mas_check.append(check)
    check = 0
    for elem in mas_check:
        if elem > 1:
            print('сеть Петри является сетью свободного выбора')
            print('сеть Петри не является маркированным графом')
            print('сеть Петри не является бесконфликтной сетью')
            break
        else:
            check += 1
    if check == len(mas_check):
        print('сеть Петри не является сетью свободного выбора')
        print('сеть Петри является маркированным графом')
        print('сеть Петри является бесконфликтной сетью')


def automatic_net(D_input, D_output):
    mas_check_input = []
    mas_check_output = []
    for i in range(len(D_input)):
        mas_check_input.append(sum(D_input[i]))
        mas_check_output.append(sum(D_output[i]))
    if max(mas_check_output) > 1 or max(mas_check_input) > 1:
        print('сеть Петри является не автоматной')
    else:
        print('сеть Петри является автоматной')


# задача достижимости
def task_reachabillity(tree, marker):
    if marker in tree:
        print('Достижение', marker, ' возможно')
    else:
        print('Достижение', marker, ' невозможно')


def main():
    path = 'D:\\Matlab\\toolbox\\petrinet2.4\\newnet\\'
    print('Схема №1')
    matrix_marking, matrix_transition = get_marking(path + 'Log1.html')
    tree, matrix_diagram = create_diagram(matrix_marking, matrix_transition)
    D_output_1 = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 1]])

    D_input_1 = np.array([[0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0],
                          [1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0]])
    D_1 = D_output_1 - D_input_1
    print('Характеристики по динамическим ограничениям:')
    max_elem, sum_mas, num_tree = k_boundedness(tree)
    boundedness(sum_mas, max_elem)
    safety(max_elem)
    conservative_and_stability(num_tree, matrix_diagram, sum_mas)
    print('Матрица инцидентности: ')
    print(D_1)
    print('Характеристики по статистическим ограничениям:')
    free_choice_net_and_marked_graph(D_input_1)
    automatic_net(D_input_1, D_output_1)
    task_reachabillity(tree, '11000')
    task_reachabillity(tree, '10001')
    print('Схема №2')
    matrix_marking, matrix_transition = get_marking(path + 'Log2.html')
    tree, matrix_diagram = create_diagram(matrix_marking, matrix_transition)
    print('Характеристики по динамическим ограничениям:')
    max_elem, sum_mas, num_tree = k_boundedness(tree)
    boundedness(sum_mas, max_elem)
    safety(max_elem)
    conservative_and_stability(num_tree, matrix_diagram, sum_mas)
    D_input_2 = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1]])

    D_output_2 = np.array([[0, 1, 1, 0, 0, 1],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [1, 0, 0, 0, 0, 0]])
    D_2 = D_output_2 - D_input_2
    print('Матрица инцидентности: ')
    print(D_2)
    print('Характеристики по статистическим ограничениям:')
    free_choice_net_and_marked_graph(D_input_2)
    automatic_net(D_input_2, D_output_2)
    task_reachabillity(tree, '010012')
    task_reachabillity(tree, '110002')


if __name__ == '__main__':
    main()

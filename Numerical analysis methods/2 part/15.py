import numpy as np
import plotly.graph_objects as go

A = 200
B = 140
R = 40
height = 2
P = 70 * 10 ** 9
E = 60 * 10 ** 6
coef_j = 0.28
n = 76
m = 76
h = A / n
l = B / m
D = (E * height ** 3) / (12 * (1 - coef_j ** 2))
Rad_right = A / 2 + R
Rad_left = A / 2 - R
list_x = [h * i for i in range(n)]
list_y = [l * i for i in range(m)]
half_round = lambda x: (R ** 2 - (x - A / 2) ** 2) ** 0.5

def create_map():
    type_node = np.zeros((m, n))
    index_node = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            type_node[i][0] = 1
            type_node[i][m - 1] = 1
            type_node[n - 1][j] = 1
            type_node[0][j] = 1
            index_node[i][j] = 1
    counter = 0
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if (list_x[j] < Rad_left or list_x[j] > Rad_right):
                index_node[i][j] = counter
                counter += 1
                continue
            y_round = half_round(list_x[j])
            y_cur = list_y[i]
            if (y_cur > y_round):
                index_node[i][j] = counter
                counter += 1
            else:
                type_node[i][j] = 2
                index_node[i][j] = 2
    return type_node, index_node, counter

def create_system(type_node, index_node, counter):
    system = np.zeros((counter, counter))
    counter = 0
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if type_node[i][j] != 2:
                system[counter][int(index_node[i][j])] = -2 * (l ** 2 + h ** 2)
                if type_node[i][j - 1] == 0:
                    system[counter][int(index_node[i][j - 1])] = h ** 2
                if type_node[i][j + 1] == 0:
                    system[counter][int(index_node[i][j + 1])] = h ** 2
                if type_node[i - 1][j] == 0:
                    system[counter][int(index_node[i - 1][j])] = l ** 2
                if type_node[i + 1][j] == 0:
                    system[counter][int(index_node[i + 1][j])] = l ** 2
                counter += 1
    return system

def create_dataplot(type_node, system_solve):
    system_values = np.zeros((m, n))
    counter = 0
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if type_node[i][j] == 2:
                continue
            system_values[i][j] = system_solve[counter]
            counter += 1
    return system_values

def create_3dplot(system_values):
    fig = go.Figure(go.Surface(x=list_x, y=list_y, z=system_values,
                               colorscale=[[0.0, "rgb(49,54,149)"],
                                           [0.1111111111111111, "rgb(69,117,180)"],
                                           [0.2222222222222222, "rgb(116,173,209)"],
                                           [0.3333333333333333, "rgb(171,217,233)"],
                                           [0.4444444444444444, "rgb(224,243,248)"],
                                           [0.5555555555555556, "rgb(254,224,144)"],
                                           [0.6666666666666666, "rgb(253,174,97)"],
                                           [0.7777777777777778, "rgb(244,109,67)"],
                                           [0.8888888888888888, "rgb(215,48,39)"],
                                           [1.0, "rgb(165,0,38)"]]))
    fig.update_layout(title='Прогиб пластины', autosize=True,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

def main():
    type_node, index_node, counter = create_map()
    print(type_node)
    print(index_node)
    system = create_system(type_node, index_node, counter)
    list_f = [P * h ** 2 * l ** 2 / D for i in range(counter)]
    system_solve = np.linalg.solve(system, list_f)
    system_values = create_dataplot(type_node, system_solve)
    create_3dplot(system_values)

if __name__ == '__main__':
    main()
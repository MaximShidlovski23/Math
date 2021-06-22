class p(object):
    def __init__(self, start, end, flow, cap, cost):
        self.start = start
        self.end = end
        self.flow = flow
        self.cap = cap
        self.cost = cost


def add_p(start, end, capacity, cost, graph):
    graph[start].append(p(end, len(graph[end]), 0, capacity, cost))
    graph[end].append(p(start, len(graph[start]) - 1, 0, 0, -cost))


def dijkstra(s, t, nn, k, used_mas, d_mas, pot, parent, graph):
    d_mas[s] = 0
    while True:
        v = -1
        for j in range(nn):
            if v < 0 or d_mas[v] > d_mas[j]:
                if not used_mas[j] and d_mas[j] < k:
                    v = j
        if v == -1 or d_mas[v] == k:
            break
        used_mas[v] = 1
        for j in range(len(graph[v])):
            if graph[v][j].cap > graph[v][j].flow:
                a = graph[v][j].start
                w = d_mas[v] + graph[v][j].cost + pot[v] - pot[a]
                if not used_mas[a] and d_mas[a] > w:
                    d_mas[a] = w
                    parent[a] = graph[v][j].end


def min_cost_max_flow(s, t, nn, k, graph):
    res_min_cost = 0
    pot = [0 for _ in range(nn)]
    parent = [0 for _ in range(nn)]
    while True:
        used_mas = [0 for _ in range(nn)]
        d_mas = [k for _ in range(nn)]
        dijkstra(s, t, nn, k, used_mas, d_mas, pot, parent, graph)
        if not used_mas[t]:
            break
        for i in range(nn):
            if used_mas[i]:
                pot[i] += d_mas[i]
            else:
                pot[i] += d_mas[t]
        delta_flow = k
        temp_cost = 0
        i = t
        while i != s:
            start = graph[i][parent[i]].start
            end = graph[i][parent[i]].end
            delta_flow = min(delta_flow, graph[start][end].cap - graph[start][end].flow)
            temp_cost += graph[start][end].cost
            i = start
        i = t
        while i != s:
            start = graph[i][parent[i]].start
            end = graph[i][parent[i]].end
            graph[start][end].flow += delta_flow
            graph[i][parent[i]].flow -= delta_flow
            i = start
        res_min_cost += delta_flow * temp_cost
    return res_min_cost


def main():
    f_in = open('input.txt')
    lines = f_in.readlines()
    n, m = map(int, lines[0].split())
    a = list(map(int, lines[1].split()))
    b = list(map(int, lines[2].split()))
    c = []
    for i in range(n):
        c.append(list(map(int, lines[i + 3].split())))
    k = 9223372036854775807
    nn = n + m + 2
    graph = []
    for i in range(nn):
        graph.append([])
    for i in range(n):
        add_p(0, i + 1, a[i], 0, graph)
        for j in range(m):
            add_p(i + 1, j + n + 1, k, c[i][j], graph)
    for j in range(m):
        add_p(j + n + 1, nn - 1, b[j], 0, graph)
    res_min_cost = min_cost_max_flow(0, nn - 1, nn, k, graph)
    f_out = open('output.txt', 'w')
    f_out.write(str(res_min_cost))


if __name__ == "__main__":
    main()
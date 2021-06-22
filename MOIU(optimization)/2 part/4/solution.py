class p(object):
    def __init__(self, start, end, cap, flow):
        self.start = start
        self.end = end
        self.cap = cap
        self.flow = flow


def add_p(start, end, cap, graph, p_ob):
    p_1 = p(start, end, cap, 0)
    p_2 = p(end, start, 0, 0)
    graph[start].append(len(p_ob))
    p_ob.append(p_1)
    graph[end].append(len(p_ob))
    p_ob.append(p_2)
    return graph, p_ob


def bfs(s, t, queue, graph, p_ob, n):
    qh = 0
    qt = 0
    qt += 1
    queue[qt] = s
    d = [-1 for _ in range(n)]
    d[s] = 0
    while qh < qt and d[t] == -1:
        qh += 1
        v = queue[qh]
        for i in range(len(graph[v])):
            id = graph[v][i]
            to = p_ob[id].end
            if d[to] == -1 and p_ob[id].flow < p_ob[id].cap:
                qt += 1
                queue[qt] = to
                d[to] = d[v] + 1
    return d[t] != -1, d


def dfs(virtex, flow, t, ptr, graph, p_ob, d):
    if flow == 0:
        return 0
    if virtex == t:
        return flow
    while ptr[virtex] < len(graph[virtex]):
        id = graph[virtex][ptr[virtex]]
        to = p_ob[id].end
        if d[to] != d[virtex] + 1:
            ptr[virtex] += 1
            continue
        check = dfs(to, min(flow, p_ob[id].cap - p_ob[id].flow), t, ptr, graph, p_ob, d)
        if check > 0:
            ti = p_ob[id]
            p_ob[id] = p(ti.start, ti.end, ti.cap, ti.flow + check)
            ti = p_ob[id ^ 1]
            p_ob[id ^ 1] = p(ti.start, ti.end, ti.cap, ti.flow - check)
            return check
        ptr[virtex] += 1
    return 0


def main():
    f_in = open('flow.in')
    lines = f_in.readlines()
    n, m = map(int, lines[0].split())
    s, t = map(int, lines[1].split())
    s = s - 1
    t = t - 1
    queue = [0 for _ in range(n)]
    p_ob = []
    graph = []
    for i in range(n):
        graph.append([])
        queue.append(0)
    for i in range(m):
        start, end, size = map(int, lines[2 + i].split())
        add_p(start - 1, end - 1, size, graph, p_ob)
    flow = 0
    while True:
        check_bfs, d = bfs(s, t, queue, graph, p_ob, n)
        if not check_bfs:
            break
        ptr = [0 for _ in range(n)]
        while True:
            check = dfs(s, 9223372036854775807, t, ptr, graph, p_ob, d)
            if check <= 0:
                break
            flow += check
    f_out = open('flow.out', 'w')
    f_out.write(str(flow) + '\n')
    for i, j in enumerate(p_ob):
        if i % 2 == 0:
            f_out.write(str(p_ob[i].flow) + '\n')


if __name__ == "__main__":
    main()
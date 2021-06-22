import random
import sys

p = []


def dsu_get(v):
    if v == p[v]:
        return v
    else:
        p[v] = dsu_get(p[v])
        return p[v]


def dsu_unite(a, b):
    a = dsu_get(a)
    b = dsu_get(b)
    if random.randint(0, 32767) & 1:
        a, b = b, a
    if a != b:
        p[a] = b


def main():
    n, m, k = map(int, sys.stdin.readline().split())
    cities = dict()
    for i in range(n):
        city = sys.stdin.readline().rstrip()
        cities[city] = i
        p.append(i)
    graph = []
    for i in range(m):
        c1, c2, length = sys.stdin.readline().rstrip().split()
        graph.append((int(length), (cities[c1], cities[c2])))
    cost = 0
    res = []
    graph.sort()
    visit = set()
    for i in range(m):
        c1 = graph[i][1][0]
        c2 = graph[i][1][1]
        length = graph[i][0]
        if dsu_get(c1) != dsu_get(c2):
            cost += length
            res.append(length)
            visit.add(c1)
            visit.add(c2)
            dsu_unite(c1, c2)
    res.sort()
    ans = 0
    res_len = len(res)
    if k >= res_len:
        print(0)
    else:
        res_len -= k
        for i in range(res_len):
            ans += res[i]
        if len(visit) != n:
            print(-1)
        else:
            print(ans)


if __name__ == "__main__":
    main()
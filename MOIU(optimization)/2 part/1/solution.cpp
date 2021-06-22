#include <iostream>
#include <vector>
#include <fstream>
#include <queue>	

using namespace std;

const long long int inf = LLONG_MAX;

int main()
{
	long long int n, m;
	ifstream fin("input.txt");
	fin >> n >> m;
	vector<vector<pair<long long int, long long int>>> graph(n);
	for (long long int i = 0; i < m; i++)
	{
		long long int start, end, length;
		fin >> start >> end >> length;
		graph[start - 1].push_back(make_pair(end - 1, length));
		graph[end - 1].push_back(make_pair(start - 1, length));
	}
	vector<long long int> B(n, inf);
	long long int s = 0;
	B[s] = 0;
	priority_queue <pair<long long int, long long int>> pq;
	pq.push(make_pair(0, s));
	while (!pq.empty())
	{
		long long int f = -pq.top().first;
		long long int v = pq.top().second;
		pq.pop();
		if (f > B[v])
		{
			continue;
		}
		for (pair<long long int, long long int> elem: graph[v])
		{
			long long int to = elem.first;
			long long int C = elem.second;
			if (B[to] > B[v] + C)
			{
				B[to] = B[v] + C;
				pq.push(make_pair(-B[to], to));
			}
		}
	}
	ofstream fout("output.txt");
	fout << B[n - 1];
	fout.close();
}
---
title: '[Note] Một số code ICPC phổ biến'
date: 2023-09-01
permalink: /posts/2023/09/01/mot-so-code-icpc-pho-bien/
tags:
  - research
  - writing
  - icpc
  - acm
--- 

Segment tree
======

Step 1

```c++
#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long

struct segtree {
    int size;
    vector<int> sums;

    void init(int n){
        size = 1;
        while(size < n) size *= 2;
        sums.assign(2 * size, 0LL);
    }
    void set(int i, int v, int x, int lx, int rx){
        if(rx - lx == 1){
            sums[x] = v;
            return;
        }
        int m = (lx + rx) / 2;
        if(i < m){
            set(i, v, 2 * x + 1, lx, m);
        }
        else {
            set(i, v, 2 * x + 2, m, rx);
        }
        sums[x] = sums[2 * x + 1] + sums[2 * x + 2];
    }
    void set(int i, int v){
        set(i, v, 0, 0, size);
    }
    int sum(int l, int r, int x, int lx, int rx){
        if(lx >= r || l >= rx) return 0;
        if(lx >= l && rx <= r) return sums[x];
        int m = (lx + rx) / 2;
        int s1 = sum(l, r, 2 * x + 1, lx, m);
        int s2 = sum(l, r, 2 * x + 2, m, rx);
        return s1 + s2;
    }
    int sum(int l, int r){
        return sum(l, r, 0, 0, size);
    }
};

int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n, m;
    cin >> n >> m;

    segtree st;
    st.init(n);

    for(int i = 0; i < n; i++){
        int v; cin >> v;
        st.set(i, v);
    }
    while(m--){
        int op; cin >> op;
        if(op == 1){
            int i, v; cin >> i >> v;
            st.set(i, v);
        }
        else {
            int l, r; cin >> l >> r;
            cout << st.sum(l, r) << endl;
        }
    }
    return 0;
}

// https://codeforces.com/edu/course/2/lesson/4/1
```


```c++

#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long

struct segtree {
    int size;
    vector<int> sums;

    void init(int n){
        size = 1;
        while(size < n) size *= 2;
        sums.assign(2 * size, 0LL);
    }

    void build(vector<int> &a, int x, int lx, int rx){
        if(rx - lx == 1){
            if(lx < (int)a.size()){
                sums[x] = a[lx];
            }
            return;
        }
        int m = (lx + rx) / 2;
        build(a, 2 * x + 1, lx, m);
        build(a, 2 * x + 2, m, rx);
        sums[x] = sums[2 * x + 1] + sums[2 * x + 2];
    }

    void build(vector<int> &a){
        build(a, 0, 0, size);
    }

    void set(int i, int v, int x, int lx, int rx){
        if(rx - lx == 1){
            sums[x] = v;
            return;
        }
        int m = (lx + rx) / 2;
        if(i < m){
            set(i, v, 2 * x + 1, lx, m);
        }
        else {
            set(i, v, 2 * x + 2, m, rx);
        }
        sums[x] = sums[2 * x + 1] + sums[2 * x + 2];
    }
    void set(int i, int v){
        set(i, v, 0, 0, size);
    }
    int sum(int l, int r, int x, int lx, int rx){
        if(lx >= r || l >= rx) return 0;
        if(lx >= l && rx <= r) return sums[x];
        int m = (lx + rx) / 2;
        int s1 = sum(l, r, 2 * x + 1, lx, m);
        int s2 = sum(l, r, 2 * x + 2, m, rx);
        return s1 + s2;
    }
    int sum(int l, int r){
        return sum(l, r, 0, 0, size);
    }
};

int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n, m;
    cin >> n >> m;

    segtree st;
    st.init(n);

    vector<int> a(n);

    for(int i = 0; i < n; i++){
        cin >> a[i];
    }
    st.build(a);

    while(m--){
        int op; cin >> op;
        if(op == 1){
            int i, v; cin >> i >> v;
            st.set(i, v);
        }
        else {
            int l, r; cin >> l >> r;
            cout << st.sum(l, r) << endl;
        }
    }
    return 0;
}

// https://codeforces.com/edu/course/2/lesson/4/1
```

```c++

#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long

struct segtree {
    int size;
    vector<int> values;

    void init(int n){
        size = 1;
        while(size < n) size *= 2;
        values.assign(2 * size, 0LL);
    }

    void build(vector<int> &a, int x, int lx, int rx){
        if(rx - lx == 1){
            if(lx < (int)a.size()){
                values[x] = a[lx];
            }
            return;
        }
        int m = (lx + rx) / 2;
        build(a, 2 * x + 1, lx, m);
        build(a, 2 * x + 2, m, rx);
        values[x] = min(values[2 * x + 1], values[2 * x + 2]);
    }

    void build(vector<int> &a){
        build(a, 0, 0, size);
    }

    void set(int i, int v, int x, int lx, int rx){
        if(rx - lx == 1){
            values[x] = v;
            return;
        }
        int m = (lx + rx) / 2;
        if(i < m){
            set(i, v, 2 * x + 1, lx, m);
        }
        else {
            set(i, v, 2 * x + 2, m, rx);
        }
        values[x] = min(values[2 * x + 1], values[2 * x + 2]);
    }
    void set(int i, int v){
        set(i, v, 0, 0, size);
    }
    int calc(int l, int r, int x, int lx, int rx){
        if(lx >= r || l >= rx) return INT_MAX;
        if(lx >= l && rx <= r) return values[x];
        int m = (lx + rx) / 2;
        int s1 = calc(l, r, 2 * x + 1, lx, m);
        int s2 = calc(l, r, 2 * x + 2, m, rx);
        return min(s1, s2);
    }
    int calc(int l, int r){
        return calc(l, r, 0, 0, size);
    }
};

int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n, m;
    cin >> n >> m;

    segtree st;
    st.init(n);

    vector<int> a(n);

    for(int i = 0; i < n; i++){
        cin >> a[i];
    }
    st.build(a);

    while(m--){
        int op; cin >> op;
        if(op == 1){
            int i, v; cin >> i >> v;
            st.set(i, v);
        }
        else {
            int l, r; cin >> l >> r;
            cout << st.calc(l, r) << endl;
        }
    }
    return 0;
}

// https://codeforces.com/edu/course/2/lesson/4/1
```

```c++
#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long

struct item {
    int m, c;
};

struct segtree {
    int size;
    vector<item> values;
    item NEURAL_ELEMENT = {INT_MAX, 0};

    item merge(item a, item b){
        if(a.m < b.m) return a;
        if(a.m > b.m) return b;
        return {a.m, a.c + b.c};
    }

    item single(int v){
        return {v, 1};
    }

    void init(int n){
        size = 1;
        while(size < n) size *= 2;
        values.resize(2 * size);
    }

    void build(vector<int> &a, int x, int lx, int rx){
        if(rx - lx == 1){
            if(lx < (int)a.size()){
                values[x] = single(a[lx]);
            }
            return;
        }
        int m = (lx + rx) / 2;
        build(a, 2 * x + 1, lx, m);
        build(a, 2 * x + 2, m, rx);
        values[x] = merge(values[2 * x + 1], values[2 * x + 2]);
    }

    void build(vector<int> &a){
        build(a, 0, 0, size);
    }

    void set(int i, int v, int x, int lx, int rx){
        if(rx - lx == 1){
            values[x] = single(v);
            return;
        }
        int m = (lx + rx) / 2;
        if(i < m){
            set(i, v, 2 * x + 1, lx, m);
        }
        else {
            set(i, v, 2 * x + 2, m, rx);
        }
        values[x] = merge(values[2 * x + 1], values[2 * x + 2]);
    }
    void set(int i, int v){
        set(i, v, 0, 0, size);
    }
    item calc(int l, int r, int x, int lx, int rx){
        if(lx >= r || l >= rx) return NEURAL_ELEMENT;
        if(lx >= l && rx <= r) return values[x];
        int m = (lx + rx) / 2;
        item s1 = calc(l, r, 2 * x + 1, lx, m);
        item s2 = calc(l, r, 2 * x + 2, m, rx);
        return merge(s1, s2);
    }
    item calc(int l, int r){
        return calc(l, r, 0, 0, size);
    }
};

int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n, m;
    cin >> n >> m;

    segtree st;
    st.init(n);

    vector<int> a(n);

    for(int i = 0; i < n; i++){
        cin >> a[i];
    }
    st.build(a);

    while(m--){
        int op; cin >> op;
        if(op == 1){
            int i, v; cin >> i >> v;
            st.set(i, v);
        }
        else {
            int l, r; cin >> l >> r;
            auto res = st.calc(l, r);
            cout << res.m << " " << res.c << endl;
        }
    }
    return 0;
}

// https://codeforces.com/edu/course/2/lesson/4/1

```

Step 2

```c++
#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long

struct item {
    int seg, pref, suf, sum;
};

struct segtree {
    int size;
    vector<item> values;
    item NEURAL_ELEMENT = {0, 0, 0, 0};

    item merge(item a, item b){
        return  {
            max(a.seg, max(b.seg, a.suf + b.pref)),
            max(a.pref, a.sum + b.pref),
            max(b.suf, a.suf + b.sum),
            a.sum + b.sum
        };
    }

    item single(int v){
        if(v > 0){
            return {v, v, v, v};
        }
        else {
            return {0, 0, 0, v};
        }
    }

    void init(int n){
        size = 1;
        while(size < n) size *= 2;
        values.resize(2 * size);
    }

    void build(vector<int> &a, int x, int lx, int rx){
        if(rx - lx == 1){
            if(lx < (int)a.size()){
                values[x] = single(a[lx]);
            }
            return;
        }
        int m = (lx + rx) / 2;
        build(a, 2 * x + 1, lx, m);
        build(a, 2 * x + 2, m, rx);
        values[x] = merge(values[2 * x + 1], values[2 * x + 2]);
    }

    void build(vector<int> &a){
        build(a, 0, 0, size);
    }

    void set(int i, int v, int x, int lx, int rx){
        if(rx - lx == 1){
            values[x] = single(v);
            return;
        }
        int m = (lx + rx) / 2;
        if(i < m){
            set(i, v, 2 * x + 1, lx, m);
        }
        else {
            set(i, v, 2 * x + 2, m, rx);
        }
        values[x] = merge(values[2 * x + 1], values[2 * x + 2]);
    }
    void set(int i, int v){
        set(i, v, 0, 0, size);
    }
    item calc(int l, int r, int x, int lx, int rx){
        if(lx >= r || l >= rx) return NEURAL_ELEMENT;
        if(lx >= l && rx <= r) return values[x];
        int m = (lx + rx) / 2;
        item s1 = calc(l, r, 2 * x + 1, lx, m);
        item s2 = calc(l, r, 2 * x + 2, m, rx);
        return merge(s1, s2);
    }
    item calc(int l, int r){
        return calc(l, r, 0, 0, size);
    }
};

int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n, m;
    cin >> n >> m;

    segtree st;
    st.init(n);

    vector<int> a(n);

    for(int i = 0; i < n; i++){
        cin >> a[i];
    }
    st.build(a);
    cout << st.calc(0, n).seg << endl;

    while(m--){
        int i, v; cin >> i >> v;
        st.set(i, v);
        cout << st.calc(0, n).seg << endl;
    }
    return 0;
}

// https://codeforces.com/edu/course/2/lesson/4/1
```

```c++

#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long

struct segtree {
    int size;
    vector<int> operations;

    void init(int n){
        size = 1;
        while(size < n) size *= 2;
        operations.assign(2 * size, 0LL);
    }

    void add(int l, int r, int v, int x, int lx, int rx){
        if(lx >= r || l >= rx) return;
        if(lx >= l && rx <= r){
            operations[x] += v;
            return;
        }
        int m = (lx + rx) / 2;
        add(l, r, v, 2 * x + 1, lx, m);
        add(l, r, v, 2 * x + 2, m, rx);
    }
    void add(int l, int r, int v){
        add(l, r, v, 0, 0, size);
    }
    int get(int i, int x, int lx, int rx){
        if(rx - lx == 1){
            return operations[x];
        }
        int m = (lx + rx) / 2;
        int res = 0;
        if(i < m){
            res = get(i, 2 * x + 1, lx, m);
        }
        else {
            res = get(i, 2 * x + 2, m, rx);
        }
        return res + operations[x];
    }

    int get(int i){
        return get(i, 0, 0, size);
    }
};

int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n, m;
    cin >> n >> m;

    segtree st;
    st.init(n);

    while(m--){
        int op; cin >> op;
        if(op == 1){
            int l, r, v; cin >> l >> r >> v;
            st.add(l, r, v);
        }
        else {
            int i; cin >> i;
            cout << st.get(i) << endl;
        }
    }
    return 0;
}

// https://codeforces.com/edu/course/2/lesson/5/1
```

```c++

#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long

struct segtree {
    int size;
    vector<int> operations;

    int operation(int a, int b){
        return max(a, b);
    }

    void init(int n){
        size = 1;
        while(size < n) size *= 2;
        operations.assign(2 * size, 0LL);
    }

    void add(int l, int r, int v, int x, int lx, int rx){
        if(lx >= r || l >= rx) return;
        if(lx >= l && rx <= r){
            operations[x] = operation(operations[x], v);
            return;
        }
        int m = (lx + rx) / 2;
        add(l, r, v, 2 * x + 1, lx, m);
        add(l, r, v, 2 * x + 2, m, rx);
    }
    void add(int l, int r, int v){
        add(l, r, v, 0, 0, size);
    }
    int get(int i, int x, int lx, int rx){
        if(rx - lx == 1){
            return operations[x];
        }
        int m = (lx + rx) / 2;
        int res = 0;
        if(i < m){
            res = get(i, 2 * x + 1, lx, m);
        }
        else {
            res = get(i, 2 * x + 2, m, rx);
        }
        return operation(res, operations[x]);
    }

    int get(int i){
        return get(i, 0, 0, size);
    }
};

int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n, m;
    cin >> n >> m;

    segtree st;
    st.init(n);

    while(m--){
        int op; cin >> op;
        if(op == 1){
            int l, r, v; cin >> l >> r >> v;
            st.add(l, r, v);
        }
        else {
            int i; cin >> i;
            cout << st.get(i) << endl;
        }
    }
    return 0;
}

// https://codeforces.com/edu/course/2/lesson/5/1
```


Matrix power
======

```c++
#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long
#define MOD ((int)1e9 + 7)

struct Matrix {
    vector< vector<int> > data;
    int m, n;

    Matrix(int m, int n) : m(m), n(n){
        data.resize(m);
        for(int i = 0; i < m; i++){
            data[i].resize(n);
        }
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                data[i][j] = 0;
            }
        }
    }

    Matrix(vector<int> &a, int m, int n) : m(m), n(n){
        data.resize(m);
        for(int i = 0; i < m; i++){
            data[i].resize(n);
        }
        int k = 0;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                data[i][j] = a[k++];
            }
        }
    }

    void print(){
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }
    Matrix operator *(const Matrix &b) {
        Matrix a = *this;
        assert(a.n == b.m);
        Matrix res(a.m, b.n);

        for (int i = 0; i < a.m; i++){
            for (int j = 0; j < b.n; j++){
                res.data[i][j] = 0;
                for (int k = 0; k < a.n; k++){
                    res.data[i][j] += ((a.data[i][k] % MOD) * (b.data[k][j] % MOD) ) % MOD;
                    res.data[i][j] %= MOD;
                }
            }
        }
        return res;
    }

    Matrix pow(int k){
        assert(this->m == this->n);
        Matrix base = *this;
        if(k == 1) return base;
        if(k % 2) return base * pow(k - 1);
        Matrix tmp = pow(k/2);
        return tmp * tmp;
    }
};

/*
f1
f0

1 1     f1    f2
1 0     f0    f1

1 1     f2    f3
1 0     f1    f2

1 1   1 1   2 1
1 0   1 0   1 1
*/


int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    /*
    vector<int> a; int m, n; cin >> m >> n;
    a.resize(m * n);
    for(int &x : a) cin >> x;
    for(int i = 0; i < m * n; i++){
        cout << a[i] << endl;
    }
    */
    vector<int> data = {1, 1, 1, 0};
    auto mt = Matrix(data, 2, 2);

    int f1 = 1, f0 = 0;
    vector<int> b = {f1, f0};
    auto base = Matrix(b, 2, 1);

    int n; cin >> n;
    if(n == 0) {cout << 0; return 0;}
    if(n == 0) {cout << 1; return 0;}

    auto c = mt.pow(n - 1) * base;
    cout << c.data[0][0] << endl;


    return 0;
}

```

Graph
======

Code Dinic:
```c++

#include <bits/stdc++.h>
using namespace std;

#define INF 1000000000

struct Edge {
    int v, flow, cap, rev;
};

void addEdge(vector<vector<Edge>>& graph, int u, int v, int cap) {
    Edge forward = {v, 0, cap, graph[v].size()};
    Edge backward = {u, 0, 0, graph[u].size()};
    graph[u].push_back(forward);
    graph[v].push_back(backward);
}

bool bfs(vector<vector<Edge>>& graph, int s, int t, vector<int>& level) {
    fill(level.begin(), level.end(), -1);
    queue<int> q;
    q.push(s);
    level[s] = 0;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (Edge& e : graph[u]) {
            if (level[e.v] < 0 && e.flow < e.cap) {
                level[e.v] = level[u] + 1;
                q.push(e.v);
            }
        }
    }
    return level[t] >= 0;
}

int dfs(vector<vector<Edge>>& graph, vector<int>& ptr, vector<int>& level, int u, int t, int flow) {
    if (u == t || flow == 0) return flow;
    for (int& i = ptr[u]; i < graph[u].size(); i++) {
        Edge& e = graph[u][i];
        if (level[e.v] == level[u] + 1 && e.flow < e.cap) {
            int pushed = dfs(graph, ptr, level, e.v, t, min(flow, e.cap - e.flow));
            if (pushed > 0) {
                e.flow += pushed;
                graph[e.v][e.rev].flow -= pushed;
                return pushed;
            }
        }
    }
    return 0;
}

int dinicMaxFlow(vector<vector<Edge>>& graph, int s, int t) {
    int maxFlow = 0;
    vector<int> level(graph.size()), ptr(graph.size());
    while (bfs(graph, s, t, level)) {
        fill(ptr.begin(), ptr.end(), 0);
        while (int flow = dfs(graph, ptr, level, s, t, INF)) {
            maxFlow += flow;
        }
    }
    return maxFlow;
}

int main() {
    int V = 6;
    vector<vector<Edge>> graph(V);
    addEdge(graph, 0, 1, 16);
    addEdge(graph, 0, 2, 13);
    addEdge(graph, 1, 2, 10);
    addEdge(graph, 1, 3, 12);
    addEdge(graph, 2, 1, 4);
    addEdge(graph, 2, 4, 14);
    addEdge(graph, 3, 2, 9);
    addEdge(graph, 3, 5, 20);
    addEdge(graph, 4, 3, 7);
    addEdge(graph, 4, 5, 4);

    int source = 0, sink = 5;
    cout << "Max flow from source to sink: " << dinicMaxFlow(graph, source, sink) << endl;

    return 0;
}

```


Code dijkstra

```c++

#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

typedef pair<int, int> pii;
const int INF = numeric_limits<int>::max();

void dijkstra(vector<vector<pii>>& graph, int start, vector<int>& dist) {
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        pii top = pq.top(); pq.pop();
        int d = top.first, u = top.second;
        if (d > dist[u]) continue;
        for (auto& neighbor : graph[u]) {
            int v = neighbor.first, w = neighbor.second;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
}

int main() {
    int V = 5;
    vector<vector<pii>> graph(V);
    graph[0].emplace_back(1, 10);
    graph[0].emplace_back(2, 5);
    graph[1].emplace_back(2, 3);
    graph[1].emplace_back(3, 1);
    graph[2].emplace_back(1, 2);
    graph[2].emplace_back(3, 9);
    graph[2].emplace_back(4, 2);
    graph[3].emplace_back(4, 4);
    graph[4].emplace_back(3, 6);

    vector<int> dist(V, INF);
    dijkstra(graph, 0, dist);

    for (int i = 0; i < V; ++i) {
        cout << "Distance from source to " << i << " is " << dist[i] << endl;
    }

    return 0;
}

```

Code floyd

```c++
#include <iostream>
#include <vector>
#include <limits>

using namespace std;

const int INF = numeric_limits<int>::max();

void floydWarshall(vector<vector<int>>& graph, int V) {
    vector<vector<int>> dist = graph;

    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == INF) {
                cout << "INF ";
            } else {
                cout << dist[i][j] << " ";
            }
        }
        cout << endl;
    }
}

int main() {
    int V = 4;
    vector<vector<int>> graph = {
        {0, 5, INF, 10},
        {INF, 0, 3, INF},
        {INF, INF, 0, 1},
        {INF, INF, INF, 0}
    };

    floydWarshall(graph, V);

    return 0;
}


```

Code prim

```c++
#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

#define INF numeric_limits<int>::max()

int primMST(vector<vector<pair<int, int>>>& graph) {
    int V = graph.size();
    vector<int> key(V, INF), parent(V, -1);
    vector<bool> inMST(V, false);

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, 0});
    key[0] = 0;

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;
        for (auto& neighbor : graph[u]) {
            int v = neighbor.first, weight = neighbor.second;
            if (!inMST[v] && weight < key[v]) {
                parent[v] = u;
                key[v] = weight;
                pq.push({key[v], v});
            }
        }
    }

    int minCost = 0;
    for (int i = 1; i < V; i++) {
        minCost += key[i];
    }
    return minCost;
}

int main() {
    int V = 5;
    vector<vector<pair<int, int>>> graph(V);
    graph[0].emplace_back(1, 2);
    graph[0].emplace_back(3, 6);
    graph[1].emplace_back(0, 2);
    graph[1].emplace_back(2, 3);
    graph[1].emplace_back(3, 8);
    graph[1].emplace_back(4, 5);
    graph[2].emplace_back(1, 3);
    graph[2].emplace_back(4, 7);
    graph[3].emplace_back(0, 6);
    graph[3].emplace_back(1, 8);
    graph[4].emplace_back(1, 5);
    graph[4].emplace_back(2, 7);

    cout << "Minimum cost of MST: " << primMST(graph) << endl;

    return 0;
}
```

```c++

#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

typedef pair<int, int> pii;
const int INF = numeric_limits<int>::max();

void dijkstra(vector<vector<pii>>& graph, int start, int k) {
    int V = graph.size();
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    vector<vector<int>> dist(V, vector<int>(k, INF));
    pq.push({0, start});
    dist[start][0] = 0;

    while (!pq.empty()) {
        pii top = pq.top(); pq.pop();
        int d = top.first, u = top.second;
        if (d > dist[u][k - 1]) continue;
        for (auto& neighbor : graph[u]) {
            int v = neighbor.first, w = neighbor.second;
            for (int i = 0; i < k; ++i) {
                int cost = dist[u][i] + w;
                if (cost < dist[v][k - 1]) {
                    dist[v][k - 1] = cost;
                    sort(dist[v].begin(), dist[v].end());
                    pq.push({dist[v][k - 1], v});
                }
            }
        }
    }

    cout << "Shortest path of length " << k << " from source " << start << ":" << endl;
    for (int i = 0; i < V; ++i) {
        cout << "To " << i << ": " << dist[i][k - 1] << endl;
    }
}

int main() {
    int V = 5;
    vector<vector<pii>> graph(V);
    graph[0].emplace_back(1, 10);
    graph[0].emplace_back(2, 5);
    graph[1].emplace_back(2, 3);
    graph[1].emplace_back(3, 1);
    graph[2].emplace_back(1, 2);
    graph[2].emplace_back(3, 9);
    graph[2].emplace_back(4, 2);
    graph[3].emplace_back(4, 4);
    graph[4].emplace_back(3, 6);

    int k = 3; // Độ dài của đường đi muốn tìm
    dijkstra(graph, 0, k);

    return 0;
}

```


Cây khung

```c++
#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

#define INF numeric_limits<int>::max()

int primMST(vector<vector<pair<int, int>>>& graph) {
    int V = graph.size();
    vector<int> key(V, INF), parent(V, -1);
    vector<bool> inMST(V, false);

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, 0});
    key[0] = 0;

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;
        for (auto& neighbor : graph[u]) {
            int v = neighbor.first, weight = neighbor.second;
            if (!inMST[v] && weight < key[v]) {
                parent[v] = u;
                key[v] = weight;
                pq.push({key[v], v});
            }
        }
    }

    int minCost = 0;
    for (int i = 1; i < V; i++) {
        minCost += key[i];
    }
    return minCost;
}

int main() {
    int V = 5;
    vector<vector<pair<int, int>>> graph(V);
    graph[0].emplace_back(1, 2);
    graph[0].emplace_back(3, 6);
    graph[1].emplace_back(0, 2);
    graph[1].emplace_back(2, 3);
    graph[1].emplace_back(3, 8);
    graph[1].emplace_back(4, 5);
    graph[2].emplace_back(1, 3);
    graph[2].emplace_back(4, 7);
    graph[3].emplace_back(0, 6);
    graph[3].emplace_back(1, 8);
    graph[4].emplace_back(1, 5);
    graph[4].emplace_back(2, 7);

    cout << "Minimum cost of MST: " << primMST(graph) << endl;

    return 0;
}

```

Thuật toán Tarjan: Tìm thành phần liên thông mạnh của đồ thị có hướng.

```c++
#include <iostream>
#include <vector>

using namespace std;

struct Edge {
  int u, v;
};

class Graph {
 public:
  int n;
  vector<vector<Edge>> adj;

  Graph(int n) : n(n) {
    adj.resize(n);
  }

  void addEdge(int u, int v) {
    adj[u].push_back({u, v});
  }
};

void dfs(Graph& g, int u, vector<bool>& visited, vector<int>& low, int& time) {
  visited[u] = true;
  low[u] = time++;

  for (Edge e : g.adj[u]) {
    int v = e.v;
    if (!visited[v]) {
      dfs(g, v, visited, low, time);
      low[u] = min(low[u], low[v]);
    } else if (e.u != g.adj[v][0].v) {
      low[u] = min(low[u], low[v]);
    }
  }
}

vector<vector<int>> tarjan(Graph& g) {
  int n = g.n;
  vector<bool> visited(n, false);
  vector<int> low(n, -1);
  int time = 0;

  for (int u = 0; u < n; ++u) {
    if (!visited[u]) {
      dfs(g, u, visited, low, time);
    }
  }

  vector<vector<int>> components;
  for (int u = 0; u < n; ++u) {
    int p = g.adj[u][0].v;
    if (low[u] == low[p]) {
      components.push_back({u});
      for (int v = u + 1; v < n; ++v) {
        if (low[v] == low[p]) {
          components.back().push_back(v);
        }
      }
    }
  }
  return components;
}

int main() {
  int n, m;
  cin >> n >> m;

  Graph g(n);
  for (int i = 0; i < m; ++i) {
    int u, v;
    cin >> u >> v;
    g.addEdge(u, v);
  }

  vector<vector<int>> components = tarjan(g);

  for (vector<int> component : components) {
    for (int v : component) {
      cout << v << " ";
    }
    cout << endl;
  }

  return 0;
}

```

Thuật toán Topological Sort: Sắp xếp đồ thị có hướng.

```c++
#include <iostream>
#include <vector>

using namespace std;

struct Edge {
  int u, v;
};

class Graph {
 public:
  int n;
  vector<vector<Edge>> adj;

  Graph(int n) : n(n) {
    adj.resize(n);
  }

  void addEdge(int u, int v) {
    adj[u].push_back({u, v});
  }
};

void dfs(Graph& g, int u, vector<bool>& visited) {
  visited[u] = true;

  for (Edge e : g.adj[u]) {
    int v = e.v;
    if (!visited[v]) {
      dfs(g, v, visited);
    }
  }
}

vector<int> topologicalSort(Graph& g) {
  int n = g.n;
  vector<bool> visited(n, false);
  vector<int> order;

  for (int u = 0; u < n; ++u) {
    if (!visited[u]) {
      dfs(g, u, visited);
    }
  }

  for (int u = n - 1; u >= 0; --u) {
    order.push_back(u);
  }

  return order;
}

int main() {
  int n, m;
  cin >> n >> m;

  Graph g(n);
  for (int i = 0; i < m; ++i) {
    int u, v;
    cin >> u >> v;
    g.addEdge(u, v);
  }

  vector<int> order = topologicalSort(g);

  for (int u : order) {
    cout << u << " ";
  }
  cout << endl;

  return 0;
}

```

Ref1: 

[Link1](https://phuchaodo.github.io/posts/2023/09/05/tong-hop-mot-so-tai-lieu-cho-icpc/)

[Link2](https://phuchaodo.github.io/posts/2023/09/01/mot-so-code-icpc-pho-bien/)



Hết.

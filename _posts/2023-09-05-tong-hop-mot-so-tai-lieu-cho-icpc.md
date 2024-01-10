---
title: '[Note] Tổng hợp một số tài liệu cho ICPC'
date: 2023-09-05
permalink: /posts/2023/09/05/tong-hop-mot-so-tai-lieu-cho-icpc/
tags:
  - research
  - writing
  - icpc
  - acm
--- 

Tổng hợp với số tài liệu và kiến thức cơ bản cho cuộc thi ICPC hằng năm.

Các bước để giải một bài toán quy hoạch động
======

## Bước 1

* Cố gắng giải bài toán theo hướng đệ quy
* Ở bước này, chúng ta không cần quan tâm đến độ phức tạp
* Quan trọng cách tiếp cận để giải nó trước. Thường ở bước này độ phức tạp cực lớn.

## Bước 2

* Vì khi sử dụng đệ quy, sẽ có nhiều hàm gọi nhiều lần, nên chúng ta sẽ dùng thêm bộ nhớ để lưu trữ.
* Ở bước này gọi là Top-Down DP (làm từ n đến 0, từ trên xuống)

## Bước 3

* Bottom-up DP
* Thay vì dùng đệ quy, chúng ta sử dụng mảng để chứa các giá trị và thực hiện tính toán trên mảng đó.

## Bước 4

* Dựa vào cái mảng từ bước 3, chúng ta có thể phát hiện được những giá trị chỉ sử dụng 1 hoặc 2 lần.
* Cải thiện về không gian.

List bài tập và tư duy về đệ quy được liệt kê ở cuối blog này.

Template cho các cuộc thi ICPC
======

## Ngôn ngữ python

* Code mẫu 01

```python
'''input

'''
import sys
import math
import bisect
from sys import stdin,stdout
from math import gcd,floor,sqrt,log
from collections import defaultdict as dd
from bisect import bisect_left as bl,bisect_right as br

sys.setrecursionlimit(100000000)

inp    =lambda: int(input())
strng  =lambda: input().strip()
jn     =lambda x,l: x.join(map(str,l))
strl   =lambda: list(input().strip())
mul    =lambda: map(int,input().strip().split())
mulf   =lambda: map(float,input().strip().split())
seq    =lambda: list(map(int,input().strip().split()))

ceil   =lambda x: int(x) if(x==int(x)) else int(x)+1
ceildiv=lambda x,d: x//d if(x%d==0) else x//d+1

flush  =lambda: stdout.flush()
stdstr =lambda: stdin.readline()
stdint =lambda: int(stdin.readline())
stdpr  =lambda x: stdout.write(str(x))

mod=1000000007


#main code

```

* Code mẫu 02

```python
import random
import math
from collections import defaultdict, Counter, deque, OrderedDict
from heapq import heapify, heappush, heappop
from functools import lru_cache, reduce
from bisect import bisect_left, bisect_right
from types import GeneratorType
import sys

input = lambda : sys.stdin.readline().strip()

class SegmentTree:
    def __init__(self, arr, func = lambda x, y : x + y, defaultvalue = 0) :
        self.n = len(arr)
        self.segmentTree = [0]*self.n + arr
        self.func = func
        self.defaultvalue = defaultvalue
        self.buildSegmentTree(arr)

    def buildSegmentTree(self, arr) :   
        for i in range(self.n -1, 0, -1) :
            self.segmentTree[i] = self.func(self.segmentTree[2*i] , self.segmentTree[2*i+1])         
            
    def query(self, l, r) :
        l += self.n
        r += self.n
        res = self.defaultvalue
        while l < r :
            if l & 1 :   
                res = self.func(res, self.segmentTree[l])
                l += 1
            l >>= 1
            if r & 1 :  
                r -= 1      
                res = self.func(res, self.segmentTree[r]) 
            r >>= 1
        return res

    def update(self, i, value) :
        i += self.n
        self.segmentTree[i] = value  
        while i > 1 :
            i >>= 1         
            self.segmentTree[i] = self.func(self.segmentTree[2*i] , self.segmentTree[2*i+1])


class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = list(range(n))
        self.count = [1]*n
    def find(self, x):
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]
    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.parents[x] = y
            self.count[y] += self.count[x]

dire = [0,1,0,-1,0]

def is_prime(n):
    if n==1:
        return False
    for i in range(2,int(n**0.5)+1):
        if n%i==0:
            return False
    return True

def ncr(n, r, p):
    num = den = 1
    for i in range(r):
        num = (num * (n - i)) % p
        den = (den * (i + 1)) % p
    return (num * pow(den,
            p - 2, p)) % p

def case(t):
    print("Case #{}:".format(t), end=" ")

# For codeforces - hashmap to avoid TLE
RANDOM = random.randrange(2**62)
def mapping_wrapper(x):
  return x ^ RANDOM

class HashMap(dict):
    def __setitem__(self, key, value):
        super().__setitem__(mapping_wrapper(key), value)
    def __getitem__(self, key):
        return super().__getitem__(mapping_wrapper(key))
    def __contains__(self, key):
        return super().__contains__(mapping_wrapper(key))


MOD = 10**9 + 7

def binpow(a, b):
    if b==0:
        return 1
    res = binpow(a,b//2)
    res = pow(res,2,MOD)
    if b%2:
        return (res*a)%MOD
    return res

def mod_inverse(a):
    return binpow(a,MOD-2)

MAX = 2*(10**5)+5

def factors(n): 
    if n==0:
        return set()   
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

# factors = [factors(i) for i in range(MAX)]

# factorial and inverse factorial

# fact = [1]*MAX
# invfact = [1]*MAX
# for i in range(1,MAX):
#     fact[i] = (fact[i-1]*i)%MOD
#     invfact[i] = (invfact[i-1]*mod_inverse(i))%MOD

# recursion limit fix decorator, change 'return' to 'yield' and add 'yield' before calling the function
def bootstrap(f):  
    stack = []
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

###############################################################################

def solve():
    pass

###############################################################################

for t in range(int(input())):
    # case(t+1)
    solve()
```

## Ngôn ngữ C++

```c++
/* source for codeblocks */
#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>
#define PI 3.141592653589793238462643383279502884197

using namespace std;

int x, y;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> x >> y;

    if(x == 0) {
        float hs = 0;

        if(y > 0) {
            hs = .5;
        }
        else if(y < 0) {
            hs = 1.5;
        }
        float res = hs * PI;
        printf("%.10f", res); return 0;
    }
    if(x > 0){
        if(y > 0){
            float tmp = atan(1.0 * y / x);
            printf("%.10f", tmp);
            return 0;
        }
        if(y < 0){
            float tmp = atan(-1.0 * y / x);
            printf("%.10f", (2 * PI - tmp));
            return 0;
        }

        printf("%.10f", 0.0);
        return 0;
    }
    if(x < 0){
        if(y == 0){
            printf("%.10f", PI); return 0;
        }
        if(y > 0){
            float tmp = atan(-1.0 * y / x);
            printf("%.10f", (PI - tmp));
            return 0;
        }
        if(y < 0){
            float tmp = atan(1.0 * y / x);
            printf("%.10f", (PI + tmp));
            return 0;
        }
    }

    return 0;
}
```

```c++
#include <bits/stdc++.h>

using namespace std;

template<typename A, typename B> ostream& operator<<(ostream &os, const pair<A, B> &p) { return os << '(' << p.first << ", " << p.second << ')'; }
template<typename T_container, typename T = typename enable_if<!is_same<T_container, string>::value, typename T_container::value_type>::type> ostream& operator<<(ostream &os, const T_container &v) { os << '{'; string sep; for (const T &x : v) os << sep << x, sep = ", "; return os << '}'; }
void dbg_out() { cerr << endl; }
template<typename Head, typename... Tail> void dbg_out(Head H, Tail... T) { cerr << ' ' << H; dbg_out(T...); }
#ifdef LOCAL
#define dbg(...) cerr << "(" << #__VA_ARGS__ << "):", dbg_out(__VA_ARGS__)
#else
#define dbg(...)
#endif

#define ar array
#define ll long long
#define ld long double
#define sza(x) ((int)x.size())
#define all(a) (a).begin(), (a).end()

const int MAX_N = 1e5 + 5;
const ll MOD = 1e9 + 7;
const ll INF = 1e9;
const ld EPS = 1e-9;



void solve() {
    
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    int tc = 1;
    // cin >> tc;
    for (int t = 1; t <= tc; t++) {
        // cout << "Case #" << t << ": ";
        solve();
    }
}
```

## Ngôn ngữ Java

```java
import java.io.*;
import java.util.*;
 
public class FastIOTemplate{
   public static void main(String[] args) {
      MyScanner sc = new MyScanner();
      out = new PrintWriter(new BufferedOutputStream(System.out));
      
      // Start writing your solution here. -------------------------------------
   
      /*
      int n      = sc.nextInt();        // read input as integer
      long k     = sc.nextLong();       // read input as long
      double d   = sc.nextDouble();     // read input as double
      String str = sc.next();           // read input as String
      String s   = sc.nextLine();       // read whole line as String

      int result = 3*n;
      out.println(result);                    // print via PrintWriter
      */

      // Stop writing your solution here. -------------------------------------
      out.close();
   }

     

   //-----------PrintWriter for faster output---------------------------------
   public static PrintWriter out;
      
   //-----------MyScanner class for faster input----------
   public static class MyScanner {
      BufferedReader br;
      StringTokenizer st;
 
      public MyScanner() {
         br = new BufferedReader(new InputStreamReader(System.in));
      }
 
      String next() {
          while (st == null || !st.hasMoreElements()) {
              try {
                  st = new StringTokenizer(br.readLine());
              } catch (IOException e) {
                  e.printStackTrace();
              }
          }
          return st.nextToken();
      }
 
      int nextInt() {
          return Integer.parseInt(next());
      }
 
      long nextLong() {
          return Long.parseLong(next());
      }
 
      double nextDouble() {
          return Double.parseDouble(next());
      }
 
      String nextLine(){
          String str = "";
	  try {
	     str = br.readLine();
	  } catch (IOException e) {
	     e.printStackTrace();
	  }
	  return str;
      }

   }
   //--------------------------------------------------------
}
```

Một số nguồn tài liệu phổ biến
======

## Thuật toán tìm kiếm nhị phân

[Tim kiem nhi phan](https://youtu.be/88kyUs70kJQ?list=PLDgptIulgMt59Jrt89y6ztjoCpBtP2d7f)

## Cấu trúc dữ liệu phổ biến

[Cau truc du lieu pho bien](https://youtu.be/3oXwgDX9L9s?list=PLDgptIulgMt7s240EpyOZjuUGPWOcze13)

## Quy hoạch động cơ bản

- Lop bai toan: Quy hoach dong (dem cach)

- Lop bai toan: Quy hoach dong (dung chi phi thap nhat)

[Hieu ve quy hoach dong chi tiet](https://youtu.be/htMQeQYb8Yc?list=PLDgptIulgMt5hmL8-H9lLrgIYxgaQixGk)


## Heap Min

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2*i + 1

    def right_child(self, i):
        return 2*i + 2

    def insert(self, value):
        self.heap.append(value)
        index = len(self.heap) - 1

        while index > 0 and self.heap[self.parent(index)] > self.heap[index]:
            self.heap[self.parent(index)], self.heap[index] = self.heap[index], self.heap[self.parent(index)]
            index = self.parent(index)

    def extract_min(self):
        if not self.heap:
            return None

        min_element = self.heap[0]
        last_element = self.heap.pop()

        if len(self.heap) > 0:
            self.heap[0] = last_element
            self.heapify(0)

        return min_element

    def heapify(self, i):
        smallest = i
        left = self.left_child(i)
        right = self.right_child(i)

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left

        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self.heapify(smallest)
            
    def remove_element(self, value):
        index = self.heap.index(value)

        # Thay thế phần tử cần xóa bằng phần tử cuối cùng của heap
        self.heap[index] = self.heap[-1]
        self.heap.pop()

        # Gọi lại heapify từ vị trí thay thế
        self.heapify(index)

# Tạo một heap min
min_heap = MinHeap()

# Thêm các phần tử vào heap
min_heap.insert(4)
min_heap.insert(1)
min_heap.insert(7)
min_heap.insert(3)
min_heap.insert(13)
min_heap.insert(-3)
min_heap.insert(0)


# Trích xuất phần tử nhỏ nhất
min_element = min_heap.extract_min()

print(f"Phần tử nhỏ nhất là: {min_element}")



print("Heap trước khi xóa:", min_heap.heap)

# Xóa phần tử có giá trị 7
min_heap.remove_element(4)

print("Heap sau khi xóa:", min_heap.heap)

```

## Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    def display(self, node, prefix=""):
        if node.is_end_of_word:
            print(prefix)
        for char, child_node in node.children.items():
            self.display(child_node, prefix + char)


# Sử dụng Trie
trie = Trie()

# Thêm các từ vào trie
words = ["hello", "world", "trie", "python", "code"]
for word in words:
    trie.insert(word)

# Hiển thị thông tin của cây Trie
trie.display(trie.root)


# Kiểm tra các từ trong trie
print(trie.search("python"))  # True
print(trie.search("worlds"))  # False
print(trie.starts_with("cod"))  # True
print(trie.starts_with("abc"))  # False
```

Kiến thức về cây cơ bản
======

## Tree gồm nhiều nút và mỗi nút gồm nhiều nút con

```python
class Node:
    def __init__(self, data):
        self.children = []
        self.data = data

def find_node(root, target_data):
    if root is None:
        return None

    if root.data == target_data:
        return root

    for child in root.children:
        result = find_node(child, target_data)
        if result:
            return result

    return None

def find_path(root, target, path):
    if root is None:
        return False

    path.append(root)

    if root.data == target.data:
        return True

    for child in root.children:
        if find_path(child, target, path):
            return True

    path.pop()
    return False

def find_lca(root, node1, node2):
    path_to_node1 = []
    path_to_node2 = []

    if not (find_path(root, node1, path_to_node1) and find_path(root, node2, path_to_node2)):
        return None

    i = 0
    while i < len(path_to_node1) and i < len(path_to_node2):
        if path_to_node1[i].data != path_to_node2[i].data:
            break
        i += 1

    return path_to_node1[i - 1]

# Example usage
# Build the tree:
#        1
#       /|\
#      2 3 4
#     / \
#    5   6

root = Node(1)
root.children = [Node(2), Node(3), Node(4)]
root.children[0].children = [Node(5), Node(6)]

node1 = root.children[2]  # Node 2
node2 = root.children[0].children[0]  # Node 4

lca = find_lca(root, node1, node2)
if lca:
    print("LCA of", node1.data, "and", node2.data, "is:", lca.data)
else:
    print("LCA not found")

```

## Cách tạo cây nhị phân

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    if root is None:
        return TreeNode(val)

    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)

    return root

def preorder_traversal(root):
    if root:
        print(root.val, end=" ")
        preorder_traversal(root.left)
        preorder_traversal(root.right)

# Example usage
num_array = [5, 2, 8, 1, 3, 6, 9]

root = None
for num in num_array:
    root = insert(root, num)

print("Pre-order traversal:")
preorder_traversal(root)

'''
      5
 2        8
1  3    6   9
'''
```

## Cơ bản về segment tree

```python
a = [0, 1, 3, 5, 2, 4, 6, 8, 7]
n = 8
t = [0] * 100

def build(id, l, r):
    if l == r:
        t[id] = a[l]
        print(f"{id}, {l}, {r}, {t[id]}")
        return
    
    mid = (l + r) // 2
    build(id * 2, l, mid)
    build(id * 2 + 1, mid + 1, r)
    
    t[id] = t[id * 2] + t[id * 2 + 1]
    print(f"{id}, {l}, {r}, {t[id]}")

def get(id, l, r, u, v):
    if r < u or v < l:
        return 0
    if u <= l and r <= v:
        print(f"->{id}, {l}, {r}, {t[id]}")
        return t[id]
    
    mid = (l + r) // 2
    t1 = get(id * 2, l, mid, u, v)
    t2 = get(id * 2 + 1, mid + 1, r, u, v)
    print(f"{id}, {l}, {r}, {t1 + t2}")
    
    return t1 + t2

def update(id, l, r, pos, val):
    if pos < l or r < pos:
        return
    if l == r:
        t[id] = val
        a[l] = val
        return
    
    mid = (l + r) // 2
    update(id * 2, l, mid, pos, val)
    update(id * 2 + 1, mid + 1, r, pos, val)
    t[id] = t[id * 2] + t[id * 2 + 1]

build(1, 1, n)
print("====")
print(get(1, 1, n, 2, 5))
print("======")
update(1, 1, n, 4, 10)
print(get(1, 1, n, 2, 5))

```

Cách sử dụng vector
======

```c++

#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long


int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n;
    cin >> n; vector<int> v(n);
    for(int& x : v) cin >> x;

    int res = 0;
    for(int i = 1; i < n - 1; i++){
        res = max(res, v[i+1] - v[i-1]);
    }
    cout << res;
    return 0;
}
```

Cách sử dụng lower_bound and upper_bound
======

```c++
#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>

using namespace std;
#define int long long

int n, k;
vector<int> v;

int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);

    vector<int> vec = {1, 2, 4, 4, 4, 6, 7, 8, 9};

    int x = 4;
    auto it1 = upper_bound(vec.begin(), vec.end(), x);

    if (it1 != vec.end()) {
        cout << "First element greater than " << x << " is at position: " << distance(vec.begin(), it1) << ", and value = " << *it1 << ", and value1 = " << vec[distance(vec.begin(), it1)] << std::endl;
    } else {
        std::cout << "No element greater than " << x << " found." << std::endl;
    }

    auto it2 = std::lower_bound(vec.begin(), vec.end(), x);

    if (it2 != vec.end()) {
        std::cout << "First element not less than " << x << " is at position: " << std::distance(vec.begin(), it2) << ", and value = " << *it2 << ", and value1 = " << vec[distance(vec.begin(), it2)]  << std::endl;
    } else {
        std::cout << "No element not less than " << x << " found." << std::endl;
    }



    return 0;
}
```

Code binary search cơ bản
======

```c++
#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>
 
using namespace std;
#define int long long
 
int w, h, n;
 
bool good(int x){
    if((x / w) * (x / h) >= n) return true;
    return false;
}
 
int32_t main() {
    ios::sync_with_stdio(0); cin.tie(0);
    cin >> w >> h >> n;
    int l = 0, r = 1;
    while(!good(r)) r *= 2;
 
    while(r > l + 1){
        int m = (l + r) / 2;
        if(good(m)) {
            r = m;
        }
        else {
            l = m;
        }
    }
    cout << r << endl;
    return 0;
}
 
// x * 3 + y * 2 >= 10
// 9/3 * 9/2 >= 10
 
// k*k / (3*2) >= 10
```


Recursive Problems List
======

1. Tính giai thừa: Tính n! (n giai thừa) với n >= 0.
2. Fibonacci: Tìm số Fibonacci thứ n trong dãy Fibonacci.
3. Tính lũy thừa: Tính a^n với a và n là các số nguyên dương.
4. Tìm ước chung lớn nhất (UCLN) và bội chung nhỏ nhất (BCNN) của hai số nguyên dương a và b.
5. Quy hoạch động: Các bài toán như tìm dãy con tăng dài nhất, tìm số cách di chuyển từ điểm (0,0) đến điểm (m,n) trên lưới bằng cách di chuyển theo các bước (1,0) hoặc (0,1).
6. Tìm các tập con của một tập hợp.
7. Tìm các tuyến đường trong đồ thị.

8. Tính tổ hợp: Tính toán các tổ hợp chập k của n phần tử (nCk).
9. Tính tổng các phần tử trong một mảng.
10. Tìm kiếm nhị phân: Tìm kiếm một phần tử trong một mảng đã được sắp xếp.
11. Sắp xếp lại một mảng (ví dụ: sắp xếp mảng theo thứ tự tăng dần hoặc giảm dần).
12. Tính đường đi ngắn nhất trong đồ thị (Dijkstra's algorithm hoặc Bellman-Ford algorithm).
13. Tìm kiếm chuỗi con chung dài nhất giữa hai chuỗi.
14. Tính tập hợp con có tổng bằng một số x (Subset Sum Problem).
15. Tìm đường đi trong mê cung.
16. Tính số Fibonacci sử dụng ma trận.
17. Kiểm tra xem một chuỗi có phải là chuỗi Palindrome không.


8. Tính tổ hợp: Tính toán các tổ hợp chập k của n phần tử (nCk).
9. Tính tổng các phần tử trong một mảng.
10. Tìm kiếm nhị phân: Tìm kiếm một phần tử trong một mảng đã được sắp xếp.
11. Sắp xếp lại một mảng (ví dụ: sắp xếp mảng theo thứ tự tăng dần hoặc giảm dần).
12. Tính đường đi ngắn nhất trong đồ thị (Dijkstra's algorithm hoặc Bellman-Ford algorithm).
13. Tìm kiếm chuỗi con chung dài nhất giữa hai chuỗi.
14. Tính tập hợp con có tổng bằng một số x (Subset Sum Problem).
15. Tìm đường đi trong mê cung.
16. Tính số Fibonacci sử dụng ma trận.
17. Kiểm tra xem một chuỗi có phải là chuỗi Palindrome không.


18. Tính toán các số Catalan: Các số Catalan xuất hiện trong nhiều vấn đề kết cấu dữ liệu và tổ hợp.

19. Tìm tất cả các hoán vị của một chuỗi ký tự.

20. Tìm các tập con không giao nhau với tổng lớn nhất (Maximum Sum Subarray Problem).

21. Tìm tất cả các cách chia một số nguyên dương thành tổng các số nguyên dương nhỏ hơn.

22. Tìm đường đi Hamilton trong đồ thị.

23. Tính số Bell: Đếm số cách chia một tập hợp gồm n phần tử thành các phân lớp khác nhau.

24. Tính tổ hợp lặp lại: Tính toán tổ hợp chập k của n phần tử với lặp lại.

25. Tìm dãy con dài nhất không giảm (Longest Increasing Subsequence).

26. Tìm tất cả các hoán vị của một mảng.

27. Tìm các cách sắp xếp n phần tử để tạo thành các cặp.


28. Tìm chuỗi con dài nhất có các ký tự duy nhất (Longest Substring Without Repeating Characters).

29. Tìm tất cả các tập con có tổng bằng một số x (Subset Sum Problem).

30. Tìm chuỗi con chung dài nhất của ba chuỗi.

31. Tính số cách tạo ra một số nguyên dương bằng tổng các số nguyên dương nhỏ hơn n.

32. Tìm tất cả các cách chia một số nguyên dương thành tổng các số nguyên dương lẻ.

33. Tìm đường đi trong đồ thị có trọng số âm (Bellman-Ford algorithm).

34. Tìm tất cả các dãy con không giao nhau có tổng lớn nhất (Maximum Sum Non-Adjacent Subsequence).

35. Tính số Fibonacci sử dụng đệ quy đuôi (tail recursion).

36. Tính toán tất cả các cách sắp xếp n phần tử để tạo thành các tập con.


Tài liệu tham khảo
======

* Tài liệu try hard code và Note book của ngthanhtrung23.github.io

[Link tham khảo](https://drive.google.com/drive/u/0/folders/14YzYldXfRDTc2RS2YHlGjSSMxCLEsDkX)


[Link 2](https://codelessons.dev/ru/set-i-multiset-v-c-chto-eto-takoe-i-kak-s-nimi-rabotat/)

[Link 3](https://codeforces.com/blog/entry/18169)

[Link 4](https://cp-algorithms.com/index.html)

Một số Std C++ phổ biến:
======

Những thành phần phổ biến:  priority_queue, map, set, multiset

[unorderd multiset 1](https://workat.tech/problem-solving/tutorial/cpp-stl-unordered-multiset-complete-guide-3p4zlex3ecgq)

[unordered multiset 2](https://www.geeksforgeeks.org/cpp-unordered_multiset/)

[std2 - top coder 1](https://www.topcoder.com/thrive/articles/Power%20up%20C++%20with%20the%20Standard%20Template%20Library%20Part%20One)

[std2 - top coder 2](https://www.topcoder.com/thrive/articles/Power%20up%20C++%20with%20the%20Standard%20Template%20Library%20Part%20Two:%20Advanced%20Uses)

Một số bài tập cần phải làm
======

Ref: (phải học, hiểu bản chất và code được)

[cp algorithm](https://cp-algorithms.com/index.html)

[cf education](https://codeforces.com/edu/course/2)

[vnoid edu 1](https://oj.vnoi.info/problems/?category=22&point_start=&point_end=)

[vnoid edu 2](https://oj.vnoi.info/problems/?category=28&point_start=&point_end=&page=2)

Ref1: 

[Link1](https://phuchaodo.github.io/posts/2023/09/05/tong-hop-mot-so-tai-lieu-cho-icpc/)

[Link2](https://phuchaodo.github.io/posts/2023/09/01/mot-so-code-icpc-pho-bien/)

Hết.

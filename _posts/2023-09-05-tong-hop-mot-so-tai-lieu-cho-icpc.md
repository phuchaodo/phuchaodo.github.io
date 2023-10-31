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


Tài liệu tham khảo
======

* Tài liệu try hard code và Note book của ngthanhtrung23.github.io

[Link tham khảo](https://drive.google.com/drive/u/0/folders/14YzYldXfRDTc2RS2YHlGjSSMxCLEsDkX)


Hết.

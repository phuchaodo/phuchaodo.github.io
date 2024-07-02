---
title: '[Note] Các thuật toán phổ biến cần phải hiểu và sử dụng'
date: 2024-07-02
permalink: /posts/2024/07/02/cac-thuat-toan-pho-bien-can-phai-hieu-va-su-dung/
tags:
  - Algorithm
  - Pytorch
--- 

Hiểu hơn về các thuật toán phổ biến cần phải biết và sử dụng


𝟭𝟬 𝗔𝗹𝗴𝗼𝗿𝗶𝘁𝗵𝗺𝘀 𝗘𝘃𝗲𝗿𝘆 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿 𝗦𝗵𝗼𝘂𝗹𝗱 𝗞𝗻𝗼𝘄:


𝟬.💡 𝗕𝗿𝗲𝗮𝗱𝘁𝗵-𝗙𝗶𝗿𝘀𝘁 𝗦𝗲𝗮𝗿𝗰𝗵 (𝗕𝗙𝗦): 
Explore a graph level by level, starting from the root, which is great for finding the shortest path in unweighted graphs. 
➡️ Useful when: You're designing web crawlers or analyzing social networks.

𝟭.💡 𝗧𝘄𝗼 𝗛𝗲𝗮𝗽𝘀: 
Uses a min-heap and max-heap to manage dynamic datasets efficiently, maintaining median and priority. 
➡️ Useful when: You need to manage a priority queue or dynamic datasets.

𝟮.💡 𝗧𝘄𝗼 𝗣𝗼𝗶𝗻𝘁𝗲𝗿𝘀: 
This technique takes 2 points in a sequence and performs logic based on the problem.
➡️ Useful when: You are implementing sorting or searching functions.

𝟯.💡 𝗦𝗹𝗶𝗱𝗶𝗻𝗴 𝗪𝗶𝗻𝗱𝗼𝘄: 
Optimizes the computation by reusing the state from the previous subset of data. 
➡️ Useful when: You're handling network congestion or data compression.

𝟰.💡 𝗗𝗲𝗽𝘁𝗵-𝗙𝗶𝗿𝘀𝘁 𝗦𝗲𝗮𝗿𝗰𝗵 (𝗗𝗙𝗦): 
Explores each path to the end, ideal for situations that involve exploring all options like in puzzles. 
➡️ Useful when: You're working with graph structures or need to generate permutations.

𝟱.💡 𝗧𝗼𝗽𝗼𝗹𝗼𝗴𝗶𝗰𝗮𝗹 𝗦𝗼𝗿𝘁: 
Helps in scheduling tasks based on their dependencies. 
➡️ Useful when: You are determining execution order in project management or compiling algorithms.

𝟲.💡 𝗠𝗲𝗿𝗴𝗲 𝗜𝗻𝘁𝗲𝗿𝘃𝗮𝗹𝘀: 
Optimizes overlapping intervals to minimize the number of intervals. 
➡️ Useful when: Scheduling resources or managing calendars.

𝟳.💡 𝗕𝗮𝗰𝗸𝘁𝗿𝗮𝗰𝗸𝗶𝗻𝗴: 
It explores all potential solutions systematically and is perfect for solving puzzles and optimization problems. 
➡️ Useful when: Solving complex logical puzzles or optimizing resource allocations.

𝟴.💡 𝗧𝗿𝗶𝗲 (𝗣𝗿𝗲𝗳𝗶𝘅 𝗧𝗿𝗲𝗲): 
A tree-like structure that manages dynamic sets of strings efficiently, often used for searching. 
➡️ Useful when: Implementing spell-checkers or autocomplete systems.

𝟵.💡 𝗙𝗹𝗼𝗼𝗱 𝗙𝗶𝗹𝗹: 
It fills a contiguous area for features like the 'paint bucket' tool. 
➡️ Useful when: Working in graphics editors or game development.

𝟭𝟬.💡 𝗦𝗲𝗴𝗺𝗲𝗻𝘁 𝗧𝗿𝗲𝗲: 
Efficiently manages intervals or segments and is useful for storing information about intervals and querying over them. 
➡️ Useful when: Dealing with database range queries or statistical calculations.



## Breadth-First Search (BFS) Explained

Breadth-First Search (BFS) is a graph traversal algorithm that explores a graph level by level, starting from a given root node. It systematically visits all the neighbors of the current node before moving to the next level of nodes. This makes it ideal for finding the shortest path in unweighted graphs.

**How BFS works:**

1. **Initialization:** Start with a queue and add the root node to it. Mark the root node as visited.
2. **Iteration:** While the queue is not empty:
    * **Dequeue:** Remove the front node from the queue.
    * **Explore Neighbors:** For each unvisited neighbor of the current node:
        * Mark the neighbor as visited.
        * Add the neighbor to the queue.
3. **Termination:** The algorithm ends when all reachable nodes from the root have been visited.

**Key Features of BFS:**

* **Level by level exploration:**  It explores the graph in a systematic manner, visiting nodes at the same level before moving to the next level.
* **Shortest path for unweighted graphs:** BFS guarantees that the first time a node is visited is also the shortest path from the root node to that node.
* **Uses a queue:** BFS utilizes a queue data structure to maintain the order of node exploration.

**Applications of BFS:**

* **Web Crawlers:**  To systematically crawl websites and index pages.
* **Social Network Analysis:** To find connections between users and analyze network structure.
* **Shortest Path Algorithms:** To find the shortest path in unweighted graphs.
* **Game Solving:** To find the optimal path in games like Pac-Man.
* **Routing Algorithms:** To find the best routes in maps and networks.

**Code Implementation (C++ & Python):**

### C++

```c++
#include <iostream>
#include <queue>
#include <vector>

using namespace std;

// Adjacency list representation of the graph
vector<vector<int>> graph = {
    {1, 2},
    {0, 3},
    {0, 3},
    {1, 2}
};

// BFS implementation
void bfs(int startNode) {
    vector<bool> visited(graph.size(), false);
    queue<int> q;

    visited[startNode] = true;
    q.push(startNode);

    while (!q.empty()) {
        int currentNode = q.front();
        q.pop();

        cout << currentNode << " ";

        for (int neighbor : graph[currentNode]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

int main() {
    int startNode = 0;
    bfs(startNode);

    return 0;
}
```

### Python

```python
from collections import defaultdict

# Adjacency list representation of the graph
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

def bfs(startNode):
    visited = set()
    queue = [startNode]

    while queue:
        currentNode = queue.pop(0)

        if currentNode not in visited:
            print(currentNode, end=" ")
            visited.add(currentNode)

            for neighbor in graph[currentNode]:
                if neighbor not in visited:
                    queue.append(neighbor)

# Run BFS from node 0
bfs(0)
```

## Two Heaps Implementation: C++ and Python

The Two Heaps technique is a clever way to manage dynamic datasets efficiently, particularly when you need to maintain a median value or manage priorities within the data. It utilizes a combination of a min-heap and a max-heap to achieve this.

**Concept:**

The dataset is split into two heaps:

* **Min-Heap:** Stores the larger half of the dataset, ensuring the smallest element is at the root.
* **Max-Heap:** Stores the smaller half of the dataset, ensuring the largest element is at the root.

These heaps are balanced to ensure the median can be retrieved easily.

**Implementation (C++):**

```c++
#include <iostream>
#include <queue>

using namespace std;

class TwoHeaps {
private:
    priority_queue<int, vector<int>, greater<int>> minHeap; // Min-heap
    priority_queue<int> maxHeap; // Max-heap

    void balanceHeaps() {
        if (minHeap.size() > maxHeap.size() + 1) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        } else if (maxHeap.size() > minHeap.size() + 1) {
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        }
    }

public:
    void insert(int value) {
        if (minHeap.empty() || value >= minHeap.top()) {
            minHeap.push(value);
        } else {
            maxHeap.push(value);
        }
        balanceHeaps();
    }

    double getMedian() {
        if (minHeap.size() > maxHeap.size()) {
            return minHeap.top();
        } else if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        } else {
            return (double)(minHeap.top() + maxHeap.top()) / 2;
        }
    }
};

int main() {
    TwoHeaps twoHeaps;
    int values[] = {5, 10, 3, 8, 1, 7};
    for (int value : values) {
        twoHeaps.insert(value);
        cout << "Median after inserting " << value << ": " << twoHeaps.getMedian() << endl;
    }
    return 0;
}
```

**Implementation (Python):**

```python
import heapq

class TwoHeaps:
    def __init__(self):
        self.minHeap = []  # Min-heap
        self.maxHeap = []  # Max-heap

    def balanceHeaps(self):
        if len(self.minHeap) > len(self.maxHeap) + 1:
            heapq.heappush(self.maxHeap, -heapq.heappop(self.minHeap))
        elif len(self.maxHeap) > len(self.minHeap) + 1:
            heapq.heappush(self.minHeap, -heapq.heappop(self.maxHeap))

    def insert(self, value):
        if not self.minHeap or value >= self.minHeap[0]:
            heapq.heappush(self.minHeap, value)
        else:
            heapq.heappush(self.maxHeap, -value)
        self.balanceHeaps()

    def getMedian(self):
        if len(self.minHeap) > len(self.maxHeap):
            return self.minHeap[0]
        elif len(self.maxHeap) > len(self.minHeap):
            return -self.maxHeap[0]
        else:
            return (self.minHeap[0] - self.maxHeap[0]) / 2

# Example Usage:
twoHeaps = TwoHeaps()
values = [5, 10, 3, 8, 1, 7]
for value in values:
    twoHeaps.insert(value)
    print(f"Median after inserting {value}: {twoHeaps.getMedian()}")
```


## Two Pointers Technique: Explained with C++ and Python Examples

The Two Pointers technique is a powerful strategy for solving problems involving sequences, often lists or arrays.  It involves using two pointers that iterate through the sequence, allowing you to compare elements, manipulate data, and achieve efficient solutions.

**How It Works:**

1. **Initialization:** You start with two pointers, typically named `left` and `right`, pointing to the beginning and end of the sequence, respectively.
2. **Iteration:** The pointers move towards each other, often in a loop, comparing elements and executing logic based on the problem's requirements.
3. **Logic:** Depending on the problem, the logic could involve:
    * **Sorting:** Swapping elements to achieve a sorted order.
    * **Searching:** Finding specific elements or pairs that satisfy a condition.
    * **Subarray Manipulation:**  Identifying subarrays with desired properties.
    * **Compression:** Removing duplicates or merging overlapping intervals.
4. **Termination:** The pointers continue moving until they meet or reach a certain condition that indicates the completion of the task.

**Advantages of Two Pointers:**

* **Efficiency:** Two Pointers often result in linear time complexity (O(n)), making it highly efficient for handling sequences.
* **Simplicity:** The concept is straightforward, making it easier to understand and implement compared to more complex algorithms.
* **Versatility:** Two Pointers can be adapted to solve a wide range of problems involving sequences.

**Use Cases:**

* **Sorting:**  Merge sort, quicksort, and partition algorithms often leverage Two Pointers to organize elements efficiently.
* **Searching:**  Finding pairs with specific sums or identifying elements in a sorted sequence can be done efficiently with Two Pointers.
* **Array Manipulation:**  Removing duplicates, compressing sequences, and finding subarrays meeting certain criteria are common applications.
* **String Processing:**  String matching, palindrome detection, and substring manipulation can benefit from Two Pointers.

**Code Implementation (C++ & Python):**

### C++

```c++
#include <iostream>
#include <vector>

using namespace std;

// Example: Finding two numbers that sum to a target value in a sorted array
vector<int> findPair(vector<int> arr, int target) {
    int left = 0;
    int right = arr.size() - 1;

    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) {
            return {left, right};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return {-1, -1}; // No pair found
}

int main() {
    vector<int> arr = {1, 2, 3, 4, 5, 6};
    int target = 7;
    vector<int> result = findPair(arr, target);
    if (result[0] != -1) {
        cout << "Pair found at indices: " << result[0] << ", " << result[1] << endl;
    } else {
        cout << "Pair not found" << endl;
    }
    return 0;
}
```

### Python

```python
# Example: Removing duplicates from a sorted array
def removeDuplicates(nums):
    left = 0
    right = 1

    while right < len(nums):
        if nums[left] != nums[right]:
            left += 1
            nums[left] = nums[right]
        right += 1

    return left + 1

nums = [1, 1, 2, 2, 3, 3, 4]
newLength = removeDuplicates(nums)
print(f"Array after removing duplicates: {nums[:newLength]}")
```

## Sliding Window Technique: Explained with C++ and Python Examples

The Sliding Window technique is a powerful algorithm design pattern used to optimize computations by reusing information from previous calculations. It's particularly useful when dealing with problems involving sequences where you need to analyze or process a specific portion (window) of the data at a time. 

**How It Works:**

1. **Window Definition:** You define a "window" of a fixed size that slides across the sequence.
2. **Initial State:** The window is positioned at the beginning of the sequence.
3. **Iteration:** The window slides one step at a time, moving from left to right. 
4. **State Update:** For each window position, you perform calculations or operations, updating the window's state (e.g., minimum, maximum, sum) based on the new data it encompasses.
5. **Reuse of Previous State:** The key optimization comes from reusing the state from the previous window position. Instead of recalculating everything from scratch, you only update the state based on the new element added to the window and the element removed from it. 

**Advantages of Sliding Window:**

* **Efficiency:** Sliding Window typically reduces the time complexity from O(n^2) to O(n), making it significantly faster for large sequences.
* **Simple Implementation:** The concept is straightforward and often requires less complex code compared to other techniques.
* **Wide Applicability:** Sliding Windows are applicable to various problems involving sequences, including finding maximum sums, counting occurrences, and data compression.

**Use Cases:**

* **Finding Maximum Subarray Sum:**  Efficiently calculating the maximum sum of a contiguous subarray of a given size.
* **Data Compression:** Utilizing sliding windows to identify patterns and compress data effectively.
* **Network Congestion Control:**  Monitoring and controlling network traffic flow by analyzing data within a sliding window.
* **String Matching:**  Optimizing string matching algorithms by comparing substrings within a sliding window.

**Code Implementation (C++ & Python):**

### C++

```c++
#include <iostream>
#include <vector>

using namespace std;

// Example: Finding the maximum sum of a contiguous subarray of size k
int maxSubarraySum(vector<int> arr, int k) {
    int maxSum = INT_MIN;
    int windowSum = 0;
    int windowStart = 0;

    for (int windowEnd = 0; windowEnd < arr.size(); windowEnd++) {
        windowSum += arr[windowEnd];

        if (windowEnd >= k - 1) {
            maxSum = max(maxSum, windowSum);
            windowSum -= arr[windowStart];
            windowStart++;
        }
    }

    return maxSum;
}

int main() {
    vector<int> arr = {2, 1, 5, 1, 3, 2};
    int k = 3;
    int maxSum = maxSubarraySum(arr, k);
    cout << "Maximum sum of subarray of size " << k << ": " << maxSum << endl;
    return 0;
}
```

### Python

```python
# Example: Finding the maximum sum of a contiguous subarray of size k
def maxSubarraySum(arr, k):
    maxSum = float('-inf')
    windowSum = 0
    windowStart = 0

    for windowEnd in range(len(arr)):
        windowSum += arr[windowEnd]

        if windowEnd >= k - 1:
            maxSum = max(maxSum, windowSum)
            windowSum -= arr[windowStart]
            windowStart += 1

    return maxSum

arr = [2, 1, 5, 1, 3, 2]
k = 3
maxSum = maxSubarraySum(arr, k)
print(f"Maximum sum of subarray of size {k}: {maxSum}")
```

**Explanation:**

* **Both C++ and Python Examples:**
    * The `maxSubarraySum` function calculates the maximum sum of a contiguous subarray of size `k`.
    * `windowSum` keeps track of the sum of elements within the current window.
    * `windowStart` and `windowEnd` pointers represent the start and end of the sliding window.
    * The loop iterates through the array, adding new elements to the window and removing old elements as it slides.
    * The maximum sum is updated whenever the window reaches the size of `k`.

**Key Points:**

* **Window Size:** The window size is determined by the problem's requirements and dictates the number of elements considered for each calculation.
* **State Update:** The logic for updating the window's state depends on the specific problem being solved.
* **Optimization:** The key to efficiency lies in reusing the previous state, avoiding unnecessary recalculations.


## Depth-First Search (DFS) Explained: C++ and Python Implementation

Depth-First Search (DFS) is a graph traversal algorithm that explores the graph by going as deep as possible along each branch before backtracking. It's ideal for situations where you need to explore all possible paths in a graph, such as solving puzzles, finding connected components, or generating permutations.

**How DFS Works:**

1. **Initialization:** Start with a stack (or recursion) and add the root node to it. Mark the root node as visited.
2. **Iteration:** While the stack is not empty:
    * **Pop:** Remove the top node from the stack.
    * **Explore Neighbors:** For each unvisited neighbor of the current node:
        * Mark the neighbor as visited.
        * Push the neighbor onto the stack.
3. **Termination:** The algorithm ends when all reachable nodes from the root have been visited.

**Key Features of DFS:**

* **Depth-First Exploration:**  It explores the graph by going deep into each branch before exploring other branches.
* **Uses Stack/Recursion:** DFS typically utilizes a stack data structure (or recursion) to keep track of the nodes to visit.
* **Backtracking:** When a dead-end is reached, the algorithm backtracks to the previous node and explores other unvisited branches.

**Applications of DFS:**

* **Graph Traversal:**  To explore all nodes and edges in a graph.
* **Puzzle Solving:** To find solutions to puzzles like Sudoku, maze solving, or finding all possible moves in a game.
* **Connected Components:** To identify groups of interconnected nodes in a graph.
* **Topological Sorting:** To arrange nodes in a graph in a specific order based on dependencies.
* **Cycle Detection:** To determine if a graph contains cycles.

**Code Implementation (C++ & Python):**

### C++

```c++
#include <iostream>
#include <stack>
#include <vector>

using namespace std;

// Adjacency list representation of the graph
vector<vector<int>> graph = {
    {1, 2},
    {0, 3},
    {0, 3},
    {1, 2}
};

// DFS implementation
void dfs(int startNode) {
    vector<bool> visited(graph.size(), false);
    stack<int> s;

    visited[startNode] = true;
    s.push(startNode);

    while (!s.empty()) {
        int currentNode = s.top();
        s.pop();

        cout << currentNode << " ";

        for (int neighbor : graph[currentNode]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                s.push(neighbor);
            }
        }
    }
}

int main() {
    int startNode = 0;
    dfs(startNode);

    return 0;
}
```

### Python

```python
from collections import defaultdict

# Adjacency list representation of the graph
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

def dfs(startNode):
    visited = set()
    stack = [startNode]

    while stack:
        currentNode = stack.pop()

        if currentNode not in visited:
            print(currentNode, end=" ")
            visited.add(currentNode)

            for neighbor in graph[currentNode]:
                if neighbor not in visited:
                    stack.append(neighbor)

# Run DFS from node 0
dfs(0)
```


## Topological Sort: Scheduling Tasks with Dependencies

Topological Sort is a graph algorithm that produces a linear ordering of vertices in a directed acyclic graph (DAG) such that for every directed edge (u, v), vertex u comes before vertex v in the ordering. It's particularly useful for scheduling tasks that have dependencies on each other, ensuring that tasks are executed in the correct order.

**How Topological Sort Works:**

1. **Initialization:** Create an array to store the sorted order and initialize a queue. 
2. **Find Starting Nodes:** Find nodes with no incoming edges (in-degree = 0). These nodes can be executed first. Add these starting nodes to the queue.
3. **Iterative Process:** While the queue is not empty:
    * **Dequeue:** Remove the first node (u) from the queue and add it to the sorted order array.
    * **Remove Incoming Edges:** For each neighbor (v) of the dequeued node (u), decrement the in-degree of v.
    * **Add to Queue:** If the in-degree of v becomes 0 after decrementing, add v to the queue.
4. **Termination:** Repeat steps 3 until the queue is empty.
5. **Result:** The sorted order array now contains the vertices in a valid topological order.

**Applications of Topological Sort:**

* **Project Management:**  To determine the order in which tasks should be completed in a project with dependencies.
* **Course Scheduling:** To schedule courses based on prerequisites.
* **Compilation:**  To determine the order in which modules should be compiled in a program.
* **Dependency Resolution:**  To resolve dependencies between software packages or libraries.
* **Data Analysis:**  To analyze dependencies between data points in a graph.

**Code Implementation (C++ & Python):**

### C++

```c++
#include <iostream>
#include <queue>
#include <vector>

using namespace std;

// Adjacency list representation of the graph
vector<vector<int>> graph = {
    {1, 2},
    {2, 3},
    {3},
    {}
};

// Function to calculate in-degree of each node
vector<int> calculateInDegrees(vector<vector<int>> &graph) {
    vector<int> inDegrees(graph.size(), 0);
    for (int i = 0; i < graph.size(); i++) {
        for (int neighbor : graph[i]) {
            inDegrees[neighbor]++;
        }
    }
    return inDegrees;
}

// Topological Sort implementation
vector<int> topologicalSort(vector<vector<int>> &graph) {
    vector<int> inDegrees = calculateInDegrees(graph);
    queue<int> q;
    vector<int> sortedOrder;

    for (int i = 0; i < inDegrees.size(); i++) {
        if (inDegrees[i] == 0) {
            q.push(i);
        }
    }

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        sortedOrder.push_back(u);

        for (int v : graph[u]) {
            inDegrees[v]--;
            if (inDegrees[v] == 0) {
                q.push(v);
            }
        }
    }

    return sortedOrder;
}

int main() {
    vector<int> sortedOrder = topologicalSort(graph);
    cout << "Topological Sort: ";
    for (int node : sortedOrder) {
        cout << node << " ";
    }
    cout << endl;
    return 0;
}
```

### Python

```python
from collections import defaultdict

# Adjacency list representation of the graph
graph = {
    0: [1, 2],
    1: [2, 3],
    2: [3],
    3: []
}

def topologicalSort(graph):
    inDegrees = {node: 0 for node in graph}

    # Calculate in-degree for each node
    for node in graph:
        for neighbor in graph[node]:
            inDegrees[neighbor] += 1

    queue = [node for node in inDegrees if inDegrees[node] == 0]
    sortedOrder = []

    while queue:
        u = queue.pop(0)
        sortedOrder.append(u)

        for v in graph[u]:
            inDegrees[v] -= 1
            if inDegrees[v] == 0:
                queue.append(v)

    return sortedOrder

sortedOrder = topologicalSort(graph)
print(f"Topological Sort: {sortedOrder}")
```

**Explanation:**

* **`calculateInDegrees()` (C++):** This function calculates the in-degree of each node in the graph (number of incoming edges).
* **`topologicalSort()` (C++ and Python):**
    * It initializes an array to store the sorted order, a queue, and a dictionary (Python) or array (C++) to keep track of in-degrees.
    * It finds nodes with in-degree 0 and adds them to the queue.
    * The loop iterates until the queue is empty.
    * It removes a node from the queue, adds it to the sorted order array, decrements the in-degree of its neighbors, and adds those neighbors with in-degree 0 to the queue.


## Merge Intervals: Optimizing Overlapping Intervals

Merge Intervals is an algorithm that takes a collection of intervals as input and merges overlapping intervals to produce a new collection of non-overlapping intervals that cover the same range as the original collection. This optimization is useful in various applications where you need to efficiently manage and represent intervals, such as:

* **Scheduling Resources:**  Merging time slots to create a consolidated schedule of events.
* **Calendar Management:**  Combining overlapping appointments to avoid conflicts.
* **Data Compression:**  Representing a series of data points using a smaller set of non-overlapping intervals.
* **Geometric Algorithms:**  Simplifying geometric shapes by merging overlapping line segments or polygons.

**How Merge Intervals Works:**

1. **Sorting:** Sort the input intervals in ascending order based on their start times.
2. **Merging:**  Iterate through the sorted intervals:
    * If the current interval overlaps with the previous interval (i.e., the start time of the current interval is less than or equal to the end time of the previous interval), merge them by updating the end time of the previous interval to the maximum end time of both intervals.
    * If the intervals do not overlap, add the current interval to the result list.
3. **Result:** The resulting list contains the merged non-overlapping intervals.

**Code Implementation (C++ & Python):**

### C++

```c++
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct Interval {
    int start;
    int end;
};

vector<Interval> mergeIntervals(vector<Interval> &intervals) {
    if (intervals.empty()) {
        return {};
    }

    sort(intervals.begin(), intervals.end(), [](const Interval &a, const Interval &b) {
        return a.start < b.start;
    });

    vector<Interval> mergedIntervals;
    mergedIntervals.push_back(intervals[0]);
    int i = 1;

    while (i < intervals.size()) {
        Interval &lastInterval = mergedIntervals.back();
        if (lastInterval.end >= intervals[i].start) {
            lastInterval.end = max(lastInterval.end, intervals[i].end);
        } else {
            mergedIntervals.push_back(intervals[i]);
        }
        i++;
    }

    return mergedIntervals;
}

int main() {
    vector<Interval> intervals = {{1, 3}, {2, 6}, {8, 10}, {15, 18}};
    vector<Interval> mergedIntervals = mergeIntervals(intervals);

    cout << "Merged Intervals: " << endl;
    for (const Interval &interval : mergedIntervals) {
        cout << "[" << interval.start << ", " << interval.end << "] " << endl;
    }

    return 0;
}
```

### Python

```python
class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x.start)

    merged_intervals = [intervals[0]]
    for i in range(1, len(intervals)):
        last_interval = merged_intervals[-1]
        if last_interval.end >= intervals[i].start:
            last_interval.end = max(last_interval.end, intervals[i].end)
        else:
            merged_intervals.append(intervals[i])

    return merged_intervals

# Example usage
intervals = [Interval(1, 3), Interval(2, 6), Interval(8, 10), Interval(15, 18)]
merged_intervals = merge_intervals(intervals)

print("Merged Intervals:")
for interval in merged_intervals:
    print(f"[{interval.start}, {interval.end}]")
```

**Explanation:**

* **`Interval` Struct/Class:**  Defines a structure or class to represent intervals with `start` and `end` properties.
* **`mergeIntervals` Function (C++) / `merge_intervals` Function (Python):**
    * Sorts the input intervals based on their start times.
    * Initializes a result list to store the merged intervals.
    * Iterates through the sorted intervals, comparing each interval with the previous one.
    * If overlapping, merges them by updating the end time of the previous interval.
    * If not overlapping, adds the current interval to the result list.
    * Returns the list of merged intervals.

**Example Output:**

Both the C++ and Python code will output:

```
Merged Intervals:
[1, 6]
[8, 10]
[15, 18]
```

**Key Points:**

* **Time Complexity:** The Merge Intervals algorithm has a time complexity of O(n log n) due to the sorting step.
* **Space Complexity:** The space complexity is O(n) in the worst case, where all intervals need to be merged into a single interval.
* **Applications:** Merge Intervals is a versatile algorithm with wide applications in various domains, especially when dealing with time-based data or geometric objects.


## Explanation of Backtracking

Backtracking is a general algorithmic technique that considers searching every possible combination in order to solve computational problems. It is especially useful for solving problems that can be broken down into smaller sub-problems, and where the solution can be constructed incrementally.

### How it works:
1. **Choice:** At each step, choose among multiple options.
2. **Constraint:** Check if the current choice is valid.
3. **Goal:** Check if the current choice leads to a solution.
4. **Backtrack:** If the current choice is invalid or does not lead to a solution, undo the choice (backtrack) and try another option.

This method is often used for problems like:
- Solving puzzles (e.g., Sudoku, crosswords)
- Combinatorial problems (e.g., finding all subsets, permutations)
- Optimization problems (e.g., the knapsack problem)

### C++ Code Example: N-Queens Problem

The N-Queens problem is a classic example of backtracking. The goal is to place N queens on an N×N chessboard such that no two queens threaten each other.

```cpp
#include <iostream>
#include <vector>

using namespace std;

bool isSafe(vector<string>& board, int row, int col, int n) {
    for (int i = 0; i < col; i++) {
        if (board[row][i] == 'Q') {
            return false;
        }
    }
    for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q') {
            return false;
        }
    }
    for (int i = row, j = col; i < n && j >= 0; i++, j--) {
        if (board[i][j] == 'Q') {
            return false;
        }
    }
    return true;
}

bool solveNQueensUtil(vector<string>& board, int col, int n) {
    if (col >= n) {
        return true;
    }
    for (int i = 0; i < n; i++) {
        if (isSafe(board, i, col, n)) {
            board[i][col] = 'Q';
            if (solveNQueensUtil(board, col + 1, n)) {
                return true;
            }
            board[i][col] = '.';
        }
    }
    return false;
}

void solveNQueens(int n) {
    vector<string> board(n, string(n, '.'));
    if (solveNQueensUtil(board, 0, n)) {
        for (const auto& row : board) {
            cout << row << endl;
        }
    } else {
        cout << "No solution exists" << endl;
    }
}

int main() {
    int n = 8;  // Change this value for different sizes
    solveNQueens(n);
    return 0;
}
```

### Python Code Example: N-Queens Problem

The N-Queens problem solved using backtracking in Python:

```python
def is_safe(board, row, col, n):
    for i in range(col):
        if board[row][i] == 'Q':
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 'Q':
            return False
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 'Q':
            return False
    return True

def solve_nqueens_util(board, col, n):
    if col >= n:
        return True
    for i in range(n):
        if is_safe(board, i, col, n):
            board[i][col] = 'Q'
            if solve_nqueens_util(board, col + 1, n):
                return True
            board[i][col] = '.'
    return False

def solve_nqueens(n):
    board = [['.' for _ in range(n)] for _ in range(n)]
    if solve_nqueens_util(board, 0, n):
        for row in board:
            print(' '.join(row))
    else:
        print("No solution exists")

if __name__ == "__main__":
    n = 8  # Change this value for different sizes
    solve_nqueens(n)
```

### Summary

- **Backtracking** systematically explores all potential solutions.
- It is effective for problems where solutions are constructed incrementally and checked for validity.
- Examples include the N-Queens problem, where queens must be placed on a chessboard without threatening each other.
- The provided C++ and Python codes illustrate solving the N-Queens problem using backtracking.


## Explanation of Trie (Prefix Tree)

A Trie, also known as a Prefix Tree, is a special type of tree used to store associative data structures. A common application of a Trie is storing a predictive text or autocomplete dictionary, where the keys are strings.

### Characteristics:
1. **Nodes and Edges:** Each node represents a single character of a string, and the path from the root to any node represents a prefix of a string stored in the Trie.
2. **Root Node:** The Trie has a root node which does not contain any character.
3. **Children:** Each node can have multiple children, one for each possible character.
4. **End of Word:** A special marker is used to denote the end of a word.

### Operations:
1. **Insertion:** Adding a word involves traversing the tree according to the characters of the word, creating nodes as necessary.
2. **Search:** To check if a word exists, traverse the tree according to the characters of the word. If the traversal completes at a node marked as the end of a word, the word exists.
3. **Prefix Search:** Similar to search, but we check if we can traverse the tree according to the characters of the prefix without necessarily reaching an end-of-word marker.

### C++ Code Example: Implementing a Trie

Here is a C++ implementation of a Trie:

```cpp
#include <iostream>
#include <unordered_map>

using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord;
    TrieNode() : isEndOfWord(false) {}
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const string& word) {
        TrieNode* node = root;
        for (char ch : word) {
            if (node->children.find(ch) == node->children.end()) {
                node->children[ch] = new TrieNode();
            }
            node = node->children[ch];
        }
        node->isEndOfWord = true;
    }

    bool search(const string& word) {
        TrieNode* node = root;
        for (char ch : word) {
            if (node->children.find(ch) == node->children.end()) {
                return false;
            }
            node = node->children[ch];
        }
        return node->isEndOfWord;
    }

    bool startsWith(const string& prefix) {
        TrieNode* node = root;
        for (char ch : prefix) {
            if (node->children.find(ch) == node->children.end()) {
                return false;
            }
            node = node->children[ch];
        }
        return true;
    }
};

int main() {
    Trie trie;
    trie.insert("hello");
    trie.insert("world");

    cout << "Search 'hello': " << trie.search("hello") << endl;
    cout << "Search 'hell': " << trie.search("hell") << endl;
    cout << "Starts with 'wor': " << trie.startsWith("wor") << endl;
    cout << "Starts with 'worl': " << trie.startsWith("worl") << endl;

    return 0;
}
```

### Python Code Example: Implementing a Trie

Here is a Python implementation of a Trie:

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

# Example usage:
trie = Trie()
trie.insert("hello")
trie.insert("world")

print("Search 'hello':", trie.search("hello"))  # True
print("Search 'hell':", trie.search("hell"))    # False
print("Starts with 'wor':", trie.starts_with("wor"))  # True
print("Starts with 'worl':", trie.starts_with("worl"))  # True
```

### Summary

- **Trie (Prefix Tree)** is an efficient tree-based data structure for storing strings.
- **Insertion, search, and prefix search** are the primary operations performed on a Trie.
- Useful applications include **spell-checkers and autocomplete systems**.
- Provided C++ and Python implementations illustrate the basic operations of a Trie.



## Explanation of Flood Fill

Flood fill is an algorithm used to determine the area connected to a given node in a multi-dimensional array. It is commonly used in graphics editors to implement the "paint bucket" tool, which fills a contiguous area with a single color. It is also useful in game development for tasks like identifying connected regions in a grid.

### How it works:
1. **Initial Position:** Start from a given pixel.
2. **Check Adjacency:** Check adjacent pixels (up, down, left, right) to see if they match the original color.
3. **Fill:** Change the color of the starting pixel and all matching adjacent pixels.
4. **Recursion/Iteration:** Repeat the process for each pixel that was changed.

The algorithm can be implemented using recursion (depth-first search) or iteration (using a stack or queue for breadth-first search).

### C++ Code Example: Flood Fill

Here is a C++ implementation of the flood fill algorithm:

```cpp
#include <iostream>
#include <vector>

using namespace std;

void floodFillUtil(vector<vector<int>>& screen, int x, int y, int prevC, int newC) {
    int rows = screen.size();
    int cols = screen[0].size();

    // Base cases
    if (x < 0 || x >= rows || y < 0 || y >= cols)
        return;
    if (screen[x][y] != prevC)
        return;

    // Replace the color at (x, y)
    screen[x][y] = newC;

    // Recur for north, east, south and west
    floodFillUtil(screen, x+1, y, prevC, newC);
    floodFillUtil(screen, x-1, y, prevC, newC);
    floodFillUtil(screen, x, y+1, prevC, newC);
    floodFillUtil(screen, x, y-1, prevC, newC);
}

void floodFill(vector<vector<int>>& screen, int x, int y, int newC) {
    int prevC = screen[x][y];
    if (prevC == newC) return;
    floodFillUtil(screen, x, y, prevC, newC);
}

int main() {
    vector<vector<int>> screen = {
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 2, 2, 2, 1, 1, 1},
        {1, 2, 2, 2, 2, 1, 1, 1},
        {1, 1, 2, 2, 2, 2, 2, 1},
        {1, 1, 1, 1, 1, 1, 1, 1}
    };

    int x = 2, y = 2, newC = 3;
    floodFill(screen, x, y, newC);

    cout << "Updated screen after flood fill:\n";
    for (const auto& row : screen) {
        for (int pixel : row) {
            cout << pixel << " ";
        }
        cout << endl;
    }

    return 0;
}
```

### Python Code Example: Flood Fill

Here is a Python implementation of the flood fill algorithm:

```python
def flood_fill_util(screen, x, y, prevC, newC):
    rows, cols = len(screen), len(screen[0])

    # Base cases
    if x < 0 or x >= rows or y < 0 or y >= cols:
        return
    if screen[x][y] != prevC:
        return

    # Replace the color at (x, y)
    screen[x][y] = newC

    # Recur for north, east, south and west
    flood_fill_util(screen, x+1, y, prevC, newC)
    flood_fill_util(screen, x-1, y, prevC, newC)
    flood_fill_util(screen, x, y+1, prevC, newC)
    flood_fill_util(screen, x, y-1, prevC, newC)

def flood_fill(screen, x, y, newC):
    prevC = screen[x][y]
    if prevC == newC:
        return
    flood_fill_util(screen, x, y, prevC, newC)

# Example usage:
screen = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 2, 2, 2, 1, 1, 1],
    [1, 2, 2, 2, 2, 1, 1, 1],
    [1, 1, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

x, y, newC = 2, 2, 3
flood_fill(screen, x, y, newC)

print("Updated screen after flood fill:")
for row in screen:
    print(row)
```

### Summary

- **Flood Fill** is used to fill a contiguous area with a single color.
- It works by checking and filling adjacent pixels recursively or iteratively.
- The algorithm is useful in graphics editors and game development.
- Provided C++ and Python implementations demonstrate the basic flood fill algorithm.


## Explanation of Segment Tree

A Segment Tree is a data structure that allows efficient storage and querying of intervals or segments. It is particularly useful for answering range queries and updating values over intervals in logarithmic time complexity. 

### Characteristics:
1. **Structure:** A Segment Tree is a binary tree where each node represents an interval or segment.
2. **Leaf Nodes:** Each leaf node represents a single element from the array.
3. **Internal Nodes:** Each internal node represents the union of its children's intervals.
4. **Build:** The tree is built in O(n) time, where n is the number of elements.
5. **Query and Update:** Both operations are performed in O(log n) time.

### Operations:
1. **Build:** Construct the tree from a given array.
2. **Query:** Find the sum/minimum/maximum of elements in a given range.
3. **Update:** Modify an element and update the tree accordingly.

### C++ Code Example: Segment Tree for Range Sum Queries

Here is a C++ implementation of a Segment Tree for range sum queries:

```cpp
#include <iostream>
#include <vector>

using namespace std;

class SegmentTree {
private:
    vector<int> tree;
    int n;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;
            build(arr, leftChild, start, mid);
            build(arr, rightChild, mid + 1, end);
            tree[node] = tree[leftChild] + tree[rightChild];
        }
    }

    void update(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;
            if (idx <= mid) {
                update(leftChild, start, mid, idx, val);
            } else {
                update(rightChild, mid + 1, end, idx, val);
            }
            tree[node] = tree[leftChild] + tree[rightChild];
        }
    }

    int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) {
            return 0;  // Outside range
        }
        if (l <= start && end <= r) {
            return tree[node];  // Inside range
        }
        int mid = (start + end) / 2;
        int leftChild = 2 * node + 1;
        int rightChild = 2 * node + 2;
        int sumLeft = query(leftChild, start, mid, l, r);
        int sumRight = query(rightChild, mid + 1, end, l, r);
        return sumLeft + sumRight;
    }

public:
    SegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 0, 0, n - 1);
    }

    void update(int idx, int val) {
        update(0, 0, n - 1, idx, val);
    }

    int query(int l, int r) {
        return query(0, 0, n - 1, l, r);
    }
};

int main() {
    vector<int> arr = {1, 3, 5, 7, 9, 11};
    SegmentTree segTree(arr);

    cout << "Sum of values in given range (1, 3): " << segTree.query(1, 3) << endl;

    segTree.update(1, 10);
    cout << "Updated sum of values in given range (1, 3): " << segTree.query(1, 3) << endl;

    return 0;
}
```

### Python Code Example: Segment Tree for Range Sum Queries

Here is a Python implementation of a Segment Tree for range sum queries:

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)

    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            leftChild = 2 * node + 1
            rightChild = 2 * node + 2
            self.build(arr, leftChild, start, mid)
            self.build(arr, rightChild, mid + 1, end)
            self.tree[node] = self.tree[leftChild] + self.tree[rightChild]

    def update(self, idx, val, node, start, end):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            leftChild = 2 * node + 1
            rightChild = 2 * node + 2
            if idx <= mid:
                self.update(idx, val, leftChild, start, mid)
            else:
                self.update(idx, val, rightChild, mid + 1, end)
            self.tree[node] = self.tree[leftChild] + self.tree[rightChild]

    def query(self, l, r, node, start, end):
        if r < start or end < l:
            return 0  # Outside range
        if l <= start and end <= r:
            return self.tree[node]  # Inside range
        mid = (start + end) // 2
        leftChild = 2 * node + 1
        rightChild = 2 * node + 2
        sumLeft = self.query(l, r, leftChild, start, mid)
        sumRight = self.query(l, r, rightChild, mid + 1, end)
        return sumLeft + sumRight

    def update_index(self, idx, val):
        self.update(idx, val, 0, 0, self.n - 1)

    def range_query(self, l, r):
        return self.query(l, r, 0, 0, self.n - 1)

# Example usage:
arr = [1, 3, 5, 7, 9, 11]
seg_tree = SegmentTree(arr)

print("Sum of values in given range (1, 3):", seg_tree.range_query(1, 3))

seg_tree.update_index(1, 10)
print("Updated sum of values in given range (1, 3):", seg_tree.range_query(1, 3))
```

### Summary

- **Segment Tree** is a binary tree used for storing intervals or segments and efficiently performing range queries and updates.
- **Operations:** Building the tree, querying a range, and updating values can be performed efficiently.
- Useful in scenarios requiring frequent range queries, such as database queries or statistical calculations.
- Provided C++ and Python implementations illustrate basic operations of a Segment Tree for range sum queries.


Hết.

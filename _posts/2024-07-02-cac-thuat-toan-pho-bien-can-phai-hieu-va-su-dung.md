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


Dưới đây là một danh sách các bài tập sử dụng tư duy "đệ quy" để luyện tập C++:

1. **Tính giai thừa của một số**:
   - Viết hàm đệ quy để tính giai thừa của một số nguyên dương n (n! = n * (n-1) * ... * 1).
   
   ```cpp
   int factorial(int n) {
       if (n <= 1)
           return 1;
       return n * factorial(n - 1);
   }
   ```

2. **Tính số Fibonacci**:
   - Viết hàm đệ quy để tính số Fibonacci thứ n (F(n) = F(n-1) + F(n-2) với F(0) = 0 và F(1) = 1).

   ```cpp
   int fibonacci(int n) {
       if (n <= 1)
           return n;
       return fibonacci(n - 1) + fibonacci(n - 2);
   }
   ```

3. **Tìm ước số chung lớn nhất (GCD)**:
   - Viết hàm đệ quy để tìm ước số chung lớn nhất của hai số nguyên a và b sử dụng thuật toán Euclid.

   ```cpp
   int gcd(int a, int b) {
       if (b == 0)
           return a;
       return gcd(b, a % b);
   }
   ```

4. **Đảo ngược một chuỗi**:
   - Viết hàm đệ quy để đảo ngược một chuỗi.

   ```cpp
   void reverseString(string &str, int start, int end) {
       if (start >= end)
           return;
       swap(str[start], str[end]);
       reverseString(str, start + 1, end - 1);
   }
   ```

5. **Kiểm tra chuỗi Palindrome**:
   - Viết hàm đệ quy để kiểm tra xem một chuỗi có phải là palindrome hay không.

   ```cpp
   bool isPalindrome(string str, int start, int end) {
       if (start >= end)
           return true;
       if (str[start] != str[end])
           return false;
       return isPalindrome(str, start + 1, end - 1);
   }
   ```

6. **Sắp xếp bằng thuật toán QuickSort**:
   - Viết hàm đệ quy để sắp xếp một mảng số nguyên sử dụng thuật toán QuickSort.

   ```cpp
   int partition(int arr[], int low, int high) {
       int pivot = arr[high];
       int i = (low - 1);

       for (int j = low; j <= high - 1; j++) {
           if (arr[j] < pivot) {
               i++;
               swap(arr[i], arr[j]);
           }
       }
       swap(arr[i + 1], arr[high]);
       return (i + 1);
   }

   void quickSort(int arr[], int low, int high) {
       if (low < high) {
           int pi = partition(arr, low, high);
           quickSort(arr, low, pi - 1);
           quickSort(arr, pi + 1, high);
       }
   }
   ```

7. **Giải bài toán tháp Hà Nội**:
   - Viết hàm đệ quy để giải bài toán tháp Hà Nội với n đĩa.

   ```cpp
   void towerOfHanoi(int n, char from_rod, char to_rod, char aux_rod) {
       if (n == 1) {
           cout << "Move disk 1 from rod " << from_rod << " to rod " << to_rod << endl;
           return;
       }
       towerOfHanoi(n - 1, from_rod, aux_rod, to_rod);
       cout << "Move disk " << n << " from rod " << from_rod << " to rod " << to_rod << endl;
       towerOfHanoi(n - 1, aux_rod, to_rod, from_rod);
   }
   ```

8. **Đếm số bước di chuyển trong mê cung**:
   - Viết hàm đệ quy để đếm số bước di chuyển từ một vị trí bắt đầu đến vị trí kết thúc trong một mê cung 2D.

   ```cpp
   bool isSafe(int maze[][N], int x, int y) {
       return (x >= 0 && x < N && y >= 0 && y < N && maze[x][y] == 1);
   }

   bool solveMazeUtil(int maze[][N], int x, int y, int sol[][N]) {
       if (x == N - 1 && y == N - 1 && maze[x][y] == 1) {
           sol[x][y] = 1;
           return true;
       }

       if (isSafe(maze, x, y)) {
           if (sol[x][y] == 1)
               return false;

           sol[x][y] = 1;

           if (solveMazeUtil(maze, x + 1, y, sol))
               return true;
           if (solveMazeUtil(maze, x, y + 1, sol))
               return true;
           if (solveMazeUtil(maze, x - 1, y, sol))
               return true;
           if (solveMazeUtil(maze, x, y - 1, sol))
               return true;

           sol[x][y] = 0;
           return false;
       }
       return false;
   }

   bool solveMaze(int maze[][N]) {
       int sol[N][N] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };

       if (!solveMazeUtil(maze, 0, 0, sol)) {
           cout << "Solution doesn't exist";
           return false;
       }

       printSolution(sol);
       return true;
   }
   ```

Những bài tập này sẽ giúp bạn rèn luyện tư duy đệ quy và kỹ năng lập trình C++ của mình. Chúc bạn học tốt!

Dưới đây là một số bài tập đệ quy khác để bạn luyện tập thêm:

9. **Tính tổng các số từ 1 đến n**:
   - Viết hàm đệ quy để tính tổng các số từ 1 đến n.

   ```cpp
   int sum(int n) {
       if (n == 0)
           return 0;
       return n + sum(n - 1);
   }
   ```

10. **Tính lũy thừa**:
    - Viết hàm đệ quy để tính x mũ y (x^y).

    ```cpp
    int power(int x, int y) {
        if (y == 0)
            return 1;
        return x * power(x, y - 1);
    }
    ```

11. **Tìm phần tử lớn nhất trong mảng**:
    - Viết hàm đệ quy để tìm phần tử lớn nhất trong mảng số nguyên.

    ```cpp
    int findMax(int arr[], int n) {
        if (n == 1)
            return arr[0];
        return max(arr[n - 1], findMax(arr, n - 1));
    }
    ```

12. **Đếm số chữ số của một số nguyên**:
    - Viết hàm đệ quy để đếm số chữ số của một số nguyên.

    ```cpp
    int countDigits(int n) {
        if (n == 0)
            return 0;
        return 1 + countDigits(n / 10);
    }
    ```

13. **In tất cả các hoán vị của một chuỗi**:
    - Viết hàm đệ quy để in ra tất cả các hoán vị của một chuỗi.

    ```cpp
    void permute(string str, int l, int r) {
        if (l == r)
            cout << str << endl;
        else {
            for (int i = l; i <= r; i++) {
                swap(str[l], str[i]);
                permute(str, l + 1, r);
                swap(str[l], str[i]); // backtrack
            }
        }
    }
    ```

14. **Sắp xếp bằng thuật toán MergeSort**:
    - Viết hàm đệ quy để sắp xếp một mảng số nguyên sử dụng thuật toán MergeSort.

    ```cpp
    void merge(int arr[], int l, int m, int r) {
        int n1 = m - l + 1;
        int n2 = r - m;
        int L[n1], R[n2];

        for (int i = 0; i < n1; i++)
            L[i] = arr[l + i];
        for (int j = 0; j < n2; j++)
            R[j] = arr[m + 1 + j];

        int i = 0, j = 0, k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            } else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }

        while (i < n1) {
            arr[k] = L[i];
            i++;
            k++;
        }

        while (j < n2) {
            arr[k] = R[j];
            j++;
            k++;
        }
    }

    void mergeSort(int arr[], int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }
    ```

15. **Giải bài toán N-Queens**:
    - Viết hàm đệ quy để giải bài toán N-Queens, đặt N quân hậu lên bàn cờ NxN sao cho không có hai quân hậu nào tấn công nhau.

    ```cpp
    #define N 4

    void printSolution(int board[N][N]) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                cout << board[i][j] << " ";
            cout << endl;
        }
    }

    bool isSafe(int board[N][N], int row, int col) {
        for (int i = 0; i < col; i++)
            if (board[row][i])
                return false;
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
            if (board[i][j])
                return false;
        for (int i = row, j = col; j >= 0 && i < N; i++, j--)
            if (board[i][j])
                return false;
        return true;
    }

    bool solveNQUtil(int board[N][N], int col) {
        if (col >= N)
            return true;
        for (int i = 0; i < N; i++) {
            if (isSafe(board, i, col)) {
                board[i][col] = 1;
                if (solveNQUtil(board, col + 1))
                    return true;
                board[i][col] = 0;
            }
        }
        return false;
    }

    bool solveNQ() {
        int board[N][N] = { { 0, 0, 0, 0 },
                            { 0, 0, 0, 0 },
                            { 0, 0, 0, 0 },
                            { 0, 0, 0, 0 } };
        if (!solveNQUtil(board, 0)) {
            cout << "Solution does not exist";
            return false;
        }
        printSolution(board);
        return true;
    }
    ```

16. **Tính số tổ hợp C(n, k)**:
    - Viết hàm đệ quy để tính số tổ hợp C(n, k).

    ```cpp
    int binomialCoeff(int n, int k) {
        if (k == 0 || k == n)
            return 1;
        return binomialCoeff(n - 1, k - 1) + binomialCoeff(n - 1, k);
    }
    ```

17. **Tìm số đường đi trong lưới MxN**:
    - Viết hàm đệ quy để tính số đường đi từ góc trên bên trái đến góc dưới bên phải của lưới MxN chỉ bằng cách di chuyển xuống hoặc sang phải.

    ```cpp
    int numberOfPaths(int m, int n) {
        if (m == 1 || n == 1)
            return 1;
        return numberOfPaths(m - 1, n) + numberOfPaths(m, n - 1);
    }
    ```

Những bài tập này sẽ giúp bạn củng cố và nâng cao kỹ năng lập trình đệ quy trong C++. Chúc bạn thành công trong việc học tập và luyện tập!

Chắc chắn rồi! Dưới đây là một số bài tập đệ quy khác nữa để bạn có thêm cơ hội luyện tập:

18. **Tìm số lượng bước đi của bài toán cầu thang**:
    - Viết hàm đệ quy để tính số cách leo lên cầu thang với n bậc, mỗi lần có thể leo 1 hoặc 2 bậc.

    ```cpp
    int countWays(int n) {
        if (n <= 1)
            return 1;
        return countWays(n - 1) + countWays(n - 2);
    }
    ```

19. **Tính tổng các phần tử trong mảng**:
    - Viết hàm đệ quy để tính tổng các phần tử trong mảng số nguyên.

    ```cpp
    int sumArray(int arr[], int n) {
        if (n == 0)
            return 0;
        return arr[n - 1] + sumArray(arr, n - 1);
    }
    ```

20. **Chuyển đổi số từ hệ thập phân sang hệ nhị phân**:
    - Viết hàm đệ quy để chuyển đổi một số từ hệ thập phân sang hệ nhị phân.

    ```cpp
    void decimalToBinary(int n) {
        if (n == 0)
            return;
        decimalToBinary(n / 2);
        cout << n % 2;
    }
    ```

21. **Kiểm tra xem một số có phải là số nguyên tố không**:
    - Viết hàm đệ quy để kiểm tra xem một số nguyên dương có phải là số nguyên tố không.

    ```cpp
    bool isPrime(int n, int i = 2) {
        if (n <= 2)
            return (n == 2) ? true : false;
        if (n % i == 0)
            return false;
        if (i * i > n)
            return true;
        return isPrime(n, i + 1);
    }
    ```

22. **Đếm số lần xuất hiện của một phần tử trong mảng**:
    - Viết hàm đệ quy để đếm số lần xuất hiện của một phần tử x trong mảng số nguyên.

    ```cpp
    int countOccurrences(int arr[], int n, int x) {
        if (n == 0)
            return 0;
        return (arr[n - 1] == x) + countOccurrences(arr, n - 1, x);
    }
    ```

23. **Tìm giá trị nhỏ nhất trong mảng**:
    - Viết hàm đệ quy để tìm giá trị nhỏ nhất trong mảng số nguyên.

    ```cpp
    int findMin(int arr[], int n) {
        if (n == 1)
            return arr[0];
        return min(arr[n - 1], findMin(arr, n - 1));
    }
    ```

24. **In các số từ 1 đến n theo thứ tự tăng dần và giảm dần**:
    - Viết hai hàm đệ quy, một hàm để in các số từ 1 đến n theo thứ tự tăng dần, và một hàm để in theo thứ tự giảm dần.

    ```cpp
    void printIncreasing(int n) {
        if (n == 0)
            return;
        printIncreasing(n - 1);
        cout << n << " ";
    }

    void printDecreasing(int n) {
        if (n == 0)
            return;
        cout << n << " ";
        printDecreasing(n - 1);
    }
    ```

25. **Đảo ngược mảng**:
    - Viết hàm đệ quy để đảo ngược một mảng số nguyên.

    ```cpp
    void reverseArray(int arr[], int start, int end) {
        if (start >= end)
            return;
        swap(arr[start], arr[end]);
        reverseArray(arr, start + 1, end - 1);
    }
    ```

26. **Tìm giá trị lớn thứ hai trong mảng**:
    - Viết hàm đệ quy để tìm giá trị lớn thứ hai trong mảng số nguyên.

    ```cpp
    int findSecondLargest(int arr[], int n, int largest = INT_MIN, int secondLargest = INT_MIN) {
        if (n == 0)
            return secondLargest;

        if (arr[n - 1] > largest) {
            secondLargest = largest;
            largest = arr[n - 1];
        } else if (arr[n - 1] > secondLargest && arr[n - 1] != largest) {
            secondLargest = arr[n - 1];
        }

        return findSecondLargest(arr, n - 1, largest, secondLargest);
    }
    ```

27. **Đếm số node trong cây nhị phân**:
    - Viết hàm đệ quy để đếm số node trong một cây nhị phân.

    ```cpp
    struct Node {
        int data;
        Node* left;
        Node* right;
        Node(int val) : data(val), left(nullptr), right(nullptr) {}
    };

    int countNodes(Node* root) {
        if (root == nullptr)
            return 0;
        return 1 + countNodes(root->left) + countNodes(root->right);
    }
    ```

28. **Tìm chiều cao của cây nhị phân**:
    - Viết hàm đệ quy để tìm chiều cao của cây nhị phân.

    ```cpp
    int treeHeight(Node* root) {
        if (root == nullptr)
            return 0;
        int leftHeight = treeHeight(root->left);
        int rightHeight = treeHeight(root->right);
        return 1 + max(leftHeight, rightHeight);
    }
    ```

29. **Tìm giá trị lớn nhất trong cây nhị phân**:
    - Viết hàm đệ quy để tìm giá trị lớn nhất trong cây nhị phân.

    ```cpp
    int findMax(Node* root) {
        if (root == nullptr)
            return INT_MIN;
        int res = root->data;
        int leftRes = findMax(root->left);
        int rightRes = findMax(root->right);
        return max(res, max(leftRes, rightRes));
    }
    ```

30. **Kiểm tra xem cây nhị phân có phải là cây cân bằng không**:
    - Viết hàm đệ quy để kiểm tra xem cây nhị phân có phải là cây cân bằng không (chênh lệch chiều cao của hai cây con không vượt quá 1).

    ```cpp
    bool isBalanced(Node* root, int* height) {
        if (root == nullptr) {
            *height = 0;
            return true;
        }

        int leftHeight = 0, rightHeight = 0;
        bool leftBalanced = isBalanced(root->left, &leftHeight);
        bool rightBalanced = isBalanced(root->right, &rightHeight);

        *height = 1 + max(leftHeight, rightHeight);

        if (abs(leftHeight - rightHeight) > 1)
            return false;

        return leftBalanced && rightBalanced;
    }

    bool isBalanced(Node* root) {
        int height = 0;
        return isBalanced(root, &height);
    }
    ```

Những bài tập này sẽ giúp bạn rèn luyện tư duy đệ quy và kỹ năng lập trình trong C++ thêm sâu sắc. Chúc bạn luyện tập hiệu quả!

Chắc chắn rồi, dưới đây là thêm một số bài tập đệ quy khác nữa để bạn luyện tập:

31. **Kiểm tra chuỗi đối xứng**:
    - Viết hàm đệ quy để kiểm tra xem một chuỗi có đối xứng không (có thể dùng cách so sánh ký tự đầu với ký tự cuối).

    ```cpp
    bool isPalindrome(string str) {
        int n = str.length();
        if (n <= 1)
            return true;
        if (str[0] != str[n - 1])
            return false;
        return isPalindrome(str.substr(1, n - 2));
    }
    ```

32. **Tính số đường đi trong mê cung với chướng ngại vật**:
    - Viết hàm đệ quy để đếm số đường đi từ một vị trí bắt đầu đến vị trí kết thúc trong một mê cung 2D có chướng ngại vật.

    ```cpp
    #define N 4

    bool isSafe(int maze[N][N], int x, int y) {
        return (x >= 0 && x < N && y >= 0 && y < N && maze[x][y] == 1);
    }

    int countPaths(int maze[N][N], int x, int y, int destX, int destY) {
        if (x == destX && y == destY)
            return 1;
        if (!isSafe(maze, x, y))
            return 0;

        int count = 0;
        maze[x][y] = 0; // Mark as visited

        // Move right
        count += countPaths(maze, x + 1, y, destX, destY);
        // Move down
        count += countPaths(maze, x, y + 1, destX, destY);
        // Move left
        count += countPaths(maze, x - 1, y, destX, destY);
        // Move up
        count += countPaths(maze, x, y - 1, destX, destY);

        maze[x][y] = 1; // Mark as unvisited

        return count;
    }
    ```

33. **Tìm tổng các node trong cây nhị phân**:
    - Viết hàm đệ quy để tính tổng các giá trị của các node trong cây nhị phân.

    ```cpp
    int sumNodes(Node* root) {
        if (root == nullptr)
            return 0;
        return root->data + sumNodes(root->left) + sumNodes(root->right);
    }
    ```

34. **Đếm số lượng node lá trong cây nhị phân**:
    - Viết hàm đệ quy để đếm số lượng node lá (node không có con) trong cây nhị phân.

    ```cpp
    int countLeafNodes(Node* root) {
        if (root == nullptr)
            return 0;
        if (root->left == nullptr && root->right == nullptr)
            return 1;
        return countLeafNodes(root->left) + countLeafNodes(root->right);
    }
    ```

35. **Tìm giá trị nhỏ nhất trong cây nhị phân**:
    - Viết hàm đệ quy để tìm giá trị nhỏ nhất trong cây nhị phân.

    ```cpp
    int findMin(Node* root) {
        if (root == nullptr)
            return INT_MAX;
        int res = root->data;
        int leftRes = findMin(root->left);
        int rightRes = findMin(root->right);
        return min(res, min(leftRes, rightRes));
    }
    ```

36. **Tìm chiều sâu của một node trong cây nhị phân**:
    - Viết hàm đệ quy để tìm chiều sâu của một node có giá trị x trong cây nhị phân (chiều sâu là số cạnh từ gốc đến node đó).

    ```cpp
    int findDepth(Node* root, int x, int depth = 0) {
        if (root == nullptr)
            return -1;
        if (root->data == x)
            return depth;

        int leftDepth = findDepth(root->left, x, depth + 1);
        if (leftDepth != -1)
            return leftDepth;

        return findDepth(root->right, x, depth + 1);
    }
    ```

37. **Kiểm tra hai cây nhị phân có giống nhau không**:
    - Viết hàm đệ quy để kiểm tra xem hai cây nhị phân có giống nhau không (có cùng cấu trúc và giá trị các node).

    ```cpp
    bool areIdentical(Node* root1, Node* root2) {
        if (root1 == nullptr && root2 == nullptr)
            return true;
        if (root1 == nullptr || root2 == nullptr)
            return false;
        return (root1->data == root2->data &&
                areIdentical(root1->left, root2->left) &&
                areIdentical(root1->right, root2->right));
    }
    ```

38. **Chuyển đổi biểu thức tiền tố sang biểu thức hậu tố**:
    - Viết hàm đệ quy để chuyển đổi một biểu thức tiền tố (prefix) thành biểu thức hậu tố (postfix).

    ```cpp
    bool isOperator(char x) {
        switch (x) {
            case '+':
            case '-':
            case '*':
            case '/':
                return true;
        }
        return false;
    }

    void prefixToPostfix(string prefix, int &index, string &postfix) {
        if (index < 0)
            return;

        char ch = prefix[index--];
        if (isOperator(ch)) {
            prefixToPostfix(prefix, index, postfix);
            prefixToPostfix(prefix, index, postfix);
            postfix += ch;
        } else {
            postfix += ch;
        }
    }
    ```

39. **Giải bài toán Sudoku**:
    - Viết hàm đệ quy để giải bài toán Sudoku 9x9.

    ```cpp
    #define N 9

    bool isSafe(int grid[N][N], int row, int col, int num) {
        for (int x = 0; x < N; x++)
            if (grid[row][x] == num || grid[x][col] == num)
                return false;

        int startRow = row - row % 3, startCol = col - col % 3;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (grid[i + startRow][j + startCol] == num)
                    return false;

        return true;
    }

    bool solveSudoku(int grid[N][N], int row, int col) {
        if (row == N - 1 && col == N)
            return true;
        if (col == N) {
            row++;
            col = 0;
        }
        if (grid[row][col] != 0)
            return solveSudoku(grid, row, col + 1);

        for (int num = 1; num <= N; num++) {
            if (isSafe(grid, row, col, num)) {
                grid[row][col] = num;
                if (solveSudoku(grid, row, col + 1))
                    return true;
                grid[row][col] = 0;
            }
        }
        return false;
    }
    ```

40. **Kiểm tra xem cây nhị phân có phải là cây tìm kiếm nhị phân (BST) không**:
    - Viết hàm đệ quy để kiểm tra xem cây nhị phân có phải là cây tìm kiếm nhị phân không.

    ```cpp
    bool isBSTUtil(Node* root, int min, int max) {
        if (root == nullptr)
            return true;
        if (root->data < min || root->data > max)
            return false;
        return isBSTUtil(root->left, min, root->data - 1) &&
               isBSTUtil(root->right, root->data + 1, max);
    }

    bool isBST(Node* root) {
        return isBSTUtil(root, INT_MIN, INT_MAX);
    }
    ```

Những bài tập này sẽ giúp bạn rèn luyện tư duy đệ quy và kỹ năng lập trình trong C++ thêm sâu sắc. Chúc bạn luyện tập hiệu quả!

Hết.

---
title: '[Note] Code relax'
date: 2022-02-15
permalink: /posts/2022/02/15/code-relax/
tags:
  - research
  - proposal
  - code
--- 

Code relax
======

```c++
1. https://leetcode.com/problems/two-sum/

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        unordered_map<int, int> m;
        for(int i = 0; i < n; i++){
            m.insert(make_pair(nums[i], i));
        }

        vector<int> res;
        for(int i = 0; i < n; i++){
            int x = target - nums[i];
            if(m.find(x) != m.end()){
                if(i != m.find(x)->second) {
                    res.push_back(i); res.push_back(m.find(x)->second);
                    return res;
                }
            }
        }
        return res;
    }
};

2. https://leetcode.com/problems/powx-n/

class Solution {
public:
    double pow2(double x, long long n){
        if(n == 0) return 1.0;
        if(n == 1) return x;
        if(n % 2) return x * pow2(x, n - 1);
        double tmp = pow2(x, n / 2);
        return tmp * tmp;
    }
    double myPow(double x, int n) {
        double res = 0;
        if(n == 0) res = 1;
        else {
            long long nn = 1LL * n;
            if(nn < 0) res = 1.0 / pow2(x, -nn);
            else res = pow2(x, nn);
        }
        return res;
    }
};

3. https://leetcode.com/problems/climbing-stairs/


class Solution {
public:
    int climbStairs(int n) {
        if(n <= 1) return 1;
        int pre = 1, curr = 1;
        for(int i = 2; i <= n; i++){
            int tmp = curr;
            curr += pre;
            pre = tmp;
        }
        return curr;
    }
};

4. https://leetcode.com/problems/plus-one/
class Solution {
public:
    vector<int> plusOne(vector<int>& d) {
        int n = d.size();
        int carry = 0;
        int i =  n - 1;
        do {
            d[i] += 1;
            if(d[i] >= 10) {
                d[i] = 0;
            }
            else {
                break;
            }
            i--;
        }while(i >= 0);
        if(i < 0) d.insert(d.begin(), 1);
        return d;
    }
};

5. https://leetcode.com/problems/isomorphic-strings/

class Solution {
public:
    bool check(string s, string t){
        int n = s.size();
        unordered_map<char, char> m;
        for(int i = 0; i < n; i++){
            if(m.find(s[i]) != m.end()){
                if(t[i] != m.find(s[i])->second) return false;
            }
            else {
                m.insert(make_pair(s[i], t[i]));
            }
        }
        return true;
    }
    bool isIsomorphic(string s, string t) {
        bool ok1 = check(s, t);
        bool ok2 = check(t, s);
        return ok1 && ok2;
    }
};

6.https://leetcode.com/problems/summary-ranges/

class Solution {
public:
    string f1(int n){
        return to_string(n);
    }
    string f2(int i, int j){
        return to_string(i) + "->" + to_string(j);
    }
    vector<string> summaryRanges(vector<int>& nums) {
        int n = nums.size();
        vector<string> res;
        
        if(n <= 0) return res;
        int i = 0;
        do {
            int j = i;
            while(j < n - 1 && nums[j] == nums[j+1] - 1) j++;
            if(j == i){
                res.push_back(f1(nums[i]));
                i++;
            }
            else {
                res.push_back(f2(nums[i], nums[j]));
                i = j + 1;
            }
        }while(i < n);
        return res;
    }
};

7.https://leetcode.com/problems/power-of-two/

class Solution {
public:
    bool isPowerOfTwo(int n) {
        if(n <= 0) return false;
        if(n == 1) return true;
        long long res = 1;
        while(res <= n){
            res *= 2;
            if(res == n) return true;
        }
        return false;
    }
};

8. https://leetcode.com/problems/ugly-number/

class Solution {
public:
    bool isUgly(int n) {
        if(n <= 0) return false;

        int tmp = (int)sqrt(n);
        int i = 2;
        unordered_map<int, int> m;

        do {
            int cnt = 0;
            while(n % i == 0) {
                cnt++;
                n /= i;
            }
            if(cnt > 0)m.insert(make_pair(i, cnt));
            i++;
        }while(i <= tmp);
        if(n > 1) m.insert(make_pair(n, 1));
        for (auto &[key, value] : m){
            cout << key << ", " << value << endl;
            if (key > 5) return false;
        }

        return true;
    }
};

9. https://leetcode.com/problems/missing-number/submissions/1158421641/

class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> m;
        for(int i = 0; i < n; i++){
            m.insert(make_pair(nums[i], 1));
        }
        for(int i = 0; i <= n; i++){
            if(m.find(i) == m.end()){
                return i;
            }
        }
        return 0;
    }
};

10. https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/?envType=study-plan-v2&envId=binary-search

class Solution {
public:
    vector<int> searchRange(vector<int>& a, int x) {
        vector<int> ans(2, -1);
        if(a.size() == 0) return ans;

        int idx1 = lower_bound(a.begin(), a.end(), x) - a.begin();
        int idx2 = upper_bound(a.begin(), a.end(), x) - a.begin();
        int n = a.size();
        if(idx1 == idx2) return ans;
        ans[0] = idx1;
        ans[1] = idx2 - 1;
        return ans;
    }
};


11.https://leetcode.com/problems/palindrome-number/

class Solution {
public:
    bool isPalindrome(int x) {
        if(x < 0) return 0;
        vector<int> v;
        while(x > 0){
            v.push_back(x % 10);
            x /= 10;
        }
        int n = v.size();
        for(int i = 0; i < n; i++){
            if(v[i] != v[n - i - 1]) return 0;
        }
        return 1;
    }
};
#define ll long long int

class Solution {
public:
    bool isPalindrome(int x) {
        if(x < 0) return 0;
        ll rev, tmp = x;
        while(tmp != 0){
            int digit = tmp % 10;
            rev = rev * 10 + digit;
            tmp /= 10;
        }
        return rev == x;
    }
};

12. https://leetcode.com/problems/roman-to-integer/

class Solution {
public:
    int romanToInt(string s) {
        int n = s.size();
        unordered_map<char, int> m;
        m['I'] = 1;
        m['V'] = 5;
        m['X'] = 10;
        m['L'] = 50;
        m['C'] = 100;
        m['D'] = 500;
        m['M'] = 1000;
        
        int res = 0;
        for(int i = 0; i < n; i++){
            if(m[s[i]] < m[s[i+1]]){
                res -= m[s[i]];
            }
            else {
                res += m[s[i]];
            }
        }
        return res;
    }
};

13. https://leetcode.com/problems/longest-common-prefix/description/
class Solution {
public:
    int min(int a, int b){
        if(a < b) return a;
        return b;
    }
    string longestCommonPrefix(vector<string>& a) {
        int n = a.size();
        int mn = INT_MAX;
        for(int i = 0; i < n; i++){
            mn = min(mn, a[i].size());
        }
        int tmp_mn = mn;
        cout << tmp_mn << endl;

        do {
            bool ok = true;
            for(int i = 0; i < n - 1; i++){
                string tmp1 = a[i].substr(0, tmp_mn);
                string tmp2 = a[i+1].substr(0, tmp_mn);
                if(tmp1 != tmp2){
                    ok = false;
                    break;
                }
            }
            if(ok) break;
            tmp_mn--;
        }while(tmp_mn > 0);
        string res = "";
        for(int i = 0; i < tmp_mn; i++){
            res += a[0][i];
        }
        return res;
    }
};

14. https://leetcode.com/problems/valid-parentheses/
class Solution {
public:
    bool check(char c1, char c2){
        if(c1 == ')' && c2 == '(') return 1;
        if(c1 == '}' && c2 == '{') return 1;
        if(c1 == ']' && c2 == '[') return 1;
        return 0;
    }
    bool isValid(string s) {
        int n = s.size();
        stack<char> st;
        for(int i = 0; i < n; i++){
            if(s[i] == '(' || s[i] == '{' || s[i] == '['){
                cout << "push" << endl;
                st.push(s[i]);
            }
            else {
                if(st.empty() == 1){
                    return 0;
                }
                else {
                    char c = st.top();
                    cout << "c: " << c << endl;
                    if(check(s[i], c)){
                        st.pop();
                    } 
                    else {
                        cout << c << ", " << s[i] << endl;
                        return 0;
                    }
                }
            }
        }
        return st.empty() == true;
    }
};

15. https://leetcode.com/problems/base-7/
class Solution {
public:    
    string convertToBase7(int n) {
        int tmp = n;
        if(n < 0){
            tmp = -n;
        }
        string res = "";
        while(tmp > 0){
            int v = tmp % 7;
            res += to_string(v);
            tmp /= 7;
        }
        reverse(res.begin(), res.end());
        if(n < 0) res = "-" + res;
        if(res.size() == 0) res = "0";
        return res;
    }
};

16. https://leetcode.com/problems/construct-the-rectangle/description/
class Solution {
public:
    vector<int> constructRectangle(int s) {
        int can = (int)sqrt(s);
        int res = 1;
        for(int i = can; i >= 1; i--){
            if(s % i == 0){
                res = i; break;
            }
        }
        
        return {s/res, res};
    }
};

17.https://leetcode.com/problems/perfect-number/description/
class Solution {
public:
    bool checkPerfectNumber(int n) {
        int res = 1;
        int can = (int)sqrt(n);
        if(n == 1) return 0;
        for(int i = 2; i <= can; i++){
            if(n % i == 0){
                res += i;
                if(n / i != i){
                    res += n/i;
                }
            }
        }
        return res == n;
    }
};

18. https://leetcode.com/problems/self-dividing-numbers/description/
class Solution {
public:
    vector<int> selfDividingNumbers(int left, int right) {
        vector<int> res;
        for(int i = left; i <= right; i++){
            int tmp = i;
            vector<int> v;
            while(tmp > 0){
                v.push_back(tmp % 10);
                tmp /= 10;
            }
            bool ok = true;
            for(int j = 0; j < v.size(); j++){
                if(v[j] == 0) {
                    ok = false; break;
                }
                else if(i % v[j] != 0){
                    ok = false; break;
                }
            }
            if(ok) res.push_back(i);
        }
        return res;
    }
};

19. https://leetcode.com/problems/add-two-numbers/description/

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *res = new ListNode(); // constructor
        ListNode *tmp = res; // copy or ?

        int carry = 0;
        while(l1 != NULL || l2 != NULL){
            int sum = 0;
            if(l1 != NULL){
                sum += l1->val;
                l1 = l1->next;
            }
            if(l2 != NULL){
                sum += l2->val;
                l2 = l2->next;
            }
            sum += carry;
            carry = sum / 10;
            ListNode* newNode = new ListNode(sum % 10);
            tmp->next = newNode; // --> assign 
            tmp = tmp->next; // res changed.
        }

        if(carry){
            ListNode *newNode = new ListNode(carry);
            tmp->next = newNode;
            tmp = tmp->next;
        }
        
        res = res->next;
        return res;
    }
};

20. https://leetcode.com/problems/merge-two-sorted-lists/description/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* res = new ListNode();
        ListNode* tmp = res;
        while(list1 != NULL && list2 !=  NULL){
            if(list1 != NULL && list2 != NULL){
                if(list1->val < list2->val){
                    ListNode* newNode = new ListNode(list1->val);
                    tmp->next = newNode;
                    tmp = tmp->next;
                    list1 = list1->next;
                }
                else {
                    ListNode* newNode = new ListNode(list2->val);
                    tmp->next = newNode;
                    tmp = tmp->next;
                    list2 = list2->next;
                }
            }
            cout << "rest" << endl;
        }
        cout << "OK" << endl;
        while(list1 != NULL){
            ListNode* newNode = new ListNode(list1->val);
            tmp->next = newNode;
            tmp = tmp->next;
            list1 = list1->next;
        }
        while(list2 != NULL){
            ListNode* newNode = new ListNode(list2->val);
            tmp->next = newNode;
            tmp = tmp->next;
            list2 = list2->next;
        }
        res = res->next;
        return res;
    }
};

21. https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/


class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n = nums.size();
        int j = 1;
        for(int i = 1; i < n; i++){
            if(nums[i] != nums[i-1]){
                nums[j] = nums[i];
                j++;
            }
        }
        return j;
    }
};

22. https://leetcode.com/problems/remove-element/description/
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int n = nums.size();
        int j = 0;
        for(int i = 0; i < n; i++){
            if(nums[i] == val){
            }
            else {
                nums[j] = nums[i];
                j++;
            }
        }
        return j;
    }
};

23. https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/
class Solution {
public:
    int strStr(string s, string p) {
        int idx = s.find(p);
        if(idx != string::npos) return idx;
        return -1;
    }
};

24. https://leetcode.com/problems/remove-duplicates-from-sorted-list/

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* tmp = head;
        while(tmp != NULL && tmp->next != NULL){
            if(tmp->val == tmp->next->val){
                tmp->next = tmp->next->next;
            }
            else {
                tmp = tmp->next;
            }
        }
        return head;
    }
};

25. https://leetcode.com/problems/maximum-depth-of-binary-tree/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root == NULL) return 0;
        if(root->left == NULL && root->right == NULL) return 1;
        int maxLeft = maxDepth(root->left);
        int maxRight = maxDepth(root->right);
        return max(maxLeft, maxRight) + 1;
    }
};

26. https://leetcode.com/problems/valid-parentheses/
class Solution {
public:
    bool cmp(char c1, char c2){
        if(c1 == '(' && c2 == ')') return true;
        if(c1 == '[' && c2 == ']') return true;
        if(c1 == '{' && c2 == '}') return true;
        return false;
    }
    bool isValid(string s) {
        int n = s.size();
        stack<char> st;
        for(int i = 0; i < n; i++){
            if(s[i] == '(' || s[i] == '[' || s[i] == '{'){
                st.push(s[i]);
            }
            else {
                char tmp = ' ';
                if(!st.empty()) {
                    tmp = st.top();
                    st.pop();
                    if(!cmp(tmp, s[i])) return false;
                }
                else {
                    return false;
                }
            }
        }
        if(!st.empty()) return false;
        return true;
    }
};

27. https://leetcode.com/problems/3sum/
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        int target = 0;
        sort(nums.begin(), nums.end());

        set<vector<int>> sset;
        vector<vector<int>> ans;

        for(int i = 0; i < n; i++){
            int j = i + 1;
            int k = n - 1;
            while(j < k){
                int sum = nums[i] + nums[j] + nums[k];
                if(sum == target){
                    sset.insert({nums[i], nums[j], nums[k]});
                    j++; k--;
                }
                else if(sum < target){
                    j++;
                }
                else {
                    k--;
                }
            }
        }
        for(auto tmp : sset){
            ans.push_back(tmp);
        }
        return ans;
    }
};

28. https://leetcode.com/problems/4sum/

class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int n = nums.size();
        set<vector<int>> sset;
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        for(int i = 0; i < n - 3; i++){
            for(int j = i + 1; j < n - 2; j++){
                int k = j + 1;
                int r = n - 1;
                long long newTarget = (long long)target - (long long)nums[i] - (long long)nums[j];
                while(k < r){
                    long long sum = nums[k] + nums[r];
                    if(sum == newTarget){
                        sset.insert({nums[i], nums[j], nums[k], nums[r]});
                        k++; r--;
                    }
                    else if(sum < newTarget){
                        k++;
                    }
                    else {
                        r--;
                    }
                }
            }
        }
        for(auto tmp : sset){
            ans.push_back(tmp);
        }
        return ans;
    }
};

29. https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n = s.size();
        unordered_map<char, int> mm;
        int left = 0, right = 0;
        int ans = 0;
        while(right < n){
            auto it = mm.find(s[right]);
            if(it != mm.end() && it->second >= left){
                left = it->second + 1;
            }
            else {
                ans = max(ans, right - left + 1);
            }
            mm[s[right]] = right;
            right++;
        }
        return ans;
    }
};

30. https://leetcode.com/problems/median-of-two-sorted-arrays/

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n1 = nums1.size();
        int n2 = nums2.size();
        int i = 0, j = 0, start = 0;
        vector<int> ans(n1 + n2);
        while(i < n1 && j < n2){
            if(nums1[i] < nums2[j]){
                ans[start++] = nums1[i++];
            }
            else {
                ans[start++] = nums2[j++];
            }
        }
        while(i < n1){
            ans[start++] = nums1[i++];
        }
        while(j < n2){
            ans[start++] = nums2[j++];
        }
        int sn = ans.size();
        int mid = (sn-1) / 2;
        if(sn % 2) return 1.0 * ans[mid];
        return 1.0 * (ans[mid] + ans[mid+1]) / 2;
    }
};

31. https://leetcode.com/problems/longest-palindromic-substring/

class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        if(n <= 1) return s;

        auto expand_from_center = [&](int left, int right){
            while(left >= 0 && right < n && s[left] == s[right]){
                left--; right++;
            }
            return s.substr(left + 1, right - left - 1);
        };
        string max_str = s.substr(0, 1);
        for(int i = 0; i < n - 1; i++){
            string odd = expand_from_center(i, i);
            string even = expand_from_center(i, i + 1);
            if(odd.size() > max_str.size()){
                max_str = odd;
            }
            if(even.size() > max_str.size()){
                max_str = even;
            }
        }
        return max_str;
    }
};

32. https://leetcode.com/problems/zigzag-conversion/

class Solution {
public:
    string convert(string s, int numRows) {
        int n = s.size();
        if(n == 0) return "";
        if(numRows == 1 || n == 1) return s;
        string ans;
        vector<string> rows(min(numRows, n));
        int currR = 0;
        bool goDown = false;
        for(char c : s){
            rows[currR] += c;
            if(currR == 0 || currR == numRows - 1){
                goDown = !goDown;
            }
            if(goDown) currR += 1;
            else currR -= 1;
        }
        for(string row : rows){
            ans += row;
        }
        return ans;
    }
};

33. https://leetcode.com/problems/reverse-integer/

class Solution {
public:
    int reverse(int x) {
        long ans = 0;
        while(x != 0){
            ans = ans * 10 + x % 10;
            x /= 10;
        }
        if(ans > INT_MAX || ans < INT_MIN) {
            return 0;
        }
        return ans;
    }
};

34. https://leetcode.com/problems/string-to-integer-atoi/

class Solution {
public:
    int myAtoi(string s) {
        int first_pos_not_space =  s.find_first_not_of(' ');
        s.erase(0, first_pos_not_space);
        int i = 0, sign = 1;

        long long ans = 0;

        if(s[0] == '+' || s[0] == '-'){
            sign = (s[0] == '-') ? -1 : 1;
            i += 1;
        }
        int n = s.size();
        while(i < n && isdigit(s[i])){
            int dig = int(s[i] - '0');
            if(ans > (INT_MAX - dig) / 10) {
                return (sign == 1) ? INT_MAX : INT_MIN;
            }
            ans = ans * 10 + dig;
            i++;
        }
        return sign * ans;
    }
};

35. https://leetcode.com/problems/divide-two-integers/

class Solution {
public:
    int divide(int a, int b) {
        if(a == INT_MIN && b == -1) return INT_MAX;
        long long la = labs(a), lb = labs(b);
        long long ans = 0;
        int sign = ((a > 0)^(b > 0)) == 0 ? 1 : -1;
        while(la >= lb){
            long long tmp = lb, mul = 1;
            while(tmp << 1 <= la){
                tmp <<= 1;
                mul <<= 1;
            }
            la -= tmp;
            ans += mul;
        }
        return ans * sign;
    }
};

36. https://leetcode.com/problems/combination-sum/description/

class Solution {
public:
    void find(vector<int>& nums, int idx, vector<vector<int>>& lst, vector<int> v, int target, int sum){
        if(idx == nums.size()){
            if(sum == target){
                lst.push_back(v);
            }
            return;
        }
        if(sum == target){
            lst.push_back(v);
            return;
        }
        if(sum > target){
            return;
        }

        sum += nums[idx];
        v.push_back(nums[idx]);
        find(nums, idx, lst, v, target, sum);

        v.pop_back();
        sum -= nums[idx];
        find(nums, idx + 1, lst, v, target, sum);
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> lst; 
        vector<int> v;
        find(candidates, 0, lst, v, target, 0);
        return lst;
    }
};

37. https://leetcode.com/problems/count-and-say/description/

class Solution {
public:
    string countAndSay(int n) {
        if(n == 1) return "1";
        string tmp = countAndSay(n - 1);
        string ans = "";
        int sz = tmp.size();
        int i = 0;
        while(i < sz){
            char c = tmp[i];
            int cnt = 0;
            while(i < sz && tmp[i] == c){
                cnt++;
                i++;
            }
            ans += '0' + cnt;
            ans += c;
            cout << ans << endl;
        }
        return ans;
    }
};

38. https://leetcode.com/problems/combination-sum-ii/description/

class Solution {
public:
    void help(vector<int>& nums, int idx, vector<vector<int>>& lst, vector<int>& v, int target){
        if(target == 0){
            lst.push_back(v); 
            return;
        }
        for(int i = idx; i < nums.size(); i++){
            if(nums[i] > target) break;
            if(i > idx && nums[i] == nums[i-1]){
                continue;
            }
            v.push_back(nums[i]);
            help(nums, i + 1, lst, v, target - nums[i]);
            v.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> lst; 
        vector<int> v;
        help(candidates, 0, lst, v, target);
        return lst;
    }
};

39. https://leetcode.com/problems/multiply-strings/description/

class Solution {
public:
    string multiply(string num1, string num2) {
        int n1 = num1.size();
        int n2 = num2.size();
        if(num1 == "0" || num2 == "0") return "0";

        vector<int> ans(n1 + n2, 0);
        for(int i = n1 - 1; i >= 0; i--){
            for(int j = n2 - 1; j >= 0; j--){
                ans[i+j+1] += (num1[i] - '0') * (num2[j] - '0');
                ans[i+j] += ans[i+j+1] / 10;
                ans[i+j+1] %= 10;
            }
        }
        int i = 0;
        string res = "";
        while(ans[i] == 0)i++;
        while(i < ans.size()) res += to_string(ans[i++]);
        return res;
    }
};

40. https://leetcode.com/problems/jump-game-ii/description/

class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        int ans = 0, end = 0;
        int farthest = 0;

        for(int i = 0; i < n - 1; i++){
            farthest = max(farthest, i + nums[i]);
            if(farthest >= n - 1){
                ans++; break;
            }
            if(i == end){
                ans++;
                end = farthest;
            }
        }
        return ans;
    }
};

41. https://leetcode.com/problems/permutations/description/

class Solution {
public:
    void dfs(vector<int>& nums, int start, vector<vector<int>>& ans){
        if(start == nums.size()){
            ans.push_back(nums);
            return;
        }
        for(int i = start; i < nums.size(); i++){
            swap(nums[i], nums[start]);
            dfs(nums, start + 1, ans);
            swap(nums[i], nums[start]);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans;
        dfs(nums, 0, ans);
        return ans;
    }
};

42. https://leetcode.com/problems/permutations-ii/description/

class Solution {
public:
    void dfs(vector<int>& nums, int start, set<vector<int>> &ans){
        if(start >= nums.size()){
            ans.insert(nums);
            return;
        }
        for(int i = start; i < nums.size(); i++){
            swap(nums[i], nums[start]);
            dfs(nums, start + 1, ans);
            swap(nums[i], nums[start]);
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        set<vector<int>> ans;
        vector<vector<int>> res;
        int start = 0;
        dfs(nums, start, ans);
        for(auto &x : ans) {
            res.push_back(x);
        }
        return res;
    }
};

43. https://leetcode.com/problems/same-tree/description/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(p == nullptr && q == nullptr) return true;
        if(p == nullptr || q == nullptr) return false;
        if(p->val != q->val) return false;
        bool b_right = isSameTree(p->right, q->right);
        bool b_left = isSameTree(p->left, q->left);
        return b_right && b_left;
    }
};

44. https://leetcode.com/problems/interleaving-string/description/

class Solution {
public:
    map<pair<int, int>, bool> dp;

    bool dfs(string s1, string s2, string s3, int idx1, int idx2){
        if(idx1 == s1.size() && idx2 == s2.size()){
            return true;
        }
        if(dp.find({idx1, idx2}) != dp.end()){
            return dp[{idx1, idx2}];
        }
        if(idx1 < s1.size() && s1[idx1] == s3[idx1 + idx2] && dfs(s1, s2, s3, idx1 + 1, idx2)){
            return true;
        }
        if(idx2 < s2.size() && s2[idx2] == s3[idx1 + idx2] && dfs(s1, s2, s3, idx1, idx2 + 1)){
            return true;
        }
        dp[{idx1, idx2}] = false;
        return dp[{idx1, idx2}];
    }
    bool isInterleave(string s1, string s2, string s3) {
        if(s3.size() != s1.size() + s2.size()) return false;
        return dfs(s1, s2, s3, 0, 0);
    }
};

45. https://leetcode.com/problems/validate-binary-search-tree/description/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool check(TreeNode* root, long long left, long long right){
        if(root == nullptr) return true;
        if(root->val < right && root->val > left) {
            return check(root->left, left, root->val) && check(root->right, root->val, right);
        }
        return false;
    }
    bool isValidBST(TreeNode* root) {
        long long mn = -1e18, mx = 1e18;
        return check(root, mn, mx);
    }
};


46. https://leetcode.com/problems/next-permutation/

class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        int idx = -1;
        for(int i = n - 2; i >= 0; i--){
            if(nums[i] < nums[i+1]){
                idx = i;
                break;
            }
        }
        if(idx != -1){
            for(int j = n - 1; j >= idx + 1; j--){
                if(nums[j] > nums[idx]){
                    swap(nums[j], nums[idx]);
                    break;
                }
            }
        }
        sort(nums.begin() + idx + 1, nums.end());
    }
};

47. https://leetcode.com/problems/add-binary/description/
class Solution {
public:
    string addBinary(string a, string b) {
        string ans;
        int na = a.size();
        int nb = b.size();
        int carry = 0;
        int i = na - 1, j = nb - 1;
        while(i >= 0 || j >= 0){
            int sum = carry;
            if(i >= 0) sum += (a[i--] - '0');
            if(j >= 0) sum += (b[j--] - '0');
            carry = sum > 1 ? 1 : 0;
            ans += to_string(sum % 2);
        }
        if(carry) ans += to_string(carry);
        reverse(ans.begin(), ans.end());
        return ans;
    }
};

48.
```


Hết.

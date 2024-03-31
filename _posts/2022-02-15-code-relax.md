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


48. https://leetcode.com/problems/binary-tree-level-order-traversal/
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
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if(root == NULL) {
            return ans;
        }
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            int levelSize = q.size();
            vector<int> output;
            for(int i = 0; i < levelSize; i++){
                TreeNode* front = q.front();
                q.pop();
                output.push_back(front->val);

                if(front->left != nullptr){
                    q.push(front->left);
                }
                if(front->right != nullptr){
                    q.push(front->right);
                }
            }
            ans.push_back(output);
        }
        return ans;
    }
};
49. https://leetcode.com/problems/symmetric-tree/description/
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
    bool check(TreeNode* left, TreeNode* right){
        if(left == NULL && right == NULL) return true;
        if(left == NULL || right == NULL) return false;
        bool cond1 = (left->val == right->val) ? true : false;
        bool cond2 = check(left->left, right->right);
        bool cond3 = check(left->right, right->left);
        return cond1 && cond2 && cond3;
    }
    bool isSymmetric(TreeNode* root) {
        return check(root->left, root->right);
    }
};

50. https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
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
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        queue<TreeNode*> q;
        if(root == NULL) return ans;
        q.push(root);
        int sz = 0;
        bool direction = 1;
        while(!q.empty()){
            sz = q.size();
            vector<int> output;
            while(sz-- > 0){
                root = q.front();
                q.pop();
                if(root->left != nullptr){
                    q.push(root->left);
                }
                if(root->right != nullptr){
                    q.push(root->right);
                }
                output.push_back(root->val);
            }
            ans.push_back(output);
            if(!direction) {
                reverse(ans[ans.size() - 1].begin(), ans[ans.size() - 1].end());
            }
            direction = !direction;
        }
        return ans;
    }
};

51.https://leetcode.com/problems/single-number/description/
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> mm;
        for(int i = 0; i < n; i++){
            if(mm.find(nums[i]) != mm.end()){
                mm.erase(mm.find(nums[i]));
                mm.insert(make_pair(nums[i], 2));
            }
            else {
                mm.insert(make_pair(nums[i], 1));
            }
        }
        int mn = -3 * 1e4;
        int mx = 3 * 1e4;
        for(int i = mn; i <= mx; i++){
            if(mm.find(i) != mm.end()){
                cout << mm.find(i)->first << ", " << mm.find(i)->second << endl;
                if(mm.find(i)->second == 1){
                    return mm.find(i)->first;
                }
            }
        }
        return 0;
    }
};
52. https://leetcode.com/problems/single-number-ii/

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> mm;
        for(int i = 0; i < n; i++){
            if(mm.find(nums[i]) != mm.end()){
                int val = mm.find(nums[i])->second;
                mm.erase(mm.find(nums[i]));
                mm.insert(make_pair(nums[i], val + 1));
            }
            else {
                mm.insert(make_pair(nums[i], 1));
            }
        }
        
        for(int i = 0; i < n; i++){
            if(mm.find(nums[i]) != mm.end()){
                int val = mm.find(nums[i])->second;
                if(val == 1){
                    return nums[i];
                }
            }
        }
        
        return 0;
    }
};
53. https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/
class Solution {
public:
    bool isValidSerialization(string s) {
        stringstream ss(s);
        string curr;
        int nodes = 1;
        while(getline(ss, curr, ',')){
            nodes--;
            if(nodes < 0) return false;
            if(curr != "#") nodes += 2;
        }
        return nodes == 0;
    }
};
54. https://leetcode.com/problems/binary-search-tree-iterator/description/
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
class BSTIterator {
public:
    stack<int> st;
    void itr(TreeNode* root){
        if(root == NULL) return;
        if(root->right != NULL) itr(root->right);
        st.push(root->val);
        if(root->left != NULL) itr(root->left);
    }  
    int next(){
        int val = st.top(); 
        st.pop();
        return val;
    }  
    bool hasNext(){
        if(st.size() > 0) return true;
        return false;
    }
    BSTIterator(TreeNode* root) {
        itr(root);
    }
};

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */
55. https://leetcode.com/problems/binary-tree-right-side-view/description/
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
    void findRightSideView(TreeNode* node, vector<int>& ans){
        if(node == NULL) return;
        queue<pair<TreeNode*, int>> q;
        int currL = -1;
        q.push({node, 0});
        while(!q.empty()){
            auto element = q.front();
            q.pop();
            node = element.first;
            int lvl = element.second;
            if(currL < lvl){
                ans.push_back(node->val);
                currL = lvl;
            }
            if(node->right != nullptr) q.push({node->right, lvl + 1});
            if(node->left != nullptr) q.push({node->left, lvl + 1});
        }
    }
    vector<int> rightSideView(TreeNode* root) {
        vector<int> ans;
        findRightSideView(root, ans);
        return ans;
    }
};
56. https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/
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
    void inorder(TreeNode* root, vector<int>& ans){
        if(root == NULL) return;
        inorder(root->left, ans);
        ans.emplace_back(root->val);
        inorder(root->right, ans);
    }
    int kthSmallest(TreeNode* root, int k) {
        vector<int> ans;
        inorder(root, ans);
        return ans[k-1];
    }
};
57. https://leetcode.com/problems/single-number-iii/description/
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> mm;
        for(int i = 0; i < n; i++){
            auto tmp = mm.find(nums[i]);
            if(tmp != mm.end()){
                int val = tmp->second;
                mm.erase(tmp);
                mm.insert(make_pair(nums[i], val + 1));
            }
            else {
                mm.insert(make_pair(nums[i], 1));
            }
        }
        vector<int> ans;
        for(int i = 0; i < n; i++){
            auto tmp = mm.find(nums[i]);
            if(tmp != mm.end()){
                if(tmp->second == 1){
                    ans.push_back(nums[i]);
                }
            }
        }
        return ans;
    }
};
58. https://leetcode.com/problems/minimum-height-trees/description/
class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        if(edges.size() == 0) return {0};
        vector<int> ans;
        vector<vector<int>> G(n);
        for(auto& e : edges){
            G[e[0]].push_back(e[1]);
            G[e[1]].push_back(e[0]);
        }
        vector<int> newLeaves, inDegree;
        for(int i = 0; i < n; i++){
            if(G[i].size() == 1){
                ans.push_back(i);
            }
            inDegree.push_back(G[i].size());
        }
        while(n > 2){
            for(auto& leaf : ans){
                for(auto& adj : G[leaf]){
                    if(--inDegree[adj] == 1){
                        newLeaves.push_back(adj);
                    }
                }
            }
            n -= ans.size();
            ans = move(newLeaves);
        }
        return ans;
    }
};

59. https://leetcode.com/problems/range-sum-query-mutable/
class NumArray {
public:
    vector<int> seg;
    int n;
    void build(vector<int>& nums, int pos, int left, int right){
        if(left == right){
            seg[pos] = nums[left];
            return;
        }
        int mid = (left + right) / 2;
        build(nums, 2 * pos + 1, left, mid);
        build(nums, 2 * pos + 2, mid + 1, right);
        seg[pos] = seg[2 * pos + 1] + seg[2 * pos + 2];
    }
    void update(int pos, int left, int right, int idx, int val){
        if(idx < left || idx > right) return;
        if(left == right){
            if(left == idx) {
                seg[pos] = val;
            }
            return;
        }
        int mid = (left + right) / 2;
        update(2 * pos + 1, left, mid, idx, val);
        update(2 * pos + 2, mid + 1, right, idx, val);
        seg[pos] = seg[2 * pos + 1] + seg[2 * pos + 2];
    }
    int sum(int queryL, int queryH, int left, int right, int pos){
        if(queryL <= left && queryH >= right) {
            return seg[pos];
        }
        if(queryL > right || queryH < left) {
            return 0;
        }
        int mid = (left + right) / 2;
        return sum(queryL, queryH, left, mid, 2 * pos + 1) + sum(queryL, queryH, mid + 1, right, 2 * pos + 2);
    }
    NumArray(vector<int>& nums) {
        if(nums.size() > 0){
            n = nums.size();
            seg.resize(4 * n, 0);
            build(nums, 0, 0, n - 1);
        }
    }
    
    void update(int index, int val) {
        if(n == 0) return;
        update(0, 0, n - 1, index, val);
    }
    
    int sumRange(int left, int right) {
        if(n == 0) return 0;
        return sum(left, right, 0, n - 1, 0);
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * obj->update(index,val);
 * int param_2 = obj->sumRange(left,right);
 */
 
 60. https://leetcode.com/problems/range-frequency-queries/
 class RangeFreqQuery {
public:
    vector<unordered_map<int, int>> tree;
    int n = 0;
    void build(vector<int>& nums, int pos, int left, int right){
        if(left > right) return;
        if(left == right) {
            tree[pos].insert({nums[left], 1});
            return;
        }
        int mid = (left + right) / 2;
        build(nums, 2 * pos + 1, left, mid);
        build(nums, 2 * pos + 2, mid + 1, right);
        for(auto itr : tree[2 * pos + 1]){
            tree[pos][itr.first] += itr.second;
        }
        for(auto itr : tree[2 * pos + 2]){
            tree[pos][itr.first] += itr.second;
        }
    }
    RangeFreqQuery(vector<int>& arr) {
        if(arr.size() > 0){
            n = arr.size();
            int x = int(ceil(log2(n)));
            int mx_size = 2 * int(pow(2, x)) - 1;
            tree.resize(mx_size);
            build(arr, 0, 0, n - 1);
        }
    }
    int frequency(int pos, int left, int right, int queryL, int queryH, int val){
        if(queryL > right || queryH < left) return 0;
        if(queryL <= left && queryH >= right){
            return tree[pos].find(val) != tree[pos].end() ? tree[pos][val] : 0;
        }
        int mid = (left + right) / 2;
        return frequency(2 * pos + 1, left, mid, queryL, queryH, val) + frequency(2 * pos + 2, mid + 1, right, queryL, queryH, val);
    }
    int query(int left, int right, int value) {
        return frequency(0, 0, n - 1, left, right, value);
    }
};

/**
 * Your RangeFreqQuery object will be instantiated and called as such:
 * RangeFreqQuery* obj = new RangeFreqQuery(arr);
 * int param_1 = obj->query(left,right,value);
 */
 
 61. https://leetcode.com/problems/balanced-binary-tree/description/
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
    int getHeight(TreeNode* root){
        if(root == nullptr) return 0;
        int leftH = getHeight(root->left);
        int rightH = getHeight(root->right);
        if(leftH == -1 || rightH == -1 || abs(leftH - rightH) > 1) return -1;
        int h = 1 + max(leftH, rightH);
        return h;
    }
    bool isBalanced(TreeNode* root) {
        if(root == nullptr) return true;
        if(getHeight(root) == -1) return false;
        return true;
    }
};
62. https://leetcode.com/problems/minimum-depth-of-binary-tree/description/
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
    void solve(TreeNode* root, int& ans, int tmp){
        if(root == NULL) return;
        if(root->left == NULL && root->right == NULL){
            ans = min(ans, tmp);
            tmp = 0;
        }
        tmp++;
        solve(root->left, ans, tmp);
        solve(root->right, ans, tmp);
        return;
    }
    int minDepth(TreeNode* root) {
        if(root == NULL) return 0;
        int ans = INT_MAX;
        int tmp = 1;
        solve(root, ans, tmp);
        return ans;
    }
};
63. https://leetcode.com/problems/path-sum/description/
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
    bool hasPathSum(TreeNode* root, int targetSum) {
        if(root == nullptr) return false;
        if(root->left == NULL && root->right == NULL){
            return root->val == targetSum;
        }
        bool left_sum = hasPathSum(root->left, targetSum - root->val);
        bool right_sum = hasPathSum(root->right, targetSum - root->val);
        return left_sum || right_sum;
    }
};
64. https://leetcode.com/problems/path-sum-ii/description/
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
    void solve(TreeNode* root, int targetSum, vector<vector<int>>& ans, vector<int> tmp, int sum){
        if(root == NULL) return;
        sum += root->val;
        tmp.push_back(root->val);
        if(root->left == NULL && root->right == NULL){
            if(sum == targetSum){
                ans.push_back(tmp);
            }
            else {
                return;
            }
        }
        solve(root->left, targetSum, ans, tmp, sum);
        solve(root->right, targetSum, ans, tmp, sum);
        return;
    }
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> ans;
        vector<int> tmp;
        int sum = 0;
        solve(root, targetSum, ans, tmp, sum);
        return ans;
    }
};
65. https://leetcode.com/problems/count-primes/description/
class Solution {
public:
    int countPrimes(int n) {
        if(n <= 1) return 0;
        vector<bool> primes(n + 1, 1);
        primes[0] = primes[1] = false;
        int cnt = 1;
        int sqrt_n = sqrt(n);
        for(int i = 2; i <= sqrt_n; i++){
            if(primes[i]) {
                for(int j = i * i; j < n; j+=i){
                    if(primes[j]) cnt++;
                    primes[j] = false;
                }
            }
        }
        return n - cnt - 1;
    }
};
66. https://leetcode.com/problems/add-digits/description/
class Solution {
public:
    int addDigits(int num) {
        if(num == 0) return 0;
        else if(num % 9 == 0) return 9;
        return num % 9;
    }
};

67. https://leetcode.com/problems/clone-graph/
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/

class Solution {
public:
    Node* dfs(Node* curr, unordered_map<Node*, Node*>& mm){
        vector<Node*> neighbour;
        Node* clone = new Node(curr->val);
        mm[curr] = clone;
        for(auto it : curr->neighbors){
            if(mm.find(it) != mm.end()){
                neighbour.push_back(mm[it]);
            }
            else {
                neighbour.push_back(dfs(it, mm));
            }
        }
        clone->neighbors = neighbour;
        return clone;
    }
    Node* cloneGraph(Node* node) {
        unordered_map<Node*, Node*> mm;
        if(node == NULL) return NULL;
        if(node->neighbors.size() == 0){
            Node* clone = new Node(node->val);
            return clone;
        }
        return dfs(node, mm);
    }
};
68. https://leetcode.com/problems/minimum-absolute-difference-in-bst/
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
    vector<int> nodes;
    void inorder_dfs(TreeNode* root){
        if(root == NULL) return;
        inorder_dfs(root->left);
        nodes.push_back(root->val);
        inorder_dfs(root->right);
    }
    int getMinimumDifference(TreeNode* root) {
        inorder_dfs(root);
        int n = nodes.size();
        int ans = nodes[n - 1];
        for(int i = 1; i < n; i++){
            ans = min(ans, nodes[i] - nodes[i-1]);
        }
        return ans;
    }
};
69. https://leetcode.com/problems/subsets/
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        int n = pow(2, nums.size()) - 1;
        for(int i = 0; i <= n; i++){
            vector<int> tmp;
            int idx = 0, res = n & i;
            while(idx < nums.size() && res){
                cout << res << endl;
                if((res & 1) == 1) {
                    tmp.push_back(nums[idx]);
                }
                res >>= 1;
                idx++;
            }
            ans.push_back(tmp);
        }
        return ans;
    }
};
70. https://leetcode.com/problems/subsets-ii/
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        unordered_set<string> seen;
        for(int mask = 0, sz = 1 << n; mask < sz; mask++){
            vector<int> subset;
            string hashcode = "";
            for(int i = 0; i < n; i++){
                int bit = (mask >> i) & 1;
                if(bit == 1){
                    subset.push_back(nums[i]);
                    hashcode += to_string(nums[i]) + ",";
                }
            }
            if(seen.count(hashcode) == 0){
                ans.push_back(subset);
                seen.insert(hashcode);
            }
        }
        return ans;
    }
};
71. https://leetcode.com/problems/counting-bits/
class Solution {
public:
    int solve(int num){
        int ans = 0;
        while(num > 0){
            if(num & 1 == 1) ans++;
            num >>= 1;
        }
        return ans;
    }
    vector<int> countBits(int n) {
        vector<int> ans;
        for(int i = 0; i <= n; i++){
            ans.push_back(solve(i));
        }
        return ans;
    }
};
72. https://leetcode.com/problems/power-of-four/
class Solution {
public:
    bool isPowerOfFour(int n) {
        if(n <= 0) return 0;
        int numOfbit1 = __builtin_popcount(n);
        int numOftrailingZero = __builtin_ctz(n);
        if(numOfbit1  == 1 && numOftrailingZero % 2 == 0){
            return 1;
        }
        return 0;
    }
};
73. https://leetcode.com/problems/sum-of-two-integers/
class Solution {
public:
    int getSum(int a, int b) {
        while(b != 0){
            int carry = a & b;
            a = a ^ b;
            b = carry << 1;
        }
        return a;
    }
};
74. https://leetcode.com/problems/repeated-dna-sequences/
class Solution {
public:
    vector<string> findRepeatedDnaSequences(string s) {
        int n = s.size();
        vector<string> ans;
        unordered_map<string, int> mm;
        for(int i = 0; i < n - 9; i++){
            string tmp = s.substr(i, 10);
            auto it = mm.find(tmp);
            if(it != mm.end()){
                int val = it->second;
                mm.erase(it);
                mm.insert(make_pair(tmp, val + 1));
            }
            else {
                mm.insert(make_pair(tmp, 1));
            }
        }
        for(auto& x : mm){
            if(x.second > 1){
                ans.push_back(x.first);
            }
        }
        return ans;
    }
};
75. https://leetcode.com/problems/reverse-bits/
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t ans = 0;
        for(int i = 0; i < 32; i++){
            ans <<= 1;
            ans |= (n & 1);
            n >>= 1;
        }
        return ans;
    }
};
76. https://leetcode.com/problems/number-of-1-bits/
class Solution {
public:
    int hammingWeight(uint32_t n) {
        return __builtin_popcount(n);
    }
};
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int ans = 0;
        while(n){
            ans += (n % 2);
            n >>= 1;
        }
        return ans;
    }
};
77. https://leetcode.com/problems/bitwise-and-of-numbers-range/description/
class Solution {
public:
    int rangeBitwiseAnd(int left, int right) {
        int cnt = 0;
        while(left != right){
            left >>= 1;
            right >>= 1;
            cnt++;
        }
        int ans = left << cnt;
        return ans;
    }
};
78. https://leetcode.com/problems/count-complete-tree-nodes/
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
    int countNodes(TreeNode* root) {
        if(root == nullptr) return 0;
        int lH = 0, rH = 0;
        TreeNode* leftNode = root;
        TreeNode* rightNode = root;
        while(leftNode != nullptr){
            leftNode = leftNode->left;
            lH++;
        }
        while(rightNode != nullptr){
            rightNode = rightNode->right;
            rH++;
        }
        if(lH == rH){
            return pow(2, lH) - 1;
        }
        return 1 + countNodes(root->left) + countNodes(root->right);
    }
};
79. https://leetcode.com/problems/find-the-duplicate-number/
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> mm;
        for(int i = 0; i < n; i++){
            auto it = mm.find(nums[i]);
            if(it != mm.end()){
                int val = it->second;
                mm.erase(it);
                mm.insert(make_pair(nums[i], val + 1));
            }
            else {
                mm.insert(make_pair(nums[i], 1));
            }
        }
        for(auto& x : mm){
            if(x.second > 1){
                return x.first;
            }
        }
        return 0;
    }
};
80. https://leetcode.com/problems/maximum-product-of-word-lengths/
class Solution {
public:
    bool check(vector<int>& a, vector<int>& b){
        int n = 26;
        for(int i = 0; i < n; i++){
            if(a[i] > 0 && b[i] > 0){
                return true;
            }
        }
        return false;
    }
    int maxProduct(vector<string>& words) {
        int n = words.size();
        int ans = 0;
        vector<vector<int>> v(n, vector<int>(26, 0));
        for(int i = 0; i < n; i++){
            string tmp = words[i];
            for(auto ch : tmp){
                v[i][ch - 'a']++;
            }
            for(int j = 0; j < i; j++){
                if(!check(v[i], v[j])){
                    int sz = words[i].size() * words[j].size();
                    ans = max(ans, sz);
                }
            }
        }
        return ans;
    }
};
81. https://leetcode.com/problems/find-if-path-exists-in-graph/
class Solution {
public:
    void dfs(int node, vector<vector<int>>& adj, vector<int>& vis){
        vis[node] = 1;
        for(auto it : adj[node]){
            if(!vis[it]){
                dfs(it, adj, vis);
            }
        }
    }
    bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
        vector<vector<int>> adj(n);
        for(auto it : edges){
            adj[it[0]].push_back(it[1]);
            adj[it[1]].push_back(it[0]);
        }
        vector<int> vis(n, 0);
        dfs(source, adj, vis);
        if(vis[destination] == 0){
            return false;
        }
        return true;
    }
};
82. https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/
#define ll long long int
#define pll pair<ll, ll>
#define MOD (int)(1e9 + 7)

class Solution {
public:
    int dijk(const vector<vector<pll>>& graph, int n, int src){
        vector<ll> dist(n, LONG_MAX);
        vector<ll> ways(n);
        ways[src] = 1;
        dist[src] = 0;
        priority_queue<pll, vector<pll>, greater<>> minHeap;
        minHeap.push({0, 0});
        while(!minHeap.empty()){
            auto [d, u] = minHeap.top(); minHeap.pop();
            if(d  > dist[u]) continue;
            for(auto [v, time] : graph[u]){
                if(dist[v] > d + time){
                    dist[v] = d + time;
                    ways[v] = ways[u];
                    minHeap.push({dist[v], v});
                }
                else if(dist[v] == d + time) {
                    ways[v] = (ways[v] + ways[u]) % MOD;
                }
            }
        }
        return ways[n-1] % MOD;
    }

    int countPaths(int n, vector<vector<int>>& roads) {
        vector<vector<pll>> graph(n);
        for(auto& road : roads){
            ll u = road[0], v = road[1], time = road[2];
            graph[u].push_back({v, time});
            graph[v].push_back({u, time});
        }
        return dijk(graph, n, 0);
    }
};

83. https://leetcode.com/problems/find-pivot-index/
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        int sum = 0;
        int leftS = 0;
        for(int i = 0; i < nums.size(); i++){
            sum += nums[i];
        }
        for(int i = 0; i < nums.size(); i++){
            sum -= nums[i];
            if(sum == leftS) {
                return i;
            }
            leftS += nums[i];
        }
        return -1;
    }
};
84. https://leetcode.com/problems/check-if-all-the-integers-in-a-range-are-covered/
class Solution {
public:
    bool isCovered(vector<vector<int>>& ranges, int left, int right) {
        int n = ranges.size();
        for(int i = left; i <= right; i++){
            bool seen = false;
            for(int j = 0; j < n && !seen; j++){
                if(i >= ranges[j][0] && i <= ranges[j][1]) {
                    seen = true;
                    break;
                }
            }
            if(!seen) return 0;
        }
        return 1;
    }
};
85. https://leetcode.com/problems/minimum-size-subarray-sum/
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size();
        int ans = INT_MAX;
        long long sum = 0;
        int idx = 0;
        for(int i = 0; i < n; i++){
            sum += nums[i];
            while(sum - nums[idx] >= target){
                sum -= nums[idx];
                idx++;
            }
            if(sum >= target){
                ans = min(ans, i - idx + 1);
            }
        }
        return ans == INT_MAX ? 0 : ans;
    }
};
86. https://leetcode.com/problems/product-of-array-except-self/
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> ans(n, 1);
        int prefix = 1;
        for(int i = 0; i < n; i++){
            ans[i] = prefix;
            prefix *= nums[i];
        }
        
        int postfix = 1;
        for(int i = n - 1; i >= 0; i--){
            ans[i] *= postfix;
            postfix *= nums[i];
        }

        return ans;
    }
};
87. https://leetcode.com/problems/range-sum-query-immutable/
class NumArray {
private:
    vector<int> prefixSum;
public:
    NumArray(vector<int>& nums) {
        prefixSum.resize(nums.size());
        prefixSum[0] = nums[0];
        for(int i = 1; i < nums.size(); i++){
            prefixSum[i] = prefixSum[i-1] + nums[i];
        }
    }
    
    int sumRange(int left, int right) {
        if(left == 0){
            return prefixSum[right];
        }
        return prefixSum[right] - prefixSum[left-1];
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * int param_1 = obj->sumRange(left,right);
 */
 88. https://leetcode.com/problems/range-sum-query-2d-immutable/
 class NumMatrix {
private:
    int row, col;
    vector<vector<int>> sums;
public:
    NumMatrix(vector<vector<int>>& matrix) {
        row = matrix.size();
        col = row > 0 ? matrix[0].size() : 0;
        sums = vector<vector<int>>(row + 1, vector<int>(col+1, 0));
        for(int i = 1; i <= row; i++){
            for(int j = 1; j <= col; j++){
                sums[i][j] = matrix[i-1][j-1] + sums[i-1][j] + sums[i][j-1] - sums[i-1][j-1];
            }
        }
    }
    
    int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2 + 1][col2 + 1] - sums[row2 + 1][col1] - sums[row1][col2 + 1] + sums[row1][col1];
    }
};

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix* obj = new NumMatrix(matrix);
 * int param_1 = obj->sumRegion(row1,col1,row2,col2);
 */
 89. https://leetcode.com/problems/subarray-sum-equals-k/
 class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        unordered_map<int, int> mm;
        mm[0] = 1;
        int cnt = 0;
        int prefixSum = 0;
        for(int i = 0; i < n; i++){
            prefixSum += nums[i];
            int toRemove = prefixSum - k;
            cnt += mm[toRemove];
            mm[prefixSum]++;
        }
        return cnt;
    }
};
90. https://leetcode.com/problems/contiguous-array/
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> mm;
        mm[0] = -1;
        int sum = 0, ans = 0;
        for(int i = 0; i < n; i++){
            sum += (nums[i] == 0) ? -1 : 1;
            auto it = mm.find(sum);
            if(it != mm.end()){
                ans = max(ans, i - mm[sum]);
            }
            else {
                mm[sum] = i;
            }
        }
        return ans;
    }
};


94. https://leetcode.com/problems/count-of-range-sum/description/
95. https://leetcode.com/problems/queue-reconstruction-by-height/description/
96. https://leetcode.com/problems/reverse-pairs/description/
97. https://leetcode.com/problems/number-of-longest-increasing-subsequence/description/
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int ans = INT_MIN;
        int n = nums.size();
        vector<pair<int, int>> dp (n, {1, 1});
        for(int i = 0; i < n; i++){
            for(int j = 0; j < i; j++){
                if(nums[i] > nums[j]){
                    if(dp[i].first < dp[j].first + 1) {
                        dp[i].second = 0;
                    }
                    dp[i].first = max(dp[i].first, dp[j].first + 1);
                    if(dp[i].first <= dp[j].first + 1){
                        dp[i].second += dp[j].second;
                    }
                    ans = max(ans, dp[i].first);
                }
            }
        }
        if(ans == INT_MIN) return n;
        int cnt = 0;
        for(auto [sub, count] : dp){
            if(sub == ans){
                cnt += count;
            }
        }
        return cnt;
    }
};

98. https://leetcode.com/problems/minimum-distance-between-bst-nodes/description/
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
    int dfs(TreeNode* curr, int right, int left){
        if(curr == NULL) return INT_MAX;
        int a = right == -1 ? INT_MAX : right - (curr->val);
        int b = left == -1 ? INT_MAX : (curr->val) - left;
        a = min(a, b);
        b = min(dfs(curr->left, curr->val, left), dfs(curr->right, right, curr->val));
        return min(a, b);
    }
    int minDiffInBST(TreeNode* root){
        return dfs(root, -1, -1);
    }
};
99. https://leetcode.com/problems/increasing-order-search-tree/description/
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
    void InOrder(TreeNode* root, queue<TreeNode*> &q){
        if(root == NULL) return;
        InOrder(root->left, q);
        q.push(root);
        InOrder(root->right, q);
    }
    TreeNode* increasingBST(TreeNode* root) {
        if(root == NULL) return NULL;
        queue<TreeNode*> q;
        InOrder(root, q);
        root = q.front(); q.pop();
        TreeNode* tmp = root;
        while(!q.empty()){
            TreeNode* node = q.front();
            q.pop();
            tmp->left = NULL;
            tmp->right = node;
            tmp = node;
        }
        tmp->left = NULL;
        tmp->right = NULL;
        return root;
    }
};
100. https://leetcode.com/problems/range-sum-of-bst/description/
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
    int rangeSumBST(TreeNode* root, int low, int high) {
        if(root == NULL) return 0;
        int r = rangeSumBST(root->right, low, high);
        int l = rangeSumBST(root->left, low, high);
        if(root->val <= high && root->val >= low){
            return root->val + r + l;
        }
        return l + r;
    }
};
101. https://leetcode.com/problems/search-in-a-binary-search-tree/description/
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
    TreeNode* searchBST(TreeNode* root, int val) {
        if(root == NULL) return NULL;
        if(root->val == val) return root;
        if(val < root->val){
            return searchBST(root->left, val);
        }
        return searchBST(root->right, val);
    }
};
102. https://leetcode.com/problems/two-sum-iv-input-is-a-bst/description/
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
    void InOrder(TreeNode* root, vector<int>& v){
        if(root == NULL) return;
        InOrder(root->left, v);
        v.push_back(root->val);
        InOrder(root->right, v);
    }
    bool findTarget(TreeNode* root, int k) {
        vector<int> v;
        InOrder(root, v);
        int i = 0, j = v.size() - 1;
        while(i < j){
            int s = v[i] + v[j];
            if(s == k) return true;
            else if(s < k)i++;
            else j--;
        }
        return false;
    }
};
103. https://leetcode.com/problems/find-mode-in-binary-search-tree/description/
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
    void helper(TreeNode* root, unordered_map<int, int> &mp){
        if(root == NULL) return;
        helper(root->left, mp);
        mp[root->val]++;
        helper(root->right, mp);
    }
    vector<int> findMode(TreeNode* root) {
        unordered_map<int, int> mp;
        helper(root, mp);
        vector<int> ans;
        int currFreq = 0;
        for(auto it = mp.begin(); it != mp.end(); it++){
            if(it->second > currFreq){
                ans.clear();
                ans.push_back(it->first);
                currFreq = it->second;
            }
            else if(it->second == currFreq){
                ans.push_back(it->first);
            }
        }
        return ans;
    }
};
104. https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/
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
    TreeNode* helper(vector<int> &nums, int l, int r){
        if(l > r) return NULL;
        int mid = (l + r) / 2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = helper(nums, l, mid - 1);
        root->right = helper(nums, mid + 1, r);
        return root;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return helper(nums, 0, nums.size() - 1);
    }
};


105. https://leetcode.com/problems/largest-triangle-area/description/
class Solution {
public:
    double dis(vector<int>& A, vector<int>& B){
        return sqrt(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2));
    }
    double area(vector<int>& A, vector<int>& B, vector<int>& C){
        double a = dis(A, B);
        double b = dis(A, C);
        double c = dis(B, C);
        double p = (a + b + c)/2;
        double res = sqrt(p * (p - a) * (p - b) * (p - c));
        return res;
    }
    double largestTriangleArea(vector<vector<int>>& points) {
        int n = points.size();
        double ans = 0;
        for(int i = 0; i < n - 2; i++){
            for(int j = i + 1; j < n - 1; j++){
                for(int k = j + 1; k < n; k++){
                    ans = max(ans, area(points[i], points[j], points[k]));
                }
            }
        }
        return ans;
    }
};
106. https://leetcode.com/problems/rectangle-overlap/description/
class Solution {
public:
    bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2) {
        int f_x1 = rec1[0], f_y1 = rec1[1];
        int f_x2 = rec1[2], f_y2 = rec1[3];
        
        int s_x1 = rec2[0], s_y1 = rec2[1];
        int s_x2 = rec2[2], s_y2 = rec2[3];
        
        if((s_y1 >= f_y2) || (f_y1 >= s_y2) || (f_x1 >= s_x2) || (f_x2 <= s_x1)){
            return false;
        }
        return true;
    }
};
107. https://leetcode.com/problems/projection-area-of-3d-shapes/description/
108. https://leetcode.com/problems/surface-area-of-3d-shapes/description/
109. https://leetcode.com/problems/matrix-cells-in-distance-order/description/
110. https://leetcode.com/problems/valid-boomerang/description/
111. https://leetcode.com/problems/check-if-it-is-a-straight-line/description/
112. https://leetcode.com/problems/minimum-time-visiting-all-points/description/
113. https://leetcode.com/problems/minimum-cuts-to-divide-a-circle/description/

114. https://bkdnoj.com/problem/CSES47_SUBARRAYSUMSII

#define _GLIBCXX_FILESYSTEM
#include <bits/stdc++.h>
using namespace std;
#define int long long int

int solve(vector<int>& arr, int targetSum) {
    unordered_map<int, int> prefixSumFreq;
    prefixSumFreq[0] = 1;
    int currentSum = 0;
    int count = 0;

    for (int num : arr) {
        currentSum += num;
        if (prefixSumFreq.find(currentSum - targetSum) != prefixSumFreq.end()) {
            count += prefixSumFreq[currentSum - targetSum];
        }
        prefixSumFreq[currentSum]++;
    }

    return count;
}

int32_t main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int n, x;
    cin >> n >> x;
    vector<int> v(n);
    for(int i = 0; i < n; i++) {
        cin >> v[i];
    }
    cout << solve(v, x) << endl;
    return 0;
}

```


Hết.

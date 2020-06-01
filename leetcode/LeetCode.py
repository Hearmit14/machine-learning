# LeetCode

（1）


class Solution:
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i]+nums[j] == target:
                    return i, j


（2）


class Solution:
    def twoSum(self, nums, target):
        d = {}
        for i, item in enumerate(nums):
            tmp = target - item
            for key, value in d.items():
                if value == tmp:
                    return [key, i]
            d[i] = item
            # 只遍历一遍
        return None


(3)


def lengthOfLongestSubstring(s):
    if s == "":
        return 0
    else:
        m = 1
        for i in range(len(s)):
            l = []
            # for j in range(i+1,len(s)):
            for j in range(i, len(s)):
                if s[j] not in l:
                    l.append(s[j])
                else:
                    break
            if len(l) > m:
                m = len(l)
        print(m)

# 实现 strStr() 函数。
# 给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置(从0开始)。如果不存在，则返回 - 1。


def strStr(haystack, needle):
    if needle == '':
        return 0
    i, j = 0, 0
    while i <= len(haystack) - 1:
        print(i, i, j)
        while j <= len(needle) - 1:
            print(i, j)
            if haystack[i] == needle[0oj]:
                i += 1
                j += 1
            else:
                i = i - j + 1
                j = 0
            if j == len(needle):
                return i - j
        return -1


def strStr(haystack, needle):
    if needle == '':
        return 0
    i, j = 0, 0
    while i <= len(haystack) - 1 and j <= len(needle) - 1:
        if haystack[i] == needle[j]:
            i += 1
            j += 1
        else:
            i = i - j + 1
            j = 0
    if j == len(needle):
        return i - j
    else:
        return -1


strStr('aaaaaab', 'ba')

# 爬楼梯
# 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

# 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

# 注意：给定 n 是一个正整数。


class Solution:
    def climbStairs(self, n: int) -> int:
        cache = {}

        def NewclimbStairs(self, n):
            if n in cache:
                return cache[n]
            if n <= 2:
                cache[n] = n
            else:
                cache[n] = self.NewclimbStairs(n-1)+self.NewclimbStairs(n-2)

        return cache[n]


TODO: 这种写法有问题！！！


class Solution:
    def climbStairs(self, n: int) -> int:
        cache = {}

        def NewclimbStairs(n):
            if n in cache:
                return cache[n]
            if n <= 2:
                cache[n] = n
            else:
                cache[n] = NewclimbStairs(n-1)+NewclimbStairs(n-2)
            return cache[n]
        return NewclimbStairs(n)


TODO: 了解啥时候需要self

# 实现 pow(x, n) ，即计算 x 的 n 次幂函数。


class Solution:
    def myPow(self, x: float, n: int) -> float:
        dict = {}

        def newPow(x, n):
            if n in dict:
                return dict[n]
            if n == 0:
                dict[n] = 1
            if n == 1:
                dict[n] = x
            i, j = abs(n)//2, abs(n) % 2
            if n < 0:
                dict[n] = 1/(newPow(x, i)*newPow(x, i)*newPow(x, j))
            else:
                dict[n] = newPow(x, i)*newPow(x, i)*newPow(x, j)
            return dict[n]
        return newPow(x, n)


# 给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]


# 给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
# 先确定起始位置，再循环结束位置
# 有问题，利用了未来的信息
# def longestPalindrome(s):
#     n = len(s)
#     dp = [[0] * n for _ in range(n)]
#     maxlen = 0
#     res = ''
#     for i in range(n):
#         for j in range(i, n):
#             if i == j:
#                 dp[i][j] = 1
#             elif j - i <= 2 and s[i] == s[j]:
#                 dp[i][j] = 1
#             elif s[i] == s[j] and dp[i+1][j-1]:
#                 dp[i][j] = 1
#             if dp[i][j] and j - i > maxlen:
#                 maxlen = j - i
#                 res = s[i:j+1]
#     return dp

# 先确定结束位置，再循环起始位置
def longestPalindrome(s):
    n = len(s)
    if n <= 1:
        return s
    dp = [[0] * n for _ in range(n)]
    maxlen = 0
    res = ''
    for j in range(n):
        for i in range(j+1):
            if i == j:
                dp[i][j] = 1
            elif j-i <= 2 and s[i] == s[j]:
                dp[i][j] = 1
            elif s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = 1
            if dp[i][j] and j+1-i > maxlen:
                maxlen = j+1-i
                res = s[i:j+1]
    return res


longestPalindrome('acdcad')

# 买卖股票的最佳时机
# 假如计划在第 i 天卖出股票，那么最大利润的差值一定是在[0, i-1] 之间选最低点买入；所以遍历数组，依次求每个卖出时机的的最大差值，再从中取最大值。


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        maxProfit = 0
        minprice = prices[0]
        for i in prices:
            minprice = min(minprice, i)
            maxProfit = max(maxProfit, i-minprice)
        return maxProfit

# 计数质数
# 统计所有小于非负整数 n 的质数的数量。


class Solution:
    def countPrimes(self, n: int) -> int:
        if n <= 1:
            return 0
        if n <= 4:
            return n-2
        l = [i for i in range(2, n)]
        j = 2
        while j < n:
            for i in range(2, n//j+1):
                if j*i in l:
                    l.remove(j*i)
            j += 1
        return len(l)

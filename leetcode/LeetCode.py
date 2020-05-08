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

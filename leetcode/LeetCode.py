LeetCode
1：
l = [2, 11, 15]
for i in range(len(l)-1):
    print i

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


3:
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

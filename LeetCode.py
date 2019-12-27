LeetCode
1：
l=[2,11,15]
for i in range(len(l)-1):
    print i

（1）
class Solution:
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i]+nums[j]==target:
                    return i,j
（2）
class Solution:
    def twoSum(self, nums, target):
        d = {}
        for i, item in enumerate(nums):
            tmp  = target - item         
            for key, value in d.items():
                if value == tmp:
                    return [key, i]
            d[i] = item
            # 只遍历一遍
        return None


3:
    def lengthOfLongestSubstring(s):
        if s=="":return 0
        else:
            m=1
            for i in range(len(s)):
                l=[]
                # for j in range(i+1,len(s)):
                for j in range(i,len(s)):
                    if s[j] not in l:
                        l.append(s[j])
                    else:
                        break
                if len(l)>m:
                    m=len(l)
            print(m)


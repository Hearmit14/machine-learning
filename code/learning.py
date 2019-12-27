#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:31:37 2017

@author: hejinyang
"""

#使用list和tuple
list(range(5))

L = ['Bart', 'Lisa', 'Adam']
len(L)

L[0]
L[-1]

L.append('Hero')
L.pop()
L[1]='Lily'

for x in L:
   print('Hello,',x)
   
t = (1, 2)
t = (1,)



#条件判断
bmi=27

bmi1=input('bmi:')
bmi=int(bmi)
if bmi<18.5:
    print('过轻')
elif bmi<25:
    print('正常')
elif bmi<28:
    print('过重')
elif bmi<32:
    print('肥胖')
else:
    print('严重肥胖')


#循环
#for遍历循环
names = ['Michael', 'Bob', 'Tracy']
for name in names:
    print(name)

sum=0
l=list(range(101))
for x in l:
    sum=sum+x
print(sum)

#while条件循环
sum=0
n=0
while(n<=100):
    sum=sum+n
    n=n+1
print(sum,n)



#函数
#定义函数
def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x

my_abs(-20)


#返回多个值
#函数可以返回多个值，但其实这只是一种假象，原来返回值是一个tuple！
import math
def quadratic(a, b, c):
    if b**2-4*a*c>=0:
        x1=(-b+math.sqrt(b**2-4*a*c))/2*a
        x2=(-b-math.sqrt(b**2-4*a*c))/2*a
        return x1,x2
    else:
        pass
    print('no answer')
 
#默认参数
def power(x,n=2):
    u=1
    while n>0:
        n=n-1
        u=x*u
    return u
	

#可变参数
def calc(number):
    sum=0
    for x in number:
    	sum+=x**2
    return sum

calc([1,2,3])

def calc(*number):
    sum=0
    for x in number:
    	sum+=x**2
    return sum
    
calc(1,2,3)
nums=[1,2,3]
calc(nums[0],nums[1],nums[2])
calc(*nums)


#递归函数
def fact(n):
	if n==1:
		return 1
	else:
		return n*fact(n-1)

def fact2(n):
	s=1
	i=1
	while i<=n:
		s=s*i
		i=i+1
	return s

#汉诺塔
def move(n,a,b,c):
	if n==1:
		print(a,'-->',c)
	else:
	    move(n-1,a,c,b)
	    move(1,a,b,c)
	    move(n-1,b,a,c)
	
move(3,'a','b','c')

#构造列表
#while需要对n进行初始化
l=[]
n=1
while n<=30:
    l.append(n)
    n+=2


#切片
L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']
[L[0],L[1],L[2]]


#for不需要对i进行初始化
l=[]
n=3
for i in range(n):
    l.append(L[i])
l

l=[]
n=3
while n>0:
    l.append(L[3-n])
    n=n-1
l


L[0:3]
L[:3]
L[-2:]
L[-2:-1]
L[-3:-1]
L[-1:]

l=list(range(100))
l[::3]

#Python没有针对字符串的截取函数，只需要切片一个操作就可以完成，非常简单。
a='hejinyang'
a[0:6]

#迭代
for x, y in [(1, 1), (2, 4), (3, 9)]:
	print(x, y)
	

a = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
b = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
c=[]
for i in range(60):
    c.append(a[i%10]+b[i%12])
    
c = [a[i%len(a)]+b[i%len(b)]for i in range(61)]

#列表生成式
l=[]
for x in range(10):
	l.append(x*x)
	
l=[x*x for x in range(10)]
l=[x*x for x in range(10) if x%2==0]
[m + n for m in 'ABC' for n in 'XYZ']#注意次序

L = ['Hello', 'World', 18, 'Apple', None]

#[s.lower() for s in L if isinstance(s,str) else s]#错误
[s.lower() if isinstance(s,str) else s for s in L]#正确


#生成器generator
L = [x * x for x in range(10)]
G=(x * x for x in range(10))#generator
next(G)

for n in G:
	print(n)

#类似sas的retain
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        t=(b,a+b)
        a=t[0]
        b=t[1]
        n = n + 1
    return 'done'


def fibb(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

#有问题    
def fib1(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        b=a+b
        a=b
        n = n + 1
    return 'done'
    
#杨辉三角
def triangle1(n):
	l=[]
	for i in range(n):
		l.append(1)
	return l
	
#有问题，列表赋值为指针
def triangle2(n):
	l=[]
	g=[]
	for i in range(n):
		for k in range(1,i):
			l[k]=g[k]+g[k-1]
		l.append(1)
		g=l
	return g 
	
#修改赋值方式
def triangle3(n):
	l=[]
	g=[]
	for i in range(n):
		for k in range(1,i):
			l[k]=g[k]+g[k-1]
		l.append(1)
		g=l[:]
	return l 


#高阶函数
def add(x, y, f):
	return f(x) + f(y)



#map reduce
#map
def f(x):
    return x*x


f(4)
r=map(f,[1,2,3,4])
list(r)

r=map(str,[1,2,4,6])
list(r)

#reduce
#reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
def f(x,y):
    return 10*x+y

from functools import reduce
reduce(f,[1,3,5,6,8])




r=['adam', 'LISA', 'barT']
r[0][:1]
r[0][1:]
len(r)
#map reduce 练习
k='LISA'
k.upper()

#不用map
def normalize(name):
    i=0
    l=[]
    while i<len(name):
        l.append(name[i][:1].upper()+name[i][1:].lower())
        i=i+1
    return l
    
normalize(r)

#用map
def normalize(name):
    return (name[:1].upper()+name[1:].lower())

list(map(normalize,r))



#累积
from functools import reduce

r=[3, 5, 7, 9]
def prod(x,y):
    return x*y

reduce(prod,r)
reduce(lambda x,y:x*y,r)



#字符转浮点
def char2num(s):
    return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,'.':'.'}[s]

list(map(char2num,'123.456'))
    
#有问题
#def str2float(x,y):
#    if y='.':
#        return x
#    elif x='.':
#        return y/10
#    else:
#        return 10*x+y
    
#filter   
def not_empty(s):
    return s and s.strip()
    
list(filter(not_empty, ['A', '', 'B', None, 'C', '  ']))

'a' and 'a  '
' a ' and 'a'
' ' and ''
'' and ' '
10 and 20 
10 or 20


#构造素数序列
def _odd_iter():
    n=1
    while True:
        n=n+2
        yield n


s=_odd_iter()

next(s)

for n in _odd_iter():
    if n<100:
        print(n)
    else:
        break


def _not_divisible(n):
    return lambda x: x % n > 0

def primes():
    yield 2
    it = _odd_iter() # 初始序列
    while True:
        n = next(it) # 返回序列的第一个数
        yield n
        it = filter(_not_divisible(n), it) # 构造新序列    
        
        
for n in primes():
    if n < 1000:
        print(n)
    else:
        break
 
n=3056
str(n)[0]
str(n)[3]
str(n)[4]

str(n)[::-1]

len(str(n))

list(range(10))


#构造回数序列
#有问题，回来再看
def _num_iter():
    n=1
    while True:
        n=n+1
        yield n


def _is_cycle(n):
    k=len(n)
    if k==1:
        return True
    else:
        for i in range(k/2-1):
            if str(n)[i]==str(n)[k-1-i]:
                return True
            return True


def is_palindrome(n):
    yield 1
    it=_num_iter()
    while True:
        n=next(it)
        yield n
        it=filter(_is_cycle(n), it)
    
for n in is_palindrome():
    if n < 100:
        print(n)
    else:
        break
    
    
#排序、
#冒泡排序
a=1
b=2
a,b=b,a


def bubble(bubbleList):
    t=len(bubbleList)
    while t>0:
        for i in range(t-1):
            if bubbleList[i]>bubbleList[i+1]:
                s=(bubbleList[i+1],bubbleList[i])
                bubbleList[i]=s[0]
                bubbleList[i+1]=s[1]
        t-=1    
    
      
ss= [3,4,1,2,5,8,0]  
bubble(ss)  

#快速排序   
#再看

    
sorted([36, 5, -12, 9, -21])    
sorted([36, 5, -12, 9, -21], key=abs)    
sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower)    
sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True)  

L= [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]

#高阶函数，对每一个元素起作用！
def by_name(t):
    return t[0]

#错误list(map(by_name(,L))
list(map(by_name,L))
    
sorted(L,key=by_name)
    


#闭包，返回函数
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum

lazy_sum(1,2,3,4,5)

f=lazy_sum(1,2,3,4,5)
f
f()

#匿名函数
def is_odd(n):
    return n % 2 == 1

L = list(filter(is_odd, range(1, 20)))
L

L = list(filter(lambda x:x%2==1, range(1, 20)))










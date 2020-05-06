# -*- codding:utf-8 -*-
# 单链表


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Linked_List:
    def __init__(self):
        self.head = None

    def initlist(self, data_list):  # 链表初始化函数
        self.head = Node(data_list[0])  # 创建头结点
        temp = self.head
        for i in data_list[1:]:  # 逐个为 data 内的数据创建结点, 建立链表
            node = Node(i)
            temp.next = node
            temp = temp.next

    def is_empty(self):  # 判断链表是否为空
        if self.head.next == None:
            print("Linked_list is empty")
            return True
        else:
            return False

    def get_length(self):  # 获取链表的长度
        temp = self.head  # 临时变量指向队列头部
        length = 0  # 计算链表的长度变量
        while temp != None:
            length = length+1
            temp = temp.next
        return length  # 返回链表的长度

    def insert(self, key, value):  # 链表插入数据函数
        if key < 0 or key > self.get_length()-1:
            print("insert error")
        temp = self.head
        i = 0
        while i <= key:  # 遍历找到索引值为 key 的结点后, 在其后面插入结点
            pre = temp
            temp = temp.next
            i = i+1
        node = Node(value)
        pre.next = node
        node.next = temp

    def print_list(self):  # 遍历链表，并将元素依次打印出来
        print("linked_list:")
        temp = self.head
        new_list = []
        while temp is not None:
            new_list.append(temp.data)
            temp = temp.next
        print(new_list)

    def remove(self, key):  # 链表删除数据函数
        if key < 0 or key > self.get_length()-1:
            print("insert error")
        i = 0
        temp = self.head
        while temp != None:  # 遍历找到索引值为 key 的结点
            pre = temp
            temp = temp.next
            i = i+1
            if i == key:
                pre.next = temp.next
                temp = None
                return True
        pre.next = None

    def reverse1(self):  # 迭代将链表反转
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def reverse2(self, head):  # 递归将链表反转
        if head is None or head.next is None:
            return head
        node = self.reverse2(head.next)
        head.next.next = head
        head.next = None
        return node


#   两两交换链表中的节点
# 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
# 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        else:
            p, q = head, head.next
            p.next = self.swapPairs(q.next)
            q.next = p
        return q

#   反转链表
# 反转一个单链表。


class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        p = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return p

# 上述两段代码的节点不同！！！

# 合并两个有序链表
# 将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。


class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        if l1.val > l2.val:
            l1, l2 = l2, l1
        first = l1
        while l2:
            if l1.next is None:
                l1.next = l2
                break
            elif l1.next.val > l2.val:
                k = l1.next
                l1.next = l2
                l1, l2 = l2, k
            else:
                l1 = l1.next
        return first

# 双链表


class Node(object):
    # 双向链表节点
    def __init__(self, item):
        self.item = item
        self.next = None
        self.prev = None


class DLinkList(object):
    # 双向链表
    def __init__(self):
        self._head = None

    def is_empty(self):
        # 判断链表是否为空
        return self._head == None

    def get_length(self):
        # 返回链表的长度
        cur = self._head
        count = 0
        while cur != None:
            count = count+1
            cur = cur.next
        return count

    def travel(self):
        # 遍历链表
        cur = self._head
        while cur != None:
            print(cur.item)
            cur = cur.next
        print("")

    def add(self, item):
        # 头部插入元素
        node = Node(item)
        if self.is_empty():
            # 如果是空链表，将_head指向node
            self._head = node
        else:
            # 将node的next指向_head的头节点
            node.next = self._head
            # 将_head的头节点的prev指向node
            self._head.prev = node
            # 将_head 指向node
            self._head = node

    def append(self, item):
        # 尾部插入元素
        node = Node(item)
        if self.is_empty():
            # 如果是空链表，将_head指向node
            self._head = node
        else:
            # 移动到链表尾部
            cur = self._head
            while cur.next != None:
                cur = cur.next
            # 将尾节点cur的next指向node
            cur.next = node
            # 将node的prev指向cur
            node.prev = cur

    def search(self, item):
        # 查找元素是否存在
        cur = self._head
        while cur != None:
            if cur.item == item:
                return True
            cur = cur.next
        return False

    def insert(self, pos, item):
        # 在指定位置添加节点
        if pos <= 0:
            self.add(item)
        elif pos > (self.length()-1):
            self.append(item)
        else:
            node = Node(item)
            cur = self._head
            count = 0
            # 移动到指定位置的前一个位置
            while count < (pos-1):
                count += 1
                cur = cur.next
            # 将node的prev指向cur
            node.prev = cur
            # 将node的next指向cur的下一个节点
            node.next = cur.next
            # 将cur的下一个节点的prev指向node
            cur.next.prev = node
            # 将cur的next指向node
            cur.next = node

    def remove(self, item):
        # 删除元素
        if self.is_empty():
            return
        else:
            cur = self._head
            if cur.item == item:
                # 如果首节点的元素即是要删除的元素
                if cur.next == None:
                    # 如果链表只有这一个节点
                    self._head = None
                else:
                    # 将第二个节点的prev设置为None
                    cur.next.prev = None
                    # 将_head指向第二个节点
                    self._head = cur.next
                return
            while cur != None:
                if cur.item == item:
                    # 将cur的前一个节点的next指向cur的后一个节点
                    cur.prev.next = cur.next
                    # 将cur的后一个节点的prev指向cur的前一个节点
                    cur.next.prev = cur.prev
                    break
                cur = cur.next

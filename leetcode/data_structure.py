# -*- codding:utf-8 -*-
# 栈


class Stack(object):
    def __init__(self, limit=10):
        self.stack = []  # 存放元素
        self.limit = limit  # 栈容量极限

    def push(self, data):  # 判断栈是否溢出
        if len(self.stack) >= self.limit:
            print('StackOverflowError')
            return
        self.stack.append(data)

    def pop(self):
        if self.stack:
            return self.stack.pop()
        else:
            raise IndexError('pop from an empty stack')  # 空栈不能被弹出

    def peek(self):  # 查看堆栈的最上面的元素
        if self.stack:
            return self.stack[-1]

    def is_empty(self):  # 判断栈是否为空
        return not bool(self.stack)

    def size(self):  # 返回栈的大小
        return len(self.stack)

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


# 二叉树
class Node(object):
    def __init__(self, item):
        self.item = item  # 表示对应的元素
        self.left = None  # 表示左节点
        self.right = None  # 表示右节点

    def __str__(self):
        return str(self.item)  # print 一个 Node 类时会打印 __str__ 的返回值


class Tree(object):
    def __init__(self):
        self.root = Node('root')  # 根节点定义为 root 永不删除，作为哨兵使用。

    def add(self, item):
        node = Node(item)
        if self.root is None:  # 如果二叉树为空，那么生成的二叉树最终为新插入树的点
            self.root = node
        else:
            q = [self.root]  # 将q列表，添加二叉树的根节点
            while True:
                pop_node = q.pop(0)
                if pop_node.left is None:  # 左子树为空则将点添加到左子树
                    pop_node.left = node
                    return
                elif pop_node.right is None:  # 右子树为空则将点添加到右子树
                    pop_node.right = node
                    return
                else:
                    q.append(pop_node.left)
                    q.append(pop_node.right)

    def get_parent(self, item):
        if self.root.item == item:
            return None  # 根节点没有父节点
        tmp = [self.root]  # 将tmp列表，添加二叉树的根节点
        while tmp:
            pop_node = tmp.pop(0)
            if pop_node.left and pop_node.left.item == item:  # 某点的左子树为寻找的点
                return pop_node  # 返回某点，即为寻找点的父节点
            if pop_node.right and pop_node.right.item == item:  # 某点的右子树为寻找的点
                return pop_node  # 返回某点，即为寻找点的父节点
            if pop_node.left is not None:  # 添加tmp 元素
                tmp.append(pop_node.left)
            if pop_node.right is not None:
                tmp.append(pop_node.right)
        return None

    def delete(self, item):
        if self.root is None:  # 如果根为空，就什么也不做
            return False

        parent = self.get_parent(item)
        if parent:
            del_node = parent.left if parent.left.item == item else parent.right  # 待删除节点
            if del_node.left is None:
                if parent.left.item == item:
                    parent.left = del_node.right
                else:
                    parent.right = del_node.right
                del del_node
                return True
            elif del_node.right is None:
                if parent.left.item == item:
                    parent.left = del_node.left
                else:
                    parent.right = del_node.left
                del del_node
                return True
            else:  # 左右子树都不为空
                tmp_pre = del_node
                tmp_next = del_node.right
                if tmp_next.left is None:
                    # 替代
                    tmp_pre.right = tmp_next.right
                    tmp_next.left = del_node.left
                    tmp_next.right = del_node.right

                else:
                    while tmp_next.left:  # 让tmp指向右子树的最后一个叶子
                        tmp_pre = tmp_next
                        tmp_next = tmp_next.left
                    # 替代
                    tmp_pre.left = tmp_next.right
                    tmp_next.left = del_node.left
                    tmp_next.right = del_node.right
                if parent.left.item == item:
                    parent.left = tmp_next
                else:
                    parent.right = tmp_next
                del del_node
                return True
        else:
            return False

# 二叉树的实现


class Node(object):
    """节点"""

    def __init__(self, item):
        self.elem = item
        self.lchild = None
        self.rchild = None


class Tree(object):
    """二叉树"""

    def __init__(self):
        self.root = None

    def add(self, item):
        node = Node(item)

        if self.root is None:
            self.root = node
            return
        queue = [self.root]

        while queue:
            cur_node = queue.pop(0)
            if cur_node.lchild is None:
                cur_node.lchild = node
                return
            else:
                queue.append(cur_node.lchild)
            if cur_node.rchild is None:
                cur_node.rchild = node
                return
            else:
                queue.append(cur_node.lchild)
                queue.append(cur_node.rchild)

    def bread_travel(self):
        """广度遍历"""
        if self.root is None:
            return
        queue = [self.root]

        while queue:
            cur_node = queue.pop(0)
            print(cur_node.elem, end=" ")
            if cur_node.lchild is not None:
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)

    def pre_travel(self, node):
        """前序遍历"""
        if node is None:
            return
        print(node.elem, end=" ")
        self.pre_travel(node.lchild)
        self.pre_travel(node.rchild)

    def mid_travel(self, node):
        """中序遍历"""
        if node is None:
            return
        self.mid_travel(node.lchild)
        print(node.elem, end=" ")
        self.mid_travel(node.rchild)

    def post_travel(self, node):
        """后序遍历"""
        if node is None:
            return
        self.post_travel(node.lchild)
        self.post_travel(node.rchild)
        print(node.elem, end=" ")


if __name__ == "__main__":
    tree = Tree()
    tree.add(0)
    tree.add(1)
    tree.add(2)
    tree.add(3)
    tree.add(4)
    tree.add(5)
    tree.add(6)
    tree.add(7)
    tree.add(8)
    tree.add(9)
    tree.bread_travel()
    print(" ")
    tree.pre_travel(tree.root)
    print(" ")
    tree.mid_travel(tree.root)
    print(" ")
    tree.post_travel(tree.root)

# 0 1 2 3 4 5 6 7 8 9    层次遍历
# 0 1 3 7 8 4 9 2 5 6    前序遍历
# 7 3 8 1 9 4 0 5 2 6    中序遍历
# 7 8 3 9 4 1 5 6 2 0    后序遍历


# 字典树
class TrieNode:
    def __init__(self):
        self.nodes = dict()  # 构建字典
        self.is_leaf = False

    def insert(self, word: str):
        curr = self
        for char in word:
            if char not in curr.nodes:
                curr.nodes[char] = TrieNode()
            curr = curr.nodes[char]
        curr.is_leaf = True

    def insert_many(self, words: [str]):
        for word in words:
            self.insert(word)

    def search(self, word: str):
        curr = self
        for char in word:
            if char not in curr.nodes:
                return False
            curr = curr.nodes[char]
        return curr.is_leaf

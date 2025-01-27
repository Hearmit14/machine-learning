`
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


# """广度遍历"""

    def level_queue(self, root):
        """利用队列实现树的层次遍历"""
        if root == None:
            return
        myQueue = []
        node = root
        myQueue.append(node)
        while myQueue:
            node = myQueue.pop(0)
            print(node.item)
            if node.left != None:
                myQueue.append(node.left)
            if node.right != None:
                myQueue.append(node.right)


# """深度遍历"""

    def front_digui(self, root):
        """利用递归实现树的先序遍历"""
        if root == None:
            return
        print(root.item)
        self.front_digui(root.left)
        self.front_digui(root.right)

    def middle_digui(self, root):
        """利用递归实现树的中序遍历"""
        if root == None:
            return
        self.middle_digui(root.left)
        print(root.item)
        self.middle_digui(root.right)

    def later_digui(self, root):
        """利用递归实现树的后序遍历"""
        if root == None:
            return
        self.later_digui(root.left)
        self.later_digui(root.right)
        print(root.item)

    def front_stack(self, root):
        """利用堆栈实现树的先序遍历"""
        if root == None:
            return
        myStack = []
        node = root
        while node or myStack:
            while node:  # 从根节点开始，一直找它的左子树
                print(node.item)
                myStack.append(node)
                node = node.left
            node = myStack.pop()  # while结束表示当前节点node为空，即前一个节点没有左子树了
            node = node.right  # 开始查看它的右子树

    def middle_stack(self, root):
        """利用堆栈实现树的中序遍历"""
        if root == None:
            return
        myStack = []
        node = root
        while node or myStack:
            while node:  # 从根节点开始，一直找它的左子树
                myStack.append(node)
                node = node.left
            node = myStack.pop()  # while结束表示当前节点node为空，即前一个节点没有左子树了
            print(node.item)
            node = node.right  # 开始查看它的右子树

    def later_stack(self, root):
        """利用堆栈实现树的后序遍历"""
        if root == None:
            return
        myStack1 = []
        myStack2 = []
        node = root
        myStack1.append(node)
        while myStack1:  # 这个while循环的功能是找出后序遍历的逆序，存在myStack2里面
            node = myStack1.pop()
            if node.left:
                myStack1.append(node.left)
            if node.right:
                myStack1.append(node.right)
            myStack2.append(node)
        while myStack2:  # 将myStack2中的元素出栈，即为后序遍历次序
            print(myStack2.pop().item)


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
    tree.level_queue(tree.root)
    print(" ")
    tree.front_digui(tree.root)
    print(" ")
    tree.middle_digui(tree.root)
    print(" ")
    tree.later_digui(tree.root)

# 0 1 2 3 4 5 6 7 8 9    层次遍历
# 0 1 3 7 8 4 9 2 5 6    前序遍历
# 7 3 8 1 9 4 0 5 2 6    中序遍历
# 7 8 3 9 4 1 5 6 2 0    后序遍历


# 前序遍历
# 递归
# 直接使用inorderTraversal递归会默认赋为初始值
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        queue = []

        def f(root):
            if root is None:
                return []
            queue.append(root.val)
            if root.left:
                f(root.left)
            if root.right:
                f(root.right)
        f(root)
        return queue

# 迭代


class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        stack = [root, ]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

# 中序遍历
# 递归


class solution():
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        queue = []

        def f(node):
            if node.left:
                f(node.left)
            queque.append(node.val)
            if node.right:
                f(node.right)
        f(root)
        return queue

# 迭代


class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return True
        stack = [root]
        queue = []
        node = stack.pop()
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            queue.append(node.val)
            node = node.right
        return queue


# 后序遍历
# 递归


class solution():
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        queue = []

        def f(node):
            if node.left:
                f(node.left)
            if node.right:
                f(node.right)
            queque.append(node.val)
        f(root)
        return queue

# 迭代


class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return True
        stack = [root]
        queue = []
        while stack:
            node = stack.pop()
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
            queue.append(node.val)
        return queue[::-1]


# 给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树。
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        # trees = []
        def generate(low, high):
            if low > high:
                return [None]
            trees = []  # 注意赋值的位置
            for i in range(low, high + 1):
                # root = TreeNode(i)
                for ltree in generate(low, i-1):
                    for rtree in generate(i+1, high):
                        root = TreeNode(i)  # 注意赋值的位置
                        root.left = ltree
                        root.right = rtree
                        trees.append(root)
            return trees
        return generate(1, n) if n else []


# 给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        res = []
        cur_level = [root, ]
        while cur_level:
            tmp = []
            for _ in range(len(cur_level)):
                i = cur_level.pop(0)
                if i.left:
                    cur_level.append(i.left)
                if i.right:
                    cur_level.append(i.right)
                tmp.append(i.val)
            res.append(tmp)
        return res


# 找出二叉树中某两个节点的第一个共同祖先。不得将其他的节点存储在另外的数据结构中。注意：这不一定是二叉搜索树。
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or p == root or q == root:
            return root
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        if l and r:
            return root
        return l if l else r

# 根据一棵树的前序遍历与中序遍历构造二叉树。
# 1 用preorder的指针，有问题


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def build(l, r):
            if l > r:
                return None
            root = TreeNode(preorder[l])
            k = inorder.index(preorder[l])
            root.left = build(l + 1, k)
            root.right = build(k + 1, r)
            return root
        return build(0, len(preorder) - 1)

# 1.1 用inorder的指针


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def build(l, r):
            if l > r:
                return None
            root = TreeNode(preorder.pop(0))
            k = inorder.index(root.val)
            root.left = build(l, k-1)
            root.right = build(k + 1, r)
            return root
        return build(0, len(preorder) - 1)

# 2


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def build(l1, r1, l2, r2):
            if l1 > r1 or l2 > r2:
                return None
            root = TreeNode(preorder[l1])
            k = inorder.index(preorder[l1])
            root.left = build(l1 + 1, k+l1-l2, l2, k - 1)
            root.right = build(k+l1-l2 + 1, r1, k + 1, r2)
            return root
        return build(0, len(preorder)-1, 0, len(inorder)-1)

# 根据一棵树的中序遍历与后序遍历构造二叉树。


class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        def build(l, r):
            if l > r:
                return None
            root = TreeNode(postorder.pop())
            k = inorder.index(root.val)
            root.right = build(k + 1, r)
            root.left = build(l, k-1)
            return root
        return build(0, len(postorder) - 1)


class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        def build(l1, r1, l2, r2):
            if l1 > r1 or l2 > r2:
                return None
            root = TreeNode(postorder[r2])
            k = inorder.index(postorder[r2])
            root.left = build(l1, k - 1, l2, l2 - l1 + k - 1)
            root.right = build(k + 1, r1, k + r2 - r1, r2 - 1)
            return root
        return build(0, len(inorder)-1, 0, len(postorder)-1)


# 删除二叉搜索树中的节点
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return root
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        elif root.val == key and not root.left and not root.right:
            root = None
        else:
            if not root.left:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)
            else:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)
        return root

    def successor(self, root):
        root = root.right
        while root.left:
            root = root.left
        return root.val

    def predecessor(self, root):
        root = root.left
        while root.right:
            root = root.right
        return root.val

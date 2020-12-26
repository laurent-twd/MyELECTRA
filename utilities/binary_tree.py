from collections import Counter

class BinaryTree:

    def __init__(self, f, token, left_child, right_child):

        self.index = None
        self.token = token
        self.binary_code = []
        self.parent = None
        self.frequency = f
        self.left_child = left_child
        self.right_child = right_child
        self.type = 'node'
        if left_child == None and right_child == None:
            self.type = 'leaf'
        self.n_max = None
        self.path = []

def build_tree(frequency):

    sorted_frequency = dict(Counter(frequency).most_common()[::-1])
    tokens = list(sorted_frequency.keys())
    queue = [BinaryTree(sorted_frequency[token], token, None, None) for token in tokens]
    idx2tree = dict(zip(tokens, queue))

    while len(queue) > 1:
        left_child = queue.pop(0)
        right_child = queue.pop(0)
        new_node = BinaryTree(left_child.frequency + right_child.frequency, None, left_child, right_child)
        left_child.parent = new_node
        right_child.parent = new_node
        queue.append(new_node)

    final_tree = queue[0]

    queue = [final_tree]
    index = 1

    while len(queue) > 0:
        tree = queue.pop(0)
        tree.index = index
        if tree.right_child != None:
            queue.append(tree.right_child)
            tree.right_child.binary_code = tree.binary_code + [1]
            tree.right_child.path = tree.path + [index]
        if tree.left_child != None:
            queue.append(tree.left_child)
            tree.left_child.binary_code = tree.binary_code + [-1]
            tree.left_child.path = tree.path + [index]
        index+=1  
    
    index_nodes = list(map(lambda i: idx2tree[i].path, frequency.keys()))
    binary_codes = list(map(lambda i: idx2tree[i].binary_code, frequency.keys()))

    return index_nodes, binary_codes
"""
Basic operations on trees.
"""

import numpy as np
from collections import defaultdict

import copy

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.children = list()
        self.n_children = 0

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in xrange(self.n_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.n_children>0:
            for i in xrange(self.n_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def add_child(self, child):
        child.parent = self
        self.n_children += 1
        self.children.append(child)

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def tree_to_adj(sent_l, tree, directed=False, self_loop=True):
    ret = np.zeros((sent_l, sent_l), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret

def head_to_tree(head, tokens, l):
    if not isinstance(head, list):
        tokens = tokens[:l].tolist()
        head = head[:l].tolist()
    
    root = None
    nodes_list = [Tree() for _ in head]

    for i in range(len(nodes_list)):
        h = head[i]
        nodes_list[i].idx = i
        nodes_list[i].dist = -1
        if h == 0:
            root = nodes_list[i]
        else:
            try:
                nodes_list[h-1].add_child(nodes_list[i])
            except:
                exit()

    assert root is not None
    return root


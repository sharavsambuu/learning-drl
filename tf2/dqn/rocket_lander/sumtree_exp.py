import random
import numpy

# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay
class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = numpy.zeros( 2*capacity - 1 )
        self.data     = numpy.zeros( capacity, dtype=object )
    def _propagate(self, idx, change):
        parent             = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])
    def total(self):
        return self.tree[0]
    def add(self, p, data):
        idx                   = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write           += 1
        if self.write >= self.capacity:
            self.write = 0
    def update(self, idx, p):
        change         = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx     = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


tree = SumTree(4)

tree.add(10.0, ( 1, 2, 3))
print("tree size", int(tree.write))
tree.add(1.0 , ( 2, 3, 4))
print("tree size", int(tree.write))
tree.add(11.0, ( 5, 6, 7))
print("tree size", int(tree.write))
tree.add(0.0 , ( 8, 9,10))
print("tree size", int(tree.write))
tree.add(2.0 , (11,12,13))
print("tree size", int(tree.write))
tree.add(12.0, (14,15,16))
print("tree size", int(tree.write))



batch_size = 5
batch      = []

segment    = tree.total() / batch_size
for i in range(batch_size):
    a = segment * i
    b = segment * (i + 1)
    s = random.uniform(a, b)
    (idx, p, data) = tree.get(s)
    print(idx, p, data)



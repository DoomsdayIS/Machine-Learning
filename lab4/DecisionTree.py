import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter


def entropy(y):
    _, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def inf_gain(v, left, right):
    r_v = len(v)
    r_left = len(left)
    r_right = len(right)
    ig = entropy(v) - (r_left / r_v * entropy(left) + r_right / r_v * entropy(right))
    return ig


def split(column, y):
    best_ig = -1
    best_t = None

    for value in set(column):
        y_predict = column < value
        ig = inf_gain(y, y[y_predict], y[~y_predict])
        if ig > best_ig:
            best_t, best_ig = value, ig
    return best_t, best_ig


def best_split(X, y):
    best_j, best_t = None, None
    best_ig = -1

    for i, c in enumerate(X.T):
        t, ig = split(c, y)
        if ig > best_ig:
            best_t, best_ig = t, ig
            best_j = i
    return best_t, best_ig, best_j


class Node:
    def __init__(self, number=1, depth=1):
        self.number = number
        self.depth = depth
        self.is_leaf = False
        self.left = None
        self.right = None
        self.ig = 0
        self.t = 0
        self.j = 0
        self.main_class = None

    def return_class(self):
        return self.main_class


class DecisionTree:
    def __init__(self):
        self.min_leaf_size = 1
        self.max_depth = 1000
        self.max_leaf_count = 10000
        self.purity = False
        self.leaf_count = 0
        self.tree_root = None
        self.real_depth = 0

    def _set_criteria(self, criteria):
        if criteria == "max_depth":
            self.max_depth = 5
        elif criteria == "purity":
            self.purity = True
        elif criteria == "min_leaf_size":
            self.min_leaf_size = 5
        elif criteria == "max_leaf_count":
            self.max_leaf_count = 5
        else:
            raise Exception("Incorrect criteria! Model doesn't fit")

    def _stop(self, node, y):
        if self.purity is True and len(set(y)) == 1:
            return True
        elif node.depth == self.max_depth:
            return True
        elif len(y) <= self.min_leaf_size:
            return True
        elif self.leaf_count >= self.max_leaf_count:
            return True
        else:
            return False

    def _split_node(self, node, x, y):
        if self._stop(node, y):
            if len(y) == 0:
                self.leaf_count -= 1
                return None
            if node.depth > self.real_depth:
                self.real_depth = node.depth
            node.is_leaf = True
            b = Counter(y)
            node.main_class = b.most_common(1)[0][0]
            return node
        node.t, node.ig, node.j = best_split(x, y)
        x1, y1 = x[x[:, node.j] < node.t], y[x[:, node.j] < node.t]
        x2, y2 = x[x[:, node.j] >= node.t], y[x[:, node.j] >= node.t]
        if len(y1) == 0 or len(y2) == 0:
            if node.depth > self.real_depth:
                self.real_depth = node.depth
            node.is_leaf = True
            b = Counter(y)
            node.main_class = b.most_common(1)[0][0]
            return node
        node_left = Node(node.number*2, node.depth+1)
        node_right = Node(node.number * 2 + 1, node.depth + 1)
        self.leaf_count += 1
        node.left = self._split_node(node_left, x1, y1)
        node.right = self._split_node(node_right, x2, y2)
        return node

    def fit(self, x, y, criteria):
        try:
            self._set_criteria(criteria)
        except Exception as e:
            print(e)
            return False
        root = Node()
        self.leaf_count = 1
        self.tree_root = self._split_node(root, x, y)
        if self.tree_root is None:
            return False
        else:
            return True

    def _get_prediction(self, row):
        rt = self.tree_root
        while rt.is_leaf is False:
            if row[rt.j] < rt.t:
                rt = rt.left
            elif row[rt.j] >= rt.t:
                rt = rt.right
        return rt.main_class

    def predict(self, x):
        results = np.array([0] * len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results


iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

dt = DecisionTree()
res = dt.fit(X_train, y_train, 'max_depth')
y_pred = dt.predict(X_test)

dt2 = DecisionTree()
res2 = dt2.fit(X_train, y_train, 'purity')
y2_pred = dt2.predict(X_test)

dt3 = DecisionTree()
res3 = dt3.fit(X_train, y_train, 'min_leaf_size')
y3_pred = dt3.predict(X_test)

dt4 = DecisionTree()
res4 = dt4.fit(X_train, y_train, 'max_leaf_count')
y4_pred = dt4.predict(X_test)


print("Accuracy score: max_depth ", accuracy_score(y_test, y_pred))
print("Accuracy score: purity ", accuracy_score(y_test, y2_pred))
print("Accuracy score: min_leaf_size ", accuracy_score(y_test, y3_pred))
print("Accuracy score: max_leaf_count ", accuracy_score(y_test, y3_pred))
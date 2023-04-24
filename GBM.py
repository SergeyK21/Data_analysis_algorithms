import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functools
import math
from matplotlib.colors import ListedColormap


class Node:

    def __init__(self, index, t, true_branch, false_branch):
        self.index = index  # индекс признака, по которому ведется сравнение с порогом в этом узле
        self.t = t  # значение порога
        self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле
        self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле


class Leaf:

    def __init__(self, data, labels, classes_or_values=True):
        self.data = data
        self.labels = labels
        if classes_or_values:
            self.prediction = self.__predict_classes()
        else:
            self.prediction = self.__predict_values()

    def __predict_classes(self):
        # подсчет количества объектов разных классов
        classes = {}  # сформируем словарь "класс: количество объектов"
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1

        # найдем класс, количество объектов которого будет максимальным в этом листе и вернем его
        prediction = max(classes, key=classes.get)
        return prediction

    def __predict_values(self):
        return self.labels.mean()


class Tree:
    def __init__(self, X, Y, min_samples_leaf=1, max_tree_depth=None, criterion_name='gini'):
        self.X = X
        self.Y = Y
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth
        self.criterion_name = criterion_name
        if criterion_name == 'gini':
            self.criterion = self.__gini
        elif criterion_name == 'entropy':
            self.criterion = self.__entropy
        elif criterion_name == 'mse':
            self.criterion = self.__mse_targets
        else:
            self.criterion = self.__mae_targets

        self.root = None

        self.X_test = np.array([])
        self.Y_test = np.array([])
        self.X_train = np.array([])
        self.Y_train = np.array([])

    def __entropy(self, labels):

        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        impurity = 0
        for label in classes:
            p = classes[label] / len(labels)
            if p != 0:
                impurity += p * math.log2(p)
        return -impurity

    def __gini(self, labels):
        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        #  расчет критерия
        impurity = 1
        for label in classes:
            p = classes[label] / len(labels)
            impurity -= p ** 2
        return impurity

    def __mse_targets(self, labels):
        return np.mean((labels - labels.mean()) ** 2)

    def __mae_targets(self, labels):
        return np.mean(np.abs(labels - labels.mean()))

    def __gain(self, left_labels, right_labels, root, criterion):
        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])
        return root - p * criterion(left_labels) - (1 - p) * criterion(right_labels)

    def __split(self, data, labels, column_index, t):

        left = np.where(data[:, column_index] <= t)
        right = np.where(data[:, column_index] > t)

        true_data = data[left]
        false_data = data[right]

        true_labels = labels[left]
        false_labels = labels[right]

        return true_data, false_data, true_labels, false_labels

    def __find_best_split(self, data, labels):
        root = self.criterion(labels)

        best_gain = 0
        best_t = None
        best_index = None

        n_features = data.shape[1]

        for index in range(n_features):
            t_values = np.unique(data[:, index])

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.__split(data, labels, index, t)
                if len(true_data) < self.min_samples_leaf or len(false_data) < self.min_samples_leaf:
                    continue

                current_gain = self.__gain(true_labels, false_labels, root, self.criterion)

                if current_gain > best_gain:
                    best_gain, best_t, best_index = current_gain, t, index

        return best_gain, best_t, best_index

    def __build_tree(self, data, labels, classes_or_values=True, count_tree_depth=0):
        if self.max_tree_depth and count_tree_depth > self.max_tree_depth:
            return Leaf(data, labels, classes_or_values)
        count_tree_depth += 1

        gain, t, index = self.__find_best_split(data, labels)

        if gain == 0:
            return Leaf(data, labels, classes_or_values)

        true_data, false_data, true_labels, false_labels = self.__split(data, labels, index, t)

        true_branch = self.__build_tree(true_data, true_labels, classes_or_values, count_tree_depth)

        false_branch = self.__build_tree(false_data, false_labels, classes_or_values, count_tree_depth)

        return Node(index, t, true_branch, false_branch)

    def __predict_object(self, obj, node):
        if isinstance(node, Leaf):
            answer = node.prediction
            return answer

        if obj[node.index] <= node.t:
            return self.__predict_object(obj, node.true_branch)
        else:
            return self.__predict_object(obj, node.false_branch)

    def set_standard_scale(self):
        """
        Стандартизация столбца данных
        :param index: self.data[:, index]
        :return: None -> меняет значения по адресам слолбца в self.data
        """
        means = np.mean(self.X, axis=0)
        stds = np.std(self.X, axis=0)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.X[i][j] = (self.X[i][j] - means[j]) / stds[j]

    @functools.lru_cache()
    def test_train(self, train_proportion: float = 0.7):
        """
        Перемешивает -> Разделяет на тестовую и тренеровочную выборку
        :param train_proportion: интервалы разбитья от 0 до 1
        :return: self.X_train, self.X_test, self.Y_train, self.Y_test
        """
        np.random.seed(12)
        shuffle_index = np.random.permutation(self.X.shape[0])
        X_shuffled, y_shuffled = self.X[shuffle_index, :], self.Y[shuffle_index]
        # X_shuffled, y_shuffled = self.data, self.y
        train_test_cut = int(self.X.shape[0] * train_proportion)

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            X_shuffled[:train_test_cut], \
                X_shuffled[train_test_cut:], \
                y_shuffled[:train_test_cut], \
                y_shuffled[train_test_cut:]

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def fit(self, data=None, labels=None, min_samples_leaf=None, max_tree_depth=None, criterion_name=None):
        try:
            if data == None:
                data = self.X
        except ValueError:
            self.X = data

        try:
            if labels == None:
                labels = self.Y
        except ValueError:
            self.Y = labels

        if min_samples_leaf:
            self.min_samples_leaf = min_samples_leaf

        if max_tree_depth:
            self.max_tree_depth = max_tree_depth

        if criterion_name == None:
            if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
                classes_or_values = True
            else:
                classes_or_values = False
        elif criterion_name == 'gini':
            self.criterion = self.__gini
            classes_or_values = True
        elif criterion_name == 'entropy':
            self.criterion = self.__entropy
            classes_or_values = True
        elif criterion_name == 'mse':
            self.criterion = self.__mse_targets
            classes_or_values = False
        else:
            self.criterion = self.__mae_targets
            classes_or_values = False

        self.root = self.__build_tree(data, labels, classes_or_values)
        return self.root

    def predict(self, data):
        preds = []
        for obj in data:
            prediction = self.__predict_object(obj, self.root)
            preds.append(prediction)
        return preds

    def __accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        t = correct / float(len(actual)) * 100.0
        return t

    def __accuracy_metric_mse(self, actual, predicted):
        return (np.sum((actual - predicted) ** 2)) / len(actual)

    def __accuracy_metric_mae(self, actual, predicted):
        return np.mean(np.abs(actual - predicted))

    def accuracy_errors(self, train_proportion: float = 0.7):
        if self.root:
            return None
        old_root = self.root
        old_X = self.X
        old_Y = self.Y
        self.test_train(train_proportion)

        self.fit(self.X_test, self.Y_test)
        y_test_pred = self.predict(self.X_test)

        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            error_test = self.__accuracy_metric(self.Y, y_test_pred)
        elif self.criterion_name == 'mse':
            error_test = self.__accuracy_metric_mse(self.Y, y_test_pred)
        else:
            error_test = self.__accuracy_metric_mse(self.Y, y_test_pred)

        self.fit(self.X_train, self.Y_train)
        y_train_pred = self.predict(self.X_train)

        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            error_train = self.__accuracy_metric(self.Y, y_train_pred)
        elif self.criterion_name == 'mse':
            error_train = self.__accuracy_metric_mse(self.Y, y_train_pred)
        else:
            error_train = self.__accuracy_metric_mse(self.Y, y_train_pred)

        self.root = old_root
        self.X = old_X
        self.Y = old_Y

        return error_train, error_test

class ForestTree:
    def __init__(self, X, Y, N=1, len_sample=None, min_samples_leaf=1, max_tree_depth=None, criterion_name='gini'):
        self.X = X
        self.Y = Y
        self.N = N
        if len_sample == None and (criterion == 'gini' or criterion == 'entropy'):
            self.len_sample = int(math.sqrt(self.X.shape[1]))
            if self.len_sample == 0:
                self.len_sample = 1
        elif len_sample == None:
            self.len_sample = self.X.shape[1] // 3
            if self.len_sample == 0:
                self.len_sample = 1
        elif N < len_sample:
            self.len_sample = self.X.shape[1]
        else:
            self.len_sample = len_sample

        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth

        self.criterion_name = criterion_name

        if criterion_name == 'gini':
            self.criterion = self.__gini
        elif criterion_name == 'entropy':
            self.criterion = self.__entropy
        elif criterion_name == 'mse':
            self.criterion = self.__mse_targets
        else:
            self.criterion = self.__mae_targets

        self.forest = None
        self.oob_error = None

        self.X_test = np.array([])
        self.Y_test = np.array([])
        self.X_train = np.array([])
        self.Y_train = np.array([])

    def __entropy(self, labels):
        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        impurity = 0
        for label in classes:
            p = classes[label] / len(labels)
            if p != 0:
                impurity += p * math.log2(p)
        return -impurity

    def __gini(self, labels):
        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        impurity = 1
        for label in classes:
            p = classes[label] / len(labels)
            impurity -= p ** 2
        return impurity

    def __mse_targets(self, labels):
        return np.mean((labels - labels.mean()) ** 2)

    def __mae_targets(self, labels):
        return np.mean(np.abs(labels - labels.mean()))

    def __gain(self, left_labels, right_labels, root, criterion):
        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])
        return root - p * criterion(left_labels) - (1 - p) * criterion(right_labels)

    def __split(self, data, labels, column_index, t):

        left = np.where(data[:, column_index] <= t)
        right = np.where(data[:, column_index] > t)

        true_data = data[left]
        false_data = data[right]

        true_labels = labels[left]
        false_labels = labels[right]

        return true_data, false_data, true_labels, false_labels

    def __get_bootstrap(self, data, labels, N):
        np.random.seed(42)
        n_samples = data.shape[0]  # размер совпадает с исходной выборкой
        bootstrap = []

        for i in range(N):
            sample_index = np.random.randint(0, n_samples, size=n_samples)
            temp = set(sample_index.tolist())
            oob_indexs = []
            for j in range(n_samples):
                flag = True
                for l in temp:
                    if l == j:
                        flag = False
                        break
                if flag:
                    oob_indexs.append(j)
            b_data = data[sample_index]
            b_labels = labels[sample_index]

            bootstrap.append((b_data, b_labels, oob_indexs))

        return bootstrap

    def __get_subsample(self, len_sample):
        # будем сохранять не сами признаки, а их индексы
        sample_indexes = list(range(len_sample))

        len_subsample = int(np.round(np.sqrt(len_sample)))

        subsample = np.random.choice(sample_indexes, size=len_subsample, replace=False)

        return subsample

    def __accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        t = correct / float(len(actual)) * 100.0
        return t

    def __accuracy_metric_mse(self, actual, predicted):
        return (np.sum((actual - predicted) ** 2)) / len(actual)

    def __accuracy_metric_mae(self, actual, predicted):
        return np.mean(np.abs(actual - predicted))

    def __predict_object(self, obj, node):
        if isinstance(node, Leaf):
            answer = node.prediction
            return answer

        if obj[node.index] <= node.t:
            return self.__predict_object(obj, node.true_branch)
        else:
            return self.__predict_object(obj, node.false_branch)

    def __predict(self, data, tree):
        preds = []
        for obj in data:
            prediction = self.__predict_object(obj, tree)
            preds.append(prediction)
        return preds

    def __find_best_split(self, data, labels):
        root = self.criterion(labels)
        best_gain = 0
        best_t = None
        best_index = None

        feature_subsample_indices = self.__get_subsample(self.len_sample)  # выбираем случайные признаки

        for index in feature_subsample_indices:
            t_values = np.unique(data[:, index])
            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.__split(data, labels, index, t)
                if len(true_data) < self.min_samples_leaf or len(false_data) < self.min_samples_leaf:
                    continue

                current_gain = self.__gain(true_labels, false_labels, root, self.criterion)
                if current_gain > best_gain:
                    best_gain, best_t, best_index = current_gain, t, index

        return best_gain, best_t, best_index
    def __build_tree(self, data, labels, classes_or_values, count_tree_depth=0):
        if self.max_tree_depth and count_tree_depth > self.max_tree_depth:
            return Leaf(data, labels, classes_or_values)
        count_tree_depth += 1

        gain, t, index = self.__find_best_split(data, labels)

        if gain == 0:
            return Leaf(data, labels, classes_or_values)

        true_data, false_data, true_labels, false_labels = self.__split(data, labels, index, t)

        true_branch = self.__build_tree(true_data, true_labels, classes_or_values, count_tree_depth)
        false_branch = self.__build_tree(false_data, false_labels, classes_or_values, count_tree_depth)

        return Node(index, t, true_branch, false_branch)

    def fit(self, data=None, labels=None, n_trees=None, len_sample=None, criterion_name=None):
        try:
            if data == None:
                data = self.X
        except ValueError:
            self.X = data

        try:
            if labels == None:
                labels = self.Y
        except ValueError:
            self.Y = labels

        if n_trees == None:
            n_trees = self.N
        else:
            self.N = n_trees

        if criterion_name == None:
            self.criterion = self.__gini
            self.criterion_name = 'gini'
        elif criterion_name == 'gini':
            self.criterion = self.__gini
            self.criterion_name = 'gini'
        elif criterion_name == 'entropy':
            self.criterion = self.__entropy
            self.criterion_name = 'entropy'
        elif criterion_name == 'mse':
            self.criterion = self.__mse_targets
            self.criterion_name = 'mse'
        else:
            self.criterion = self.__mae_targets
            self.criterion_name = 'mae'

        if len_sample != None and self.X.shape[1] < len_sample:
            self.len_sample = self.X.shape[1]
        elif len_sample != None:
            self.len_sample = len_sample

        forest = []

        bootstrap = self.__get_bootstrap(data, labels, n_trees)

        oob_values = []

        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            accuracy_metric = self.__accuracy_metric
        elif self.criterion_name == 'mse':
            accuracy_metric = self.__accuracy_metric_mse
        else:
            accuracy_metric = self.__accuracy_metric_mae
        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            classes_or_values = True
        else:
            classes_or_values = False
        oob_indexs= []
        for b_data, b_labels, oob_index in bootstrap:
            tree = self.__build_tree(b_data, b_labels, classes_or_values)
            forest.append(tree)
            oob_indexs.append(oob_index)


        predicted = []
        actual = []
        for i in range(data.shape[0]):
            temp = []
            for j, indexs in enumerate(oob_indexs):
                for index in indexs:
                    if i == index:
                        temp.append(self.__predict(np.array([data[index, :]]), forest[j])[0])
                        break
            if len(temp) != 0:
                if classes_or_values:
                    predicted.append(max(set(temp), key=temp.count))
                else:
                    predicted.append(np.mean(temp))
                actual.append(labels[i])


        oob_value = accuracy_metric(np.array(actual), np.array(predicted))
        self.forest = forest
        self.oob_error = oob_value

        return forest, oob_value


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    # data, labels = make_classification(n_samples=1000, n_features=2, n_informative=2,
    #                                    n_classes=2, n_redundant=0,
    #                                    n_clusters_per_class=1, random_state=3)
    data, labels = make_classification(n_samples=1000, n_features=2, n_informative=2,
                                       n_classes=2, n_redundant=0,
                                       n_clusters_per_class=1, random_state=23)

    N = 5
    len_sample = 1
    min_samples_leaf = 5

    max_tree_depth = None


    # def __init__(self, X, Y, N=1, len_sample=None, min_samples_leaf=1, max_tree_depth=None, criterion_name='gini'):

    foresttree = ForestTree(X=data, Y=labels, N=N,  len_sample=len_sample,
                            min_samples_leaf=min_samples_leaf, criterion_name='gini')

    # def fit(self, data=None, labels=None, n_trees=None, len_sample=None, criterion_name=None):

    print(foresttree.fit(data=data, labels=labels, n_trees=3, len_sample=2, criterion_name='gini')[1])
    print(foresttree.fit(data=data, labels=labels, n_trees=3, len_sample=2, criterion_name='entropy')[1])

    tree = Tree(data, labels,3)

    print(tree.accuracy_errors())
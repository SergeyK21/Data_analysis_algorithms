import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functools
import math
import seaborn as sns
from matplotlib.colors import ListedColormap


class Node:
    def __init__(self, index, t, true_branch, false_branch):
        self.index = index  # индекс признака, по которому ведется сравнение с порогом в этом узле
        self.t = t  # значение порога
        self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле
        self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле


# И класс терминального узла (листа)
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


class ForestTree:
    def __init__(self, X=None, Y=None, N=1, len_sample=None, min_samples_leaf=1, max_tree_depth=None,
                 criterion_name='gini'):
        self.X = X
        self.Y = Y
        self.N = N
        if (self.X or self.Y) and len_sample == None and (criterion_name == 'gini' or criterion_name == 'entropy'):
            self.len_sample = int(math.sqrt(self.X.shape[1]))
            if self.len_sample == 0:
                self.len_sample = 1
        elif (self.X or self.Y) and len_sample == None:
            self.len_sample = self.X.shape[1] // 3
            if self.len_sample == 0:
                self.len_sample = 1
        elif (self.X or self.Y) and N < len_sample:
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

        self.X_test = None
        self.Y_test = None
        self.X_train = None

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

    def __get_bootstrap(self, data, labels, N, oob):
        np.random.seed(42)
        n_samples = data.shape[0]  # размер совпадает с исходной выборкой
        bootstrap = []

        for i in range(N):
            sample_index = np.random.randint(0, n_samples, size=n_samples)
            temp = set(sample_index.tolist())
            oob_indexs = []
            if oob:
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

    def __accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        t = correct / float(len(actual)) * 100.0
        return t

    def __accuracy_metric_r2(self, actual, predicted):
        return 1 - (np.sum((actual - predicted) ** 2) / actual.shape[0]) \
            / (np.sum((actual - np.mean(actual)) ** 2) / actual.shape[0])

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

        feature_subsample_indices = np.random.choice(list(range(data.shape[1])), size=self.len_sample, replace=False)

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
        if self.max_tree_depth and count_tree_depth >= self.max_tree_depth:
            return Leaf(data, labels, classes_or_values)
        count_tree_depth += 1

        gain, t, index = self.__find_best_split(data, labels)

        if gain == 0:
            return Leaf(data, labels, classes_or_values)

        true_data, false_data, true_labels, false_labels = self.__split(data, labels, index, t)

        true_branch = self.__build_tree(true_data, true_labels, classes_or_values, count_tree_depth)
        false_branch = self.__build_tree(false_data, false_labels, classes_or_values, count_tree_depth)

        return Node(index, t, true_branch, false_branch)

    def fit(self, data=None, labels=None, n_trees=None, len_sample=None, criterion_name=None, oob=False):
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

        bootstrap = self.__get_bootstrap(data, labels, n_trees, oob)

        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            accuracy_metric = self.__accuracy_metric
        else:
            accuracy_metric = self.__accuracy_metric_r2

        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            classes_or_values = True
        else:
            classes_or_values = False
        oob_indexs = []
        for b_data, b_labels, oob_index in bootstrap:
            tree = self.__build_tree(b_data, b_labels, classes_or_values)
            forest.append(tree)
            oob_indexs.append(oob_index)

        if oob:
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
        else:
            self.forest = forest
            return self.forest

    def __get_meshgrid(self, data, step=.05, border=1.2):
        x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
        y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
        return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    def visualize(self, train_data, train_labels, test_data, test_labels):
        plt.figure(figsize=(16, 7))

        colors = ListedColormap(['red', 'blue'])
        light_colors = ListedColormap(['lightcoral', 'lightblue'])

        # график обучающей выборки
        plt.subplot(1, 2, 1)
        xx, yy = self.__get_meshgrid(train_data)
        mesh_predictions = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
        plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=colors)

        train_accuracy = self.__accuracy_metric_classify(train_labels, self.predict(train_data))
        plt.title(f'Train accuracy={train_accuracy:.2f}')

        # график тестовой выборки
        plt.subplot(1, 2, 2)
        plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
        plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=colors)

        test_accuracy = self.__accuracy_metric_classify(test_labels, self.predict(test_data))
        plt.title(f'Test accuracy={test_accuracy:.2f}')

    def __accuracy_metric_classify(self, actual, predicted):
        correct = 0
        for i in range(actual.shape[0]):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def predict(self, data):
        # добавим предсказания всех деревьев в список
        predictions = []
        for tree in self.forest:
            predictions.append(self.__predict(data, tree))
            # print(predictions)

        # сформируем список с предсказаниями для каждого объекта
        predictions_per_object = list(zip(*predictions))
        # print(predictions_per_object)

        # выберем в качестве итогового предсказания для каждого объекта то,
        # за которое проголосовало большинство деревьев
        voted_predictions = []
        for obj in predictions_per_object:
            if self.classes_or_values:
                voted_predictions.append(max(set(obj), key=obj.count))
            else:
                voted_predictions.append(np.mean(np.array(obj)))
        return voted_predictions


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    # data, labels = make_classification(n_samples=1000, n_features=2, n_informative=2,
    #                                    n_classes=2, n_redundant=0,
    #                                    n_clusters_per_class=1, random_state=3)
    data, labels = make_classification(n_samples=1000, n_features=2, n_informative=2,
                                       n_classes=2, n_redundant=0,
                                       n_clusters_per_class=1, random_state=23)



    foresttree = ForestTree()

    # print(foresttree.test_list)
    print(
        foresttree.fit(data=data, labels=labels, n_trees=15,  len_sample=2, criterion_name='entropy', oob=True)[
            1])
    print(
        foresttree.fit(data=data, labels=labels, n_trees=15,  len_sample=2, criterion_name='gini', oob=True)[
            1])
    print(
        foresttree.fit(data=data, labels=labels, n_trees=5, len_sample=2, criterion_name='entropy', oob=True)[
            1])
    print(
        foresttree.fit(data=data, labels=labels, n_trees=5, len_sample=2, criterion_name='gini', oob=True)[
            1])
    # from sklearn.ensemble import RandomForestClassifier
    #
    # rf = RandomForestClassifier(oob_score=True, n_estimators=5)
    # rf.fit(data, labels)
    #
    # rf.oob_score_

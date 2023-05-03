import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import ListedColormap


# Реализуем класс узла

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

    def test_train(self, data, labels, train_proportion: float = 0.7):
        """
        Перемешивает -> Разделяет на тестовую и тренеровочную выборку
        :param train_proportion: интервалы разбитья от 0 до 1
        :return: self.X_train, self.X_test, self.Y_train, self.Y_test
        """

        self.X = data

        self.Y = labels

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

        old_root = self.root
        old_X = self.X
        old_Y = self.Y
        self.test_train(self.X, self.Y, train_proportion=train_proportion)

        self.fit(self.X_test, self.Y_test)
        y_test_pred = self.predict(self.X_test)

        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            error_test = self.__accuracy_metric(self.Y, y_test_pred)
        else:
            error_test = 1 - (np.sum((self.Y_test - y_test_pred) ** 2) / self.Y_test.shape[0]) \
                         / (np.sum((self.Y_test - np.mean(self.Y_test)) ** 2) / self.Y_test.shape[0])

        self.fit(self.X_train, self.Y_train)
        y_train_pred = self.predict(self.X_train)

        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            error_train = self.__accuracy_metric(self.Y, y_train_pred)
        else:
            error_train = 1 - (np.sum((self.Y_train - y_train_pred) ** 2) / self.Y_train.shape[0]) \
                         / (np.sum((self.Y_train - np.mean(self.Y_train)) ** 2) / self.Y_train.shape[0])

        self.root = old_root
        self.X = old_X
        self.Y = old_Y

        return error_train, error_test

    def __get_meshgrid(self, data, step=.05, border=1.2):
        x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
        y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
        return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    def visualize(self, train_data, test_data, train_labels, test_labels, step=.05, border=1.2):
        plt.figure(figsize=(16, 7))

        colors = ListedColormap(['red', 'blue'])
        light_colors = ListedColormap(['lightcoral', 'lightblue'])

        # график обучающей выборки
        plt.subplot(1, 2, 1)
        xx, yy = self.__get_meshgrid(train_data, step, border)
        mesh_predictions = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()], self.root)).reshape(xx.shape)
        plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
        plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=colors)
        train_accuracy = self.accuracy_metric_classify(train_labels, self.predict(train_data, self.root))
        plt.title(f'Train accuracy={train_accuracy:.2f}')

        # график тестовой выборки
        plt.subplot(1, 2, 2)
        xx, yy = self.__get_meshgrid(test_data, step, border)
        mesh_predictions = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()], self.root)).reshape(xx.shape)
        plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
        plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=colors)
        test_accuracy = self.accuracy_metric_classify(test_labels, self.predict(test_data, self.root))
        plt.title(f'Test accuracy={test_accuracy:.2f}')
        plt.show()

    def print_tree(self, node, spacing=""):

        # Если лист, то выводим его прогноз
        if isinstance(node, Leaf):
            print(spacing + "Прогноз:", node.prediction)
            return

        print(spacing + 'Индекс', str(node.index), '<=', str(node.t))

        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

    def accuracy_metric_classify(self, actual, predicted):
        correct = 0
        for i in range(actual.shape[0]):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

if __name__ == '__main__':
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    print(X.shape, y.shape)
    tree = Tree(X, y, min_samples_leaf=3, criterion_name='mse')
    tree.fit()
    print(tree.accuracy_errors())

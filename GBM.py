import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import ListedColormap
from tqdm import tqdm


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

    def predict(self, data, tree=None):
        if tree != None:
            preds = []
            for obj in data:
                prediction = self.__predict_object(obj, tree)
                preds.append(prediction)
            return preds
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

    def fit(self, data=None, labels=None, n_trees=None, len_sample=None, criterion_name=None, oob=False,
            min_samples_leaf=None):
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

        if min_samples_leaf != None:
            self.min_samples_leaf = min_samples_leaf

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
            if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
                voted_predictions.append(max(set(obj), key=obj.count))
            else:
                voted_predictions.append(np.mean(np.array(obj)))
        return voted_predictions


class GBM(Tree):
    def __init__(self, n_trees, eta, max_tree_depth, X=None, Y=None, min_samples_leaf=1, criterion_name='gini'):
        super().__init__(X, Y, min_samples_leaf, max_tree_depth, criterion_name)
        self.n_trees = n_trees
        self.eta = eta
        self.trees_list = None
        self.train_errors = None
        self.test_errors = None

    def gb_predict(self, X, trees_list):
        # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
        # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании
        # прибавляются с шагом eta

        predictions = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            prediction = 0
            for tree in trees_list:
                prediction += self.eta * self.predict([x], tree)[0]
            predictions[i] = prediction

        # predictions = np.array(
        #     [sum([eta * alg.predict([x])[0] for alg in trees_list]) for x in X]
        # )

        return predictions

    def __mean_squared_error(self, real, prediction):
        return (sum((real - prediction) ** 2)) / len(real)

    def __residual(self, y, z):
        return y - z

    def gb_fit(self, X=None, Y=None, train_proportion: float = 0.7, stop_train_test=False):
        if not stop_train_test:
            self.test_train(X, Y, train_proportion=train_proportion)

        # Деревья будем записывать в список
        trees = []

        # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
        train_errors = []
        test_errors = []

        for i in range(self.n_trees):

            # первый алгоритм просто обучаем на выборке и добавляем в список
            if len(trees) == 0:
                # обучаем первое дерево на обучающей выборке
                tree = self.fit(self.X_train, self.Y_train)

                train_errors.append(self.__mean_squared_error(self.Y_train, self.gb_predict(self.X_train, trees)))
                test_errors.append(self.__mean_squared_error(self.Y_test, self.gb_predict(self.X_test, trees)))
            else:
                # Получим ответы на текущей композиции
                target = self.gb_predict(self.X_train, trees)

                # алгоритмы начиная со второго обучаем на сдвиг
                tree = self.fit(self.X_train, self.__residual(self.Y_train, target))

                train_errors.append(self.__mean_squared_error(self.Y_train, self.gb_predict(self.X_train, trees)))
                test_errors.append(self.__mean_squared_error(self.Y_test, self.gb_predict(self.X_test, trees)))

            trees.append(tree)
        self.trees_list = trees
        self.test_errors = test_errors
        self.train_errors = train_errors
        return trees, train_errors, test_errors

    def sgb_fit(self, X=None, Y=None, train_proportion: float = 0.7, stop_train_test=False, sample_coef=0.5):
        if not stop_train_test:
            self.test_train(X, Y, train_proportion=train_proportion)
        n_samples = self.X_train.shape[0]
        # Деревья будем записывать в список
        trees = []

        # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
        train_errors = []
        test_errors = []

        for i in range(self.n_trees):
            indices = np.random.randint(0, n_samples, int(n_samples * sample_coef))
            X_train_sampled, y_train_sampled = self.X_train[indices], self.Y_train[indices]

            # первый алгоритм просто обучаем на выборке и добавляем в список
            if len(trees) == 0:
                # обучаем первое дерево на обучающей выборке
                tree = self.fit(X_train_sampled, y_train_sampled)

                train_errors.append(self.__mean_squared_error(self.Y_train, self.gb_predict(self.X_train, trees)))
                test_errors.append(self.__mean_squared_error(self.Y_test, self.gb_predict(self.X_test, trees)))
            else:
                # Получим ответы на текущей композиции
                target = self.gb_predict(X_train_sampled, trees)

                # алгоритмы начиная со второго обучаем на сдвиг
                tree = self.fit(X_train_sampled, self.__residual(y_train_sampled, target))

                train_errors.append(self.__mean_squared_error(self.Y_train, self.gb_predict(self.X_train, trees)))
                test_errors.append(self.__mean_squared_error(self.Y_test, self.gb_predict(self.X_test, trees)))

            trees.append(tree)
        self.trees_list = trees
        self.test_errors = test_errors
        self.train_errors = train_errors
        return trees, train_errors, test_errors

    def evaluate_alg(self):
        train_prediction = self.gb_predict(self.X_train, self.trees_list)

        print(f'Ошибка алгоритма из {self.n_trees} деревьев глубиной {self.max_tree_depth} \
        с шагом {self.eta} на тренировочной выборке: {self.__mean_squared_error(self.Y_train, train_prediction)}')

        test_prediction = self.gb_predict(self.X_test, self.trees_list)

        print(f'Ошибка алгоритма из {self.n_trees} деревьев глубиной {self.max_tree_depth} \
        с шагом {self.eta} на тестовой выборке: {self.__mean_squared_error(self.Y_test, test_prediction)}')

    def get_error_plot(self):
        plt.xlabel('Iteration number')
        plt.ylabel('MSE')
        plt.xlim(0, self.n_trees)
        plt.plot(list(range(self.n_trees)), self.train_errors, label='train error')
        plt.plot(list(range(self.n_trees)), self.test_errors, label='test error')
        plt.legend(loc='upper right')
        plt.show()

    def plot_different_max_depths(self, max_depths):
        train_errors_depths = []
        test_errors_depths = []

        for max_depth in tqdm(max_depths):
            self.max_tree_depth = max_depth
            _, train_errors, test_errors = self.gb_fit(stop_train_test=True)
            train_errors_depths.append(train_errors[-1])
            test_errors_depths.append(test_errors[-1])

        print(f'Количество деревьев в бустинге {self.n_trees}')
        plt.plot(range(len(max_depths)), train_errors_depths, label='train_error')
        plt.plot(range(len(max_depths)), test_errors_depths, label='test_error')
        plt.xlabel('Глубина дерева')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # from sklearn.datasets import make_classification
    #
    # # data, labels = make_classification(n_samples=1000, n_features=2, n_informative=2,
    # #                                    n_classes=2, n_redundant=0,
    # #                                    n_clusters_per_class=1, random_state=3)
    # data, labels = make_classification(n_samples=1000, n_features=2, n_informative=2,
    #                                    n_classes=2, n_redundant=0,
    #                                    n_clusters_per_class=1, random_state=3)
    #
    # N = 5
    # len_sample = 1
    # min_samples_leaf = 5
    #
    # max_tree_depth = None
    #
    # # def __init__(self, X, Y, N=1, len_sample=None, min_samples_leaf=1, max_tree_depth=None, criterion_name='gini'):
    #
    # foresttree = ForestTree(X=data, Y=labels, N=N, len_sample=len_sample,
    #                         min_samples_leaf=min_samples_leaf, criterion_name='gini')
    #
    # # def fit(self, data=None, labels=None, n_trees=None, len_sample=None, criterion_name=None):
    #
    # print(foresttree.fit(data=data, labels=labels, n_trees=5, len_sample=2, criterion_name='gini', oob=True)[1])
    # print(foresttree.fit(data=data, labels=labels, n_trees=5, len_sample=2, criterion_name='entropy', oob=True)[1])
    #
    # tree = Tree(data, labels, 3)
    #
    # print(tree.accuracy_errors())
    ###############################################################################################################
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    print(X.shape, y.shape)

    temp = GBM(n_trees=50, eta=.1, max_tree_depth=3, min_samples_leaf=5, criterion_name='mse')

    temp.sgb_fit(X=X, Y=y, sample_coef=.1)

    temp.evaluate_alg()

    temp.gb_fit(X=X, Y=y)

    temp.evaluate_alg()

    from sklearn.datasets import make_classification

    data, labels = make_classification(n_samples=1000, n_features=2, n_informative=2,
                                       n_classes=2, n_redundant=0,
                                       n_clusters_per_class=1, random_state=23)

    foresttree = ForestTree()
    #
    print('n_trees = 15, foresttree.fit entropy error =',
          foresttree.fit(data=data, labels=labels, n_trees=15, len_sample=10, criterion_name='entropy',
                         min_samples_leaf=3,
                         oob=True)[1])
    print('n_trees = 15, foresttree.fit gini error =',
          foresttree.fit(data=data, labels=labels, n_trees=15, len_sample=10, criterion_name='gini', min_samples_leaf=3,
                         oob=True)[1])
    print('n_trees = 5, foresttree.fit entropy error =',
          foresttree.fit(data=data, labels=labels, n_trees=5, len_sample=10, criterion_name='entropy',
                         min_samples_leaf=3,
                         oob=True)[1])
    print('n_trees = 5, foresttree.fit gini error =',
          foresttree.fit(data=data, labels=labels, n_trees=5, len_sample=10, criterion_name='gini', min_samples_leaf=3,
                         oob=True)[1])


    tree = Tree(X, y, min_samples_leaf=3, criterion_name='mse')
    tree.fit()
    print(tree.accuracy_errors())

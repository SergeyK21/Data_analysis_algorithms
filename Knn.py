import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import ListedColormap
from tqdm import tqdm


class KNN:
    def __init__(self, k=5, metric=2):
        self.k = k
        self.metric = metric

        self.X = None
        self.Y = None

        self.X_test = None
        self.Y_test = None
        self.X_train = None
        self.Y_train = None

    def __metrics(self, x1, x2):
        return np.sum((x1 - x2) ** self.metric) ** (1 / self.metric)

    def predict(self, X, x_train=None, y_train=None):
        if x_train is None or y_train is None:
            x_train = self.X
            y_train = self.Y
        answers = []
        for x in X:
            test_distances = []

            for i in range(len(x_train)):
                # расчет расстояния от классифицируемого объекта до
                # объекта обучающей выборки
                distance = self.__metrics(x, x_train[i])

                # Записываем в список значение расстояния и ответа на объекте обучающей выборки
                test_distances.append((distance, y_train[i]))

            # создаем словарь со всеми возможными классами
            classes = {class_item: 0 for class_item in set(y_train)}

            # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
            for d in sorted(test_distances)[0:self.k]:
                classes[d[1]] += 1

            # Записываем в список ответов наиболее часто встречающийся класс
            answers.append(sorted(classes, key=classes.get)[-1])

        return answers

    def fit(self, X, Y, train_proportion=None):
        self.X = X
        self.Y = Y
        if train_proportion is None:
            if self.X.shape[0] < self.k:
                self.k = self.X.shape[0]
        else:
            self.test_train(X, Y, train_proportion)
            if self.X_test.shape[0] < self.k:
                self.k = self.X_test.shape[0]
            self.X = self.X_train
            self.Y = self.Y_train
            answers = self.predict(self.X_test)
            return self.__accuracy(answers, self.Y_test)




    def test_train(self, data, labels, train_proportion):
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

    def __accuracy(self, pred, y):
        return (sum(pred == y) / len(y))

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    # Для наглядности возьмем только первые два признака (всего в датасете их 4)
    X = X[:, :2]
    knn= KNN(k=9)

    print(knn.fit(X,y, 0.7))
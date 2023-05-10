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

    def _metrics(self, x1, x2):
        return np.sum((x1 - x2) ** self.metric) ** (1 / self.metric)

    def __w_coeff_1(self, q, d):
        return q ** d  # w(i) = q^i или w(i) = q^d в зависимости от того, что передать в параметре d

    def __w_coeff_2(self, q, d):
        return 1 / (d + q) ** (1 + q)  # w(i) = 1 / (d+a)^b или  w(i) = 1 / i

    def predict(self, X, x_train=None, y_train=None, q=1, wv=1):
        if x_train is None or y_train is None:
            x_train = self.X
            y_train = self.Y
        answers = []
        for x in X:
            test_distances = []

            for i in range(len(x_train)):
                distance = self._metrics(x, x_train[i])

                weight = 1

                if wv == 1:
                    weight = self.__w_coeff_1(q, len(test_distances) + 1)  # w(i) = q^i
                elif wv == 2:
                    weight = self.__w_coeff_1(q, distance)  # w(i) = q^d
                elif wv == 3:
                    weight = self.__w_coeff_2(q, distance)  # w(i) = 1 / (d+a)^b
                elif wv == 4:
                    weight = self.__w_coeff_2(q, len(test_distances) + 1)  # w(i) = 1 / i

                test_distances.append((weight * distance, y_train[i]))

            classes = {class_item: 0 for class_item in set(y_train)}

            for d in sorted(test_distances)[0:self.k]:
                classes[d[1]] += 1

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
            return self.accuracy(answers, self.Y_test)

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

    def accuracy(self, pred, y):
        return (sum(pred == y) / len(y))


class K_means(KNN):
    def __init__(self, k=5, metric=2, centroids=None):
        super().__init__(k, metric)
        self.centroids = centroids

    def kmeans(self, data, max_iterations=1, min_distance=1e-4, centroids=None):

        self.X = data

        clusters = {i: [] for i in range(self.k)}
        if centroids is None:
            centroids = [data[i] for i in range(self.k)]

        for _ in range(max_iterations):
            self.Y = []
            for x in data:
                distances = [self._metrics(x, centroid) for centroid in centroids]
                cluster = distances.index(min(distances))
                clusters[cluster].append(x)
                self.Y.append(cluster)

            old_centroids = centroids.copy()

            for cluster in clusters:
                centroids[cluster] = np.mean(clusters[cluster], axis=0)

            optimal = True
            for centroid in range(len(centroids)):
                if np.linalg.norm(centroids[centroid] - old_centroids[centroid], ord=2) > min_distance:
                    optimal = False
                    break

            if optimal:
                break

        return old_centroids, clusters

    def visualize(self, centroids, clusters):
        colors = ['r', 'g', 'b', 'orange', 'y']

        plt.figure(figsize=(7, 7))

        # нанесем на график центроиды
        for centroid in centroids:
            plt.scatter(centroid[0], centroid[1], marker='x', s=130, c='black')

        # нанесем объекты раскрашенные по классам
        for cluster_item in clusters:
            for x in clusters[cluster_item]:
                plt.scatter(x[0], x[1], color=colors[cluster_item])

        plt.show()

    def kmeans_quality(self, centroids, clusters):
        k = 0
        quality = 0
        for c in centroids:
            for x in clusters[k]:
                quality += self._metrics(x, c) ** 2
            k += 1
        return quality


if __name__ == '__main__':
    from sklearn.datasets import make_blobs, make_moons

    # X, y = make_blobs(n_samples=100, random_state=1)
    ## X, y = make_moons(n_samples=50, noise=0.02, random_state=1)

    # plt.figure(figsize=(7, 7))
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    # centroids = np.array([[-12., -2.5], [0., 5.], [-6, -10]])
    # centroids = None

    # km = K_means(k=3, metric=2)

    # centroids, clusters = km.kmeans(data=X, max_iterations=1)
    # km.visualize(centroids, clusters)
    #
    # centroids, clusters = km.kmeans(data=X, max_iterations=3)
    # km.visualize(centroids, clusters)
    #
    # centroids, clusters = km.kmeans(data=X, max_iterations=5)
    # km.visualize(centroids, clusters)
#______________________________________________________________________________
    # from sklearn.datasets import load_iris
    #
    # km = K_means(k=3, metric=2)
    # X, y = load_iris(return_X_y=True)
    # X = X[:, :2]
    # cmap = ListedColormap(['red', 'green', 'blue'])
    # plt.figure(figsize=(7, 7))
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
    # plt.show()
    #
    # print(km.fit(X, y, .7))
    #
    # print(km.X_test.shape)
    #
    # h = .2
    #
    # # Расчет пределов графика
    # x_min, x_max = km.X_train[:, 0].min() - 1, km.X_train[:, 0].max() + 1
    # y_min, y_max = km.X_train[:, 1].min() - 1, km.X_train[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    #
    # plt.figure(figsize=(28, 28))
    #
    # count = 0
    #
    # Z_tred = np.c_[xx.ravel(), yy.ravel()]
    #
    # for q in [0.25, 0.5, 0.75]:
    #     for weights_version in [1, 2, 3, 4]:
    #         count += 1
    #         predict = km.predict(km.X_test, x_train=None, y_train=None, q=q, wv=weights_version)
    #         accur = km.accuracy(predict, km.Y_test)
    #
    #         Z = km.predict(Z_tred, x_train=None, y_train=None, q=q, wv=weights_version)
    #
    #         Z = np.array(Z).reshape(xx.shape)
    #
    #         plt.subplot(4, 4, count)
    #         plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    #         plt.scatter(km.X_train[:, 0], km.X_train[:, 1], c=km.Y_train, cmap=cmap)
    #         plt.xlim(xx.min(), xx.max())
    #         plt.ylim(yy.min(), yy.max())
    #         plt.title(
    #             f"Трехклассовая kNN классификация при k = {km.k}, w = {q},\nвариант расчета веса {weights_version}, accuracy = {accur:.3f}")
    #
    # count += 1
    #
    # predict = km.predict(km.X_test, x_train=None, y_train=None, q=0, wv=0)
    # accur = km.accuracy(predict, km.Y_test)
    #
    # Z = km.predict(Z_tred, x_train=None, y_train=None, q=0, wv=0)
    # Z = np.array(Z).reshape(xx.shape)
    #
    # plt.subplot(4, 4, count)
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    #
    # plt.scatter(km.X_train[:, 0], km.X_train[:, 1], c=km.Y_train, cmap=cmap)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.title(
    #     f"Трехклассовая kNN классификация при k = {3}, w = {0},\nвариант расчета веса {0}, accuracy = {accur:.3f}")
    #
    # plt.show()
#____________________________________________________________
    from sklearn.datasets import make_blobs
    import random

    X, y = make_blobs(n_samples=500)

    plt.figure(figsize=(7, 7))
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    km = K_means(k=3, metric=2)
    centroids, clusters = km.kmeans(data=X, max_iterations=10, min_distance=1e-4)

    # km.visualize(centroids, clusters)

    kmeans_q = []
    for k in range(1, 11):
        km.k = k
        centroids, clusters = km.kmeans(data=X, max_iterations=10, min_distance=1e-4)
        kmeans_q.append(km.kmeans_quality(centroids, clusters))

    print('среднее квадратичные внутриклассовые расстояния:\n', np.round(kmeans_q))

    k = np.arange(10)
    plt.xlabel('number of clusters')
    plt.xticks(k + 1)
    plt.ylabel('cluster cohesion')
    plt.plot(k + 1, kmeans_q)
    plt.show()

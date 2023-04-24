import numpy as np
import pprint
import matplotlib.pyplot as plt
import functools
from tqdm import tqdm
import seaborn as sns


class LR:
    def __init__(self, data, y, eta=0.01, max_iter: int = 1e4,
                 min_weight_dist=1e-8):
        self.cm = np.zeros((2, 2))
        self.koef_precision = 0.5
        self.data = data
        self.y = y
        self.w = np.zeros(self.data.shape[1])
        # шаг градиентного спуска
        self.eta = eta
        # максимальное число итераций
        self.max_iter = max_iter
        # критерий сходимости (разница весов, при которой алгоритм останавливается)
        self.min_weight_dist = min_weight_dist
        self.errors = np.array([])
        self.w_list = np.array([])
        self.iters = np.array([])
        self.X_test = np.array([])
        self.Y_test = np.array([])
        self.X_train = np.array([])
        self.Y_train = np.array([])

    def show_data(self):
        """

        :return: Выводит данные
        """
        pprint(self.w.tolist())
        pprint(self.data.tolist())
        pprint(self.y.tolist())

    def show_errors_data(self):
        """

        :return: Выводит данные
        """
        for i in range(len(self.iters)):
            if i % 100 == 0:
                print(f'Iter {self.iters[i]}: error - {self.errors[i]}, weights: {self.w_list[i]}')
        print(f'В случае использования градиентного спуска функционал ошибки составляет {round(self.errors[-1], 4)}')

    def min_max_scale(self, index: int):
        """
        Нормализация столбца данных
        :param index: self.data[:, index]
        :return: None -> меняет значения по адресам слолбца в self.data
        """

        return (self.data[:, index] - self.data[:, index].min()) \
            / (self.data[:, index].max() - self.data[:, index].min())

    def standard_scale(self, index):
        """
        Стандартизация столбца данных
        :param index: self.data[:, index]
        :return: None -> меняет значения по адресам слолбца в self.data
        """
        return (self.data[:, index] - self.data[:, index].mean()) / self.data[:, index].std()

    def set_standard_scale(self):
        """
        Стандартизация столбца данных
        :param index: self.data[:, index]
        :return: None -> меняет значения по адресам слолбца в self.data
        """
        means = np.mean(self.data, axis=0)
        stds = np.std(self.data, axis=0)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.data[i][j] = (self.data[i][j] - means[j]) / stds[j]

    @functools.lru_cache()
    def test_train(self, train_proportion: float = 0.7):
        """
        Перемешивает -> Разделяет на тестовую и тренеровочную выборку
        :param train_proportion: интервалы разбитья от 0 до 1
        :return: self.X_train, self.X_test, self.Y_train, self.Y_test
        """
        np.random.seed(12)
        shuffle_index = np.random.permutation(self.data.shape[0])
        X_shuffled, y_shuffled = self.data[shuffle_index, :], self.y[shuffle_index]
        # X_shuffled, y_shuffled = self.data, self.y
        train_test_cut = int(self.data.shape[0] * train_proportion)

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            np.array(X_shuffled[:train_test_cut]), \
                np.array(X_shuffled[train_test_cut:]), \
                np.array(y_shuffled[:train_test_cut]), \
                np.array(y_shuffled[train_test_cut:])

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def log_loss(self, labels: bool = True):
        m = self.data.shape[0]
        # используем функцию сигмоиды, написанную ранее
        A = self.sigmoid(np.dot(self.data, self.w))
        eps = 1e-8
        A = np.clip(A, eps, 1 - eps)
        # labels 0, 1
        if labels:
            loss = -1.0 / m * np.sum(self.y * np.log(A) + (1 - self.y) * np.log(1 - A))

        # labels -1, 1
        else:
            temp_y = np.where(self.y == 1, 1, -1)
            loss = 1.0 / m * np.sum(np.log(1 + np.exp(-temp_y * np.dot(self.data, self.w))))

        grad = 1.0 / m * self.data.T @ (A - self.y)

        return loss, grad

    @functools.lru_cache()
    def gradient_descent(self, labels: bool = True):
        """
        градиентный спуск.

        :param size: int -> колличество векторов для расчета градиента
        :param foo: str -> функция ошибки
        :return: Вектор весов.
        """
        self.w = np.zeros(self.data.shape[1])
        W = self.w
        # список векторов весов после каждой итерации
        w_list = [W.copy()]
        # список значений ошибок после каждой итерации
        errors = []
        # список итераций
        iters = []

        weight_dist = np.inf

        for i in range(0, int(self.max_iter)):
            loss, grad = self.log_loss(labels)

            self.w -= self.eta * grad

            weight_dist = np.linalg.norm(self.w - W, ord=2)
            W = self.w

            errors.append(loss)
            w_list.append(W.tolist())
            iters.append(i)

        self.w_list = np.array(w_list)
        self.errors = errors
        self.iters = iters
        return W, iters, w_list, errors

    def predict(self, X, koef_precision: float = 0.5):
        m = X.shape[0]
        y_predicted = np.zeros(m)
        A = np.squeeze(self.sigmoid(np.dot(X, self.w)))
        for i in range(A.shape[0]):
            if (A[i] > koef_precision):
                y_predicted[i] = 1
        return y_predicted

    def predict_proba(self, X):
        m = X.shape[0]
        y_predict = np.zeros((m, 2))
        A = np.squeeze(self.sigmoid(np.dot(X, self.w)))
        for i in range(A.shape[0]):
            y_predict[i] = [1 - A[i], A[i]]
        return y_predict

    @functools.lru_cache()
    def eta_min_logloss(self):
        iterations = np.logspace(2, 4, 4, dtype=np.int)
        etas = np.linspace(1e-2, 5, 10)
        best_error = np.inf
        best_params = {}
        for iteration in tqdm(iterations):
            for eta in etas:
                W, error = self.gradient_descent(0, eta)
                if error < best_error:
                    best_error = error
                    best_params = {
                        'iteration': iteration,
                        'eta': eta
                    }
        return best_params, best_error

    def accuracy(self, y, y_pred):
        return np.mean(y == y_pred)

    def confusion_matrix(self, y, y_pred, save=True):
        cm = np.zeros((2, 2))
        for i in range(y.shape[0]):
            if y[i] == y_pred[i] == 1:  # TP
                cm[0][0] += 1
            elif y[i] == y_pred[i] == 0:  # TN
                cm[1][1] += 1
            elif y[i] != y_pred[i] and y[i] == 1:  # FN
                cm[1][0] += 1
            elif y[i] != y_pred[i] and y[i] == 0:  # FP
                cm[0][1] += 1
        if save:
            self.cm = cm
        return cm

    def show_confusion_matrix(self):
        sns.heatmap(self.cm, annot=True)
        plt.title('confusion matrix')
        plt.xlabel('prediction')
        plt.ylabel('groud truth')
        plt.show()

    def precision(self):
        TP = self.cm[0][0]
        FP = self.cm[0][1]
        if TP == 0:
            return 0
        return TP / (TP + FP)

    def recall(self):
        TP = self.cm[0][0]
        FN = self.cm[1][0]
        if TP == 0:
            return 0
        return TP / (TP + FN)

    def f_score(self, b: int = 1):
        pr = self.precision()
        rec = self.recall()
        if pr == 0 or rec == 0:
            return 0
        return (1 + b ** 2) * pr * rec / ((pr + rec) * (b ** 2))

    def get_FPR(self, cm):
        TN = cm[1][1]
        FP = cm[0][1]
        return FP / (FP + TN)

    def get_TPR(self, cm):
        TP = cm[0][0]
        FN = cm[1][0]
        return TP / (TP + FN)

    def ROC_curve(self, X, Y, step=.1):
        koefs = np.append(np.arange(0, 1, step), [1], axis=0).tolist()
        x = []
        y = []
        result_list = []
        for k in koefs:
            cm = self.confusion_matrix(Y, self.predict(X, k), save=False)
            x.append(self.get_FPR(cm))
            y.append(self.get_TPR(cm))
            result_list.append((k, self.get_FPR(cm), self.get_TPR(cm)))
        AUC_ROC = np.trapz(y, x=x, dx=step)
        plt.title('ROC curve')
        plt.ylim(0, 1.05)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid()
        plt.legend(' ', title=f'AUC-ROC={AUC_ROC:.3f}', loc='lower right')
        plt.plot(x, y)
        plt.show()
        return AUC_ROC, result_list

    def PR_corve(self, X, Y, step=.1):
        koefs = np.append(np.arange(0, 1, step), [1], axis=0).tolist()
        x = []
        y = []
        result_list = []
        for k in koefs:
            self.confusion_matrix(Y, self.predict(X, k))
            x.append(self.recall())
            y.append(self.precision())
            result_list.append((k, self.recall(), self.precision(), self.f_score()))
        AUC_PR = np.trapz(y, x=x, dx=step)
        plt.title('PR curve')
        plt.ylim(0, 1.05)
        plt.xlabel('recall')
        plt.ylabel('presision')
        plt.grid()
        plt.legend(' ', title=f'AUC-PR={AUC_PR:.3f}', loc='lower right')
        plt.plot(x, y)
        plt.show()
        return AUC_PR, result_list

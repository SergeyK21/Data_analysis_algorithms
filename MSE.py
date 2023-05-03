import numpy as np
import pprint
from scipy.stats import shapiro
from scipy.stats import kstest
import matplotlib.pyplot as plt
import statsmodels.api as sm
import functools
from mpl_toolkits.mplot3d.axes3d import Axes3D


class MSE:
    def __init__(self, data, y, eta=0.01, max_iter: int = 1e4,
                 min_weight_dist=1e-8, reg=1e-8):

        self.data = data
        self.y = y
        self.n_features = data.shape[1]
        # шаг градиентного спуска
        self.eta = eta
        # максимальное число итераций
        self.max_iter = max_iter
        # критерий сходимости (разница весов, при которой алгоритм останавливается)
        self.min_weight_dist = min_weight_dist
        self.w = np.zeros(self.data.shape[1])
        self.reg = reg
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

    def calc_mae(self, w):
        """
        Среднее абсолютное отклонение
        :param w: вектор весов
        :return: float
        """
        return np.mean(np.abs(self.y - self.data.dot(w)))

    def calc_mse(self, w):
        """
        Среднеквадратичное отклонение
        :param w: Вектор весов модели
        :return: float
        """
        return (np.sum((self.y - self.data.dot(w)) ** 2)) / len(self.y)

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

    def predict(self, X: np.array([])) -> np.array([]):
        """
        предсказание модели
        :param X: данн
        :return: np.array([])
        """
        return np.dot(X, self.w)

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
            X_shuffled[:train_test_cut], \
                X_shuffled[train_test_cut:], \
                y_shuffled[:train_test_cut], \
                y_shuffled[train_test_cut:]

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    # стохастический градиентный спуск
    @functools.lru_cache()
    def gradient_descent(self, size: int = 0, eta: float = 0.01, l2: bool = False, l1: bool = False,
                         reg: float = 1e-8, foo: str = 'mse') -> np.array([]):
        """
        градиентный спуск.

        :param size: int -> колличество векторов для расчета градиента
        :param foo: str -> функция ошибки
        :return: Вектор весов.
        """
        self.eta = eta
        self.reg = reg
        if foo == 'mse':
            foo = self.calc_mse
        else:
            foo = self.calc_mae
        W = np.random.randn(self.data.shape[1])
        n = self.data.shape[0]
        # список векторов весов после каждой итерации
        w_list = [W.copy()]
        # список значений ошибок после каждой итерации
        errors = []
        # список итераций
        iters = []

        weight_dist = np.inf

        for i in range(0, int(self.max_iter)):
            if size > 0:
                inds = np.random.randint(n, size=size)
                X_tmp = self.data[inds, :]
                y_tmp = self.y[inds]

            else:
                X_tmp = self.data
                y_tmp = self.y

            y_pred_tmp = np.dot(X_tmp, W)
            dQ = 2 / y_tmp.shape[0] * ((X_tmp.T) @ (y_pred_tmp - y_tmp))
            dReg1 = 0
            dReg2 = 0
            if l2 and l1:
                dReg1 = self.reg * np.sign(W)
                dReg2 = self.reg * W
            elif l2 and l1 == False:
                dReg2 = self.reg * W
            elif l2 == False and l1:
                dReg1 = self.reg * np.sign(W)
            new_w = W - self.eta * (dQ + dReg1 + dReg2)
            weight_dist = np.linalg.norm(new_w - W, ord=2)
            W = new_w

            errors.append(foo(W))
            w_list.append(W.tolist())
            iters.append(i)

            if weight_dist <= self.min_weight_dist:
                break;
        self.w = W
        self.w_list = np.array(w_list)
        self.errors = errors
        self.iters = iters
        return W

    @functools.lru_cache()
    def alpha(self, train_proportion: float = 0.7, n: int = 50, ridge_or_laso: bool = True,
              eta: float = 0.01):
        X_train, X_test, Y_train, Y_test = self.test_train(train_proportion)
        mse_train = []
        mse_test = []

        coeffs = np.zeros((n, X_train.shape[1]))
        alpha_list = np.logspace(-3, 2, n)
        for i, val in enumerate(alpha_list):
            temp = MSE(X_train, Y_train)
            if ridge_or_laso:
                temp.gradient_descent(eta=eta, l2=True, l1=False, reg=val)
            else:
                temp.gradient_descent(eta=eta, l2=False, l1=True, reg=val)
            coeffs[i, :] = temp.w
            coef = coeffs[i]
            y_pred_tr = np.dot(X_train, coef)
            mse_train.append(np.mean((y_pred_tr - Y_train) ** 2))
            y_pred_ts = np.dot(X_test, coef)
            mse_test.append(np.mean((y_pred_ts - Y_test) ** 2))

        for i in range(X_train.shape[1]):
            plt.plot(alpha_list, coeffs[:, i])

        plt.title(
            'Убывание абсолютных значений весов признаков\n при увеличении коэффициента регуляризации alpha')

        plt.xlabel('alpha')
        plt.ylabel('Вес признака')
        plt.show()

        plt.plot(alpha_list, mse_train, label='Тренировочные данные', color='g')
        plt.plot(alpha_list, mse_test, label='Тестовые данные', color='r')
        plt.legend()
        plt.xlabel('alpha')
        plt.ylabel('MSE');
        plt.show()

    # Визуализируем изменение функционала ошибки

    def show_errors(self):
        """
        График ошибок
        :return: None
        """
        plt.plot(range(len(self.errors)), self.errors)
        plt.title('MSE')
        plt.xlabel('Iteration number')
        plt.ylabel('MSE')
        plt.show()

    # Визуализируем изменение весов (красной точкой обозначены истинные веса, сгенерированные вначале)
    def show_weight_ch(self):
        """
        График изменения вектора весов w0 и w1.
        :return: None
        """
        plt.figure(figsize=(13, 6))
        plt.title('Gradient descent')
        plt.xlabel(r'$w_1$')
        plt.ylabel(r'$w_2$')

        plt.scatter(self.w_list[:, 0], self.w_list[:, 1])
        plt.scatter(self.w[0], self.w[1], c='r')
        plt.plot(self.w_list[:, 0], self.w_list[:, 1])

        plt.show()

    def show_data_w0_w1_y(self):
        """
        Визуализация набора данных.
        :return:
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.data[:, 0], self.data[:, 1], self.y)

        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.set_zlabel('Y')
        plt.show()

    def get_coef_cor_Pirson(self, i, j=0, X_and_y=True):
        """
        Коэффициент Пирсона -> (M(xy) - M(x)*M(y)) / (std(x) * std(y))
        :param i:  index -> data[:, i]
        :param j: index -> data[:, j]
        :param X_and_y: True(data[:, i], y[:,:]) False (data[:, i], data[:, j])
        :return: float, str
        """
        if X_and_y:
            x = self.data[:, i]
            y = self.y
        else:
            if i != j:
                x = self.data[:, i]
                y = self.data[:, j]
            else:
                return None, 'ошибка индексов'
        r2 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.std(x) * np.std(y))
        mes = ''
        if r2 > 0:
            if r2 > .5:
                mes = 'Сильная прямая корреляционная связь.'
            elif r2 > .1 and r2 <= .5:
                mes = 'Слабая прямая корреляционная связь.'
            elif r2 >= 0 and r2 <= .1:
                mes = 'Нет корреляционной связи, требует дополнительных исследований.'
        elif r2 < 0:
            if r2 < -.5:
                mes = 'Сильная обратная корреляционная связь.'
            elif r2 >= -.5 and r2 < -.1:
                mes = 'Слабая обратная корреляционная связь.'
            elif r2 >= -.1 and r2 <= 0:
                mes = 'Нет корреляционной связи, требует дополнительных исследований.'

        return r2, mes

    def get_median(self, index):
        """

        :param index: -1 -> y, остальные data
        :return: float
        """
        n = self.y.shape[0]

        X = np.array(np.c_[self.y, self.data])
        X = X[X[:, 0].argsort()]

        if n % 2 == 0:
            return (X[int(n / 2), int(index + 1)] + X[int((n / 2) + 1), int(index + 1)]) / 2
        else:
            return X[int((n + 1) / 2), int(index + 1)]

    def show_fig_QQ(self, index) -> None:
        """

        :param index: -1 -> y, остальные data
        :return: None
        """
        X = np.c_[self.y, self.data]
        X = X[X[:, 0].argsort()]
        fig = sm.qqplot(X[:, index + 1], line='45')
        fig.show()

    def shpw_hist(self, index):
        """

        :param index: -1 -> y, остальные data
        :return: None
        """
        X = np.c_[self.y, self.data]
        X = X[X[:, index + 1].argsort()]
        plt.hist(X[:, index + 1], edgecolor='black', bins=20)
        plt.show()

    def get_shapiro(self, index):
        """

        :param index:  -1 -> y, остальные data
        :return:
        """
        X = np.c_[self.y, self.data]
        X = X[X[:, index + 1].argsort()]

        return shapiro(X[:, index + 1])

    def get_kstest(self, index, txt: str = 'norm'):
        """

        :param index: -1 -> y, остальные data
        :return:
        """
        X = np.c_[self.y, self.data]
        X = X[X[:, index + 1].argsort()]

        return kstest(X[:, index + 1], txt)

    def get_log_data(self, index):
        """

        :param index:
        :return:
        """
        X = np.array(np.c_[self.y, self.data])
        X[:, index + 1] = np.log(X[:, index + 1])
        return X

    def get_sqrt_data(self, index):
        """

        :param index:
        :return:
        """
        X = np.array(np.c_[self.y, self.data])
        X[:, index + 1] = np.sqrt(X[:, index + 1])
        return X

    def get_cbrt_data(self, index):
        """

        :param index:
        :return:
        """
        X = np.array(np.c_[self.y, self.data])
        X[:, index + 1] = np.cbrt(X[:, index + 1])
        return X

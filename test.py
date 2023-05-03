# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/04 13:58
 @Author  : hanzi5
 @Email   : hanzi5@yeah.net
 @File    : Adaboost.py
 @Software: PyCharm
"""
import numpy as np
# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'SimHei'  # Используется для нормального отображения китайского
plt.rcParams['axes.unicode_minus'] = False  # Используется для нормального отображения отрицательных знаков


def splitDataSet(dataSet, i, value, types='lt'):
    """
 # Разделите набор данных, разделите дерево только один раз и верните разделенный результат
         : param dataSet: data
         : param i: характерный индекс
         : значение параметра: порог
         : типы параметров: больше или меньше чем
         : return: результат классификации, обратите внимание, что возвращаемый результат - непосредственно 1, -1 результат классификации
    """
    retArray = np.ones((np.shape(dataSet)[0], 1))  # Тип по умолчанию - 1
    if types == 'lt':  # Используйте (меньше или равно значению), чтобы разделить данные, и, если условие выполнено, измените значение результата на -1
        retArray[dataSet[:, i] <= value] = -1.0
    elif types == 'gt':  # Используйте (больше, чем значение), чтобы разделить данные, и измените значение результата на -1, если условие выполнено
        retArray[dataSet[:, i] > value] = -1.0
    return retArray


def bulidSimpleTree(dataSet, y, D):
    """
 Создать простейшее дерево, разделенное только один раз, эквивалентное пню
         : param dataSet: матрица характеристик данных
         : param y: метка-вектор
         : параметр D: весовой вектор тренировочных данных
         : return: оптимальное дерево решений, взвешенная сумма минимальной частоты ошибок, оптимальный результат прогнозирования
    """
    m, n = dataSet.shape  # Примеры строк и столбцов
    numFeatures = len(dataSet[0]) - 1  # Последний столбец - y, рассчитать количество столбцов элемента x
    numSteps = 10  # Используется для расчета размера шага, numSteps равных частей
    minError = np.inf  # Инициализировать значение потери
    bestTree = {}  # Используйте dict для сохранения оптимальной серии деревьев
    for i in range(numFeatures):  # Обходить все x столбцы объектов
        # i=0
        rangeMin = dataSet[:, i].min()  # Минимальное значение измерения XI
        rangeMax = dataSet[:, i].max()  # Максимальное значение измерения XI
        stepSize = (rangeMax - rangeMin) / numSteps  # Шаг
        for j in range(-1, int(numSteps) + 1):  # Цикл, чтобы найти оптимальную точку разреза
            # j=-1
            for inequal in ['lt', 'gt']:  # Обход (lt меньше или равно) и (gt больше чем)
                # inequal=1
                value = (rangeMin + float(j) * stepSize)  # Вырезать значение
                predictedVals = splitDataSet(dataSet, i, value, inequal)  # Получить данные разделения
                errArr = np.mat(np.ones((m, 1)))
                errArr[
                    predictedVals == y] = 0  # Коэффициент ошибок, соответствующий правильной предсказанной выборке, равен 0, в противном случае он равен 1
                weightedError = D.T * errArr  # «Статистический метод обучения» Li Hang P138, 8.1, вычисление коэффициента ошибок классификации по данным обучения
                if weightedError < minError:  # Запись оптимального классификатора дерева решений для пней
                    minError = weightedError  # Рассчитать взвешенную сумму ошибок
                    bestClasEst = predictedVals.copy()  # Лучший результат прогноза
                    bestTree['column'] = i  # Размер х
                    bestTree['splitValue'] = value  # Вырезать значение
                    bestTree['ineq'] = inequal  # Метод разделения (lt меньше или равно) и (gt больше чем)
    return bestTree, minError, bestClasEst


# Adaboost классификатор на основе однослойного дерева решений
def adaboost(dataSet, maxLoop=100):
    """
 Обучение ADA на основе однослойного дерева решений
         : param dataSet: образец x и y
         : param maxLoop: количество итераций
         : return: серия слабых классификаторов и их весов, результаты выборочной классификации
    """
    adaboostTree = []
    m, n = dataSet.shape  # Примеры строк и столбцов
    y = dataSet[:, -1].reshape((-1, 1))  # Извлечь y, чтобы облегчить расчет
    D = np.array(
        np.ones((m, 1)) / m)  # Инициализировать вес каждого образца, чтобы быть равным, столько данных, сколько д
    aggClassEst = np.mat(np.zeros((m, 1)))  # Расчетное совокупное значение каждой категории точек данных
    for i in range(maxLoop):  # maxLoop hyperparameter, общее количество итераций
        bestTree, minError, bestClasEst = bulidSimpleTree(dataSet, y, D)
        alpha = 0.5 * np.log((1 - minError) / (
                    minError + 0.00001))  # «Метод статистического обучения» Li Hang P139, 8.2, рассчитать коэффициент Gm, добавить десятичную дробь, чтобы избежать деления 0
        bestTree['alpha'] = alpha.getA()[0][0]  # Извлечь значение в матрице и добавить его в bestTree
        adaboostTree.append(bestTree)  # Хранить дерево в списке
        D = D * np.exp(-alpha.getA()[0][
            0] * y * bestClasEst)  # Обновление веса D, «Статистический метод обучения» Li Hang P139, 8,3-8,5, рассчитать коэффициент Гм (х)
        D = D / D.sum()  # Нормализованное значение веса, статистический метод обучения "Li Hang P139, 8,5, рассчитать Zm
        # Рассчитать ошибку всех классификаторов, если она равна 0, прекратить обучение
        aggClassEst += alpha.getA()[0][
                           0] * bestClasEst  # Классификация оценочной совокупной стоимости, adaboost работает линейно, вам нужно каждый раз добавлять результаты прогнозирования дерева
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(y), np.ones((m,
                                                                            1)))  # aggClassEst Символ каждого элемента представляет результат классификации, если он не равен y, это означает ошибку, статистический метод обучения "Li Hang P138, 8.8
        errorRate = aggErrors.sum() / m  # Средняя ошибка классификации
        print("total error: ", errorRate)
        if errorRate == 0.0:  # Если средняя ошибка классификации равна 0, это означает, что данные были правильно классифицированы и выпрыгивают из цикла
            break
    return adaboostTree, aggClassEst


def adaClassify(data, adaboostTree):
    """
 Классифицировать прогнозные данные
         : параметры данных: образцы прогноза х и у
         : param adaboostTree: использовать данные обучения, обученное дерево решений
         : return: предсказать результаты выборочной классификации
    """
    dataMatrix = np.mat(data)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(adaboostTree)):  # Обходить все adaboostTree, накапливать приблизительное значение
        classEst = splitDataSet(dataMatrix, adaboostTree[i]['column'], adaboostTree[i]['splitValue'],
                                adaboostTree[i]['ineq'])
        aggClassEst += adaboostTree[i][
                           'alpha'] * classEst  # Классификация оценочной совокупной стоимости, adaboost работает линейно, вам нужно каждый раз добавлять результаты прогнозирования дерева
    result = np.sign(
        aggClassEst)  # Принимайте только положительные и отрицательные признаки, «Статистический метод обучения» Li Hang P139, 8.8
    return result


def plotData(dataSet):
    """
 Рисование данных
    """
    type1_x1 = []
    type1_x2 = []
    type2_x1 = []
    type2_x2 = []

    # Возьмите два типа значений x1 и x2 для рисования
    type1_x1 = dataSet[dataSet[:, -1] == -1][:, :-1][:, 0].tolist()
    type1_x2 = dataSet[dataSet[:, -1] == -1][:, :-1][:, 1].tolist()
    type2_x1 = dataSet[dataSet[:, -1] == 1][:, :-1][:, 0].tolist()
    type2_x2 = dataSet[dataSet[:, -1] == 1][:, :-1][:, 1].tolist()

    # Точка рисования
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(type1_x1, type1_x2, marker='s', s=90)
    ax.scatter(type2_x1, type2_x2, marker='o', s=50, c='red')
    plt.title('Данные обучения Adab')

if __name__ == '__main__':
    print('\ n1, Adaboost, начало')
    dataSet = np.array([
        [1., 2.1, 1],
        [2., 1.1, 1],
        [1.3, 1., -1],
        [1., 1., -1],
        [2., 1., 1]])
    # classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    # Draw
    print('\ n2, рисование данных Adaboost')
    plotData(dataSet)

    print('\ n3, Рассчитать дерево Adaboost')
    adaboostTree, aggClassEst = adaboost(dataSet)

    # Категоризация данных
    print('\ n4. Для [5,5], [0, 0] баллов используйте Adaboost для классификации:')
    print(adaClassify([[5, 5], [0, 0]], adaboostTree))

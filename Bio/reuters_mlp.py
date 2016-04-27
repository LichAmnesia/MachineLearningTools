# -*- coding: utf-8 -*-
# @Author: Lich_Amnesia
# @Email: alwaysxiaop@gmail.com
# @Date:   2016-04-22 14:20:11
# @Last Modified time: 2016-04-26 23:29:52
# @FileName: reuters_mlp.py

'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''

from __future__ import print_function
import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import os
import AC_out
import multiprocessing

filepath = os.path.dirname(os.path.abspath(__file__))
trainsetpath = os.path.join(filepath, 'Trainset')
predictoutpath = os.path.join(filepath, 'Predictout')


class mlpTrain(object):

    def __init__(self):
        # configure MLP
        self.nb_epoch = 5
        self.batch_size = 32
        self.nb_classes = 2

    # worker function
    def worker(self, begin, end, lock):
        # lock.acquire()
        print("process: ", os.getpid())
        for setId in range(begin, end):
            self.trainProcessing(setId)
        # lock.release()

    def run(self):
        # Multi-process
        record = []
        lock = multiprocessing.Lock()
        for i in range(4):
            begin = i * 5 + 1
            end = (i + 1) * 5 + 1
            process = multiprocessing.Process(
                target=self.worker, args=(begin, end, lock))
            print("start process")
            process.start()
            record.append(process)

        for process in record:
            process.join()
        
        AC_out.main()
        return

        # ansero = 1
        # ansnum = 1
        # ansn_estimators = 0
        # for n_estimators in xrange(500, 510, 10):
        #     sumero = 0
        #     sumall = 0
        #     for setId in range(1, 2):
        #         predictX, predictY, trainX, trainY = self.getDataset(setId, 20)
        #         ero, allnum = getPredictOutputRF(
        #             setId, predictX, predictY, trainX, trainY, n_estimators)
        #         sumero += ero
        #         sumall += allnum
        #     if float(sumero) / sumall < float(ansero) / ansnum:
        #         ansero = sumero
        #         ansnum = sumall
        #         ansn_estimators = n_estimators
        #     print n_estimators, sumero, sumall, 1 - float(sumero) / sumall
        # print ansn_estimators, ansero, ansnum, 1 - float(ansero) / ansnum

    def trainProcessing(self, setId):
        # 获得初始化的Train和Test的数据集
        X_test, y_test, X_train, y_train = self.getDataset(
            setId, 20)
        # 获取分类数目
        # = np.max(y_train) + 1
        print("Convert class vector to binary class matrix "
              "(for use with categorical_crossentropy)")
        # 转换Y的形式，方便后面的output layer， 并且获得输入的shape作为input layer初始输入
        Y_train = np_utils.to_categorical(y_train, self.nb_classes)
        Y_test = np_utils.to_categorical(y_test, self.nb_classes)
        input_size = X_train.shape[1]
        print('Building model...')
        self.getPredictOutputMLP(
            setId, input_size, X_train, Y_train, X_test, Y_test)

    # 获得MLP的dataset
    def getDataset(self, setId, MaxNum):
        print("Now predict " + str(setId))
        # 形成set的主函数,for MLP
        # 测试集读取
        predictSet = np.loadtxt(os.path.join(
            trainsetpath, 'predict_' + str(setId) + '.rf'))
        predictId = setId
        predictX = predictSet[:, :-1]
        predictY = predictSet[:, -1]

        # 训练集读取
        nextSetId = (setId) % MaxNum + 1
        trainSet = np.loadtxt(os.path.join(
            trainsetpath, 'predict_' + str(nextSetId) + '.rf'))
        trainId = nextSetId
        # print(nextSetId, trainSet.shape)
        for i in range(MaxNum):
            nextSetId = (nextSetId) % MaxNum + 1
            if nextSetId == trainId or nextSetId == predictId:
                continue
            tmpTrainSet = np.loadtxt(os.path.join(
                trainsetpath, 'predict_' + str(nextSetId) + '.rf'))
            trainSet = np.append(trainSet, tmpTrainSet, axis=0)
        trainX = trainSet[:, :-1]
        trainY = trainSet[:, -1]
        return predictX, predictY, trainX, trainY

    # 进行预测
    def getPredictOutputMLP(self, setId, input_size, X_train, Y_train, X_test, Y_test):
        model = Sequential()

        model.add(Dense(550, input_shape=(input_size,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(500, input_shape=(550,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(400, input_shape=(500,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(300, input_shape=(400,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        history = model.fit(X_train, Y_train,
                            nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                            verbose=0, validation_split=0.1)
        score = model.evaluate(X_test, Y_test,
                               batch_size=self.batch_size, verbose=0)
        predict_proba_out = model.predict_proba(X_test,
                                                batch_size=self.batch_size, verbose=1)

        # print('Predict Y:', predict_proba_out)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        outFile = open(os.path.join(
            predictoutpath, 'output_' + str(setId)), 'wb')
        for i in range(predict_proba_out.shape[0]):
            outFile.write(
                str(predict_proba_out[i][1]) + ' ' + str(Y_test[i]) + '\n')
        outFile.close()


if __name__ == '__main__':
    mlp = mlpTrain()
    mlp.run()

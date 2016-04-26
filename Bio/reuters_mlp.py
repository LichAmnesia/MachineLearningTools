# -*- coding: utf-8 -*-
# @Author: Lich_Amnesia
# @Email: alwaysxiaop@gmail.com
# @Date:   2016-04-22 14:20:11
# @Last Modified time: 2016-04-26 16:28:06
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

filepath = os.path.dirname(os.path.abspath(__file__))
trainsetpath = os.path.join(filepath, 'Trainset')
predictoutpath = os.path.join(filepath, 'Predictout')


class mlpTrain(object):

    def __init__(self):
        self.X_train, self.y_train, self.X_test, self.y_test = (
            None, None, None, None)
        # configure MLP 
        self.nb_epoch = 5
        self.batch_size = 32

    def run(self):
        for setId in range(1, 2):
                # 获得初始化的Train和Test的数据集
                self.X_test, self.y_test, self.X_train, self.y_train = self.getDataset(
                    setId, 20)
                # 获取分类数目
                self.nb_classes = np.max(self.y_train) + 1
                print("Convert class vector to binary class matrix "
                    "(for use with categorical_crossentropy)")
                # 转换Y的形式，方便后面的output layer， 并且获得输入的shape作为input layer初始输入
                self.Y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
                self.Y_test = np_utils.to_categorical(self.y_test, self.nb_classes)
                self.input_size = self.X_train.shape[1]
                print('Building model...')

                self.getPredictOutputMLP(setId)
                
                
                ero, allnum = getPredictOutputRF(
                    setId, predictX, predictY, trainX, trainY, n_estimators)
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

    # 获得MLP的dataset
    def getDataset(self, setId, MaxNum):
        print("Now predict " + str(setId))
        # 形成set的主函数,for MLP
        # 测试集读取
        predictSet = np.loadtxt(os.path.join(trainsetpath, 'predict_' + str(setId) + '.rf'))
        predictId = setId
        predictX = predictSet[:, :-1]
        predictY = predictSet[:, -1]

        # 训练集读取
        nextSetId = (setId) % MaxNum + 1
        trainSet = np.loadtxt(os.path.join(trainsetpath, 'predict_' + str(nextSetId) + '.rf'))
        trainId = nextSetId
        # print(nextSetId, trainSet.shape)
        for i in range(MaxNum):
            nextSetId = (nextSetId) % MaxNum + 1
            if nextSetId == trainId or nextSetId == predictId:
                continue
            tmpTrainSet = np.loadtxt(os.path.join(trainsetpath, 'predict_' + str(nextSetId) + '.rf'))
            trainSet = np.append(trainSet, tmpTrainSet, axis=0)
        trainX = trainSet[:, :-1]
        trainY = trainSet[:, -1]
        return predictX, predictY, trainX, trainY

    # 进行预测
    def getPredictOutputMLP(self, setId):
        model = Sequential()
        
        model.add(Dense(512, input_shape=(self.input_size,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(self.X_train, self.Y_train,
                            nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                            verbose=1, validation_split=0.1)
        score = model.evaluate(self.X_test, self.Y_test,
                               batch_size=self.batch_size, verbose=1)
        predic = model.predict_proba(self.X_test,
                                     batch_size=self.batch_size, verbose=1)

        print('Predict Y:', predic)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        outFile = open(os.path.join(predictoutpath, 'output_' + str(setId)),'wb')
        cnt = 0
        for i in range(predic.shape[0]):
            outFile.write(str(predictRes[i]) + ' ' + str(predictY[i]) + '\n')
            if predictRes[i] != predictY[i]:
                cnt += 1
        outFile.close()
        
        rf = RandomForestRegressor(max_features=40,n_estimators=n_estimators)

        rf.fit(trainX,trainY)
        predictRes = rf.predict(predictX)
        outFile = open(predictoutpath + 'output_' + str(setId), 'wb')
        cnt = 0
        for i in range(predictY.shape[0]):
            outFile.write(str(predictRes[i]) + ' ' + str(predictY[i]) + '\n')
            if predictRes[i] != predictY[i]:
                cnt += 1
        outFile.close()
        return cnt, predictY.shape[0]

if __name__ == '__main__':
    mlp = mlpTrain()
    mlp.run()


np.random.seed(1337)  # for reproducibility
max_words = 1000
batch_size = 32
nb_epoch = 5

print('Loading data...')
(X_train, y_train), (X_test, y_test) = reuters.load_data(
    nb_words=max_words, test_split=0.2)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

import IPython
IPython.embed()


nb_classes = np.max(y_train) + 1
print(nb_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print(
    "Convert class vector to binary class matrix "
    "(for use with categorical_crossentropy)")
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
predic = model.predict_proba(X_test,
                             batch_size=batch_size, verbose=1)

print('Predict Y:', predic)
print('Test score:', score[0])
print('Test accuracy:', score[1])



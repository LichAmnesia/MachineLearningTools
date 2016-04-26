# -*- coding: utf-8 -*-

'''
除本id之外所有加入模型并进行predict
训练数据在Trainset/文件夹下面
预测结果后写入Predictout/put_id文件，
最终得到20个文件，就是每个predict的结果
目前使用RF
'''
import string
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVR
filepath = "D:\\Work\\bioinformatics\\hhsearch\\PDBCYS\\"
trainsetpath = filepath + "Trainset\\"
predictoutpath = filepath + "predictout\\"
# 形成set的主函数
def getSet(setId, MaxNum):
    print("Now predict " + str(setId))
    # 测试集读取
    predictSet = np.loadtxt(trainsetpath + 'predict_' + str(setId) + '.rf')
    predictId = setId

    # 修改为只有最后一个，并且把1.0的去掉
    predictX = np.array([[1]])
    predictY = np.array([])
    predictI = np.array([])
    global predictLen
    predictLen = predictSet.shape[0]
    for i in range(predictSet.shape[0]):
        if predictSet[i,-2] == 1.0:
            continue
        predictX = np.append(predictX,np.array([[predictSet[i,-2]]]), axis=0)
        predictY = np.append(predictY,predictSet[i,-1])
        predictI = np.append(predictI,i)
    predictX = predictX[1:,:]
    # print predictX
    # predictX = predictSet[:,:-1]
    # predictY = predictSet[:,-1]

    # 训练集读取
    nextSetId = (setId) % MaxNum + 1
    trainSet = np.loadtxt(trainsetpath + 'predict_' + str(nextSetId) + '.rf')
    trainId = nextSetId
    # print(nextSetId, trainSet.shape)
    for i in range(MaxNum):
        nextSetId = (nextSetId) % MaxNum + 1
        if nextSetId == trainId or nextSetId == predictId:
            continue
        tmpTrainSet = np.loadtxt(trainsetpath + 'predict_' + str(nextSetId) + '.rf')
        trainSet = np.append(trainSet,tmpTrainSet, axis=0)
        # print(nextSetId,trainSet.shape)
    # 修改为只有最后一个，并且把1.0的去掉
    print trainSet.shape,predictSet.shape
    trainX = np.array([[1]])
    trainY = np.array([])
    for i in range(trainSet.shape[0]):
        if trainSet[i,-2] == 1.0:
            continue
        trainX = np.append(trainX,np.array([[trainSet[i,-2]]]), axis=0)
        trainY = np.append(trainY,trainSet[i,-1])
    trainX = trainX[1:,:]
    print trainX.shape,trainY.shape,predictX.shape,predictY.shape,predictI.shape
    # trainX = trainSet[:,:-1]
    # trainY = trainSet[:,-1]
    return predictX,predictY,trainX,trainY,predictI

# 进行预测
def getPredict(setId,predictX,predictY,trainX,trainY, n_estimators,predictI):
    # rf = RandomForestRegressor(max_features=40,n_estimators=n_estimators)
    # Adaboost运算结果
    rf = AdaBoostClassifier(n_estimators=n_estimators)
    # print predictLen,predictI

    # print trainX.shape, trainY.shape, predictX.shape, predictY.shape

    rf.fit(trainX,trainY)
    predictRes = rf.predict(predictX)
    for i in range(predictX.shape[0]):
        if predictX[i] > 0.7:
            predictRes[i] = 0.0
    getModellerSet(setId, predictI, predictRes)
    # outFile = open(predictoutpath + 'output_' + str(setId), 'wb')
    cnt = 0
    for i in range(predictY.shape[0]):
        # outFile.write(str(predictRes[i]) + ' ' + str(predictY[i]) + '\n')
        if predictRes[i] != predictY[i]:
            cnt += 1
    # outFile.close()
    return cnt, predictY.shape[0]

# 形成set的主函数,for rf
def getPredictSet(setId, MaxNum):
    print("Now predict " + str(setId))
    # 测试集读取
    predictSet = np.loadtxt(trainsetpath + 'predict_' + str(setId) + '.rf')
    predictId = setId

    # 修改为只有最后一个，并且把1.0的去掉
    # predictX = np.array([[1]])
    # predictY = np.array([])
    # predictI = np.array([])
    # global predictLen
    # predictLen = predictSet.shape[0]
    # for i in range(predictSet.shape[0]):
    #     if predictSet[i,-2] == 1.0:
    #         continue
    #     predictX = np.append(predictX,np.array([[predictSet[i,-2]]]), axis=0)
    #     predictY = np.append(predictY,predictSet[i,-1])
    #     predictI = np.append(predictI,i)
    # predictX = predictX[1:,:]
    # # print predictX
    predictX = predictSet[:,:-1]
    predictY = predictSet[:,-1]

    # 训练集读取
    nextSetId = (setId) % MaxNum + 1
    trainSet = np.loadtxt(trainsetpath + 'predict_' + str(nextSetId) + '.rf')
    trainId = nextSetId
    # print(nextSetId, trainSet.shape)
    for i in range(MaxNum):
        nextSetId = (nextSetId) % MaxNum + 1
        if nextSetId == trainId or nextSetId == predictId:
            continue
        tmpTrainSet = np.loadtxt(trainsetpath + 'predict_' + str(nextSetId) + '.rf')
        trainSet = np.append(trainSet,tmpTrainSet, axis=0)
        # print(nextSetId,trainSet.shape)
    # 修改为只有最后一个，并且把1.0的去掉
    # trainX = np.array([[1]])
    # trainY = np.array([])
    # for i in range(trainSet.shape[0]):
    #     if trainSet[i,-2] == 1.0:
    #         continue
    #     trainX = np.append(trainX,np.array([[trainSet[i,-2]]]), axis=0)
    #     trainY = np.append(trainY,trainSet[i,-1])
    # trainX = trainX[1:,:]
    # print trainX.shape,trainY.shape,predictX.shape,predictY.shape,predictI.shape
    trainX = trainSet[:,:-1]
    trainY = trainSet[:,-1]
    return predictX,predictY,trainX,trainY

# 进行预测
def getPredictOutputRF(setId,predictX,predictY,trainX,trainY, n_estimators):
    # rf = SVR(epsilon=0.01)
    rf = RandomForestRegressor(max_features=40,n_estimators=n_estimators)
    # Adaboost运算结果
    # rf = AdaBoostClassifier(n_estimators=n_estimators)
    # print predictLen,predictI

    # print trainX.shape, trainY.shape, predictX.shape, predictY.shape

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


# 形成算每个相应的最后一个数据展示的主函数
def getModellerSet(setId, predictI, predictRes):
    modelleroutpath = filepath + 'Modeller\\'

    # print("Now predict " + str(setId))
    # 测试集读取
    predictSet = np.loadtxt(trainsetpath + 'predict_' + str(setId) + '.rf')
    outFile = open(modelleroutpath + 'output_' + str(setId), 'wb')
    for i in range(predictSet.shape[0]):
        # 表示有没有找到这个点
        flag = 0
        for j in range(predictI.shape[0]):
            if predictI[j] == i:
                flag = 1
                outFile.write(str(predictRes[j])+'\n')
        if flag != 1:
            outFile.write(str(-1.0)+'\n')
    outFile.close()


def main():
    # 算modeller的结果
    # for setId in range(1,21):
    #     getModellerSet(setId, 20)
    # return

    # ansero = 1
    # ansnum = 1
    # ansn_estimators = 0
    # for n_estimators in xrange(150,160,10):
    #     sumero = 0
    #     sumall = 0
    #     for setId in range(1,21):
    #         predictX,predictY,trainX,trainY,predictI = getSet(setId, 20)
    #         ero, allnum = getPredict(setId,predictX,predictY,trainX,trainY,n_estimators,predictI)
    #         sumero += ero
    #         sumall += allnum
    #     if float(sumero) / sumall < float(ansero)/ ansnum:
    #         ansero = sumero
    #         ansnum = sumall
    #         ansn_estimators = n_estimators
    #     print n_estimators, sumero, sumall, 1 - float(sumero) / sumall
    # print ansn_estimators, ansero, ansnum, 1 - float(ansero)/ ansnum

    ansero = 1
    ansnum = 1
    ansn_estimators = 0
    for n_estimators in xrange(500,510,10):
        sumero = 0
        sumall = 0
        for setId in range(1,21):
            predictX,predictY,trainX,trainY = getPredictSet(setId, 20)
            ero, allnum = getPredictOutputRF(setId,predictX,predictY,trainX,trainY,n_estimators)
            sumero += ero
            sumall += allnum
        if float(sumero) / sumall < float(ansero)/ ansnum:
            ansero = sumero
            ansnum = sumall
            ansn_estimators = n_estimators
        print n_estimators, sumero, sumall, 1 - float(sumero) / sumall
    print ansn_estimators, ansero, ansnum, 1 - float(ansero)/ ansnum


if __name__ == '__main__':
    main()

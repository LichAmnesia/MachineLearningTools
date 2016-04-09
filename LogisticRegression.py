# -*- coding: utf-8 -*-
# @Author: Lich_Amnesia  
# @Email: alwaysxiaop@gmail.com
# @Date:   2016-04-09 21:00:45
# @Last Modified time: 2016-04-09 22:02:56
# @FileName: LogisticRegression.py

import numpy as np
import matplotlib.pyplot as plt
import sys,os

# process the origin data to formatted train/test data like the following
# train.txt
# gre gpa rank admit
# 300 3.00 1 1
# 300 2.00 1 0
# Also random generate train/test dataset. 
# All the dataset size is 400, 40 for test set and other for training.
def preDataProcess():
	# configure file_path
	file_path = 'data/LogisticRegression_origindata.csv'
	trainFile = 'data/LogisticRegression_train.txt'
	testFile = 'data/LogisticRegression_test.txt'

	originDataSet = np.loadtxt(open(file_path,"rb"),delimiter=",",skiprows=1)
	originDataSet = np.c_[originDataSet[:,1:],originDataSet[:,0]]
	originDataSize = originDataSet.shape[0]
	randomArr = np.sort(np.random.randint(0,originDataSize,size=originDataSize/10))
	trainDataSet = np.array([])
	testDataSet = np.array([])
	j = 0
	for i in range(originDataSize):
		if j < randomArr.size and i == randomArr[j]:
			if testDataSet.size == 0:
				testDataSet = originDataSet[i]
			else :
				testDataSet = np.vstack((testDataSet, originDataSet[i]))
			j += 1
		else :
			if trainDataSet.size == 0:
				trainDataSet = originDataSet[i]
			else :
				trainDataSet = np.vstack((trainDataSet, originDataSet[i]))
	np.savetxt(trainFile,trainDataSet,fmt='%.2f %.2f %.2f %d')
	np.savetxt(testFile,testDataSet,fmt='%.2f %.2f %.2f %d')
	return trainDataSet, testDataSet

def loadData():
	trainFile = 'data/LogisticRegression_train.txt'
	testFile = 'data/LogisticRegression_test.txt'
	trainDataSet = np.loadtxt(open(trainFile,"rb"),delimiter=" ")
	testDataSet = np.loadtxt(open(testFile,"rb"),delimiter=" ")
	return trainDataSet, testDataSet


def main():
	# preDataProcess will generate random train/test dataset, you can only use loadData to get dataset.
	# trainDataSet, testDataSet = preDataProcess()

	trainDataSet, testDataSet = loadData()
	


if __name__ == '__main__':
	main()
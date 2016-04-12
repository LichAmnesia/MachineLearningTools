# -*- coding: utf-8 -*-
# @Author: Lich_Amnesia  
# @Email: alwaysxiaop@gmail.com
# @Date:   2016-04-12 15:40:14
# @Last Modified time: 2016-04-12 16:51:04
# @FileName: Perceptron.py


'''
if you do not want to generate new train/test dataset, you will not need to use preDataProcess function.
You just need to change the configuration like theta and maxCycles

this py do not add regularization.
'''


import numpy as np
import matplotlib.pyplot as plt
import sys,os

# process the origin data to formatted train/test data like the following
# train.txt
# 0.28147 0.71434 0.075309 0.9116	1
# 0.46295 0.64512 0.96324 0.31516	-1
# Also random generate train/test dataset. 
# All the dataset size is 500, 500 for test set and other for training.
# load data from train/test file.
def loadData():
	trainFile = 'data/perceptron_train.txt'
	testFile = 'data/perceptron_test.txt'
	trainDataSet = np.loadtxt(open(trainFile,"rb"))
	testDataSet = np.loadtxt(open(testFile,"rb"))
	# Add X0 to the dataSet
	X0 = np.array([1.0 for i in range(trainDataSet.shape[0])])
	trainDataSet = np.c_[X0,trainDataSet]
	X0 = np.array([1.0 for i in range(testDataSet.shape[0])])
	testDataSet = np.c_[X0,testDataSet]
	return trainDataSet, testDataSet

# normalization, 
def norm(input_x):
	mean = np.mean(input_x,axis=0)
	std = np.std(input_x,axis=0)
	n, m = input_x.shape
	for i in range(n):
		for j in range(m):
			if std[j] != 0:
				input_x[i][j] = (input_x[i][j] - mean[j]) / std[j]
	return input_x
	
# sigmoid function, the input_x's size is n * 1 and the output's size is n * 1 too.
def sigmoid(input_x):
	return (1.0 / (1.0 + np.exp(-input_x)))

# alpha: steplength maxCycles: number of iterations
def gradAscent(trainDataSet, alpha, maxCycles):
	X_parameters, Y_parameters = trainDataSet[:,:-1],trainDataSet[:,-1]
	X_parameters = norm(X_parameters)
	X_mat = np.mat(X_parameters) # size: n * m (m = 4, X0=1 now)
	y_mat = np.mat(Y_parameters).T # size: n * 1
	n,m = X_mat.shape
	W = np.zeros((m,1)) # initialize W as zero vector, W has m columns for X_i
	# do maxCycles to get W
	# this need to be changed, because if old_loss == new_loss, it can return the answer
	for i in range(maxCycles):
		input_x = np.dot(X_mat,W)
		h = sigmoid(input_x)
		error = h - y_mat  # size: n * 1
		W = W - alpha * np.dot(X_mat.T,error)
	return W

# classify test dataset and give error rate 
def classify(testDataSet, W):
	X_parameters, Y_parameters = testDataSet[:,:-1],testDataSet[:,-1]
	X_parameters = norm(X_parameters)
	X_mat = np.mat(X_parameters) # size: n * m (m = 3 now)
	y_mat = np.mat(Y_parameters).T # size: n * 1
	n, m = X_mat.shape
	h = sigmoid(np.dot(X_mat,W))
	# calculate the error rate
	error = 0.0
	for i in range(n):
		if round(h[i]) != int(y_mat[i]):
			error += 1
	# print np.c_[h,y_mat]
	print('error rate is {0:.4f}'.format(error / n))
	return

class Perceptron:
	def __init__(self, W, alpha):
		self.W = W
		# self.b = b
		self.alpha = alpha

	def loss(self, X_parameters, Y_parameters):
		return np.sum(Y_parameters * (np.dot(self.W.T,X_parameters)))

	def train(self, trainDataSet):
		X_parameters, Y_parameters = trainDataSet[:,:-1],trainDataSet[:,-1]
		# X_parameters = norm(X_parameters)
		X_mat = np.mat(X_parameters) # size: n * m (m = 5 now)
		y_mat = np.mat(Y_parameters).T # size: 1 * n
		n, m = X_mat.shape
		while True:
			M = len(X_mat) # wrong classification number
			for i in range(len())
		print len(self.W)
		return
		# while 

def main():
	trainDataSet, testDataSet = loadData()
	perceptronTrain = Perceptron(np.zeros(len(trainDataSet)),1)
	perceptronTrain.train(trainDataSet)
	# configure steplength and iterations
	alpha = 1e-3
	maxCycles = 400

	W = gradAscent(trainDataSet, alpha, maxCycles)
	classify(testDataSet,W)


if __name__ == '__main__':
	main()

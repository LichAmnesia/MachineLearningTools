# -*- coding: utf-8 -*-
# @Author: Lich_Amnesia  
# @Email: alwaysxiaop@gmail.com
# @Date:   2016-04-12 15:40:14
# @Last Modified time: 2016-04-12 18:25:20
# @FileName: Perceptron.py




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
	

class Perceptron:
	def __init__(self, W, alpha, eps = 1e-8):
		self.W = np.mat(W)
		self.alpha = alpha
		self.eps = eps

	def loss(self, x, y):
		return y * (np.dot(self.W.T,x.T))

	def sgd(self, x, y):
		self.W += (self.alpha * y * x).T

	def train(self, trainDataSet):
		X_parameters, Y_parameters = trainDataSet[:,:-1],trainDataSet[:,-1]
		# X_parameters = norm(X_parameters)
		X_mat = np.mat(X_parameters) # size: n * m (m = 6 now, has X_0)
		y_mat = np.mat(Y_parameters).T # size: 1 * n
		n, m = X_mat.shape
		while True:
			M = len(X_mat) # wrong classification number
			for i in range(len(X_mat)):
				if self.loss(X_mat[i], y_mat[i])  <= 0:
					self.sgd(X_mat[i], y_mat[i])
				else:
					M -= 1
			if M == 0:
				print('self.W is \n {0}'.format(self.W))
				break
		return self.W
	
	def classify(self, testDataSet):
		X_parameters, Y_parameters = testDataSet[:,:-1],testDataSet[:,-1]
		# X_parameters = norm(X_parameters)
		X_mat = np.mat(X_parameters) # size: n * m (m = 6 now, has X_0)
		y_mat = np.mat(Y_parameters).T # size: n * 1
		n, m = X_mat.shape
		M = len(X_mat) # wrong classification number
		for i in range(len(X_mat)):
			x = X_mat[i]
			if np.dot(self.W.T,x.T) <= 0 and y_mat[i] == -1:
				M -= 1
			elif np.dot(self.W.T,x.T) > 0 and y_mat[i] == 1 :
				M -= 1
		error = float(M) / len(X_mat)
		print('error rate is {0:.4f}'.format(error))
		return error


class Perceptron_dual:
	def __init__(self, alpha, b, ita, eps = 1e-8):
		self.alpha = alpha
		self.b = b
		self.ita = ita
		self.eps = eps

	
	def gram(self,X):
         return np.dot(X,X.T)
	
	def train(self, trainDataSet):
		X_parameters, Y_parameters = trainDataSet[:,1:-1],trainDataSet[:,-1]
		# X_parameters = norm(X_parameters)
		X_mat = np.mat(X_parameters) # size: n * m (m = 2 now,not has X_0)
		y_mat = np.mat(Y_parameters).T # size: n * 1
		# Y_parameters 1 * n
		n, m = X_mat.shape
		G = self.gram(X_mat)
		while True:
			M = len(X_mat) # wrong classification number
			for j in range(len(X_mat)):
				if y_mat[j] * (np.sum(self.alpha * Y_parameters * G[j].T) + self.b) <= 0:
					self.alpha[j] += self.ita
					self.b += self.ita * y_mat[j]
				else:
					M -= 1
			# print M
			if M == 0:
				print('self.alpha is \n {0}\nself.b is \n {1}'.format(self.alpha,self.b))
				break
		return self.alpha, self.b
	


def main():
	trainDataSet, testDataSet = loadData()
	# configure steplength and iterations
	alpha = 1
	perceptronTrain = Perceptron(np.zeros((trainDataSet.shape[1] - 1,1)),alpha)
	W = perceptronTrain.train(trainDataSet)
	perceptronTrain.classify(testDataSet)

	perceptronDualTrain = Perceptron_dual(np.zeros(trainDataSet.shape[1] - 1), 0, alpha)
	W = perceptronDualTrain.train(trainDataSet)
		

if __name__ == '__main__':
	main()

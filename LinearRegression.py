# -*- coding: utf-8 -*-
# @Author: Lich_Amnesia  
# @Email: alwaysxiaop@gmail.com
# @Date:   2016-04-08 18:35:50
# @Last Modified time: 2016-04-08 19:38:38
# @FileName: LinearRegression.py

import numpy as np
import matplotlib.pyplot as plt
import sys,os

# use numpy load from txt function to load data
def loadData():
	file_path = 'data/LR_in.txt'
	file = open(file_path)
	data_set = np.loadtxt(file)
	X0 = np.array([1.0 for i in range(data_set.shape[0])])
	return np.c_[X0,data_set[:,:-1]],data_set[:,-1] 

# use X^T * X to calculate the answer
def calculateMethod(X_parameters, Y_parameters):
	X = np.mat(X_parameters)
	# import! this y should be Y.T, you can print it to find the reason
	y = np.mat(Y_parameters).T
	tmp1 = np.dot(X.T,X).I
	tmp2 = np.dot(X.T,y)
	theta = np.dot(tmp1,tmp2)
	theta = np.array(theta)
	print(theta)
	return theta

# use calculated theta, it will returrn predict Y
def predictOut(X_parameters, theta):
	X = np.mat(X_parameters)
	theta = np.mat(theta)
	out = np.dot(X,theta)
	return out

# use matplotlib to draw X-Y axis points
def draw(X_parameters, Y_parameters,theta):
	plt.scatter(X_parameters[:,-1],Y_parameters,color='blue')
	Y_predict_out = predictOut(X_parameters,theta)
	plt.plot(X_parameters[:,-1],Y_predict_out,color='r',linewidth=4)
	plt.xlabel('Year')
	plt.ylabel('House Price')
	plt.show()
	return

def main():
	X_parameters, Y_parameters = loadData()
	theta = calculateMethod(X_parameters,Y_parameters)
	draw(X_parameters,Y_parameters,theta)
	# print(X_parameters)
	return

if __name__ == '__main__':
	main()
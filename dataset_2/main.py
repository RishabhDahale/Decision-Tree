import numpy as np
import pandas as pd
import math
import copy
import csv
import sys
from scipy import stats

def Entropy(data, target):
	classes = data[..., target]
	values, count = np.unique(classes, return_counts = True)
	probability = count/np.sum(count)
	# E = -np.sum(np.dot(probability, np.log2(probability)))
	E = -np.sum(np.multiply(probability,np.log2(probability)))
	return E

def InformationGain(data, attribute, target):
	Gain = Entropy(data, target)
	n = np.shape(data)[0]
	attr = data[..., attribute]
	types, count = np.unique(attr, return_counts=True)
	for att in attr:
		S = data[data[:,attribute] == att]
		n1 = np.shape(S)[0]
		Gain = Gain - ((n1/n)*Entropy(S, target))
	return float(Gain)

class Node:

	def __init__(self, index, isLeaf=False):
		self.index = index
		self.threshold = 0
		self.children = []
		self.isLeaf = isLeaf

#All the getter methods
	def getLeaf(self):
		return self.isLeaf

	def getChildren(self):
		return self.children

	def getThreshold(self):
		return self.threshold

	def getIndex(self):
		return self.index

#All the setter method
	def setThreshold(self, threshold):
		self.threshold = threshold

	def setChildren(self, v):
		self.children.append(v)

def makeTree(data, originalData, target):
	if np.shape(data)[1] == 1:
		value, count = np.unique(data, return_counts=True)
		# print(np.shape(count))
		# print(count)
		# print(np.shape(data))
		# i = np.argmax(count)
		leaf = Node(target, isLeaf = True)
		# leaf.setChildren(value[np.argmax(count)])
		# leaf.setChildren(np.mode(value))
		leaf.setChildren(np.mean(value))
		return leaf

	else:
		mn = np.mean(data, axis=0)
		IG = []
		for i in range(np.shape(data)[1]-1):
			data1= np.array(data)
			# if data1[...,i] > mn[i]:
			# 	data1[]
			data1[np.where(data1[..., i] >= mn[i])] = 1
			data1[np.where(data1[..., i] < mn[i])] = 0 
			IG.append(InformationGain(data1, i, target))
		# IG = [InformationGain(data, i, target) for i in range(np.shape(data))[1]]
		index = np.argmax(IG)
		node1 = Node(index, False)
		IG2 = []
		data1 = np.array(data)

		thr = mn[index]
		lessData = data[np.where(data[...,index] < thr)]
		lessData = np.delete(lessData, index, 1)
		moreData = data[np.where(data[..., index] >= thr)]
		moreData = np.delete(moreData, index, 1)
		node1.setChildren(makeTree(lessData, originalData, target-1))
		node1.setChildren(makeTree(moreData, originalData, target-1))
		node1.setThreshold = thr
		return node1

def predict(data, tree, mean):
	prediction = []
	for i in range(np.shape(data)[0]):
		row = data[i][...]
		# print(np.shape(row))
		myTree = copy.deepcopy(tree)
		
		while(np.shape(row)[0]>1):
			while myTree.getLeaf() == False:
				idx = myTree.getIndex()
				if row[idx] >= mean[idx]:
					myTree = myTree.getChildren()[1]
					row = np.delete(row, idx)
				else:
					myTree = myTree.getChildren()[0]
					row = np.delete(row, idx)
			# print(myTree.getChildren())
			prediction.append(myTree.getChildren()[0])

	with open("output.csv", "w") as csvfile:
		writerr = csv.writer(csvfile,delimiter=',')
		writerr.writerow(['id','quality'])
		ii = 0
		for p in prediction:
			ii += 1
			writerr.writerow([ii,p])
	# print(prediction)
	# csvfile.close()

def train_test_split(data):
	train_data = data[:int(0.8*np.shape(data)[0]), ...]
	test_data = data[int(0.8*np.shape(data)[0]):, ...]
	return train_data, test_data


train_data = pd.read_csv(sys.argv[2])
test_data = pd.read_csv(sys.argv[4])
train_data = np.array(train_data)
test_data = np.array(test_data)

# train_data, test_data = train_test_split(data)
mn = np.mean(train_data, axis=0)
# print(Entropy(train_data, 4))
# print(InformationGain(train_data, 4, 4))

tree = makeTree(train_data, train_data, 11)
predict(test_data, tree, mn)

# with open("test.csv", "w") as test:
# 	writerr = csv.writer(test,delimiter=',')
# 	writerr.writerow(['id','output'])
# 	ii = 0
# 	for p in test_data[...,4]:
# 		ii += 1
# 		writerr.writerow([ii,p])



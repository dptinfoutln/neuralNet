#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from operator import add
from itertools import product

config =(2,15,1)

class NLayer:
	def __init__(self,entries,activation,seuilIter,stepGenerator):
		self.entries = entries
		self.activation = activation
		(self.seuil,self.iter) = seuilIter
		self.stepGenerator = stepGenerator
		self.cpt = 0

		self.w1 = np.random.randn(config[0],config[1])
		self.wfinal = np.random.rand(config[1],config[2])
		self.weights = [self.w1,self.wfinal]
	
	def forward(self, value):
		self.a1 = np.dot(value,self.weights[0])
		self.l1 = self.activation[0](self.a1)
		self.a2 = np.dot(self.l1,self.weights[1])
		self.layoutOut = (self.a1,self.a2)
		#linear activation for final neurons, we can keep a2 as is
		return self.a2

	def computeError(self):
		self.entries=np.random.permutation(self.entries)
		self.nOutputs = []
		self.errors = []
		for values in self.entries:
			#values.item(0) : input,  values.item(1): value to learn
			self.forward(values.item(0))
			self.nOutputs.append((np.array(values.item(0)),self.l1))
			#linear activation for final neurons
			eFinal=(values.item(1)-self.a2)
			e1=self.activation[1](self.a1)*np.dot(self.weights[-1],eFinal)
			self.errors.append((e1,eFinal))
	def retroprop(self):
		step = next(self.stepGenerator)

		#computes deltas of the current layer of the current example
		result = []
		#i = n° of example
		for i in range(len(self.errors)):
			tmp=[]
			for index,(error,output) in enumerate(zip(self.errors[i],self.nOutputs[i])):
				matrix=np.array(list(map(lambda x : x[0]*x[1],product(error,output)))).reshape(np.shape(self.weights[index]))
				tmp.append(matrix)
			result.append(tmp)

		#computes the average of deltas for each layer
		dError_average = [ step*np.matrix(reduce(add, matrices))/len(matrices) for matrices in zip(*result) ]

		#we apply the error to the weigths
		for w,e in zip(self.weights,dError_average):
			w+=e
	

	def getSquaredError(self):
		self.computeError()
		self.squaredError = 0.5*np.sum(list(map(lambda x : x*x,np.array(self.errors)[:,1])))
		return self.squaredError

	def __iter__(self):
		return self	

	def __next__(self):
		self.getSquaredError()
		if self.squaredError < self.seuil or self.cpt > self.iter:
			raise StopIteration
		self.retroprop()
		self.cpt+=1


#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

entrees = np.matrix(((0,0),(0,1),(1,0),(1,1)))
sorties = np.matrix((0,1,1,0))
l2=0
lr=0.1
error=0

def relu(v):
	v[v<0]=0
	return v

def derivRelu(v):
	v[v<0]=0
	v[v>1]=1
	return v


def stepDecrease(x):
	while True:
		yield x
		x *= x

config =(2,3,1)

class NLayer:
	def __init__(self,entries,results,Nnbr,activation,seuil,stepGenerator=stepDecrease):
		self.entries = entries
		self.results = results
		self.activation = activation
		self.seuil = seuil
		self.stepGenerator = stepGenerator(0.9)

		self.w1 = np.random.rand(config[0]+1,config[1])
		self.wfinal = np.random.rand(config[1]+1,config[2])
		self.weights = [self.w1,self.wfinal]
	
	def forward(self):
		self.a1 = np.insert(self.entries,0,1,axis=1)
		print(self.a1)
		self.a1 = np.dot(self.a1,self.weights[0])
		print(self.a1)
		self.l1 = self.activation[0](self.a1)
		self.l1 = np.insert(self.l1,0,1,axis=1)
		print(self.l1)
		self.a2 = np.dot(self.l1,self.weights[1])
		self.layoutOut = (self.a1,self.a2)
#linear activation for final neurons

	def computeError(self):
		self.forward()
		self.errors = []
#linear activation for final neurons
		self.errors.append(self.results-self.a2)
		print("a1")
		print(np.shape(self.a1))
		print("w2*e2")
		print(np.shape(np.dot(self.weights[-1],self.errors[-1].transpose())))
		print("w2")
		print(np.shape(self.wfinal))
		self.errors.append(self.activation[1](self.a1)*np.dot(self.weights[-1],self.errors[-1].transpose()))

	def retroprop(self):
		step = next(self.stepGenerator)
		layerLen = len(self.errors)
		for i in layerLen:
			for k in np.shape(l1)[0]:
				error_avg=0
				for l in np.shape(l1)[1]:
					error_avg += step*self.examplaire[l]*self.errors[i].item(k,l)
				error_avg /= np.shape(l1)[1] 


	def getSquaredError(self):
		self.computeError()
		self.squaredError = np.sum(self.error**2)
		return self.squaredError

	def __iter__(self):
		return self	

	def __next__(self):
		self.getSquaredError()
		if self.squaredError < self.seuil:
			raise StopIteration
		self.retroprop()
		print("poids")
		print(self.w1)
		print("valeurs predites")
		print(self.l2)
	
net = NLayer(entrees[3],np.transpose(sorties),6,(relu,derivRelu),0.1)
for i in net:
	plt.plot(net.getSquaredError())
plt.show()

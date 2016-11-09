#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

entrees = np.matrix((((0,0),0),((0,1),1),((1,0),1),((1,1),0)))
lr=0.1
error=0

def relu(v):
	v[v<0]=0
	return v

def derivRelu(v):
	v[v<0]=0
	v[v>0]=1
	return v


def stepDecrease(x,y):
	val=x
	while True:
		yield val*y

config =(2,5,1)

class NLayer:
	def __init__(self,entries,activation,seuilIter,stepGenerator=stepDecrease):
		self.entries = entries
		self.activation = activation
		(self.seuil,self.iter) = seuilIter
		self.stepGenerator = stepGenerator(0.15,0.99)
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
		print(step)

		result = []
		#i = n° of example
		for i in range(len(self.errors)):
			tmp=[]
			#j = n° of layer
			for j in range(len(self.errors[i])):
				matrix=np.ones(np.shape(self.weights[j]))
				#k = n° of output
				for k in range(len(self.nOutputs[i][j])):
					#l = n° of error
					for l in range(len(self.errors[i][j])):
						matrix[k,l] = self.nOutputs[i][j][k] * self.errors[i][j][l]
				tmp.append(matrix)
			result.append(tmp)

		list1 = []	
		list2 = []	
		for i in range(len(result)):
			list1.append(result[i][0])
			list2.append(result[i][1])

		#we compute the average
		dError_average1 = step*np.matrix(reduce(lambda x,y: x+y, list1))/len(list1)
		dError_average2 = step*np.matrix(reduce(lambda x,y: x+y, list2))/len(list2)

		#we apply the error to the weigths
		self.w1+=dError_average1
		self.wfinal+=dError_average2
	

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
	
net = NLayer(entrees,(relu,derivRelu),(0.001,100000))
for i in net:
	print("squaredError=",net.getSquaredError())
	#plt.plot(net.getSquaredError())

print(net.forward((0,0)))
print(net.forward((1,0)))
print(net.forward((0,1)))
print(net.forward((1,1)))
#plt.show()

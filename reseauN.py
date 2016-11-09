#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

entrees = np.matrix((((0,0),0),((0,1),1),((1,0),1),((1,1),0)))
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

config =(2,10,1)

class NLayer:
	def __init__(self,entries,activation,seuilIter,stepGenerator=stepDecrease):
		self.entries = entries
		self.activation = activation
		(self.seuil,self.iter) = seuilIter
		self.stepGenerator = stepGenerator(0.99)
		self.step = 0.15
		self.cpt = 0

		self.w1 = np.random.rand(config[0],config[1])
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
			self.nOutputs.append((self.l1,self.a2))
			#linear activation for final neurons
			eFinal=(values.item(1)-self.a2)
			e1=self.activation[1](self.a1)*np.dot(self.weights[-1],eFinal.transpose())
			self.errors.append((e1,eFinal))
	def retroprop(self):
		step = self.step+next(self.stepGenerator)

		result = []
		for i in range(len(self.errors)):
			for j in range(len(self.errors[i])):
				result.append(self.errors[i][j]*self.nOutputs[i][j])
	
		list1 = []	
		list2 = []	
		for i in range(len(result)):
			if i%2 == 0:
				list1.append(result[i])
			else:
				list2.append(result[i])

		dError_average = [np.sum(list1,axis=0)/len(list1),np.sum(list2,axis=0)/len(list2)]
		eo = (e,o) = (config[0],config[2])
		for j in range(len(dError_average)):
			w=self.weights[j]
			for i in range(np.shape(w)[1]):
				w[:,i]=w[:,i]+np.ones(np.shape(w)[0])*dError_average[j][i]

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
	
net = NLayer(entrees,(relu,derivRelu),(0.1,1000))
for i in net:
	print(net.getSquaredError())
	plt.plot(net.getSquaredError())

print(net.forward((0,0)))
print(net.forward((1,0)))
print(net.forward((0,1)))
print(net.forward((1,1)))
plt.show()

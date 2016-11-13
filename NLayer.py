#!/usr/bin/python3
import numpy as np
from functools import reduce
from operator import add
from itertools import product

config =(2,10,1)

class NLayer:
	def __init__(self,data,activation,seuilIter,stepGenerator):
            self.data = data
            self.activation, self.activGradient = activation
            (self.seuil,self.iter) = seuilIter
            self.stepGenerator = stepGenerator
            self.cpt = 0
            self.subExLen = 4

            #weights random initialization
            configDec = [ (config[i],config[i+1]) for i in range(len(config)-1)]
            self.weights = []
            for weightsLayer in configDec:
                self.weights.append(np.random.randn(*weightsLayer))

	def forward(self, value):
            self.layerOut = []
            self.layerActivation = [np.array(value)]
            for layerIndex,weight in enumerate(self.weights[:-1]):
                self.layerOut.append(np.dot(self.layerActivation[-1],weight))
                self.layerActivation.append(self.activation(self.layerOut[layerIndex]))
            #linear activation for final neurons, we can keep it as is
            self.layerOut.append(np.dot(self.layerActivation[-1],self.weights[-1]))
            return self.layerOut[-1] 

	def computeError(self):
            self.data = np.random.permutation(self.data)
            self.entriesGenerator = (self.data[x:x+self.subExLen] for x in range(0,len(self.data),self.subExLen))
            for subData in self.entriesGenerator:
                self.wUpdateData = []
                self.errors = []
                for entry,output in subData:
                    self.forward(entry)
                    self.wUpdateData.append(self.layerActivation)
                    #linear activation for final neurons
                    subErrors = [output-self.layerOut[-1]]
                    for out,weight in zip(reversed(self.layerOut[:-1]),reversed(self.weights)):
                        subErrors.append(self.activGradient(out)*np.dot(weight,subErrors[-1]))
                    self.errors.append(subErrors)
	def retroprop(self):
            step = next(self.stepGenerator)

            #computes deltas of the current layer of the current example
            result = []
            #i = n° of example
            for i in range(len(self.errors)):
                    tmp=[]
                    for index,(error,output) in enumerate(zip(self.errors[i],reversed(self.wUpdateData[i]))):

                        matrix=np.array(list(map(lambda x : x[0]*x[1],product(error,output)))).reshape(np.shape(tuple(self.weights[len(self.weights)-1-index])))
                        tmp.append(matrix)
                    result.append(tmp)

            #computes the average of deltas for each layer
            dError_average = [ step*np.matrix(reduce(add, matrices))/len(matrices) for matrices in zip(*result) ]

            #we apply the error to the weigths
            for w,e in zip(reversed(self.weights),dError_average):
                w+=e
    

	def getSquaredError(self):
            self.squaredError = 0.5*np.sum(list(map(lambda x : x*x,np.array(self.errors)[:,1])))
            return self.squaredError

	def __iter__(self):
            return self	

	def __next__(self):
            self.computeError()
            self.getSquaredError()
            if self.squaredError < self.seuil or self.cpt > self.iter:
                    raise StopIteration
            self.retroprop()
            self.cpt+=1


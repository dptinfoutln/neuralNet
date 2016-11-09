#!/usr/bin/python3
import NLayer as NL
import numpy as np
import matplotlib.pyplot as plt

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

entries = np.matrix((((0,0),0),((0,1),1),((1,0),1),((1,1),0)))

net = NL.NLayer(entries,(relu,derivRelu),(0.001,100000),stepDecrease)

cpt=0
for i in net:
	plt.plot(cpt,net.getSquaredError(),'ro')
	cpt+=1

print(net.forward((0,0)))
print(net.forward((1,0)))
print(net.forward((0,1)))
print(net.forward((1,1)))
plt.show()

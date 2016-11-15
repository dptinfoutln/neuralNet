#!/usr/bin/python3
import NLayer as NL
import numpy as np
import matplotlib.pyplot as plt

def relu(v):
	v[v<0]=0
	return v

def derivRelu(v):
	v[v<=0]=0
	v[v>0]=1
	return v


def stepDecrease(x,y):
	val=x
	while True:
		yield val*y

with open('Iris.data') as f:
    mappedValues=[]
    for line in f:
        tmp=list(map(float,line.split(',')))
        mappedValues.append(((tmp[:-1]),tmp[-1]))
    net = NL.NLayer(mappedValues,(4,100,100,100,1),(relu,derivRelu),(1e-3,1e8),stepDecrease(0.15,0.99),150)

    cpt=0
    for i in net:
            plt.plot(cpt,net.getSquaredError(),'ro')
            cpt+=1

    for e,o in mappedValues:
            print('entry={0} network answer={1} correct answer={2}'.format(e,net.forward(e),o))
    print('error={0} nbIterations={1}'.format(net.getSquaredError(), net.getNumberIteration()))
    plt.show()

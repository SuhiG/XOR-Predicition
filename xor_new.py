import numpy as np 
import time

n_hidden=10
n_in=10

n_out=10

n_samples=300

learning_rate=0.01
momentum=0.9

np.random.seed(0)

def sigmoid(x):
	return(1.0/(1.0+np.exp(-x)))

def tanh_prime(x):
	return(1-np.tanh(x)**2)	

#input data,transpose,layer1,layer2,biases
def train(x,t,V,W,bw,bv):
	A=np.dot(x,V)+bv
	Z=np.tanh(A)

	B=np.dot(Z,W)+bw
	Y=sigmoid(B)

	#backprop
	Ew=Y-t
	Ev=tanh_prime(A)*np.dot(W,Ew)

	#predictining the loss
	dW=np.outer(Z,Ew)
	dV=np.outer(x,Ev)

	#cross entropy
	loss=np.mean(t*np.log(Y)+(1-t)*np.log(1-Y))

	return loss,(dV,dW,Ew,Ev)


def predict(x,V,W,bw,bv):
	A=np.dot(x,V)+bv
	B=np.dot(np.tanh(A),W)+bw
	return(sigmoid(B)>0.5).astype(int)

#creating layers

V=np.random.normal(scale=0.1,size=(n_in,n_hidden))
W=np.random.normal(scale=0.1,size=(n_hidden,n_out))

bv=np.zeros(n_hidden)
bw=np.zeros(n_out)

params=[V,W,bv,bw]

#generating data
X=np.random.binomial(1,0.5,(n_samples,n_in))
T=X^1

#trainig time

for epoch in range(100):
	err=[]
	upd=[0]*len(params)

	t0=time.clock()

	#for each data point updating weights
	for i in range(X.shape[0]):
		loss,grad=train(X[i],T[i],*params)

		for j in range(len(params)):
			params[j]-=upd[j]

		for j in range(len(params)):
			upd[j]=learning_rate*grad[j]+momentum*upd[j]

		err.append(loss)
	
	print("epoch:%d ,loss: %.8f,time:%.4fs",(epoch,np.mean(err),time.clock()-t0))		

x=np.random.binomial(1,0.5,n_in)
print("xor prediction")
print(x)
print(predict(x,*params))



import numpy as np
#initialize random nn
def gen(nodes):return [[np.random.random((a,b))-.5 for a,b in zip(nodes[1:],nodes[:-1])],[np.random.random((a,1))-.5 for a in nodes[1:]]]
#use neural network
def N(M,x):
	A=[x]
	R=lambda x:np.maximum(x,x/8)
	for i in range(len(M[0])):A.append(R(M[0][i]@A[i]+M[1][i]))
	return A
#backpropogation of error
def bprop(M,x,y):
	A=N(M,x)
	dR=lambda x:(x>=0)+(x<0)/8
	dB=[(A[-1]-y)*dR(A[-1])]
	for i in range(len(A)-2):dB.append(M[0][-i-1].T@dB[i]*dR(A[-i-2]))
	dB=dB[::-1]
	dW=[b@a.T for a,b in zip(A[:-1],dB)]
	return [dW,dB]
#train nn
def Train(M,x,y,rate,loops):
	Z=M
	D=lambda x,y:[a-rate*b for a,b in zip(x,y)]
	for z in range(loops):
		print(z)
		for i in range(len(x)):
			dM=bprop(M,x[i],y[i])
			Z[0]=D(Z[0],dM[0])
			Z[1]=D(Z[1],dM[1])
		M=Z
	return M

#make nn with specific nodes
M=gen([2,4,4,1])
#dataset desctibes OR operator
x=[np.array([[0],[0]]),np.array([[0],[1]]),np.array([[1],[0]]),np.array([[1],[1]])]
y=[np.array([0]),np.array([1]),np.array([1]),np.array([1])]
#set training specifications
Train(M,x,y,.1,1600)

#print final nn output
print(N(M,np.array([[0],[0]]))[-1])
print(N(M,np.array([[0],[1]]))[-1])
print(N(M,np.array([[1],[0]]))[-1])
print(N(M,np.array([[1],[1]]))[-1])
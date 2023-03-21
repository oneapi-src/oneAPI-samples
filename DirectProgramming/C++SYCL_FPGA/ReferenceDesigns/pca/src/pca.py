import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(51)
passedCount = 0
totalTests = 10

n = 5000 #number of samples 
p = 64 #number of features 
PComps = min(1, int(p*0.1)) #number of primary components, first 10% values will be prominent 


for test in range(totalTests):
	
	print("testing input: " + str(test))

	# target eigen vectors should be orthogonal 
	# to each other. Here we generate random vectors
	# and orthogonal vectors are obtained through QR decomposition
	# setting initial vectors

	Vecs = np.random.rand(p,p)
	q, r = np.linalg.qr(Vecs)

	# columns of q are orthogonal to each other and unit vectors
	# transpose to make rows are orthogonal to each other
	q1 = q.reshape(p*p)
	q = np.transpose(q)


	Lamdas = np.zeros(p);
	# setting the eigen values proportions 
	Lamdas[0:PComps] = [100/PComps + 100/PComps*i for i in range(PComps)]
	Lamdas[PComps:p] = abs(np.random.rand(p-PComps))
	Lamdas = abs(Lamdas)

	# soerting the eigen values (principal components) 
	lam_index = Lamdas.argsort()[::-1]
	Lamdas = Lamdas[lam_index]


	A = np.zeros((n,p))


	# generating the samples through 
	# scaled addition of orthogonal target vectors
	# variation of the scale is within the eigen value  
	for i in range(n):
		vec = np.zeros(p)
		r = np.random.rand(p)
		for j in range(p):
			vec += r[j]*Lamdas[j]*q[j]
		A[i] = vec



	# removing the mean feature 
	A_mean = A - A.mean(0)

	# Standerdization of the data modify eigen vectors
	# Hence input to PCA is not standerdized data
	# for i in range(p):
	# 	A_mean[:,i] /= np.std(A_mean[:,i])



	# Getting the principal components through the sklearn library PCA
	pca=PCA(n_components=p, svd_solver='full')
	pca.fit(A)


	# taking dot product between target and calculated eigen vectors
	# to find similarity between them. Absolute value of dot product 
	# close to one means they are similar and close to zero means
	# they are totally different 

	dotProds = [q[i]@np.transpose(pca.components_[i]) for i in range(PComps)]
	dotProds = np.absolute(dotProds)

	# checking if any of the primary componets vectors significantly deviated from 
	# target vectors 
	if(min(dotProds) < 0.95):
		print("mismatch between target and calculated in test: " + str(test) + " dot Product: " + str(min(dotProds)))
	else:
		passedCount = passedCount + 1


passedRate = passedCount/totalTests * 100
print("Passed rate is " + str(passedRate))



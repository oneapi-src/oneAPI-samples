import numpy as np 
import math
import scipy as sp 
from sklearn.decomposition import PCA
np.random.seed(100)


def hessenberg(A):
  n = A.shape[0]
  Q = np.eye(n)
  H = np.copy(A)

  for j in range(n-2):
    u = np.copy(H[j+1:n,j])
    u[0] = u[0] + np.sign(u[0])*np.linalg.norm(u)
    
    v = np.copy(u/np.linalg.norm(u)).reshape(n-1-j,1)
    H[j+1:n,:] -=  2*v@(np.transpose(v)@H[j+1:n,:])
    H[:,j+1:n] -= (H[:,j+1:n] @ (2*v)) @ np.transpose(v)
    Q[:,j+1:n] -= (Q[:,j+1:n] @ (2*v)) @ np.transpose(v)

  return H, Q





def hessqr_basic(Hin, Eig, n):
  H = Hin[0:n,0:n]
  n = H.shape[0]
  V = np.zeros((2,(n-1)*2))

# QR comuation
  for j in range(n-1):
    u = np.copy(H[j:j+2, j])
    c = u[0]/np.linalg.norm(u)
    s = u[1]/np.linalg.norm(u)

    G = np.array([[c,s], [-s, c]])
    H[j:j+2,:]  = G@H[j:j+2,:] 
    V[:, j*2:(j+1)*2] = G


    Eig[:,j:j+2]  = Eig[:,j:j+2]@np.transpose(G)

# RQ computation 
  for j in range(n-1):
    G = V[:, j*2:(j+1)*2]
    G = np.transpose(G)
    H[:,j:j+2] = H[:,j:j+2] @G

  H = np.tril(H, k=1)
  H = np.triu(H, k=-1)

  Hin[0:n,0:n] = H
  return Hin, Eig


def iterative_qrdH(C):
  # C = C.astype('float32')
  size=C.shape[0]
  
  Eig = np.eye(size)

  # H, Q = sp.linalg.hessenberg(C,  calc_q=True)
  H, Q = hessenberg(C)

  H = np.tril(H, k=1)
  H = np.triu(H, k=-1)

  # total iteration counter
  counter = 0

  # print(H)
  # QR with shift 
  threshold = 1e-6;
  ind = size
  for itr in range(size):
    for i in range(200):
      counter = counter + 1
      # Wilkinson shift
      a = H[ind-2,ind-2]
      b = H[ind-2,ind-1]
      c = H[ind-1,ind-1]
      lamda =  (c-a)/2
      mu = c - (np.sign(lamda) * b*b)/(abs(lamda) + np.sqrt(lamda*lamda+b*b))
      
      H[0:ind,0:ind] -= np.eye(ind)*mu
      H, Eig = hessqr_basic(H, Eig, ind)
      H[0:ind,0:ind] += np.eye(ind)*mu

      if(abs(H[ind-1,ind-2]) < threshold):
        # print("element at: "+str(ind) + " has became zero at internal iter: " + str(i))
        # print(H)
        ind = ind -1
        break
    if(ind <= 2):
      break

    # print(H)

  print("Total number of QR iteration is: " + str(counter))
  return H, Q@Eig
  # return H, Q@Eig


# Testing the shift based Hessenberg QR iteration

print("Eigen vector and vaues for custom matrix")

C = [[147, 686, 710, 771, 417, 363],
[686, 709, 679, 850, 229, 350],
[710, 679, 177, 663, 588, 14],
[771, 850, 663, 351, 771, 792],
[417, 229, 588, 771, 572, 277],
[363, 350, 14, 792, 277, 471]]

C = [[147, 686, 710, 771, 417, 363, 141, 120, 362, 263, 406, 541],
[686, 709, 679, 850, 229, 350, 932, 538, 465, 204, 280, 383],
[710, 679, 177, 663, 588, 14, 352, 757, 473, 166, 719, 262],
[771, 850, 663, 351, 771, 792, 114, 266, 25, 276, 642, 732],
[417, 229, 588, 771, 572, 277, 188, 943, 535, 714, 118, 127],
[363, 350, 14, 792, 277, 471, 902, 185, 608, 442, 976, 515],
[141, 932, 352, 114, 188, 902, 237, 975, 972, 466, 119, 59],
[120, 538, 757, 266, 943, 185, 975, 622, 841, 574, 384, 817],
[362, 465, 473, 25, 535, 608, 972, 841, 98, 374, 919, 356],
[263, 204, 166, 276, 714, 442, 466, 574, 374, 817, 303, 156],
[406, 280, 719, 642, 118, 976, 119, 384, 919, 303, 78, 80],
[541, 383, 262, 732, 127, 515, 59, 817, 356, 156, 80, 559],
]

# np.set_printoptions(precision=3, threshold=5, edgeitems=4, suppress=True)
C =np.array(C).reshape(12,12)
C = C.astype('float64')

# applying Hessenberg Transform 
H_eig_val, H_eig_vec = iterative_qrdH(C)
eig_val = np.array([H_eig_val[i,i] for i in range(C.shape[0])])
abs_eig_val = [abs(eig_val[i]) for i in range(C.shape[0])]
H_sort_index = np.array(abs_eig_val).argsort()[::-1]

print("Hess QR eigen values: ")
print(eig_val[H_sort_index])

print("H_Q after QR iteration: ")
print(np.transpose(H_eig_vec)[H_sort_index])


w,v = np.linalg.eig(C)
w_abs =np.array([abs(w[i]) for i in range(w.shape[0])])
w_sort_index = w_abs.argsort()[::-1]
w = w[w_sort_index]
v = np.transpose(v)
v = v[w_sort_index]

print("\nnumpy eigen values ")
print(w)

print("\nnumpy eigen vectors")
print(v)


# pca=PCA(n_components=C.shape[0], svd_solver='full')
# pca.fit(C)
# # print("Eigen vectors are")
# print(pca.components_)
import numpy as np 
import math
import sys

np.random.seed(100)

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

if __name__ == "__main__":

    threshold = 1e-4

    Matrices = int(sys.argv[1])
    N = int(sys.argv[2])

    # Testing the shift based Hessenberg QR iteration

    # print("Eigen vector and vaues for custom matrix")
    C=[]
    with open('../build/mat_A.txt', 'r') as infile:
        mat_A = infile.read()
        C = [float(i) for i in mat_A.split(',') if mat_A.strip()]

    # N = int(math.sqrt(len(C)))
    # print('Python: size of the array is: ' + str(N))


    C =np.array(C).reshape(Matrices,N, N)
    C = C.astype('float64')

    w_str = ""
    v_str = ""
    for i in range(Matrices):
        if(i == 189):
            print("input matrix is:\n")
            print(C[i, :, :])
        if(not is_invertible(C[i, :, :])):
            print("This matrix is singular")

        w,v = np.linalg.eig(C[i, :, :])
        if(i == 189):
            print(w)

        w_abs =np.array([abs(w[i]) for i in range(w.shape[0])])
        w_sort_index = w_abs.argsort()[::-1]

        w = w[w_sort_index]
        v = np.transpose(v)
        v = v[w_sort_index]



        v_list =  list(v.reshape(N*N))
        w_list = list(w)

        w_str = w_str + " " + ' '.join(map(str, w_list))
        v_str = v_str + " " + ' '.join(map(str, v_list))

    with open('../build/mat_W.txt', 'w') as Wfile:
        Wfile.write(w_str)

    with open('../build/mat_V.txt', 'w') as Vfile:
        Vfile.write(v_str)

import numpy as np 
import math
import sys

np.random.seed(100)
eigVal_threhhold = 1e-2
SHIFT_NOISE = 1e-2
KTHRESHOLD = 1e-5
KDEFLIM = 15

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]



# shift based iterative QRD 
def iterative_qrd(C):
    size=C.shape[0]
    Eig = np.eye(size)

    # total iteration counter
    counter = 0
    C1 = np.copy(C)
    all_zero = 0

    RQ_str = ""
    QQ_str = ""
    Q_str = ""
    R_str = ""
    A_str = ""
  
    # QR with shift 
    ind = size
    for itr in range(size):
        for i in range(2000):
            H = C1[0:ind,0:ind]

            A_str += "A Matrix at iteration: " + str(counter) + "\n"
            A_str += str(H) + "\n\n"
            
            # Wilkinson shift
            mu = H[ind-1,ind-1]
            mu -= mu*SHIFT_NOISE
            
            H -= np.eye(ind)*mu
            q, r = np.linalg.qr(H)

            H = np.matmul(r,q)
            H += np.eye(ind)*mu

            Q = np.eye(size)
            Q[0:ind,0:ind] = q
            Eig = np.matmul(Eig, Q)
            C1[0:ind,0:ind] = H

            RQ_str += "RQ Matrix at iteration: " + str(counter) + "\n"
            RQ_str += str(C1) + "\n\n"

            QQ_str += "QQ Matrix at iteration: " + str(counter) + "\n"
            QQ_str += str(Eig) + "\n\n"

            Q_str += "Q Matrix at iteration: " + str(counter) + "\n"
            Q_str += str(q) + "\n\n"

            R_str += "R Matrix at iteration: " + str(counter) + "\n"
            R_str += str(r) + "\n\n"

            counter = counter + 1

            all_zero = 1
            for k in range(ind-1):
                if(abs(H[ind-1][k]) > KTHRESHOLD):
                    all_zero = 0
                    break

            if(all_zero):
                ind = ind -1
                break
            

        if(ind == KDEFLIM):
            print("Total number of Shift QR iteration is: " + str(counter))
            break
    
    with open('../build/DebugNP_RQ.txt', 'w') as RQfile:
        RQfile.write(RQ_str)
    with open('../build/DebugNP_QQ.txt', 'w') as QQfile:
        QQfile.write(QQ_str)
    with open('../build/DebugNP_Q.txt', 'w') as Qfile:
        Qfile.write(Q_str)
    with open('../build/DebugNP_R.txt', 'w') as Rfile:
        Rfile.write(R_str)
    with open('../build/DebugNP_A.txt', 'w') as Afile:
        Afile.write(A_str)

    return C1, Eig


# shift based iterative QRD 
def iterative_qrd_nodebug(C):
    size=C.shape[0]
    Eig = np.eye(size)

    # total iteration counter
    counter = 0
    C1 = np.copy(C)
    all_zero = 0
  
    # QR with shift 
    ind = size
    for itr in range(size):
        for i in range(2000):
            H = C1[0:ind,0:ind]
            # Wilkinson shift
            mu = H[ind-1,ind-1]
            mu -= mu*SHIFT_NOISE
            
            H -= np.eye(ind)*mu
            q, r = np.linalg.qr(H)

            H = np.matmul(r,q)
            H += np.eye(ind)*mu

            Q = np.eye(size)
            Q[0:ind,0:ind] = q
            Eig = np.matmul(Eig, Q)
            C1[0:ind,0:ind] = H

            counter = counter + 1

            all_zero = 1
            for k in range(ind-1):
                if(abs(H[ind-1][k]) > KTHRESHOLD):
                    all_zero = 0
                    break

            if(all_zero):
                ind = ind -1
                break
            

        if(ind == KDEFLIM):
            break

    w = [C1[i][i] for i in range(size)]
    w = np.array(w)
    
    return w, Eig


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
    SmallE_str = ""
    for i in range(Matrices):
        if(i == 921):
            iterative_qrd(C[i, :, :])
            print("input matrix is:\n")
            print(C[i, :, :])
        if(not is_invertible(C[i, :, :])):
            print("This matrix is singular")

        w,v = iterative_qrd_nodebug(C[i, :, :])
        # w,v = np.linalg.eigh(C[i, :, :])
        if(i == 921):
            print(w)
            print(v)


        w_abs =np.array([abs(w[i]) for i in range(w.shape[0])])
        w_sort_index = w_abs.argsort()[::-1]

        w = w[w_sort_index]
        v = np.transpose(v)
        v = v[w_sort_index]

        if(abs(w[N-1]) < eigVal_threhhold):
            SmallE_str += "small eigen value is found at Matrix: " + str(i) + "\n"
            SmallE_str += str(w) + "\n\n\n"



        v_list =  list(v.reshape(N*N))
        w_list = list(w)

        w_str = w_str + " " + ' '.join(map(str, w_list))
        v_str = v_str + " " + ' '.join(map(str, v_list))

    with open('../build/mat_W.txt', 'w') as Wfile:
        Wfile.write(w_str)

    with open('../build/mat_V.txt', 'w') as Vfile:
        Vfile.write(v_str)

    with open('../build/smalE.txt', 'w') as smallEfile:
        smallEfile.write(SmallE_str)

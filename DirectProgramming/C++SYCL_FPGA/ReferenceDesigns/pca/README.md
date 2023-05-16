# PCA Principal Component Analysis of Samples
This reference design demonstrates PCA imlementation for real matrices on FPGA.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br>RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware                          | Intel&reg; Programmable Acceleration Card (PAC) with Intel Arria&reg; 10 GX FPGA <br> Intel&reg; FPGA Programmable Acceleration Card (PAC) D5005 (with Intel Stratix&reg; 10 SX) <br> Intel Xeon&reg; CPU E5-1650 v2 @ 3.50GHz (host machine)
| Software                          | Intel&reg; oneAPI DPC++ Compiler <br> Intel&reg; FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Implementing a FPGA version of the PCA using iterative QRD algorithm.
| Time to complete                  | 1 hr (not including compile time)


**Performance** 

Please refer to the performance disclaimer at the end of this README.

| Device                                         | Throughput
|:---                                            |:---
| Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA        | ?? matrices/s for complex matrices of size 128 * 128
| Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX)      | ?? matrices/s for complex matrices of size 256 * 256


## Purpose


Real-world datasets typically consist of multiple features or dimensions. However, not all features within the samples contain significant information, and some may be nearly identical across the dataset. In order to facilitate analysis, visualization, and storage of the data, reducing the number of features while maintaining the majority of the information is crucial. The Principal Component Analysis (PCA) approach identifies such principal features, often combinations of the original features, in descending order based on their contribution to the information within the samples.     

## Algorithm
The present FPGA reference design showcases the application of Principal Component Analysis (PCA), a fundamental operation in linear algebra, on input samples comprising a specified number of features. The design decomposes the input matrix, denoted as _A_, into Principal Components in a descending order and further provides the variance explained by each component.



The current design utilizes the subsequent algorithm to compute Principal Components. The input matrix comprises a two-dimensional array with the dimensions A\[N\]\[P\] . In this context, $N$ represents the number of samples, and $P$ corresponds to the number of features.

<!-- <br />  -->
1. Computing mean feature of the samples $$F_{\mu}\[i\] = \frac{1}{N} \sum_{j = 0}^{N-1} A\[j\]\[i\] \tag{1}$$
2. Adjusitng the sample mean to zero  $$A_{\mu}\[i\]\[j\] = A\[i\]\[j\] -  F_{\mu}\[j\] \tag{2}$$
3. Standardize data such that variance of sample will be one $$F_{var}\[i\] = \frac{1}{N} \sum_{j = 0}^{N-1} {(A_{\mu}\[j\]\[i\])}^2 \tag{3}$$ $$A_{std}\[i\]\[j\] = \frac{A_{\mu}\[i\]\[j\]}{\sqrt{F_{var}\[i\]}} \tag{4}$$
4. Computing the Covariance Matrix of size $p \times p$,  $$A_{StdCov}\[i\]\[j\] = \sum_{k = 0}^{N-1}{A_{std}\[i\]\[k\] \times A_{std}\[j\]\[k\] } \tag{5}$$
5. The next procedure involves computing the Eigenvalues and their corresponding Eigenvectors of the covariance matrix, denoted as $A_{cov}$. The Eigenvectors are then sorted in a descending order based on their corresponding Eigenvalues. The Eigenvectors are identified as Principal Components, whereas the corresponding Eigenvalues reveal the variance attributed to each Principal Component.

## Reference Design 
This design executes the PCA analysis through two kernels 
* kernel:1 does the preprocessing steps, essentially steps 1 to 4
* kernel:2 impements the step 5, computing eigen values and eigen vectors then sorting  

## kenrel:1 preprocessing and Covariance Matrix Computation
Executing the steps 1-4, one after another is not efficent as it is impossible to store the whole input samples in onchip memory if sample size is huge. 
steps 1-4 are modified and reordered such that covariance matrix in the step 4 can be computed for inputs given through the stream.  
### Modified Variance Computation 
* Standard way to compute variance  $$F_{var}\[i\] = \frac{1}{N} \sum_{j = 0}^{N-1} {(A\[j\]\[i\] -  F_{\mu}\[i\])}^2 \tag{6}$$
* Expanding the sum expression $$F_{var}\[i\] = \frac{1}{N} (\sum_{j = 0}^{N-1} {(A\[j\]\[i\])^2} - 2 \times F_{\mu}\[i\] \sum_{j = 0}^{N-1} {A\[j\]\[i\]} + N \times F_{\mu}\[i\] \times  F_{\mu}\[i\]) \tag{7}$$
* Reducing it to $$F_{var}\[i\] = \frac{1}{N} (\sum_{j = 0}^{N-1} {(A\[j\]\[i\])^2} -  N \times F_{\mu}\[i\] \times  F_{\mu}\[i\]) \tag{8}$$

### Modified Co-variance Matrix Computation 
* Step 4 can be re-written as follows $$A_{StdCov}\[i\]\[j\] = \frac{1}{\sqrt{F_{var}\[i\] \times F_{var}\[j\]}} \sum_{k = 0}^{N-1}{(A\[k\]\[i\] - F_{\mu}\[i\]) \times (A\[k\]\[j\] - F_{\mu}\[j\]) } \tag{9}$$
* It can be expanded as follows $$A_{cov}\[i\]\[j\] = \frac{1}{\sqrt{F_{var}\[i\] \times F_{var}\[j\]}} (\sum_{k = 0}^{N-1}{A\[k\]\[i\] \times A\[k\]\[j\]  - F_{\mu}\[i\] \sum_{k = 0}^{N-1} A\[k\]\[j\] - F_{\mu}\[j\] \sum_{k = 0}^{N-1} A\[k\]\[i\]  + N \times F_{\mu}\[i\] \times F_{\mu}\[j\]) } \tag{10}$$
* Reduced to $$A_{StdCov}\[i\]\[j\] = \frac{1}{\sqrt{F_{var}\[i\] \times F_{var}\[j\]}} (\sum_{k = 0}^{N-1}{A\[k\]\[i\] \times A\[k\]\[j\]  - N \times F_{\mu}\[i\] \times F_{\mu}\[j\]) }$$
* Lets Assume $$B\[i\]\[j\] = \sum_{k = 0}^{N-1}{A\[k\]\[i\] \times A\[k\]\[j\]} \tag{11}$$
* Variance can be re-written as $$F_{var}\[i\] = \frac{1}{N} (B\[i\]\[i\] -  N \times F_{\mu}\[i\] \times  F_{\mu}\[i\]) \tag{12}$$
* Covariance Matrix after standardisation $$A_{StdCov}\[i\]\[j\] = \frac{1}{\sqrt{F_{var}\[i\] \times F_{var}\[j\]}} (A_{Cov}\[i\]\[j\]  - N \times F_{\mu}\[i\] \times F_{\mu}\[j\]) \tag{13}$$

### Architecture 
It is clear that, $A_{StdCov}\[i\]\[j\]$ can be computed by computing $B\[i\]\[j\]$ and $F_{\mu}\[i\]$. $A_{cov}\[i\]\[j\]$ can be computed as follows for 8 samples with 4 features.

$$ \begin{bmatrix}
A_{0,0} & A_{1,0} & A_{2,0} & A_{3,0} & A_{4,0} & A_{5,0} & A_{6,0} & A_{7,0} \\
A_{0,1} & A_{1,1} & A_{2,1} & A_{3,1} & A_{4,1} & A_{5,1} & A_{6,1} & A_{7,1} \\
A_{0,2} & A_{1,2} & A_{2,2} & A_{3,2} & A_{4,2} & A_{5,2} & A_{6,2} & A_{7,2} \\
A_{0,3} & A_{1,3} & A_{2,3} & A_{3,3} & A_{4,3} & A_{5,3} & A_{6,3} & A_{7,3} 
\end{bmatrix} \times \begin{bmatrix}
A_{0,0} & A_{0,1} & A_{0,2} & A_{0,3} \\
A_{1,0} & A_{1,1} & A_{1,2} & A_{1,3} \\
A_{2,0} & A_{2,1} & A_{2,2} & A_{2,3} \\
A_{3,0} & A_{3,1} & A_{3,2} & A_{3,3} \\
A_{4,0} & A_{4,1} & A_{4,2} & A_{4,3} \\
A_{5,0} & A_{5,1} & A_{5,2} & A_{5,3} \\
A_{6,0} & A_{6,1} & A_{6,2} & A_{6,3} \\
A_{7,0} & A_{7,1} & A_{7,2} & A_{7,3} 
\end{bmatrix} = \begin{bmatrix}
B_{0,0} & B_{0,1} & B_{0,2} & B_{0,3} \\
B_{1,0} & B_{1,1} & B_{1,2} & B_{1,3} \\
B_{2,0} & B_{2,1} & B_{2,2} & B_{2,3} \\
B_{3,0} & B_{3,1} & B_{3,2} & B_{3,3} \\
\end{bmatrix} $$


This reference design employs blocked covariance matrix computation to support larger sample sizes. $p \times p$ block from input ( _A_) is multiplied with it's transpose and added to accumulator block. In this way this design is scalable to any sample size. 

$$ \begin{bmatrix}
A_{0,0} & A_{1,0} & A_{2,0} & A_{3,0} & \- & \- & \- & \- \\
A_{0,1} & A_{1,1} & A_{2,1} & A_{3,1} & \- & \- & \- & \- \\
A_{0,2} & A_{1,2} & A_{2,2} & A_{3,2} & \- & \- & \- & \- \\
A_{0,3} & A_{1,3} & A_{2,3} & A_{3,3} & \- & \- & \- & \-
\end{bmatrix} \times \begin{bmatrix}
A_{0,0} & A_{0,1} & A_{0,2} & A_{0,3} \\
A_{1,0} & A_{1,1} & A_{1,2} & A_{1,3} \\
A_{2,0} & A_{2,1} & A_{2,2} & A_{2,3} \\
A_{3,0} & A_{3,1} & A_{3,2} & A_{3,3} \\
\- & \- & \- & \- \\
\- & \- & \- & \- \\
\- & \- & \- & \- \\
\- & \- & \- & \- 
\end{bmatrix} + \begin{bmatrix}
ACC_{0,0} & ACC_{0,1} & ACC_{0,2} & ACC_{0,3} \\
ACC_{1,0} & ACC_{1,1} & ACC_{1,2} & ACC_{1,3} \\
ACC_{2,0} & ACC_{2,1} & ACC_{2,2} & ACC_{2,3} \\
ACC_{3,0} & ACC_{3,1} & ACC_{3,2} & ACC_{3,3}
\end{bmatrix} $$

* double buffering/ ping pong buffering is employed to hide the time to load $p \times p$ block from input stream 
* Storage for $p \times p$ block is partioned $p$ times to enable one full dot product 
* $F_{\mu}$ is computed by computing average sum of features while computing the block matrix multiplication 
* $A_{StdCov}\[i\]\[j\]$ using  $ACC\[i\]\[j\]$ and $F_{\mu}\[j\]$
* Its designed such that it can process any number of input matrices 

### Latency Model
* Total Number of blocks in $T$ input matrices 
$$N_{blks} = \lceil{\frac{N}{p}} \rceil \times T$$
* Block Matrix multiplier is in the critical path and latency to process all blocks is 
$$Clks_{bmm} = \lceil{\frac{N}{p}} \rceil \times T \times p^{2}$$
* Total latency of kernel1 adding first loading of block from stream and outputting final covariance matrix, $V$ is nuber of elements streamed per clock, ignoring the pipeline latency/ flush time of each hardware modules  
$$Clks_{kernel1} = \frac{p}{V} \times p + \lceil{\frac{N}{p}} \rceil \times T \times p^{2} + p^{2}$$

## Kernel2: Eigen Value and Eigen Vector computation and sorting
### Eigen Value compuation 
 As $A_{cov}\[i\]\[j\]=A_{cov}\[j\]\[i\]$, $A_{StdCov}$ is a symmetric square matrix. A symmetric matrix will have real eigen values and eigen vectors, those can be calculated using iterative QR decomposition.   
 
 **Set** $C_{0}=A_{StdCov}$ <br /> 
 **Set** $k=0$ <br /> 
 **do** <br /> 
    &emsp; **QR Decomposition** $C_{k−1}=Q_{k}R_{k}$ <br /> 
    &emsp; Set $C_{k}=R_{k}Q_{k}$ <br /> 
    &emsp;  $k = k+1$ <br /> 
 **while** ($C$ converges)
<br /><br />
Upon achieving convergence in matrix $C$, the diagonal values of $C$ will signify the Eigenvalues. However, the primary limitation of this unsophisticated algorithm is that it necessitates an enormous number of iterations to attain convergence. To enhance convergence, the algorithm employs matrix shifts and deflation according to the following procedure:

 **Set** $C^{F}=A_{StdCov}$ <br /> 
**for** ($size_{C} = p$; $size_{C}  > 1$; $size_{C} =size_{C} -1$) **do** <br />
&emsp; **Set** $C_{0}\[i\]\[j\]=C^{F}\[i\]\[j\]$ &emsp; $i < size_{C} $, $j < size_{C}$ <br /> 
&emsp; **Set** $k=0$ <br /> 
 &emsp; **do** <br /> 
   &emsp; &emsp; $C_{k−1} = C_{k−1} - \mu I$ <br /> 
   &emsp; &emsp; **QR Decomposition** $C_{k−1}=Q_{k}R_{k}$ <br /> 
   &emsp; &emsp; Set $C_{k}=R_{k}Q_{k} + \mu I$ <br /> 
   &emsp; &emsp; $k = k+1$ <br /> 
&emsp; **while** ($C$ converges) <br />
&emsp; **Set** $C^{F}\[i\]\[j\]=C_{k-1}\[i\]\[j\]$ &emsp; $i < size_{C}$, $j < size_{C}$ <br /> 
**endfor** <br /> 

Above algorithm computes eigen values one by one and deflate the matrix once a eigen value has been computed. $size_{C}$ represent the dimension of deflated matrix. This algorithm converges much faster, requiring around 3 iteration to compute an eigen value compared to previous naive implementation. It is assumed that matrix is converged if values indicated by \* is less than zero threshold, then $C_{3,3}$ will be eigen a value. 

$$ \begin{bmatrix}
C_{0,0} & C_{1,0} & C_{2,0} & C_{3,0} \\
C_{0,1} & C_{1,1} & C_{2,1} & C_{3,1} \\
C_{0,2} & C_{1,2} & C_{2,2} & C_{3,2} \\
\*      & \*      & \*      & C_{3,3}
    \end{bmatrix} $$


There are two options to compute the shift value $\mu$, Rayleigh quotient shifts and Wilkinson shift. Rayleigh quotient shifts is equvalent to right bottom element($C\[size_{D}-1\]\[size_{D}-1\]$) of matrix _C_.  Wilkinson shift requires bottom right $2 \times 2$ sub-matrix to compute the shift value. 

$$  \begin{bmatrix}
x & x & x & x \\
x & x & x & x \\
x & x & a & b \\
x & x & b & c \\
    \end{bmatrix} $$ 

Wilkinson shift is given by following equation 
$$\mu = c - \frac{sign(\delta) \times b^{2}}{|\delta| + \sqrt{\delta^{2} + b^{2}}}$$

Rayleigh quotient shifts based QR iteration is not always stable but Wilkinson shift is highly stable, when using double preession arithmetic. Downside is Wilkinson shift requires costly hardware IPs such as divider, sqrt and reguires many pipeline stages, leads to higher latency. This reference design target to use floating point arithmetic (It supports anytype throgh SYCL template). It is observed that above agorithm will become numerically unstable when floating point arithmetic is used (due to floating point cancellation and errors propagate from divider in QR decomposition). In order to improve the the numerical accuaracy, we assign 99.9% of Rayleigh quotient shifts as $\mu$. This avoids the diagonal values become zero even other column elements becomes zero during QR iterations.  

### Eigen vector computation 
The Eigen vectors ($E_{vec}$) computed by compounding the $Q$ matrix computed from the QR decomposition in each QR iteration as follows. <br /> 
 **Set** $C_{0}=A_{StdCov}$ <br /> 
 **Set** $k=0$ <br /> 
  **Set** $E_{vec}=I$ <br /> 
 **do** <br /> 
    &emsp; **QR Decomposition** $C_{k−1}=Q_{k}R_{k}$ <br /> 
    &emsp; Set $C_{k}=R_{k}Q{k}$ <br /> 
    &emsp;  $E_{vec} = E_{vec} Q$ <br /> 
    &emsp;  $k = k+1$ <br /> 
 **while** ($C$ converges)
<br /><br />

In the version that does shift and deflation, $Q$ will be made to $p \times p$ size by making rest of the diagonals to one and other elements left to zero as follows. 


$$ \begin{bmatrix}
Q_{0,0} & Q_{0,1} & Q_{0,2} \\
Q_{1,0} & Q_{1,1} & Q_{1,2} \\
Q_{2,0} & Q_{2,1} & Q_{2,2}
\end{bmatrix} -> \begin{bmatrix}
Q_{0,0} & Q_{0,1} & Q_{0,2} & 0 & 0 \\
Q_{1,0} & Q_{1,1} & Q_{1,2} & 0 & 0 \\
Q_{2,0} & Q_{2,1} & Q_{2,2} & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix} $$

### Implementation 
There are two main components in iterative QR loop, based on data dependency 
* $QR$ Matrix decomposition 
* $RQ$ Matrix multiplication and $E_{vec} Q$ Matrix multiplication 

In order to facilitate one full dot product computation in the above operations, memory for matrices are patitioned as follows 
* Input to and output from, QR decomposition is organized column wise and partioned column wise. An entire column can be loaded each clock cycle

$$ \begin{bmatrix}
\* & X & \. \\
\* & X & \. \\
\* & X & \.
\end{bmatrix} $$

* $R$ and $E_{vec}$ memories are partitioned row wise. An entire row can be loaded each clock cycle

$$ \begin{bmatrix}
\* & \* & \* \\
X & X & X \\
\. & \. & \.
\end{bmatrix} $$

#### Modified QR decomposition 
This design utlize the oneAPI sample source [QRD](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL_FPGA/ReferenceDesigns/qrd) and modify it to support QR decomposition of deflated matrices during the QR decomposition. The static design is made such that it can process big input matrix using one full dot product in modified gram schimdt algorithm. 

$$ \begin{bmatrix}
C_{0,0} & C_{1,0} & C_{2,0} & C_{3,0} \\
C_{0,1} & C_{1,1} & C_{2,1} & C_{3,1} \\
C_{0,2} & C_{1,2} & C_{2,2} & C_{3,2} \\
C_{0,3} & C_{1,3} & C_{2,3} & C_{3,3}
\end{bmatrix} -> \begin{bmatrix}
C_{0,0} & C_{1,0} & C_{2,0} & 0 \\
C_{0,1} & C_{1,1} & C_{2,1} & 0 \\
C_{0,2} & C_{1,2} & C_{2,2} & 0 \\
0 & 0 & 0 & 0
\end{bmatrix} $$
 
Above example illustrate when deflating 4x4 matrix into 3x3, when doing QR decomposision, elements outside 3x3 matrix will be re-interpreted as zero and only 3x3 matrix element will be updated after QR decomposition. Shift value is subtracted when loading the diagonal values on the go. 

- QR Decompostion latency model

$$ VecDepth = 8 + (4 + 3 \lceil log_{2}(p) \rceil) + 15 $$

$$ Clks_{QRD} = \sum_{i=1}^{p}{max(i,vecDepth)} $$


#### Fused $RQ$ and $E_{vec} Q$ computation 
In HLS, each loops is scheduled one after another, computing the $RQ$ and $E_{vec} Q$  matrix multiplication in separate loops will increase the latency. In this implementation both $RQ$ and $E_{vec} Q$ are computed in a single nested loop, using one full dot product for each operation. Similar to QR decompostion, masking is applied when computing $RQ$ in deflated matrices. 

While computing the $RQ$, subtracted shift is added back and the new shift value for next iteration is stored in a register. Logic to check the convergence is also implemented in this loop and boolen outcome of convergence is stored in register. This register is checked at the end of the iteration to deflate the matrix or exist the computation if 1x1 deflated matrix is reached. Further, a debug logic to detect the $QR$ decomposition failure by inspecting the orthogonolity of $Q$ matrix is implemented in this nested loop. This also require a full dot prduct to check all possible combination of vectors. 

- Fused Matrix multiplication latency model ignoring small pipeline latency

$$ Clks_{FMM} = p^{2} $$

#### Sorting 
After the QRD iteration, eigen values and eigen vectors need to be sorted. This reference design implements the selection sort, searching through all the elements and finding the next maimum element. A new memory is used to store the indexes of sorted elements, in order to elminate the latency for swapping elements. Additionally a register block is used as mask avoid checking the already sorted elements. 

- Sorting latency model ignoring small pipeline latency

$$ Clks_{Sort} = 2 \times p^{2} $$


## Hessenberg based Eigen Vector and Eigen value computation 
The standard QR iteration is $O(n^{3})$ complexity and it is reduced to $O(n^{2})$ complexity through one full dot product oneAPI qrd implementation. The time complexit can be further reduced to $O(n)$ QR iteration and resources coule be saved as through the Hessenberg transform. It is noted that Hessenberg transform comes with $O(n^{3})$ complexity but it is required only once. QR iteration is done around $3 \times n$ for a shift based QR iteration. Hessenberg QR iteration could potentially save the number of clock cycles for QR iteration for reasonably larger matrices. 


Hessenberg transform preserves eigen values, eigen vectors and symmetry of the input matrix. As such, it can be proven, hessenberg transform of symmetric matrix will be tridiagonal matrix, requires significantly less storage 

$$ \begin{bmatrix}
C_{0,0} & C_{0,1} & C_{0,2} & C_{0,3} \\
C_{0,1} & C_{1,1} & C_{1,2} & C_{1,3} \\
C_{0,2} & C_{1,2} & C_{2,2} & C_{2,3} \\
C_{0,3} & C_{1,3} & C_{2,3} & C_{3,3}
\end{bmatrix} -> \begin{bmatrix}
H_{0,0} & H_{0,1} & 0 & 0 \\
H_{0,1} & H_{1,1} & H_{1,2} & 0 \\
0 & H_{1,2} & H_{2,2} & H_{2,3} \\
0 & 0 & H_{2,3} & H_{3,3}
\end{bmatrix} $$

Hessenber QR iteration consists of two parts
* Forward iteration - Computes the _R_ matrix 
* Backward iteration - Computes the _RQ_ matrix 

### Forward Iteration 

$$ \begin{bmatrix}
H_{0,0} & H_{0,1} & 0 & 0 \\
H_{0,1} & H_{1,1} & H_{1,2} & 0 \\
0 & H_{1,2} & H_{2,2} & H_{2,3} \\
0 & 0 & H_{2,3} & H_{3,3}
\end{bmatrix} -> \begin{bmatrix}
h_{0,0}^{1} & h_{1,0}^{1} & h_{2,0}^{1} & 0 \\
0 & h_{1,1}^{1} & h_{2,1}^{1} & 0 \\
0 & h_{1,2}^{1} & h_{2,2}^{1} & h_{3,2}^{1} \\
0 & 0 & h_{2,3}^{1} & h_{3,3}^{1}
\end{bmatrix} -> \begin{bmatrix}
h_{0,0}^{2} & h_{1,0}^{2} & h_{2,0}^{2} & 0 \\
0 & h_{1,1}^{2} & h_{2,1}^{2} & h_{3,1}^{2} \\
0 & 0 & h_{2,2}^{2} & h_{3,2}^{2} \\
0 & 0 & h_{2,3}^{2} & h_{3,3}^{2}
\end{bmatrix} -> \begin{bmatrix}
h_{0,0}^{3} & h_{1,0}^{3} & h_{2,0}^{3} & 0 \\
0 & h_{1,1}^{3} & h_{2,1}^{3} & h_{3,1}^{3} \\
0 & 0 & h_{2,2}^{3} & h_{3,2}^{3} \\
0 & 0 & 0 & h_{3,3}^{3}
\end{bmatrix} $$

After the forward iteration, lower diagonal part will become zero. This is done through given rotations as follows 

$$ \begin{bmatrix}
c_{1} & s_{1} & 0 & 0 \\
-s_{1} & c_{1} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \times \begin{bmatrix}
h_{0,0} & H_{0,1} & 0 & 0 \\
H_{0,1} & H_{1,1} & H_{1,2} & 0 \\
0 & H_{1,2} & H_{2,2} & H_{2,3} \\
0 & 0 & H_{2,3} & H_{3,3}
\end{bmatrix} = \begin{bmatrix}
h_{0,0}^{1} & h_{1,0}^{1} & h_{2,0}^{1} & 0 \\
0 & h_{1,1}^{1} & h_{2,1}^{1} & 0 \\
0 & h_{1,2}^{1} & h_{2,2}^{1} & h_{3,2}^{1} \\
0 & 0 & h_{2,3}^{1} & h_{3,3}^{1}
\end{bmatrix} $$

Here 

$$ c_{1} = \frac{H_{0,0}}{\sqrt{H_{0,0} \times H_{0,0}  + H_{0,1} \times H_{0,1}}} $$ 

$$ s_{1} = \frac{H_{0,1}}{\sqrt{H_{0,0} \times H_{0,0}  + H_{0,1} \times H_{0,1}}} $$ 

Similarly, 

$$ \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & c_{2} & s_{2} & 0 \\
0 & -s_{2} & c_{2} & 0  \\
0 & 0 & 0 & 1
\end{bmatrix} \times \begin{bmatrix}
h_{0,0}^{1} & h_{1,0}^{1} & h_{2,0}^{1} & 0 \\
0 & h_{1,1}^{1} & h_{2,1}^{1} & 0 \\
0 & h_{1,2}^{1} & h_{2,2}^{1} & h_{3,2}^{1} \\
0 & 0 & h_{2,3}^{1} & h_{3,3}^{1}
\end{bmatrix} = \begin{bmatrix}
h_{0,0}^{2} & h_{1,0}^{2} & h_{2,0}^{2} & 0 \\
0 & h_{1,1}^{2} & h_{2,1}^{2} & h_{3,1}^{2} \\
0 & 0 & h_{2,2}^{2} & h_{3,2}^{2} \\
0 & 0 & h_{2,3}^{2} & h_{3,3}^{2}
\end{bmatrix} $$

Here 

$$ c_{2} = \frac{h_{1,1}^{1}}{\sqrt{h_{1,1}^{1} \times h_{1,1}^{1}  + h_{1,2}^{1} \times h_{1,2}^{1}}} $$ 

$$ s_{2} = \frac{h_{1,2}^{1}}{\sqrt{h_{1,1}^{1} \times h_{1,1}^{1}  + h_{1,2}^{1} \times h_{1,2}^{1}}}  $$ 


Coressponding Q matrix will be 

$$ Q = \begin{bmatrix}
c_{1} & -s_{1} & 0 & 0 \\
s_{1} & c_{1} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \times \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & c_{2} & -s_{2} & 0 \\
0 & s_{2} & c_{2} & 0  \\
0 & 0 & 0 & 1
\end{bmatrix} \times \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & c_{3} & -s_{3} \\
0 & 0 & s_{3} & c_{3}  \\
\end{bmatrix} $$


### Backward Iteration 
Backward Iteration computed the $RQ$ matrix, hence it can be written as 

$$ \begin{bmatrix}
h_{0,0}^{3} & h_{1,0}^{3} & h_{2,0}^{3} & 0 \\
0 & h_{1,1}^{3} & h_{2,1}^{3} & h_{3,1}^{3} \\
0 & 0 & h_{2,2}^{3} & h_{3,2}^{3} \\
0 & 0 & 0 & h_{3,3}^{3}
\end{bmatrix} \times \begin{bmatrix}
c_{1} & -s_{1} & 0 & 0 \\
s_{1} & c_{1} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \times \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & c_{2} & -s_{2} & 0 \\
0 & s_{2} & c_{2} & 0  \\
0 & 0 & 0 & 1
\end{bmatrix} \times \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & c_{3} & -s_{3} \\
0 & 0 & s_{3} & c_{3}  \\
\end{bmatrix} $$


### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.

### Matrix dimensions and FPGA resources

This design was tested on matrices upto 64 feature size on Arria 10 and design can support any number of sample size. 

## Key Implementation Details
| Kernel            | Description
---                 |---
| PCA               | Implements PCA through shift based iterative QRD algorithm.

To optimize the performance-critical loop in its algorithm, the design leverages concepts discussed in the following FPGA tutorials:
* **Triangular Loop Optimization**
* **Explicit Pipelining with `fpga_reg`** (fpga_register)
* **Loop `ivdep` Attribute** (loop_ivdep)
* **Unrolling Loops** (loop_unroll)

 The key optimization techniques used are as follows:
   1. Blocked Covariance matrix computation and out of order evaluation to support any sample size 
   2. QR decomposition on variable matrix sizes through masking 
   3. Fusing two matrix multilication in single nested loop

## Building the Reference Design

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see **Use the setvars Script** for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Code Samples in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, remember that you must specify the type of compute node and whether to run in batch or interactive mode. Compiles to FPGA are only supported on fpga_compile nodes. Executing programs on FPGA hardware is only supported on fpga_runtime nodes of the appropriate type, such as fpga_runtime:arria10 or fpga_runtime:stratix10.  Neither compiling nor executing programs on FPGA hardware are supported on the login nodes. For more information, see the Intel&reg; oneAPI Base Toolkit Get Started Guide ([https://devcloud.intel.com/oneapi/documentation/base-toolkit/](https://devcloud.intel.com/oneapi/documentation/base-toolkit/)).

When compiling for FPGA hardware, it is recommended to increase the job timeout to 24h.


### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the [Using Visual Studio Code with Intel&reg; oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System
1. Install the design into a directory `build` from the design directory by running `cmake`:

   ```
   mkdir build
   cd build
   ```

   If you are compiling for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:

   ```
   cmake ..
   ```

   If instead you are compiling for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following targets are provided, and they match the recommended development flow:

    * Compile for emulation (fast compile time, targets emulated FPGA device).

       ```
       make fpga_emu
       ```

    * Generate HTML performance report. Find the report in `qrd_report.prj/reports/report.html`directory.

       ```
       make report
       ```

    * Compile for FPGA hardware (longer compile time, targets FPGA device).

       ```
       make fpga
       ```

3. (Optional) As the above hardware compile may take several hours to complete, FPGA precompiled binaries (compatible with Linux* Ubuntu* 18.04) can be downloaded <a href="https://iotdk.intel.com/fpga-precompiled-binaries/latest/qrd.fpga.tar.gz" download>here</a>.

### On a Windows* System
1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA, run `cmake` using the command:
    ```
    cmake -G "NMake Makefiles" ..
   ```
   Alternatively, to compile for the Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX), run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:
   * Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report:
     ```
     nmake report
     ```
   * Compile for FPGA hardware (longer compile time, targets FPGA device):
     ```
     nmake fpga
     ```

*Note:* The Intel&reg; PAC with Intel Arria&reg; 10 GX FPGA and Intel&reg; FPGA PAC D5005 (with Intel Stratix&reg; 10 SX) do not yet support Windows*. Compiling to FPGA hardware on Windows* requires a third-party or custom Board Support Package (BSP) with Windows* support.<br>

*Note:* If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build. You can then run cmake from that directory, and provide cmake with the full path to your sample directory.


### Troubleshooting
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this Reference Design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [FPGA Workflows on Third-Party IDEs for Intel&reg; oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html)

## Running the Reference Design
You can perform the PCA on matrices repeatedly, as shown below. This step performs the following:
* Generates 8 random sample matrices.
* Computes the PCA of the 8 matrices.
* Repeats the decomposition multiple times (specified as a command line argument) to evaluate performance.


 1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
 Increase the amount of memory that the emulator runtime is permitted to allocate by setting the CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE environment variable before running the executable.
     ```
     export CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
     ./pca.fpga_emu           (Linux)

     set CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB
     qrd.fpga_emu.exe         (Windows)
     ```

2. Run the sample on the FPGA device.
     ```
     ./pca.fpga         (Linux)
     pca.fpga.exe       (Windows)
     ```
### Application Parameters

| Argument | Description
---        |---
| `<num>`  | Optional argument that specifies the number of times to repeat the PCA decomposition of 8 matrices. Its default value is `16` for the emulation flow and '819200' for the FPGA flow.


## Additional Design Information

### Design Parameters 

| Flag | Description
---    |---
`-RELSHIFT` | Switching between Rayleigh shift and Wilkinson shift
`-SHIFT_NOISE` | Percentage of shift value that is not subtracted from diagonals 


NOTE: The values for `seed`, `FIXED_ITERATIONS`, `ROWS_COMPONENT`, `COLS_COMPONENT` are set according to the board being targeted.

### Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [this page](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview).

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on July 29, 2020.

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

&copy; Intel Corporation.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
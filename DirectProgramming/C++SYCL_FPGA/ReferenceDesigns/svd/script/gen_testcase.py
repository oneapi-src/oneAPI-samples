import numpy as np 
import random
import time
import sys

def gen_matrix(rows, cols, rank, seed=11, min_val=0.0, max_val=1.0):
  # check input
  if (rank < 1) or (rank > cols):
    print("Rank of a matrix must be: column count >= rank >= 1")
    return None

  out_mat = np.zeros([rows, cols])
  dependent_cols = cols - rank
  random.seed(seed)
  # if randomly generated column happends to be dependent, try again
  for i in range(99):
    # fill matrix col by col
    for col in range(cols):
      # constant multipler for dependent col
      col_multipler = random.random()
      for row in range(rows):
        # generate independent columns (randomly)
        if (col < (cols - dependent_cols)):
          out_mat[row][col] = random.uniform(min_val, max_val)
        # generate dependent columns from col 0
        else:
          out_mat[row][col] = out_mat[row][0] * col_multipler
    if np.linalg.matrix_rank(out_mat) == rank:
      return out_mat
  return None
    
    
def gen_rank_deficient_matrices(rows, cols, matrix_count):
  matrices = []
  seed_start = int(time.time()) % 1000 # so we get different matrix every run
  for i in range(matrix_count):
    cur_rank = random.randrange(1, cols)  # get a ramdom deficient rank number
    matrices.append(gen_matrix(rows, cols, cur_rank, (seed_start+i)))
  return matrices
  
  
def print_cpp_matrix(matrix):
  print("{");
  for row in range(len(matrix)):
    print("{", end="")
    for col in range(len(matrix[0])):
      print("{:.8f}".format(matrix[row][col]), end="")
      if col < (len(matrix[0])-1):
        print(", ", end="")
    print("}", end="")
    if row < (len(matrix)-1):
      print(",")
  print("\n}")

def print_cpp_array(matrix):
  print("{", end="");
  for col in range(len(matrix)):
    print("{:.8f}".format(matrix[col]), end="")
    if col < (len(matrix)-1):
      print(", ", end="")
  print("}")
    

# main ========================================================
if __name__ == "__main__":
  print(np.__version__)
  print(sys.version)

  mat = gen_matrix(rows=16, cols=16, rank=16, seed=(int(time.time()) % 1000), min_val=0.0, max_val=1.0)
  print_cpp_matrix(mat)
  U, S, V  = np.linalg.svd(mat)
  print_cpp_array(S)

  
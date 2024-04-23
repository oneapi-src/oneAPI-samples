#include <iostream>

void VectorAdd(const int *a_in, const int *b_in, int *c_out, int len) {
  for (int idx = 0; idx < len; idx++) {
    int a_val = a_in[idx];
    int b_val = b_in[idx];
    int sum = a_val + b_val;
    c_out[idx] = sum;
  }
}

constexpr int kVectSize = 256;

int main() {
  // declare arrays and fill them
  int *vec_a = new int[kVectSize];
  int *vec_b = new int[kVectSize];
  int *vec_c = new int[kVectSize];
  for (int i = 0; i < kVectSize; i++) {
    vec_a[i] = i;
    vec_b[i] = (kVectSize - i);
  }

  std::cout << "add two vectors of size " << kVectSize << std::endl;

  VectorAdd(vec_a, vec_b, vec_c, kVectSize);

  // verify that vector C is correct
  bool passed = true;
  for (int i = 0; i < kVectSize; i++) {
    int expected = vec_a[i] + vec_b[i];
    if (vec_c[i] != expected) {
      std::cout << "idx=" << i << ": result " << vec_c[i] << ", expected ("
                << expected << ") A=" << vec_a[i] << " + B=" << vec_b[i]
                << std::endl;
      passed = false;
    }
  }

  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

  delete[] vec_a;
  delete[] vec_b;
  delete[] vec_c;

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
# 
# Number-theoretic transform test (Python)
# 
# Copyright (c) 2020 Project Nayuki
# All rights reserved. Contact Nayuki for licensing.
# https://www.nayuki.io/page/number-theoretic-transform-integer-dft
# 

import random, unittest
import ntt_core as ntt


class NumberTheoreticTransformTest(unittest.TestCase):
	
	def test_forward_transform(self):
		actual = ntt.transform([6, 0, 10, 7, 2], 3, 11)
		expect = [3, 7, 0, 5, 4]
		self.assertEqual(expect, actual)
	
	
	def test_inverse_transform(self):
		actual = ntt.inverse_transform([3, 7, 0, 5, 4], 3, 11)
		expect = [6, 0, 10, 7, 2]
		self.assertEqual(expect, actual)
	
	
	def test_simple_convolution(self):
		mod = 673
		root = 326
		vec0 = ntt.transform([4, 1, 4, 2, 1, 3, 5, 6], root, mod)
		vec1 = ntt.transform([6, 1, 8, 0, 3, 3, 9, 8], root, mod)
		vec2 = [(x * y % mod) for (x, y) in zip(vec0, vec1)]
		actual = ntt.inverse_transform(vec2, root, mod)
		expect = [123, 120, 106, 92, 139, 144, 140, 124]
		self.assertEqual(expect, actual)
	
	
	def test_automatic_convolution(self):
		actual = ntt.circular_convolve(
			[4, 1, 4, 2, 1, 3, 5, 6],
			[6, 1, 8, 0, 3, 3, 9, 8])
		expect = [123, 120, 106, 92, 139, 144, 140, 124]
		self.assertEqual(expect, actual)
	
	
	def test_transform_roundtrip_randomly(self):
		TRIALS = 300
		for _ in range(TRIALS):
			veclen = random.randrange(100) + 1
			maxval = random.randrange(100) + 1
			vec = [random.randrange(maxval + 1) for _ in range(veclen)]
			temp, root, mod = ntt.find_params_and_transform(vec, maxval + 1)
			inv = ntt.inverse_transform(temp, root, mod)
			self.assertEqual(vec, inv)
	
	
	def test_transform_linearity_randomly(self):
		TRIALS = 100
		for _ in range(TRIALS):
			veclen = random.randrange(100) + 1
			maxval = random.randrange(100) + 1
			vec0 = [random.randrange(maxval + 1) for _ in range(veclen)]
			vec1 = [random.randrange(maxval + 1) for _ in range(veclen)]
			out0, root, mod = ntt.find_params_and_transform(vec0, maxval + 1)
			out1 = ntt.transform(vec1, root, mod)
			out01 = [(x + y) % mod for (x, y) in zip(out0, out1)]
			vec2 = [(x + y) % mod for (x, y) in zip(vec0, vec1)]
			out2 = ntt.transform(vec2, root, mod)
			self.assertEqual(out2, out01)
	
	
	def test_convolution_randomly(self):
		TRIALS = 100
		for _ in range(TRIALS):
			veclen = random.randrange(100) + 1
			maxval = random.randrange(100) + 1
			vec0 = [random.randrange(maxval + 1) for _ in range(veclen)]
			vec1 = [random.randrange(maxval + 1) for _ in range(veclen)]
			actual = ntt.circular_convolve(vec0, vec1)
			expect = NumberTheoreticTransformTest._circular_convolve(vec0, vec1)
			self.assertEqual(expect, actual)
	
	
	@staticmethod  # Naive algorithm
	def _circular_convolve(vec0, vec1):
		assert len(vec0) == len(vec1)
		result = [0] * len(vec0)
		for (i, val0) in enumerate(vec0):
			for (j, val1) in enumerate(vec1):
				result[(i + j) % len(vec0)] += val0 * val1
		return result
	
	
	def test_transform_radix2_vs_naive(self):
		TRIALS = 300
		for _ in range(TRIALS):
			veclen = 2**random.randrange(8)
			maxval = random.randrange(100) + 1
			vec = [random.randrange(maxval + 1) for _ in range(veclen)]
			temp, root, mod = ntt.find_params_and_transform(vec, maxval + 1)
			ntt.transform_radix_2(vec, root, mod)
			self.assertEqual(temp, vec)
	
	
	def test_transform_radix2_roundtrip_randomly(self):
		TRIALS = 10
		for _ in range(TRIALS):
			veclen = 2**random.randint(0, 16)
			vallimit = 2**random.randint(1, 16)
			invec = [random.randrange(vallimit) for _ in range(veclen)]
			
			mod = ntt.find_modulus(len(invec), vallimit)
			root = ntt.find_primitive_root(len(invec), mod - 1, mod)
			vec = list(invec)
			ntt.transform_radix_2(vec, root, mod)
			
			ntt.transform_radix_2(vec, ntt.reciprocal(root, mod), mod)
			scaler = ntt.reciprocal(veclen, mod)
			vec = [(x * scaler % mod) for x in vec]
			self.assertEqual(invec, vec)



if __name__ == "__main__":
	unittest.main()

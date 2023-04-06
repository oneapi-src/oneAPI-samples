/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#ifndef HEAAN_SCHEMEALGO_H_
#define HEAAN_SCHEMEALGO_H_

#include "Common.h"
#include "EvaluatorUtils.h"
#include "Plaintext.h"
#include "SecretKey.h"
#include "Ciphertext.h"
#include "Scheme.h"

class SchemeAlgo {
public:
	Scheme& scheme;

	SchemeAlgo(Scheme& scheme) : scheme(scheme) {};


	//----------------------------------------------------------------------------------
	//   ARRAY ENCRYPTION & DECRYPTION
	//----------------------------------------------------------------------------------


	Ciphertext* encryptSingleArray(complex<double>* vals, long size);

	Ciphertext* encryptSingleArray(double* vals, long size);

	complex<double>* decryptSingleArray(SecretKey& secretKey, Ciphertext* ciphers, long size);


	//----------------------------------------------------------------------------------
	//   POWERS & PRODUCTS
	//----------------------------------------------------------------------------------


	Ciphertext powerOf2(Ciphertext& cipher,  const long logDegree);

	Ciphertext* powerOf2Extended(Ciphertext& cipher, const long logDegree);

	Ciphertext power(Ciphertext& cipher, const long degree);

	Ciphertext* powerExtended(Ciphertext& cipher, const long degree);

	Ciphertext prodOfPo2(Ciphertext* ciphers, const long logDegree);

	Ciphertext prod(Ciphertext* ciphers, const long degree);


	//----------------------------------------------------------------------------------
	//   METHODS ON ARRAYS OF CIPHERTEXTS
	//----------------------------------------------------------------------------------


	Ciphertext sum(Ciphertext* ciphers, const long size);

	Ciphertext distance(Ciphertext& cipher1, Ciphertext& cipher2);

	Ciphertext* multVec(Ciphertext* ciphers1, Ciphertext* ciphers2, const long size);

	void multAndEqualVec(Ciphertext* ciphers1, Ciphertext* ciphers2, const long size);

	Ciphertext* multAndModSwitchVec(Ciphertext* ciphers1, Ciphertext* ciphers2, const long size);

	void multModSwitchAndEqualVec(Ciphertext* ciphers1, Ciphertext* ciphers2, const long size);

	Ciphertext innerProd(Ciphertext* ciphers1, Ciphertext* ciphers2, const long size);

	Ciphertext partialSlotsSum(Ciphertext& cipher, const long slots);

	void partialSlotsSumAndEqual(Ciphertext& cipher, const long slots);


	//----------------------------------------------------------------------------------
	//   FUNCTIONS
	//----------------------------------------------------------------------------------


	Ciphertext inverse(Ciphertext& cipher, const long steps);

	Ciphertext* inverseExtended(Ciphertext& cipher, const long steps);

	Ciphertext exponent(Ciphertext& cipher, long degree = 7);

	Ciphertext sigmoid(Ciphertext& cipher, long degree = 7);

	Ciphertext function(Ciphertext& cipher, string& funcName, const long degree);

	Ciphertext* functionExtended(Ciphertext& cipher, string& funcName, const long degree);

	//----------------------------------------------------------------------------------
	//   FFT & FFT INVERSE
	//----------------------------------------------------------------------------------


	void bitReverse(Ciphertext* ciphers, const long size);

	void fft(Ciphertext* ciphers, const long size);

	void fftInvLazy(Ciphertext* ciphers, const long size);

	void fftInv(Ciphertext* ciphers, const long size);

};

#endif

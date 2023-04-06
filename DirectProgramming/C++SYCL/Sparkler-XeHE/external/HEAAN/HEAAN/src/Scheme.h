/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_SCHEME_H_
#define HEAAN_SCHEME_H_

#include <NTL/RR.h>
#include <NTL/ZZ.h>
#include <complex>

#include "BootContext.h"
#include "SecretKey.h"
#include "Ciphertext.h"
#include "Plaintext.h"
#include "Key.h"
#include "EvaluatorUtils.h"
#include "Ring.h"

using namespace std;
using namespace NTL;

static long ENCRYPTION = 0;
static long MULTIPLICATION  = 1;
static long CONJUGATION = 2;

class Scheme {
private:
public:
	Ring& ring;

	bool isSerialized;

	map<long, Key*> keyMap; ///< contain Encryption, Multiplication and Conjugation keys, if generated
	map<long, Key*> leftRotKeyMap; ///< contain left rotation keys, if generated

	map<long, string> serKeyMap; ///< contain Encryption, Multiplication and Conjugation keys, if generated
	map<long, string> serLeftRotKeyMap; ///< contain left rotation keys, if generated

	Scheme(SecretKey& secretKey, Ring& ring, bool isSerialized = false);

	virtual ~Scheme();

	//----------------------------------------------------------------------------------
	//   KEYS GENERATION
	//----------------------------------------------------------------------------------


	void addEncKey(SecretKey& secretKey);

	void addMultKey(SecretKey& secretKey);

	void addConjKey(SecretKey& secretKey);

	void addLeftRotKey(SecretKey& secretKey, long r);

	void addRightRotKey(SecretKey& secretKey, long r);

	void addLeftRotKeys(SecretKey& secretKey);

	void addRightRotKeys(SecretKey& secretKey);

	void addBootKey(SecretKey& secretKey, long logl, long logp);


	//----------------------------------------------------------------------------------
	//   ENCODING & DECODING
	//----------------------------------------------------------------------------------


	void encode(Plaintext& plain, complex<double>* vals, long n, long logp, long logq);

	void encode(Plaintext& plain, double* vals, long n, long logp, long logq);

	complex<double>* decode(Plaintext& plain);

	void encodeSingle(Plaintext& plain, complex<double> val, long logp, long logq);

	void encodeSingle(Plaintext& plain, double val, long logp, long logq);

	complex<double> decodeSingle(Plaintext& plain);


	//----------------------------------------------------------------------------------
	//   ENCRYPTION & DECRYPTION
	//----------------------------------------------------------------------------------


	void encryptMsg(Ciphertext& cipher, Plaintext& plain);

	void decryptMsg(Plaintext& plain, SecretKey& secretKey, Ciphertext& cipher);

	void encrypt(Ciphertext& cipher, complex<double>* vals, long n, long logp, long logq);

	void encrypt(Ciphertext& cipher, double* vals, long n, long logp, long logq);
	
	void encryptBySk(Ciphertext& cipher, SecretKey& secretKey, complex<double>* vals, long n, long logp, long logq, double=3.2);
	
	void encryptBySk(Ciphertext& cipher, SecretKey& secretKey, double* vals, long n, long logp, long logq, double=3.2);

	void encryptZeros(Ciphertext& cipher, long n, long logp, long logq);

	complex<double>* decrypt(SecretKey& secretKey, Ciphertext& cipher);
	
	complex<double>* decryptForShare(SecretKey& secretKey, Ciphertext& cipher, long=0);

	void encryptSingle(Ciphertext& cipher, complex<double> val, long logp, long logq);

	void encryptSingle(Ciphertext& cipher, double val, long logp, long logq);

	complex<double> decryptSingle(SecretKey& secretKey, Ciphertext& cipher);


	//----------------------------------------------------------------------------------
	//   HOMOMORPHIC OPERATIONS
	//----------------------------------------------------------------------------------

	void negate(Ciphertext& res, Ciphertext& cipher);

	void negateAndEqual(Ciphertext& cipher);

	void add(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2);

	void addAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);

	void addConst(Ciphertext& res, Ciphertext& cipher, double cnst, long logp);

	void addConst(Ciphertext& res, Ciphertext& cipher, RR& cnst, long logp);

	void addConst(Ciphertext& res, Ciphertext& cipher, complex<double> cnst, long logp);

	void addConstAndEqual(Ciphertext& cipher, double cnst, long logp);

	void addConstAndEqual(Ciphertext& cipher, RR& cnst, long logp);

	void addConstAndEqual(Ciphertext& cipher, complex<double> cnst, long logp);

	void sub(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2);

	void subAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);

	void subAndEqual2(Ciphertext& cipher1, Ciphertext& cipher2);

	void imult(Ciphertext& res, Ciphertext& cipher);

	void idiv(Ciphertext& res, Ciphertext& cipher);

	void imultAndEqual(Ciphertext& cipher);

	void idivAndEqual(Ciphertext& cipher);

	void mult(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2);

	void multAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);

	void square(Ciphertext& res, Ciphertext& cipher);

	void squareAndEqual(Ciphertext& cipher);

	void multByConst(Ciphertext& res, Ciphertext& cipher, double cnst, long logp);

	void multByConst(Ciphertext& res, Ciphertext& cipher, complex<double> cnst, long logp);

	void multByConstVec(Ciphertext& res, Ciphertext& cipher, complex<double>* cnstVec, long logp);

	void multByConstVecAndEqual(Ciphertext& cipher, complex<double>* cnstVec, long logp);

	void multByConstAndEqual(Ciphertext& cipher, double cnst, long logp);

	void multByConstAndEqual(Ciphertext& cipher, RR& cnst, long logp);

	void multByConstAndEqual(Ciphertext& cipher, complex<double> cnst, long logp);

	void multByPoly(Ciphertext& res, Ciphertext& cipher, ZZ* poly, long logp);

	void multByPolyNTT(Ciphertext& res, Ciphertext& cipher, uint64_t* rpoly, long bnd, long logp);

	void multByPolyAndEqual(Ciphertext& cipher, ZZ* poly, long logp);

	void multByPolyNTTAndEqual(Ciphertext& cipher, uint64_t* rpoly, long bnd, long logp);

	void multByMonomial(Ciphertext& res, Ciphertext& cipher, const long degree);

	void multByMonomialAndEqual(Ciphertext& cipher, const long degree);

	void leftShift(Ciphertext& res, Ciphertext& cipher, long bits);

	void leftShiftAndEqual(Ciphertext& cipher, long bits);

	void doubleAndEqual(Ciphertext& cipher);

	void divByPo2(Ciphertext& res, Ciphertext& cipher, long bits);

	void divByPo2AndEqual(Ciphertext& cipher, long bits);


	//----------------------------------------------------------------------------------
	//   RESCALING
	//----------------------------------------------------------------------------------


	void reScaleBy(Ciphertext& res, Ciphertext& cipher, long dlogq);

	void reScaleTo(Ciphertext& res, Ciphertext& cipher, long logq);

	void reScaleByAndEqual(Ciphertext& cipher, long dlogq);

	void reScaleToAndEqual(Ciphertext& cipher, long logq);

	void modDownBy(Ciphertext& res, Ciphertext& cipher, long dlogq);

	void modDownByAndEqual(Ciphertext& cipher, long dlogq);

	void modDownTo(Ciphertext& res, Ciphertext& cipher, long logq);

	void modDownToAndEqual(Ciphertext& cipher, long logq);


	//----------------------------------------------------------------------------------
	//   ROTATIONS & CONJUGATIONS
	//----------------------------------------------------------------------------------


	void leftRotateFast(Ciphertext& res, Ciphertext& cipher, long r);
	void rightRotateFast(Ciphertext& res, Ciphertext& cipher, long r);

	void leftRotateFastAndEqual(Ciphertext& cipher, long r);
	void rightRotateFastAndEqual(Ciphertext& cipher, long r);

	void conjugate(Ciphertext& res, Ciphertext& cipher);
	void conjugateAndEqual(Ciphertext& cipher);


	//----------------------------------------------------------------------------------
	//   BOOTSTRAPPING
	//----------------------------------------------------------------------------------


	void normalizeAndEqual(Ciphertext& cipher);

	void coeffToSlotAndEqual(Ciphertext& cipher);

	void slotToCoeffAndEqual(Ciphertext& cipher);

	void exp2piAndEqual(Ciphertext& cipher, long logp);

	void evalExpAndEqual(Ciphertext& cipher, long logT, long logI = 4);

	void bootstrapAndEqual(Ciphertext& cipher, long logq, long logQ, long logT, long logI = 4);
};

#endif

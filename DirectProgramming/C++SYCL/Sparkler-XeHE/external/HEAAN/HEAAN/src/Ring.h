/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_RING_H_
#define HEAAN_RING_H_

#include <NTL/ZZ.h>
#include <NTL/RR.h>
#include <complex>
#include <map>
#include "BootContext.h"
#include "RingMultiplier.h"

using namespace std;
using namespace NTL;

static RR Pi = ComputePi_RR();

class Ring {

public:

	ZZ* qpows;
	long* rotGroup;
	complex<double>* ksiPows;
	map<long, BootContext*> bootContextMap;
	RingMultiplier multiplier;

	Ring();


	//----------------------------------------------------------------------------------
	//   Encode & Decode
	//----------------------------------------------------------------------------------


	void arrayBitReverse(complex<double>* vals, long size);

	void EMB(complex<double>* vals, long size);

	void EMBInvLazy(complex<double>* vals, long size);

	void EMBInv(complex<double>* vals, long size);

	void encode(ZZ* mx, double* vals, long slots, long logp);

	void encode(ZZ* mx, complex<double>* vals, long slots, long logp);

	void decode(ZZ* mx, complex<double>* vals, long slots, long logp, long logq);


	//----------------------------------------------------------------------------------
	//   CONTEXT
	//----------------------------------------------------------------------------------


	void addBootContext(long logSlots, long logp);


	//----------------------------------------------------------------------------------
	//   MULTIPLICATION
	//----------------------------------------------------------------------------------

	long maxBits(const ZZ* f, long n);

	void CRT(uint64_t* rx, ZZ* x, const long np);

	void addNTTAndEqual(uint64_t* ra, uint64_t* rb, const long np);

	void mult(ZZ* x, ZZ* a, ZZ* b, long np, const ZZ& q);

	void multNTT(ZZ* x, ZZ* a, uint64_t* rb, long np, const ZZ& q);

	void multDNTT(ZZ* x, uint64_t* a, uint64_t* rb, long np, const ZZ& q);

	void multAndEqual(ZZ* a, ZZ* b, long np, const ZZ& q);

	void multNTTAndEqual(ZZ* a, uint64_t* rb, long np, const ZZ& q);

	void square(ZZ* x, ZZ* a, long np, const ZZ& q);

	void squareNTT(ZZ* x, uint64_t* ra, long np, const ZZ& q);

	void squareAndEqual(ZZ* a, long np, const ZZ& q);


	//----------------------------------------------------------------------------------
	//   OTHER
	//----------------------------------------------------------------------------------


	void mod(ZZ* res, ZZ* p, const ZZ& QQ);

	void modAndEqual(ZZ* p, const ZZ& QQ);

	void negate(ZZ* res, ZZ* p);

	void negateAndEqual(ZZ* p);

	void add(ZZ* res, ZZ* p1, ZZ* p2, const ZZ& QQ);

	void addAndEqual(ZZ* p1, ZZ* p2, const ZZ& QQ);

	void sub(ZZ* res, ZZ* p1, ZZ* p2, const ZZ& QQ);

	void subAndEqual(ZZ* p1, ZZ* p2, const ZZ& QQ);

	void subAndEqual2(ZZ* p1, ZZ* p2, const ZZ& QQ);

	void multByMonomial(ZZ* res, ZZ* p, long mDeg);

	void multByMonomialAndEqual(ZZ* p, long mDeg);

	void multByConst(ZZ* res, ZZ* p, ZZ& cnst, const ZZ& QQ);

	void multByConstAndEqual(ZZ* p, ZZ& cnst, const ZZ& QQ);

	void leftShift(ZZ* res, ZZ* p, const long bits, const ZZ& QQ);

	void leftShiftAndEqual(ZZ* p, const long bits, const ZZ& QQ);

	void doubleAndEqual(ZZ* p, const ZZ& QQ);

	void rightShift(ZZ* res, ZZ* p, long bits);

	void rightShiftAndEqual(ZZ* p, long bits);


	//----------------------------------------------------------------------------------
	//   ROTATION & CONJUGATION
	//----------------------------------------------------------------------------------


	void leftRotate(ZZ* res, ZZ* p, long r);

	void conjugate(ZZ* res, ZZ* p);


	//----------------------------------------------------------------------------------
	//   SAMPLING
	//----------------------------------------------------------------------------------


	void subFromGaussAndEqual(ZZ* res, const ZZ& q);
	
	void subFromGaussAndEqual(ZZ* res, const ZZ& q, double _sigma);

	void addGaussAndEqual(ZZ* res, const ZZ& q);
	
	void addGaussAndEqual(ZZ* res, const ZZ& q, double _sigma);

	void sampleHWT(ZZ* res);

	void sampleZO(ZZ* res);

	void sampleUniform2(ZZ* res, long bits);


	//----------------------------------------------------------------------------------
	//   DFT
	//----------------------------------------------------------------------------------


	void DFT(complex<double>* vals, long n);

	void IDFTLazy(complex<double>* vals, long n);

	void IDFT(complex<double>* vals, long n);

};

#endif

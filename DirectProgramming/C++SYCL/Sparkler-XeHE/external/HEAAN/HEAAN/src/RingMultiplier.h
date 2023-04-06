/*
 * Copyright (c) by CryptoLab inc.
 * This program is licensed under a
 * Creative Commons Attribution-NonCommercial 3.0 Unported License.
 * You should have received a copy of the license along with this
 * work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
 */
#ifndef HEAAN_RINGMULTIPLIER_H_
#define HEAAN_RINGMULTIPLIER_H_

#include <cstdint>
#include <vector>
#include <NTL/ZZ.h>
#include "Params.h"

using namespace std;
using namespace NTL;

class RingMultiplier {
public:

	uint64_t* pVec = new uint64_t[nprimes];
	uint64_t* prVec = new uint64_t[nprimes];
	uint64_t* pInvVec = new uint64_t[nprimes];
	uint64_t** scaledRootPows = new uint64_t*[nprimes];
	uint64_t** scaledRootInvPows = new uint64_t*[nprimes];
	uint64_t* scaledNInv = new uint64_t[nprimes];
	_ntl_general_rem_one_struct* red_ss_array[nprimes];
	mulmod_precon_t* coeffpinv_array[nprimes];

	ZZ* pProd = new ZZ[nprimes];
	ZZ* pProdh = new ZZ[nprimes];
	ZZ** pHat = new ZZ*[nprimes];
	uint64_t** pHatInvModp = new uint64_t*[nprimes];

	RingMultiplier();

	bool primeTest(uint64_t p);

	void NTT(uint64_t* a, long index);
	void INTT(uint64_t* a, long index);

	void CRT(uint64_t* rx, ZZ* x, const long np);

	void addNTTAndEqual(uint64_t* ra, uint64_t* rb, const long np);

	void reconstruct(ZZ* x, uint64_t* rx, long np, const ZZ& QQ);

	void mult(ZZ* x, ZZ* a, ZZ* b, long np, const ZZ& QQ);

	void multNTT(ZZ* x, ZZ* a, uint64_t* rb, long np, const ZZ& QQ);

	void multDNTT(ZZ* x, uint64_t* ra, uint64_t* rb, long np, const ZZ& QQ);

	void multAndEqual(ZZ* a, ZZ* b, long np, const ZZ& QQ);

	void multNTTAndEqual(ZZ* a, uint64_t* rb, long np, const ZZ& QQ);

	void square(ZZ* x, ZZ* a, long np, const ZZ& QQ);

	void squareNTT(ZZ* x, uint64_t* ra, long np, const ZZ& QQ);

	void squareAndEqual(ZZ* a, long np, const ZZ& QQ);

	void mulMod(uint64_t& r, uint64_t a, uint64_t b, uint64_t p);

	void mulModBarrett(uint64_t& r, uint64_t a, uint64_t b, uint64_t p, uint64_t pr);
	void butt(uint64_t& a, uint64_t& b, uint64_t W, uint64_t p, uint64_t pInv);
	void ibutt(uint64_t& a, uint64_t& b, uint64_t W, uint64_t p, uint64_t pInv);
	void idivN(uint64_t& a, uint64_t NScale, uint64_t p, uint64_t pInv);

	uint64_t invMod(uint64_t x, uint64_t p);

	uint64_t powMod(uint64_t x, uint64_t y, uint64_t p);

	uint64_t inv(uint64_t x);

	uint64_t pow(uint64_t x, uint64_t y);

	uint32_t bitReverse(uint32_t x);

	void findPrimeFactors(vector<uint64_t> &s, uint64_t number);

	uint64_t findPrimitiveRoot(uint64_t m);

	uint64_t findMthRootOfUnity(uint64_t M, uint64_t p);

};

#endif /* RINGMULTIPLIER_H_ */

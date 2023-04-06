/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_BOOTCONTEXT_H_
#define HEAAN_BOOTCONTEXT_H_

#include <NTL/ZZ.h>

using namespace NTL;

class BootContext {
public:

	uint64_t** rpvec;
	uint64_t** rpvecInv;
	uint64_t* rp1;
	uint64_t* rp2;

	long* bndvec;
	long* bndvecInv;
	long bnd1;
	long bnd2;

	long logp;

	BootContext(uint64_t** rpvec = NULL, uint64_t** rpvecInv = NULL, uint64_t* rp1 = NULL, uint64_t* rp2 = NULL,
			long* bndvec = NULL, long* bndvecInv = NULL, long bnd1 = 0, long bnd2 = 0, long logp = 0);

};

#endif

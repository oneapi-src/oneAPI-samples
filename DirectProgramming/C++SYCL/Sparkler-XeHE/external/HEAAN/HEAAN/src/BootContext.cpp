/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#include "BootContext.h"

BootContext::BootContext(uint64_t** rpvec, uint64_t** rpvecInv, uint64_t* rp1, uint64_t* rp2,
		long* bndvec, long* bndvecInv, long bnd1, long bnd2, long logp) :
		rpvec(rpvec), rpvecInv(rpvecInv), rp1(rp1), rp2(rp2),
		bndvec(bndvec), bndvecInv(bndvecInv), bnd1(bnd1), bnd2(bnd2), logp(logp){}

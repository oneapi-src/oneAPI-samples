/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_KEY_H_
#define HEAAN_KEY_H_

#include <NTL/ZZ.h>
#include "Params.h"

using namespace NTL;

class Key {
public:

	uint64_t* rax = new uint64_t[Nnprimes]();
	uint64_t* rbx = new uint64_t[Nnprimes]();

	Key();

	virtual ~Key();
};

#endif

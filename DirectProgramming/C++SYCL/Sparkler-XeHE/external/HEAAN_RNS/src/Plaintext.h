/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#ifndef HEAANNTT_PLAINTEXT_H_
#define HEAANNTT_PLAINTEXT_H_

#include "Common.h"

class Plaintext {

public:

	uint64_t* mx;

	long N;

	long slots;

	long l;

	Plaintext();

	Plaintext(uint64_t* mx, long N, long slots, long l);

	Plaintext(const Plaintext& ptxt);

	Plaintext& operator=(const Plaintext &o);
	
};


#endif

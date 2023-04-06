/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_PLAINTEXT_H_
#define HEAAN_PLAINTEXT_H_

#include <NTL/ZZ.h>
#include "Params.h"

using namespace std;
using namespace NTL;

class Plaintext {
public:

	ZZ* mx = new ZZ[N];

	long logp;
	long logq;
	long n;


	Plaintext(long logp = 0, long logq = 0, long n = 0);

	virtual ~Plaintext();
};

#endif

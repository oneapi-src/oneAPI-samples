/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#ifndef HEAANNTT_KEY_H_
#define HEAANNTT_KEY_H_

#include <cstdint>

using namespace std;
class Key {
public:

	uint64_t* ax;
	uint64_t* bx;

	Key(uint64_t* ax, uint64_t* bx);

	virtual ~Key();
};

#endif /* KEY_H_ */

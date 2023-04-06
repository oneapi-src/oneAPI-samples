/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_TIMEUTILS_H_
#define HEAAN_TIMEUTILS_H_

#include <iostream>

struct timeval;

using namespace std;

class TimeUtils {
public:

	struct timeval startTime, stopTime;
	double timeElapsed;

	TimeUtils();

	void start(string msg);

	void stop(string msg);

};

#endif

/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_STRINGUTILS_H_
#define HEAAN_STRINGUTILS_H_

#include <NTL/ZZ.h>

#include <complex>

using namespace std;
using namespace NTL;

class StringUtils {
public:


	//----------------------------------------------------------------------------------
	//   SHOW ARRAY
	//----------------------------------------------------------------------------------


	static void showVec(long* vals, long size);

	static void showVec(double* vals, long size);

	static void showVec(complex<double>* vals, long size);

	static void showVec(ZZ* vals, long size);


	//----------------------------------------------------------------------------------
	//   SHOW & COMPARE ARRAY
	//----------------------------------------------------------------------------------


	static void compare(double val1, double val2, string prefix);

	static void compare(complex<double> val1, complex<double> val2, string prefix);

	static void compare(double* vals1, double* vals2, long size, string prefix);

	static void compare(complex<double>* vals1, complex<double>* vals2, long size, string prefix);

	static void compare(double* vals1, double val2, long size, string prefix);

	static void compare(complex<double>* vals1, complex<double> val2, long size, string prefix);

	static void compare(double val1, double* vals2, long size, string prefix);

	static void compare(complex<double> val1, complex<double>* vals2, long size, string prefix);

};

#endif

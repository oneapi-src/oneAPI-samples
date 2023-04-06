/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#ifndef HEAAN_STRINGUTILS_H_
#define HEAAN_STRINGUTILS_H_

#include <complex>

#include "Common.h"

using namespace std;

class StringUtils {
public:


	//----------------------------------------------------------------------------------
	//   SHOW ARRAY
	//----------------------------------------------------------------------------------


	static void show(uint64_t* vals, long size);
	/**
	 * prints in console array
	 * @param[in] vals: long array
	 * @param[in] size: array size
	 */
	static void show(long* vals, long size);

	/**
	 * prints in console array
	 * @param[in] vals: double array
	 * @param[in] size: array size
	 */
	static void show(double* vals, long size);

	/**
	 * prints in console array
	 * @param[in] vals: complex array
	 * @param[in] size: array size
	 */
	static void show(complex<double>* vals, long size);


	//----------------------------------------------------------------------------------
	//   SHOW & COMPARE ARRAY
	//----------------------------------------------------------------------------------


	/**
	 * prints in console val1, val2 and (val1-val2)
	 * @param[in] val1: double value
	 * @param[in] val2: double value
	 * @param[in] prefix: string prefix
	 */
	static void showcompare(double val1, double val2, string prefix);

	/**
	 * prints in console val1, val2 and (val1-val2)
	 * @param[in] val1: complex value
	 * @param[in] val2: complex value
	 * @param[in] prefix: string prefix
	 */
	static void showcompare(complex<double> val1, complex<double> val2, string prefix);

	/**
	 * prints in console pairwise val1[i], val2[i] and (val1[i]-val2[i])
	 * @param[in] vals1: double array
	 * @param[in] vals2: double array
	 * @param[in] size: array size
	 * @param[in] prefix: string prefix
	 */
	static void showcompare(double* vals1, double* vals2, long size, string prefix);

	/**
	 * prints in console pairwise val1[i], val2[i] and (val1[i]-val2[i])
	 * @param[in] vals1: complex array
	 * @param[in] vals2: complex array
	 * @param[in] size: array size
	 * @param[in] prefix: string prefix
	 */
	static void showcompare(complex<double>* vals1, complex<double>* vals2, long size, string prefix);

	/**
	 * prints in console pairwise val1[i], val2 and (val1[i]-val2)
	 * @param[in] vals1: double array
	 * @param[in] val2: double value
	 * @param[in] size: array size
	 * @param[in] prefix: string prefix
	 */
	static void showcompare(double* vals1, double val2, long size, string prefix);

	/**
	 * prints in console pairwise val1[i], val2 and (val1[i]-val2)
	 * @param[in] vals1: complex array
	 * @param[in] val2: complex value
	 * @param[in] size: array size
	 * @param[in] prefix: string prefix
	 */
	static void showcompare(complex<double>* vals1, complex<double> val2, long size, string prefix);

	/**
	 * prints in console pairwise val1, val2[i] and (val1-val2[i])
	 * @param[in] val1: double value
	 * @param[in] vals2: double array
	 * @param[in] size: array size
	 * @param[in] prefix: string prefix
	 */
	static void showcompare(double val1, double* vals2, long size, string prefix);

	/**
	 * prints in console pairwise val1, val2[i] and (val1-val2[i])
	 * @param[in] val1: complex value
	 * @param[in] vals2: complex array
	 * @param[in] size: array size
	 * @param[in] prefix: string prefix
	 */
	static void showcompare(complex<double> val1, complex<double>* vals2, long size, string prefix);

};

#endif

/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#include "StringUtils.h"

//----------------------------------------------------------------------------------
//   SHOW ARRAY
//----------------------------------------------------------------------------------

void StringUtils::show(uint64_t* vals, long size) {
	cout << "[";
	for (long i = 0; i < size; ++i) {
		cout << vals[i] << ", ";
	}
	cout << "]" << endl;
}

void StringUtils::show(long* vals, long size) {
	cout << "[";
	for (long i = 0; i < size; ++i) {
		cout << vals[i] << ", ";
	}
	cout << "]" << endl;
}

void StringUtils::show(double* vals, long size) {
	cout << "[";
	for (long i = 0; i < size; ++i) {
		cout << vals[i] << ", ";
	}
	cout << "]" << endl;
}

void StringUtils::show(complex<double>* vals, long size) {
	cout << "[";
	for (long i = 0; i < size; ++i) {
		cout << vals[i] << ", ";
	}
	cout << "]" << endl;
}


//----------------------------------------------------------------------------------
//   SHOW & COMPARE ARRAY
//----------------------------------------------------------------------------------


void StringUtils::showcompare(double val1, double val2, string prefix) {
	cout << "---------------------" << endl;
	cout << "m" + prefix + ":" << val1 << endl;
	cout << "d" + prefix + ":" << val2 << endl;
	cout << "e" + prefix + ":" << val1-val2 << endl;
	cout << "---------------------" << endl;
}

void StringUtils::showcompare(complex<double> val1, complex<double> val2, string prefix) {
	cout << "---------------------" << endl;
	cout << "m" + prefix + ":" << val1 << endl;
	cout << "d" + prefix + ":" << val2 << endl;
	cout << "e" + prefix + ":" << val1-val2 << endl;
	cout << "---------------------" << endl;
}

void StringUtils::showcompare(double* vals1, double* vals2, long size, string prefix) {
	for (long i = 0; i < size; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << vals1[i] << endl;
		cout << "d" + prefix + ": " << i << " :" << vals2[i] << endl;
		cout << "e" + prefix + ": " << i << " :" << (vals1[i]-vals2[i]) << endl;
		cout << "---------------------" << endl;
	}
}

void StringUtils::showcompare(complex<double>* vals1, complex<double>* vals2, long size, string prefix) {
	for (long i = 0; i < size; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << vals1[i] << endl;
		cout << "d" + prefix + ": " << i << " :" << vals2[i] << endl;
		cout << "e" + prefix + ": " << i << " :" << (vals1[i]-vals2[i]) << endl;
		cout << "---------------------" << endl;
	}
}


void StringUtils::showcompare(double* vals1, double val2, long size, string prefix) {
	for (long i = 0; i < size; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << vals1[i] << endl;
		cout << "d" + prefix + ": " << i << " :" << val2 << endl;
		cout << "e" + prefix + ": " << i << " :" << vals1[i]-val2 << endl;
		cout << "---------------------" << endl;
	}
}

void StringUtils::showcompare(complex<double>* vals1, complex<double> val2, long size, string prefix) {
	for (long i = 0; i < size; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << vals1[i] << endl;
		cout << "d" + prefix + ": " << i << " :" << val2 << endl;
		cout << "e" + prefix + ": " << i << " :" << vals1[i]-val2 << endl;
		cout << "---------------------" << endl;
	}
}

void StringUtils::showcompare(double val1, double* vals2, long size, string prefix) {
	for (long i = 0; i < size; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << val1 << endl;
		cout << "d" + prefix + ": " << i << " :" << vals2[i] << endl;
		cout << "e" + prefix + ": " << i << " :" << val1-vals2[i] << endl;
		cout << "---------------------" << endl;
	}
}

void StringUtils::showcompare(complex<double> val1, complex<double>* vals2, long size, string prefix) {
	for (long i = 0; i < size; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << val1 << endl;
		cout << "d" + prefix + ": " << i << " :" << vals2[i] << endl;
		cout << "e" + prefix + ": " << i << " :" << val1-vals2[i] << endl;
		cout << "---------------------" << endl;
	}
}

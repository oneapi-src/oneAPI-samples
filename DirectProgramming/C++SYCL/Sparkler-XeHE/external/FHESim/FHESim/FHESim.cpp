// FHESim.cpp : Defines the entry point for the application.
//

#include "FHESim.h"

using namespace std;


/* Demo of NTT code
   M.Scott 21/07/2017
   gcc -O3 ntt_ref.c -o ntt_ref.exe
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Uncomment to create transcript of excesses
//#define RECORD_EXCESS

// Set some constants
//q= 12289

#define PRIME 0x3001	// q in Hex
#define LGN 10			// Degree n=2^LGN
#define ND 0xF7002FFF	// 1/(R-q) mod R
#define ONE 0x2AC8		// R mod q
#define R2MODP 0x1620	// R^2 mod q
#define L 1				// 2nq < 2^{31-L}
#define XLIM 0x2AA9C	// available excess 2^{31}/q	


//q= 8383489
/*
#define PRIME 0x7FEC01  // q in Hex
#define LGN 9           // Degree n=2^LGN
#define ND 0xBEEFEBFF   // 1/(R-q) mod R
#define ONE 0x27FE00    // R mod q
#define R2MODP 0x58F4C  // R^2 mod q
#define L 4				// 2nq < 2^{31-L}
#define XLIM 0x100		// available excess 2^{31}/q
*/
// q=39960577
/*
#define PRIME 0x261C001
#define LGN 9
#define ND 0xF261BFFF
#define ONE 0x124BF95
#define R2MODP 0xAB9F67
#define L 32
#define XLIM 0x35
*/

#define DEGREE (1<<LGN)

#define int_t int32_t			// type for internal calculation
#define uint_t uint32_t
#define int_dt int64_t			// double length type
#define uint_dt uint64_t


#define WL (8*sizeof(int_t))

/* Montgomery stuff */

inline int_t redc(uint_dt T)
{
	uint_t m = (uint_t)T * (uint_t)ND;
	return ((uint_dt)m * PRIME + T) >> WL;
}

inline int_t nres(uint_t x)
{
	return redc((uint_dt)x * R2MODP);
}

inline int_t modmul(uint_t a, uint_t b)
{
	return redc((uint_dt)a * b);
}

/* reverse bits */

uint_t reverse(uint_t v)
{
	uint_t r = v;
	int s = sizeof(v) * 8 - 1;

	for (v >>= 1; v; v >>= 1)
	{
		r <<= 1;
		r |= v & 1;
		s--;
	}
	r <<= s;
	r >>= (WL - LGN);
	return r;
}

/* some number theory functions borrowed from MIRACL */

int spmd(int x, int n, int m)
{ /*  returns x^n mod m  */
	int r, sx;
	x %= m;
	r = 0;
	if (x == 0) return r;
	r = 1;
	if (n == 0) return r;
	sx = x;
	for (;;)
	{
		if (n % 2 != 0) r = ((int_dt)r * sx) % m;
		n /= 2;
		if (n == 0) return r;
		sx = ((int_dt)sx * sx) % m;
	}
}

int invers(int x, int y)
{ /* returns inverse of x mod y */
	int r, s, q, t, p, pos;
	if (y != 0) x %= y;
	r = 1;
	s = 0;
	p = y;
	pos = 1;

	while (p != 0)
	{ /* main euclidean loop */
		q = x / p;
		t = r + s * q; r = s; s = t;
		t = x - p * q; x = p; p = t;
		pos = !pos;
	}
	if (!pos) r = y - r;
	return r;
}

int sqrmp(int x, int m)
{ /* square root mod a small prime =1 mod 8 by Shanks method  *
   * returns 0 if root does not exist or m not prime */
	int z, y, v, w, t, q, i, e, n, r, pp;
	x %= m;
	if (x == 0) return 0;
	if (x == 1) return 1;
	if (spmd(x, (m - 1) / 2, m) != 1) return 0;    /* Legendre symbol not 1   */

	q = m - 1;
	e = 0;
	while (q % 2 == 0)
	{
		q /= 2;
		e++;
	}
	if (e == 0) return 0;      /* even m */
	for (r = 2;; r++)
	{ /* find suitable z */
		z = spmd(r, q, m);
		if (z == 1) continue;
		t = z;
		pp = 0;
		for (i = 1; i < e; i++)
		{ /* check for composite m */
			if (t == (m - 1)) pp = 1;
			t = ((int_dt)t * t) % m;
			if (t == 1 && !pp) return 0;
		}
		if (t == (m - 1)) break;
		if (!pp) return 0;   /* m is not prime */
	}
	y = z;
	r = e;
	v = spmd(x, (q + 1) / 2, m);
	w = spmd(x, q, m);
	while (w != 1)
	{
		t = w;
		for (n = 0; t != 1; n++) t = ((int_dt)t * t) % m;
		if (n >= r) return 0;
		y = spmd(y, (1 << (r - n - 1)), m);
		v = ((int_dt)v * y) % m;
		y = ((int_dt)y * y) % m;
		w = ((int_dt)w * y) % m;
		r = n;
	}
	return v;
}

/* NTT code */

/* precompute roots of unity and its powers */
void init_ntt2(int_t* roots, int_t* iroots)
{
	int q = PRIME;
	int i, j, proot = q - 1;
	for (j = 0; j < LGN; j++) proot = sqrmp(proot, q);

	roots[0] = 1;   /* build table of powers */
	for (j = 1; j < DEGREE; j++) roots[reverse(j)] = ((int64_t)proot * roots[reverse(j - 1)]) % q;
	for (j = 0; j < DEGREE; j++)
	{
		iroots[j] = invers(roots[j], q);
		roots[j] = nres(roots[j]);
		iroots[j] = nres(iroots[j]);
	}
}

/* Cooley-Tukey NTT */

void ntt(int_t* roots, int_t* x)
{
	int m, i, j, k, t = DEGREE / 2;
	int_t S, U, V, W, q = PRIME;
#ifdef RECORD_EXCESS
	int xes[DEGREE];
	for (j = 0; j < DEGREE; j++)
		xes[j] = 2;
#endif
	/* Convert to Montgomery form */
	for (j = 0; j < DEGREE; j++)
		x[j] = nres(x[j]);

	m = 1;
	while (m < DEGREE)
	{
		k = 0;
		for (i = 0; i < m; i++)
		{
			S = roots[m + i];
			for (j = k; j < k + t; j++)
			{
				U = x[j];
				V = modmul(x[j + t], S);
				x[j] = U + V;

#ifdef RECORD_EXCESS
				xes[j] += 2;
				xes[j + t] = xes[j];
				printf("NTT MUL XES= %d XLIM %d\n", xes[j + t], XLIM);
				if (xes[j + t] > XLIM) { printf("******* Possible Overflow *******\n"); exit(0); }
				printf("NTT ADD XES= %d XLIM %d\n", xes[j], XLIM);
				if (xes[j] > XLIM) { printf("******* Possible Overflow *******\n"); exit(0); }
#endif
				x[j + t] = U + 2 * q - V;
			}
			k += 2 * t;
		}
		t /= 2;
		m *= 2;
	}
}

/* Gentleman-Sande INTT */

void intt(int_t inv, int_t invpr, int_t* iroots, int_t* x)
{
	int m, i, j, k, n, lim, t = 1;
	int_t S, U, V, W, q = PRIME;
#ifdef RECORD_EXCESS
	int xes[DEGREE];
	int eU, eV;
	for (j = 0; j < DEGREE; j++)
		xes[j] = 2;
#endif

	m = DEGREE / 2;
	n = LGN;
	while (m > 1)
	{
		//lim=L/(2*m);
		lim = L >> n;
		n--;
		k = 0;
		for (i = 0; i < m; i++)
		{
			S = iroots[m + i];
			for (j = k; j < k + t; j++)
			{
				if (m < L && j < k + lim)   // This should be unwound for timings
				{ // need to knock back excesses. Never used if L=1.
					U = modmul(x[j], ONE);
					V = modmul(x[j + t], ONE);
				}
				else
				{
					U = x[j];
					V = x[j + t];
				}
				x[j] = U + V;

#ifdef RECORD_EXCESS
				if (m < L && j < k + lim)
				{ // need to knock back excesses. Never used if L=1.
					eU = eV = 2;
				}
				else
				{
					eU = xes[j]; eV = xes[j + t];
				}
				printf("INTT ADD XES= %d XLIM %d \n", eU + eV, XLIM);
				if (eU + eV > XLIM) { printf("******* Possible Overflow *******\n");  exit(0); }
				printf("INTT MUL XES= %d XLIM %d\n", eU + (DEGREE / L), XLIM);
				if (eU + (DEGREE / L) > XLIM) { printf("******* Possible Overflow *******\n");  exit(0); }

				if (eV > (DEGREE / L)) { printf("******* Possible Overflow *******\n");  exit(0); }

				xes[j] = eU + eV;
				xes[j + t] = 2;
#endif

				W = U + (DEGREE / L) * q - V;
				x[j + t] = modmul(W, S);
			}
			k += 2 * t;
		}
		t *= 2;
		m /= 2;
	}

	/* Last iteration merged with n^-1 */

	t = DEGREE / 2;
	for (j = 0; j < t; j++)
	{
		if (j < L / 2)
		{ // need to knock back excesses. Never used if L=1.
			U = modmul(x[j], ONE);
			V = modmul(x[j + t], ONE);
		}
		else
		{
			U = x[j];
			V = x[j + t];
		}

#ifdef RECORD_EXCESS
		if (j < L / 2)
		{ // need to knock back excesses. Never used if L=1.
			eU = eV = 2;
		}
		else
		{
			eU = xes[j]; eV = xes[j + t];
		}
		printf("INTT ADD XES= %d XLIM %d\n", eU + eV, XLIM);
		if (eU + eV > XLIM) { printf("******* Possible Overflow *******\n"); /*exit(0);*/ }
		printf("INTT MUL XES= %d XLIM %d\n", eU + (DEGREE / L), XLIM);
		if (eU + (DEGREE / L) > XLIM) { printf("******* Possible Overflow *******\n"); /* exit(0);*/ }
		xes[j] = eU + eV;
		xes[j + t] = 2;
#endif

		W = U + (DEGREE / L) * q - V;
		x[j + t] = modmul(W, invpr);
		x[j] = modmul(U + V, inv);
	}
	/* convert back from Montgomery to "normal" form */
	for (j = 0; j < DEGREE; j++)
	{
		x[j] = redc(x[j]);
		x[j] -= q;
		x[j] += (x[j] >> (WL - 1)) & q;
	}
}

int main()
{
	cout << "Hello NTT" << endl;
	int i, j, k, err, q = PRIME;
	int_t a[DEGREE], b[DEGREE], roots[DEGREE], iroots[DEGREE];
	int_t inv, invpr;

	srand(3); /* initialise random number generator */

/* pre-calculate powers of roots of unity */
	init_ntt2(roots, iroots);
	inv = nres(invers(DEGREE, q));
	invpr = modmul(iroots[1], inv);
	inv -= q; inv += (inv >> (WL - 1)) & q;
	invpr -= q; invpr += (invpr >> (WL - 1)) & q;

	/* generate some random data */
	for (i = 0; i < DEGREE; i++)
	{
		a[i] = rand() % q;
		b[i] = a[i];
	}

	printf("q= %d\n", q);

	ntt(roots, b);

	for (i = 0; i < DEGREE; i++) b[i] = modmul(b[i], ONE);

	intt(inv, invpr, iroots, b);

	err = 0;
	for (i = 0; i < DEGREE; i++)
		if (a[i] != b[i]) err++;
	if (!err)	printf("Success x == iNTT(NTT(x))\n");
	else		printf("Failed \n");

	return 0;
}


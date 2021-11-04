from math import log, sqrt, exp, erf
import numpy as np

invsqrt = lambda x: 1.0 / sqrt(x)


def black_scholes_python(nopt, price, strike, t, rate, vol, call, put):
    mr = -rate
    sig_sig_two = vol * vol * 2

    for i in range(nopt):
        P = float(price[i])
        S = strike[i]
        T = t[i]

        a = log(P / S)
        b = T * mr

        z = T * sig_sig_two
        c = 0.25 * z
        y = invsqrt(z)

        w1 = (a - b + c) * y
        w2 = (a - b - c) * y

        d1 = 0.5 + 0.5 * erf(w1)
        d2 = 0.5 + 0.5 * erf(w2)

        Se = exp(b) * S

        call[i] = P * d1 - Se * d2
        put[i] = call[i] - P + Se
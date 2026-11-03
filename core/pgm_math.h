#ifndef PGM_MATH_H_
#define PGM_MATH_H_

/* Header-only numerics shared by every pGM implementation: pgm_core (the reference engine), the OpenMM
 * plugin's host code and the GROMACS module.  One copy, so the implementations cannot drift apart.
 *
 *   erfKernels        radial kernels B_n of erf(a r)/r (the Gaussian-screened multipole kernels)
 *   erfcKernels       the same of erfc(a r)/r
 *   ewaldPairKernels  the PME direct-space pair kernels B(a) - B(beta), in the stable form for each r
 *   bsplineModuli     |b(m)|^-2 of the smooth-PME influence function
 *   ewaldCoefficient  beta with erfc(beta rc) = rtol, in double
 *   minImage          minimum image in a reduced (GROMACS/OpenMM) triclinic box
 */

#include <cmath>
#include <vector>

#include "pgm_constants.h"

namespace pgmcore {

constexpr double SQRT_PI = 1.7724538509055160273;
constexpr double PI = 3.14159265358979323846;

/**
 * B_0..B_{nmax-1} of erf(a r)/r, B_n = -(1/r) d/dr B_{n-1}: the series
 * B_n = (2a/sqrt(pi)) (2a^2)^n sum_m (-s)^m / (m! (2m+2n+1)), s = (a r)^2, for s < 1 (15 terms, no
 * cancellation at short range), the upward recursion B_n = ((2n-1) B_{n-1} - (2a^2)^{n-1} (2a/sqrt(pi)) e^{-s})/r^2
 * elsewhere.  B must hold nmax values.
 */
inline void erfKernels(double a, double r, int nmax, double* B) {
    const double x = a*r, s = x*x;
    const double c0 = 2.0*a/SQRT_PI, a2 = 2.0*a*a;
    if (s < 1.0) {
        double t[15];
        t[0] = 1.0;
        for (int m = 1; m < 15; m++)
            t[m] = t[m-1]*(-s)/m;
        double pw = 1.0;
        for (int k = 0; k < nmax; k++) {
            double sum = 0.0;
            for (int m = 14; m >= 0; m--)
                sum += t[m]/(2*m+2*k+1);
            B[k] = c0*pw*sum;
            pw *= a2;
        }
    }
    else {
        const double c = c0*std::exp(-s), ir2 = 1.0/(r*r);
        B[0] = std::erf(x)/r;
        double pw = 1.0;
        for (int k = 1; k < nmax; k++) {
            B[k] = ((2*k-1)*B[k-1]-pw*c)*ir2;
            pw *= a2;
        }
    }
}

/**
 * B_0..B_{nmax-1} of erfc(a r)/r: Bc_0 = erfc(a r)/r, Bc_n = ((2n-1) Bc_{n-1} + (2a^2)^{n-1} (2a/sqrt(pi)) e^{-(a r)^2})/r^2
 * (all terms positive: no cancellation).  The Ewald pair kernel for a r >= 1 is G = Bc(beta) - Bc(a).
 */
inline void erfcKernels(double a, double r, int nmax, double* B) {
    const double x = a*r, c = (2.0/SQRT_PI)*a*std::exp(-x*x), ir2 = 1.0/(r*r), a2 = 2.0*a*a;
    B[0] = std::erfc(x)/r;
    double pw = 1.0;
    for (int k = 1; k < nmax; k++) {
        B[k] = ((2*k-1)*B[k-1]+pw*c)*ir2;
        pw *= a2;
    }
}

/**
 * The PME direct-space kernels of a pair, G_n = B_n[erf(a r)/r] - B_n[erf(beta r)/r] (n < nmax <= 4):
 * for a r < 1 from the series of both (no cancellation at short range), else Bc(beta) - Bc(a) in the
 * erfc form (no cancellation of the 1/r^(2n+1) parts at long range).
 */
inline void ewaldPairKernels(double a, double beta, double r, int nmax, double* G) {
    double B[4];
    if (a*r < 1.0) {
        erfKernels(a, r, nmax, G);
        erfKernels(beta, r, nmax, B);
        for (int k = 0; k < nmax; k++)
            G[k] -= B[k];
    }
    else {
        erfcKernels(beta, r, nmax, G);
        erfcKernels(a, r, nmax, B);
        for (int k = 0; k < nmax; k++)
            G[k] -= B[k];
    }
}

/**
 * |sum_k M_p(k+1) exp(2 pi i m k / K)|^2 for m = 0..K-1 (M_p the cardinal B-spline of order p); the
 * zeros (even p, m = K/2) are replaced by the mean of their neighbours, as OpenMM and pmemd do.
 */
inline std::vector<double> bsplineModuli(int K, int p) {
    std::vector<double> M(p+1, 0.0), tmp(p+1, 0.0);
    M[0] = 1.0;   // M_1 at fraction 0; M_k(j) for j = 0..k-1 below
    for (int k = 2; k <= p; k++) {
        for (int j = 0; j < k; j++) {
            const double cur = (j < k-1 ? M[j] : 0.0), prev = (j > 0 ? M[j-1] : 0.0);
            tmp[j] = (j*cur+(k-j)*prev)/(k-1);
        }
        for (int j = 0; j < k; j++)
            M[j] = tmp[j];
    }
    std::vector<double> data(K, 0.0);
    for (int k = 0; k < p-1 && k < K; k++)
        data[k] = M[k+1];
    std::vector<double> mod(K);
    for (int m = 0; m < K; m++) {
        double sc = 0, ss = 0;
        for (int k = 0; k < K; k++) {
            const double arg = 2*PI*m*k/K;
            sc += data[k]*std::cos(arg);
            ss += data[k]*std::sin(arg);
        }
        mod[m] = sc*sc+ss*ss;
    }
    for (int m = 0; m < K; m++)
        if (mod[m] < 1e-7)
            mod[m] = 0.5*(mod[(m-1+K)%K]+mod[(m+1)%K]);
    return mod;
}

/** The Ewald coefficient beta with erfc(beta rc) = rtol (bisection in double, as GROMACS's calc_ewaldcoeff_q). */
inline double ewaldCoefficient(double rc, double rtol) {
    double beta = 5, low = 0, high;
    int n = 0;
    do {
        n++;
        beta *= 2;
    } while (std::erfc(beta*rc) > rtol);
    n += 60;
    high = beta;
    for (int i = 0; i < n; i++) {
        beta = 0.5*(low+high);
        if (std::erfc(beta*rc) > rtol)
            low = beta;
        else
            high = beta;
    }
    return beta;
}

/**
 * Minimum image of a difference vector v in a reduced triclinic box (rows a, b, c with a = (ax, 0, 0),
 * b = (bx, by, 0), the GROMACS and OpenMM convention): c first, then b, then a.  V: any type with
 * v[0..2]; Box: box[row][column].
 */
template<class V, class Box>
inline void minImage(V& v, const Box& box) {
    for (int c = 2; c >= 0; c--) {
        const double k = std::round(v[c]/box[c][c]);
        for (int j = 0; j < 3; j++)
            v[j] -= k*box[c][j];
    }
}

} // namespace pgmcore

#endif

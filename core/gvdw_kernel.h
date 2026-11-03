#ifndef PGM_GVDW_KERNEL_H_
#define PGM_GVDW_KERNEL_H_

/* GVDW: the Gaussian-density van der Waals of Huang, Luo and Duan (pmemd-pgm igvdw = 1, pGM-JAX vdw.py).
 *
 * With the Gaussian pair exponent of the pGM electrostatics b_ij = 1/sqrt(2 (R_i^2 + R_j^2)) and
 * y = b_ij r, the pair energy is
 *
 *     U_ij(r) = A_ij exp(-B_ij y^2)      (Gaussian repulsion, pmemd gvdw_rep_form = 0)
 *             | A_ij exp(-B_ij y)        (Slater / Born-Mayer, gvdw_rep_form = 1)
 *               - C6_ij b_ij^6 G(y),     G(y) = F(y)/y^6,
 *
 *     F(y) = A(y)^2 + g(y)^2/2,  A(y) = erf(y) - (2y/sqrt(pi))(1 + 2y^2/3) e^{-y^2},
 *                                 g(y) = (4 y^3/(3 sqrt(pi))) e^{-y^2},
 *
 * finite at r = 0 (G(0) = 8/(9 pi)).  Combination: A_ij = a_i a_j, C6_ij = c_i c_j,
 * B_ij = (b_i + b_j)/2 from per-atom a = sqrt(A), c = sqrt(C6), b.  G is evaluated from its Taylor
 * series in y^2 (15 terms, < 1e-13 relative) for y < 0.6 and in closed form above (pGM-JAX's split).
 *
 * Units: nm, kJ/mol; A [kJ/mol], C6 [kJ/mol nm^6], B dimensionless.
 */

#include <cmath>

namespace pgmcore {

enum GVDWRepulsion { GVDWGaussian = 0, GVDWSlater = 1 };

/** G(y) = F(y)/y^6 and H(y) = G'(y)/y. */
inline void gvdwG(double y, double& G, double& H) {
    static const double C0 = 8.0/(9.0*M_PI);
    static const double c[15] = {1.0, -2.0, 58.0/25, -188.0/105, 2222.0/2205, -1532.0/3465, 64244.0/405405,
            -162056.0/3378375, 1587734.0/126351225, -211108.0/72747675, 11902564.0/19860115275.0,
            -611992.0/5465775315.0, 755028172.0/39529267903125.0, -213335384.0/71152682225625.0,
            15291334888.0/35078272337233125.0};
    if (y < 0.6) {
        double y2 = y*y, g = 0, h = 0;
        for (int k = 14; k >= 0; k--)
            g = g*y2+c[k];
        for (int k = 14; k >= 1; k--)
            h = h*y2+2.0*k*c[k];
        G = C0*g;
        H = C0*h;
        return;
    }
    const double isp = 1.0/std::sqrt(M_PI);
    double y2 = y*y, e = std::exp(-y2);
    double A = std::erf(y)-2.0*isp*y*(1.0+2.0*y2/3.0)*e;
    double g = (4.0/3.0)*isp*y2*y*e;
    double F = A*A+0.5*g*g;
    double Ap = (8.0/3.0)*isp*y2*y2*e;
    double gp = 4.0*isp*y2*(1.0-2.0*y2/3.0)*e;
    double Fp = 2.0*A*Ap+g*gp;
    double y6 = y2*y2*y2;
    G = F/y6;
    H = (Fp/y6-6.0*F/(y6*y))/y;
}

/**
 * Energy of one pair [kJ/mol]; dEdR = -(1/r) dU/dr [kJ/mol/nm^2] (force on i = dEdR (r_i - r_j)).
 * bij = 1/sqrt(2 (R_i^2 + R_j^2)) [1/nm], the Gaussian pair exponent (not the Ewald coefficient).
 * At r = 0 the energy is finite; the Slater force there is set to 0 (its direction is undefined).
 */
inline double gvdwPair(double r, double bij, double A, double C6, double B, int rep, double& dEdR) {
    double y = bij*r, b2 = bij*bij, b6 = b2*b2*b2;
    double erep, drep;   // drep = (1/r) dU_rep/dr
    if (rep == GVDWGaussian) {
        erep = A*std::exp(-B*y*y);
        drep = -2.0*B*b2*erep;
    }
    else {
        erep = A*std::exp(-B*y);
        drep = (r > 0 ? -B*bij*erep/r : 0.0);
    }
    double G, H;
    gvdwG(y, G, H);
    dEdR = -(drep-C6*b6*b2*H);
    return erep-C6*b6*G;
}

/** Dispersion tail beyond rc (Amber vdwmeth = 1 for the r^-6 term, F = 1): -2 pi (sum c)^2/(3 V rc^3). */
inline double gvdwTail(double sumSqrtC6, double volume, double rc) {
    return -2.0*M_PI*sumSqrtC6*sumSqrtC6/(3.0*volume*rc*rc*rc);
}

} // namespace pgmcore

#endif

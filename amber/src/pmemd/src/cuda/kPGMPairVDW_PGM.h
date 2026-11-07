//---------------------------------------------------------------------------------------------
// kPGMPairVDW_PGM.h - van der Waals pair term of the pGM direct-space kernels (LJ or GVDW).
// Include at file scope after the cSim declaration, before the kernels that include
// kNLCPNE_PGM.h.
//---------------------------------------------------------------------------------------------
#undef PGM_EXP
#ifdef use_DPFP
#define PGM_EXP(x) exp(x)
#else
#define PGM_EXP(x) __expf(x)
#endif
#ifndef PGM_PAIR_VDW_DEFINED
#define PGM_PAIR_VDW_DEFINED
//---------------------------------------------------------------------------------------------
// PGMPairVDW: van der Waals energy e and force factor df = (-dU/dr)/r for one pair.
//   cSim.pgmIGVDW == 0: Lennard-Jones 12-6 (ljterm.x = 12 A, ljterm.y = 6 B, AMBER convention)
//   cSim.pgmIGVDW == 1: Gaussian VDW, U = A_ij exp(-b y^n) - C6_ij F(y)/r^6, y = kappa_ij r,
//                       A_ij = ljterm.x/12, C6_ij = ljterm.y/6 (topology ACOEF/BCOEF), n = 2 or 1
//---------------------------------------------------------------------------------------------
__device__ __forceinline__ void PGMPairVDW(PMEFloat2 ljterm, PMEFloat r2inv, PMEFloat rinv,
                                           PMEFloat kappa, PMEFloat y, PMEFloat erf_y,
                                           PMEFloat &e, PMEFloat &df)
{
  PMEFloat r6inv = r2inv * r2inv * r2inv;
  if (cSim.pgmIGVDW == 0) {
    PMEFloat f6  = ljterm.y * r6inv;
    PMEFloat f12 = ljterm.x * r6inv * r6inv;
    df = (f12 - f6) * r2inv;
    e  = f12 * (PMEFloat)(1.0 / 12.0) - f6 * (PMEFloat)(1.0 / 6.0);
    return;
  }
  const PMEFloat ispi = (PMEFloat)0.56418958354775628695;
  PMEFloat Arep = ljterm.x * (PMEFloat)(1.0 / 12.0);
  PMEFloat C6   = ljterm.y * (PMEFloat)(1.0 / 6.0);
  PMEFloat b    = cSim.pgmBRepScale;
  PMEFloat y2   = y * y;
  PMEFloat ey2  = PGM_EXP(-y2);
  PMEFloat erep;
  if (cSim.pgmGVDWRepForm == 0) {
    erep = Arep * PGM_EXP(-b * y2);
    df   = (PMEFloat)2.0 * b * kappa * kappa * erep;
  } else {
    erep = Arep * PGM_EXP(-b * y);
    df   = b * kappa * erep * rinv;
  }
  PMEFloat g  = (PMEFloat)(4.0 / 3.0) * ispi * y * y2 * ey2;
  PMEFloat A  = erf_y - (PMEFloat)2.0 * ispi * y * ey2 - g;
  PMEFloat F  = A * A + (PMEFloat)0.5 * g * g;
  if (y > (PMEFloat)0.01) {
    PMEFloat Ap = (PMEFloat)(8.0 / 3.0) * ispi * y2 * y2 * ey2;
    PMEFloat gp = (PMEFloat)4.0 * ispi * y2 * ((PMEFloat)1.0 - (PMEFloat)(2.0 / 3.0) * y2) * ey2;
    PMEFloat Fp = (PMEFloat)2.0 * A * Ap + g * gp;
    e   = erep - C6 * F * r6inv;
    df += C6 * (kappa * Fp * rinv - (PMEFloat)6.0 * F * r2inv) * r6inv;
  } else {
    PMEFloat k3 = kappa * kappa * kappa;
    e = erep - C6 * (PMEFloat)(8.0 / (9.0 * 3.14159265358979323846)) * k3 * k3;
  }
}
#endif

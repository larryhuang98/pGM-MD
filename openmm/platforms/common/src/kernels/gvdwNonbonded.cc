// GVDW pairs from the combination rule, inserted into OpenMM's nonbonded tile kernel (NonbondedUtilities).
// Per-atom parameters PARAMS1 / PARAMS2 (real4): (R^2, sqrt(A), sqrt(C6), b).  Exception pairs are
// excluded here and evaluated by gvdwExceptions.cc.
#ifdef USE_CUTOFF
if (!isExcluded && r2 < CUTOFF_SQUARED) {
#else
if (!isExcluded) {
#endif
    const real4 gp1 = PARAMS1;
    const real4 gp2 = PARAMS2;
    const real gA = gp1.y*gp2.y;
    const real gC6 = gp1.z*gp2.z;
    if (gA != 0 || gC6 != 0) {
        const real gBeta = RSQRT(2*(gp1.x+gp2.x));
        const real gB = (real) 0.5*(gp1.w+gp2.w);
        const real gR = SQRT(r2);                          // not OpenMM's r = r2*invR: NaN at r = 0
        const real gInvR = (r2 > 0 ? RSQRT(r2) : (real) 0);
        real gEnergy, gDEdR;
        GVDW_PAIR
        tempEnergy += gEnergy;
        dEdR += gDEdR;
    }
}

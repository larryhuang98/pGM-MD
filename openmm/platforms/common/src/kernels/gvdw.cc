// GVDW pair interaction, inserted into OpenMM's nonbonded tile kernel (NonbondedUtilities).
// Per-atom parameters PARAMS (real4): (R^2, sqrt(A), sqrt(C6), b); see core/gvdw_kernel.h.
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
        const real gy = gBeta*r;
        const real gy2 = gy*gy;
        const real gb2 = gBeta*gBeta;
        const real gb6 = gb2*gb2*gb2;
#if GVDW_SLATER
        const real gErep = gA*EXP(-gB*gy);
        const real gDrep = -gB*gBeta*gErep*invR;      // (1/r) dU_rep/dr
#else
        const real gErep = gA*EXP(-gB*gy2);
        const real gDrep = -2*gB*gb2*gErep;
#endif
        real gG, gH;                                   // G = F/y^6, H = G'/y
        if (gy < (real) 0.6) {
            gG = (real) (15291334888.0/35078272337233125.0);
            gG = gG*gy2+(real) (-213335384.0/71152682225625.0);
            gG = gG*gy2+(real) (755028172.0/39529267903125.0);
            gG = gG*gy2+(real) (-611992.0/5465775315.0);
            gG = gG*gy2+(real) (11902564.0/19860115275.0);
            gG = gG*gy2+(real) (-211108.0/72747675.0);
            gG = gG*gy2+(real) (1587734.0/126351225.0);
            gG = gG*gy2+(real) (-162056.0/3378375.0);
            gG = gG*gy2+(real) (64244.0/405405.0);
            gG = gG*gy2+(real) (-1532.0/3465.0);
            gG = gG*gy2+(real) (2222.0/2205.0);
            gG = gG*gy2+(real) (-188.0/105.0);
            gG = gG*gy2+(real) (58.0/25.0);
            gG = gG*gy2+(real) (-2.0);
            gG = gG*gy2+(real) 1.0;
            gH = (real) (28*15291334888.0/35078272337233125.0);
            gH = gH*gy2+(real) (26*-213335384.0/71152682225625.0);
            gH = gH*gy2+(real) (24*755028172.0/39529267903125.0);
            gH = gH*gy2+(real) (22*-611992.0/5465775315.0);
            gH = gH*gy2+(real) (20*11902564.0/19860115275.0);
            gH = gH*gy2+(real) (18*-211108.0/72747675.0);
            gH = gH*gy2+(real) (16*1587734.0/126351225.0);
            gH = gH*gy2+(real) (14*-162056.0/3378375.0);
            gH = gH*gy2+(real) (12*64244.0/405405.0);
            gH = gH*gy2+(real) (10*-1532.0/3465.0);
            gH = gH*gy2+(real) (8*2222.0/2205.0);
            gH = gH*gy2+(real) (6*-188.0/105.0);
            gH = gH*gy2+(real) (4*58.0/25.0);
            gH = gH*gy2+(real) (2*-2.0);
            gG *= (real) GVDW_C0;
            gH *= (real) GVDW_C0;
        }
        else {
            const real ge = EXP(-gy2);
            const real gAy = erf(gy)-(real) GVDW_2_SQRTPI*gy*(1+(real) (2.0/3.0)*gy2)*ge;
            const real ggy = (real) (2.0/3.0*GVDW_2_SQRTPI)*gy2*gy*ge;
            const real gF = gAy*gAy+(real) 0.5*ggy*ggy;
            const real gAp = (real) (4.0/3.0*GVDW_2_SQRTPI)*gy2*gy2*ge;
            const real ggp = (real) (2*GVDW_2_SQRTPI)*gy2*(1-(real) (2.0/3.0)*gy2)*ge;
            const real gFp = 2*gAy*gAp+ggy*ggp;
            const real giy = RECIP(gy);
            const real giy6 = giy*giy*giy*giy*giy*giy;
            gG = gF*giy6;
            gH = (gFp*giy6-6*gF*giy6*giy)*giy;
        }
        tempEnergy += gErep-gC6*gb6*gG;
        dEdR -= gDrep-gC6*gb6*gb2*gH;
    }
}

/**
 * Per-atom kernels of pGM: covalent dipoles, the conjugate-gradient solve of the induced dipoles, the
 * self / polarization energies and the covalent-dipole forces.
 *
 * Layouts: real vectors 3*i+k; fixed-point field buffers k*PADDED_NUM_ATOMS+i (scale 2^32).
 * The operator of the solve is A = alpha^-1 - T with T mu = (direct + reciprocal field of mu) + CSELF mu.
 * Scalars of the CG (mixed): [0] r.z, [1] mean|alpha b|, [2] max|alpha r|/norm, [3] CG step, [4] beta.
 */

#define FIELD_SCALE (1/(real) 0x100000000)

DEVICE real3 loadField(GLOBAL const mm_long* RESTRICT f, int i) {
    return make_real3(FIELD_SCALE*(real) f[i], FIELD_SCALE*(real) f[i+PADDED_NUM_ATOMS], FIELD_SCALE*(real) f[i+2*PADDED_NUM_ATOMS]);
}

DEVICE real3 loadVec(GLOBAL const real* RESTRICT v, int i) {
    return make_real3(v[3*i], v[3*i+1], v[3*i+2]);
}

DEVICE void storeVec(GLOBAL real* RESTRICT v, int i, real3 x) {
    v[3*i] = x.x;
    v[3*i+1] = x.y;
    v[3*i+2] = x.z;
}

/**
 * p_i = sum over the covalent dipoles of atom i of c unit(r_partner - r_i) (minimum image).
 */
KERNEL void computePermanentDipoles(GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT covStart,
        GLOBAL const int* RESTRICT covPartner, GLOBAL const real* RESTRICT covStrength, GLOBAL real* RESTRICT permDipole
#ifdef USE_PERIODIC
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {
    for (int i = GLOBAL_ID; i < PADDED_NUM_ATOMS; i += GLOBAL_SIZE) {
        real3 p = make_real3(0);
        if (i < NUM_ATOMS) {
            real4 pos1 = posq[i];
            for (int m = covStart[i]; m < covStart[i+1]; m++) {
                real4 pos2 = posq[covPartner[m]];
                real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                p += delta*(covStrength[m]*RSQRT(dot(delta, delta)));
            }
        }
        storeVec(permDipole, i, p);
    }
}

/**
 * Initial guess of the induced dipoles: mode 0 alpha (E_fixed + CSELF p), 1 the last dipoles,
 * 2 quadratic (3 h0 - 3 h1 + h2), 3 cubic (4 h0 - 6 h1 + 4 h2 - h3) extrapolation.
 */
KERNEL void initialGuess(GLOBAL const mm_long* RESTRICT fixedField, GLOBAL const real* RESTRICT permDipole,
        GLOBAL const real* RESTRICT polarizability, GLOBAL const real* RESTRICT h0, GLOBAL const real* RESTRICT h1,
        GLOBAL const real* RESTRICT h2, GLOBAL const real* RESTRICT h3, GLOBAL real* RESTRICT x, int mode) {
    for (int i = GLOBAL_ID; i < PADDED_NUM_ATOMS; i += GLOBAL_SIZE) {
        real3 v;
        real a = polarizability[i];
        if (mode == 0)
            v = (loadField(fixedField, i) + loadVec(permDipole, i)*((real) CSELF))*a;
        else if (mode == 1)
            v = loadVec(h0, i);
        else if (mode == 2)
            v = loadVec(h0, i)*3 - loadVec(h1, i)*3 + loadVec(h2, i);
        else
            v = loadVec(h0, i)*4 - loadVec(h1, i)*6 + loadVec(h2, i)*4 - loadVec(h3, i);
        if (a == 0 || i >= NUM_ATOMS)
            v = make_real3(0);
        storeVec(x, i, v);
    }
}

// ---------------------------------------------------------------- single-work-group CG kernels

DEVICE void reduceSum(LOCAL_ARG mixed* buf, int n) {
    for (int step = n/2; step > 0; step >>= 1) {
        if (LOCAL_ID < step)
            buf[LOCAL_ID] += buf[LOCAL_ID+step];
        SYNC_THREADS;
    }
}

DEVICE void reduceMax(LOCAL_ARG mixed* buf, int n) {
    for (int step = n/2; step > 0; step >>= 1) {
        if (LOCAL_ID < step)
            buf[LOCAL_ID] = max(buf[LOCAL_ID], buf[LOCAL_ID+step]);
        SYNC_THREADS;
    }
}

/**
 * b = E_fixed + CSELF p; r = b - (x/alpha - T x) with T x = inducedField + CSELF x; z = alpha r; p_dir = z.
 * One work group of CG_GROUP_SIZE threads.
 */
KERNEL void cgInit(GLOBAL const mm_long* RESTRICT fixedField, GLOBAL const mm_long* RESTRICT inducedField,
        GLOBAL const real* RESTRICT permDipole, GLOBAL const real* RESTRICT polarizability, GLOBAL const real* RESTRICT x,
        GLOBAL real* RESTRICT r, GLOBAL real* RESTRICT z, GLOBAL real* RESTRICT pdir, GLOBAL mixed* RESTRICT scalars) {
    LOCAL mixed s1[CG_GROUP_SIZE], s2[CG_GROUP_SIZE], s3[CG_GROUP_SIZE];
    mixed rz = 0, nrm = 0, err = 0;
    for (int i = LOCAL_ID; i < NUM_ATOMS; i += CG_GROUP_SIZE) {
        real a = polarizability[i];
        real3 rr = make_real3(0);
        real3 b = loadField(fixedField, i) + loadVec(permDipole, i)*((real) CSELF);
        if (a > 0) {
            real3 xi = loadVec(x, i);
            real3 Tx = loadField(inducedField, i) + xi*((real) CSELF);
            rr = b - (xi*RECIP(a) - Tx);
        }
        real3 zz = rr*a;
        storeVec(r, i, rr);
        storeVec(z, i, zz);
        storeVec(pdir, i, zz);
        rz += dot(rr, zz);
        nrm += fabs(a*b.x)+fabs(a*b.y)+fabs(a*b.z);
        err = max(err, (mixed) max(fabs(zz.x), max(fabs(zz.y), fabs(zz.z))));
    }
    s1[LOCAL_ID] = rz;
    s2[LOCAL_ID] = nrm;
    s3[LOCAL_ID] = err;
    SYNC_THREADS;
    reduceSum(s1, CG_GROUP_SIZE);
    reduceSum(s2, CG_GROUP_SIZE);
    reduceMax(s3, CG_GROUP_SIZE);
    if (LOCAL_ID == 0) {
        mixed norm = s2[0]/(3*NUM_ATOMS)+1e-30;
        scalars[0] = s1[0];
        scalars[1] = norm;
        scalars[2] = s3[0]/norm;
    }
}

/**
 * One preconditioned CG iteration, given inducedField = direct + reciprocal field of p_dir.
 * Ap = p/alpha - T p; step = rz / p.Ap; x += step p; r' = r - step Ap; z' = alpha r';
 * beta = z'.(r' - r) / rz (flexible, Polak-Ribiere); p = z' + beta p.
 */
KERNEL void cgStep(GLOBAL const mm_long* RESTRICT inducedField, GLOBAL const real* RESTRICT polarizability,
        GLOBAL real* RESTRICT x, GLOBAL real* RESTRICT r, GLOBAL real* RESTRICT z, GLOBAL real* RESTRICT pdir,
        GLOBAL real* RESTRICT Ap, GLOBAL mixed* RESTRICT scalars) {
    LOCAL mixed s1[CG_GROUP_SIZE], s2[CG_GROUP_SIZE], s3[CG_GROUP_SIZE];
    mixed pAp = 0;
    for (int i = LOCAL_ID; i < NUM_ATOMS; i += CG_GROUP_SIZE) {
        real a = polarizability[i];
        real3 v = make_real3(0);
        if (a > 0) {
            real3 pi = loadVec(pdir, i);
            v = pi*RECIP(a) - (loadField(inducedField, i) + pi*((real) CSELF));
            pAp += dot(pi, v);
        }
        storeVec(Ap, i, v);
    }
    s1[LOCAL_ID] = pAp;
    SYNC_THREADS;
    reduceSum(s1, CG_GROUP_SIZE);
    mixed rzOld = scalars[0];
    mixed denom = s1[0];
    real step = (real) (rzOld/(denom > 1e-30 ? denom : 1e-30));
    SYNC_THREADS;
    mixed num = 0, rz = 0, err = 0;
    for (int i = LOCAL_ID; i < NUM_ATOMS; i += CG_GROUP_SIZE) {
        real a = polarizability[i];
        real3 pi = loadVec(pdir, i);
        real3 ri = loadVec(r, i);
        storeVec(x, i, loadVec(x, i) + pi*step);
        real3 rn = ri - loadVec(Ap, i)*step;
        real3 zn = rn*a;
        num += dot(zn, rn-ri);
        rz += dot(rn, zn);
        err = max(err, (mixed) max(fabs(zn.x), max(fabs(zn.y), fabs(zn.z))));
        storeVec(r, i, rn);
        storeVec(z, i, zn);
    }
    s1[LOCAL_ID] = num;
    s2[LOCAL_ID] = rz;
    s3[LOCAL_ID] = err;
    SYNC_THREADS;
    reduceSum(s1, CG_GROUP_SIZE);
    reduceSum(s2, CG_GROUP_SIZE);
    reduceMax(s3, CG_GROUP_SIZE);
    real beta = (real) (s1[0]/(rzOld > 1e-30 ? rzOld : 1e-30));
    for (int i = LOCAL_ID; i < NUM_ATOMS; i += CG_GROUP_SIZE)
        storeVec(pdir, i, loadVec(z, i) + loadVec(pdir, i)*beta);
    if (LOCAL_ID == 0) {
        scalars[0] = s2[0];
        scalars[2] = s3[0]/scalars[1];
        scalars[3] = step;
        scalars[4] = beta;
    }
}

// ---------------------------------------------------------------- after the solve

/** x += peek alpha r (the final SOR step); copy x into the newest history slot. */
KERNEL void cgFinish(GLOBAL real* RESTRICT x, GLOBAL const real* RESTRICT r, GLOBAL const real* RESTRICT polarizability,
        GLOBAL real* RESTRICT hist, real peek) {
    for (int i = GLOBAL_ID; i < PADDED_NUM_ATOMS; i += GLOBAL_SIZE) {
        real3 v = loadVec(x, i) + loadVec(r, i)*(peek*polarizability[i]);
        storeVec(x, i, v);
        storeVec(hist, i, v);
    }
}

KERNEL void totalDipoles(GLOBAL const real* RESTRICT permDipole, GLOBAL const real* RESTRICT induced, GLOBAL real* RESTRICT total) {
    for (int i = GLOBAL_ID; i < 3*PADDED_NUM_ATOMS; i += GLOBAL_SIZE)
        total[i] = permDipole[i]+induced[i];
}

/**
 * Self energy -(beta/sqrt(pi)) q^2 - (CSELF/2) |d|^2 (PME) and polarization energy |mu|^2 / (2 alpha).
 */
KERNEL void selfEnergy(GLOBAL const real* RESTRICT charge, GLOBAL const real* RESTRICT total, GLOBAL const real* RESTRICT induced,
        GLOBAL const real* RESTRICT polarizability, GLOBAL mixed* RESTRICT energyBuffer) {
    mixed energy = 0;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        real a = polarizability[i];
        real3 d = loadVec(total, i);
        real3 mu = loadVec(induced, i);
#ifdef USE_EWALD
        real q = charge[i];
        energy -= (EWALD_ALPHA/SQRT_PI)*q*q + 0.5f*CSELF*dot(d, d);
#endif
        if (a > 0)
            energy += dot(mu, mu)/(2*a);
    }
    energyBuffer[GLOBAL_ID] += ENERGY_SCALE*energy;
}

/**
 * Forces of the covalent dipoles: with g = dE/dd_i = -KE E_i (total field at fixed dipoles, incl. self),
 * the dipole m = (i, j), v = r_j - r_i, adds dE/dr_j = c (g - (g.u) u)/|v| and dE/dr_i = -dE/dr_j.
 */
KERNEL void covalentForces(GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT covAtom, GLOBAL const int* RESTRICT covPartner,
        GLOBAL const real* RESTRICT covStrength, GLOBAL const mm_long* RESTRICT totalField, GLOBAL const real* RESTRICT total,
        GLOBAL mm_ulong* RESTRICT forceBuffer
#ifdef USE_PERIODIC
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {
    for (int m = GLOBAL_ID; m < NUM_COVALENT; m += GLOBAL_SIZE) {
        int i = covAtom[m], j = covPartner[m];
        real4 pos1 = posq[i];
        real4 pos2 = posq[j];
        real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(delta)
#endif
        real invL = RSQRT(dot(delta, delta));
        real3 u = delta*invL;
        real3 E = loadField(totalField, i) + loadVec(total, i)*((real) CSELF);
        real3 g = E*(-(real) ENERGY_SCALE);
        real3 t = (g - u*dot(g, u))*(covStrength[m]*invL);
        ATOMIC_ADD(&forceBuffer[j], (mm_ulong) realToFixedPoint(-t.x));
        ATOMIC_ADD(&forceBuffer[j+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-t.y));
        ATOMIC_ADD(&forceBuffer[j+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-t.z));
        ATOMIC_ADD(&forceBuffer[i], (mm_ulong) realToFixedPoint(t.x));
        ATOMIC_ADD(&forceBuffer[i+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(t.y));
        ATOMIC_ADD(&forceBuffer[i+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(t.z));
    }
}

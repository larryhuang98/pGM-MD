/**
 * pGM pair kernels over the nonbonded tiles (all pairs, no exclusions: PGMForce requests none, so the
 * exclusion tiles of the context, which belong to other forces, are computed in full except i == j).
 *
 * Compiled three times:
 *   PAIR_KIND 0   field of dipoles only (the induced-dipole matrix-vector product)
 *   PAIR_KIND 1   field of charges and dipoles (the permanent field)
 *   PAIR_KIND 2   energy, forces and field of charges and dipoles
 *
 * Kernel G_n = B_n[erf(a r)/r] - B_n[erf(beta r)/r] (PME) or B_n[erf(a r)/r] (no cutoff), with
 * a = 1/sqrt(rg_i + rg_j), rg = 2 R^2.  x = r_1 - r_2.  Fields in e/nm^2 (no Coulomb constant),
 * forces and energies scaled by ENERGY_SCALE (the Coulomb constant).
 */

#if PAIR_KIND == 2
    #define NUM_B 4
#else
    #define NUM_B 3
#endif

/**
 * B_0..B_{NUM_B-1} of erf(a r)/r: Taylor series for (a r)^2 < 1, closed form elsewhere, and the point
 * Coulomb kernel when erfc(a r) is negligible (a r > DAMP_CUT).
 */
DEVICE void pgmKernels(real a, real r, real invR, real* B) {
    real x = a*r;
    real s = x*x;
    real c0 = (2/SQRT_PI)*a;
    real a2 = 2*a*a;
    if (s < 1) {
        real t = 1, pw = 1;
        real sum[NUM_B];
        for (int k = 0; k < NUM_B; k++)
            sum[k] = 0;
        for (int m = 0; m < NSERIES; m++) {
            for (int k = 0; k < NUM_B; k++)
                sum[k] += t/(2*m+2*k+1);
            t *= -s/(m+1);
        }
        for (int k = 0; k < NUM_B; k++) {
            B[k] = c0*pw*sum[k];
            pw *= a2;
        }
    }
    else if (x > DAMP_CUT) {
        real invR2 = invR*invR;
        B[0] = invR;
        for (int k = 1; k < NUM_B; k++)
            B[k] = (2*k-1)*B[k-1]*invR2;
    }
    else {
        real c = c0*EXP(-s);
        real invR2 = invR*invR;
        B[0] = erf(x)*invR;
        real pw = 1;
        for (int k = 1; k < NUM_B; k++) {
            B[k] = ((2*k-1)*B[k-1]-pw*c)*invR2;
            pw *= a2;
        }
    }
}

/**
 * B_0..B_{NUM_B-1} of erfc(a r)/r: Bc_n = ((2n-1) Bc_{n-1} + (2a^2)^(n-1) 2a exp(-a^2 r^2)/sqrt(pi)) / r^2.
 */
DEVICE void erfcKernels(real a, real r, real invR, real* B) {
    real x = a*r;
    real c = (2/SQRT_PI)*a*EXP(-x*x);
    real invR2 = invR*invR;
    real a2 = 2*a*a;
    B[0] = erfc(x)*invR;
    real pw = 1;
    for (int k = 1; k < NUM_B; k++) {
        B[k] = ((2*k-1)*B[k-1]+pw*c)*invR2;
        pw *= a2;
    }
}

/**
 * G_n of a pair.  With Ewald: for a r >= 1, G = Bc(beta) - Bc(a) from erfc (no cancellation of the
 * 1/r^(2n+1) parts, which matters in single precision), Bc(a) dropped beyond DAMP_CUT; for a r < 1,
 * G = B(a) - B(beta) from the series.  Without Ewald: B(a).
 */
DEVICE void pairKernels(real rg1, real rg2, real r, real invR, real* G) {
    real a = RSQRT(rg1+rg2);
#ifdef USE_EWALD
    real x = a*r;
    real Bb[NUM_B];
    if (x >= 1) {
        erfcKernels((real) EWALD_ALPHA, r, invR, G);
        if (x <= DAMP_CUT) {
            erfcKernels(a, r, invR, Bb);
            for (int k = 0; k < NUM_B; k++)
                G[k] -= Bb[k];
        }
    }
    else {
        pgmKernels(a, r, invR, G);
        pgmKernels((real) EWALD_ALPHA, r, invR, Bb);
        for (int k = 0; k < NUM_B; k++)
            G[k] -= Bb[k];
    }
#else
    pgmKernels(a, r, invR, G);
#endif
}

typedef struct {
    real x, y, z;
    real q, rg;
    real dx, dy, dz;
    real ex, ey, ez;      // field accumulated on this atom
#if PAIR_KIND == 2
    real fx, fy, fz;      // force accumulated on this atom
#endif
} AtomData;

/**
 * One pair: delta = r_2 - r_1.  Adds the field (and force, energy) on atom 1 to the registers and on
 * atom 2 to its local data unless isDiagonal.
 */
DEVICE void computePair(real3 delta, real r2, real q1, real rg1, real3 d1, AtomData* data2, bool isDiagonal,
        real3* field1
#if PAIR_KIND == 2
        , real3* force1, mixed* energy
#endif
        ) {
    // finite at r = 0 (two sites on top of each other): the kernels use the series there
    real r = SQRT(r2);
    real invR = (r2 > 0 ? RSQRT(r2) : (real) 0);
    real G[NUM_B];
    pairKernels(rg1, data2->rg, r, invR, G);
    real3 x = make_real3(-delta.x, -delta.y, -delta.z);
    real3 d2 = make_real3(data2->dx, data2->dy, data2->dz);
    real d1x = dot(d1, x), d2x = dot(d2, x);
#if PAIR_KIND == 0
    real3 e1 = x*(d2x*G[2]) - d2*G[1];
    real3 e2 = x*(d1x*G[2]) - d1*G[1];
#else
    real q2 = data2->q;
    real3 e1 = x*(q2*G[1]+d2x*G[2]) - d2*G[1];
    real3 e2 = x*(-q1*G[1]+d1x*G[2]) - d1*G[1];
#endif
    *field1 += e1;
#if PAIR_KIND == 2
    real d1d2 = dot(d1, d2);
    real cross = q1*d2x-q2*d1x;
    real e = q1*q2*G[0] + cross*G[1] + d1d2*G[1] - d1x*d2x*G[2];
    real sc = -q1*q2*G[1] - cross*G[2] - d1d2*G[2] + d1x*d2x*G[3];
    real3 dedx = x*sc + (d2*q1 - d1*q2)*G[1] - (d1*d2x + d2*d1x)*G[2];
    dedx *= (real) ENERGY_SCALE;
    *force1 -= dedx;
    *energy += (isDiagonal ? 0.5f : 1.0f)*ENERGY_SCALE*e;
#endif
    if (!isDiagonal) {
        data2->ex += e2.x;
        data2->ey += e2.y;
        data2->ez += e2.z;
#if PAIR_KIND == 2
        data2->fx += dedx.x;
        data2->fy += dedx.y;
        data2->fz += dedx.z;
#endif
    }
}

DEVICE void loadAtom(AtomData* data, real4 pos, int atom, GLOBAL const real* RESTRICT charge, GLOBAL const real* RESTRICT rgauss,
        GLOBAL const real* RESTRICT dipole) {
    data->x = pos.x;
    data->y = pos.y;
    data->z = pos.z;
#if PAIR_KIND == 0
    data->q = 0;
#else
    data->q = charge[atom];
#endif
    data->rg = rgauss[atom];
    data->dx = dipole[3*atom];
    data->dy = dipole[3*atom+1];
    data->dz = dipole[3*atom+2];
    data->ex = 0;
    data->ey = 0;
    data->ez = 0;
#if PAIR_KIND == 2
    data->fx = 0;
    data->fy = 0;
    data->fz = 0;
#endif
}

KERNEL void computePairs(GLOBAL const real4* RESTRICT posq, GLOBAL const int2* RESTRICT exclusionTiles,
        GLOBAL mm_ulong* RESTRICT fieldBuffer,
#if PAIR_KIND == 2
        GLOBAL mm_ulong* RESTRICT forceBuffer, GLOBAL mixed* RESTRICT energyBuffer,
#endif
        GLOBAL const real* RESTRICT charge, GLOBAL const real* RESTRICT rgauss, GLOBAL const real* RESTRICT dipole,
#ifdef USE_CUTOFF
        GLOBAL const int* RESTRICT tiles, GLOBAL const unsigned int* RESTRICT interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, GLOBAL const real4* RESTRICT blockCenter,
        GLOBAL const real4* RESTRICT blockSize, GLOBAL const unsigned int* RESTRICT interactingAtoms
#else
        unsigned int numTiles
#endif
        ) {
    const unsigned int totalWarps = (GLOBAL_SIZE)/TILE_SIZE;
    const unsigned int warp = (GLOBAL_ID)/TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1);
    const unsigned int tbx = LOCAL_ID - tgx;
    LOCAL AtomData localData[THREAD_BLOCK_SIZE];
#if PAIR_KIND == 2
    mixed energy = 0;
#endif

    // First loop: the tiles that contain exclusions of other forces (computed in full here).

    const unsigned int firstExclusionTile = warp*NUM_TILES_WITH_EXCLUSIONS/totalWarps;
    const unsigned int lastExclusionTile = (warp+1)*NUM_TILES_WITH_EXCLUSIONS/totalWarps;
    for (int tile = firstExclusionTile; tile < lastExclusionTile; tile++) {
        const int2 tileIndices = exclusionTiles[tile];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        real3 field = make_real3(0);
#if PAIR_KIND == 2
        real3 force = make_real3(0);
#endif
        unsigned int atom1 = x*TILE_SIZE + tgx;
        real4 pos1 = posq[atom1];
        AtomData data1;
        loadAtom(&data1, pos1, atom1, charge, rgauss, dipole);
        real3 d1 = make_real3(data1.dx, data1.dy, data1.dz);
        if (x == y) {
            // A tile on the diagonal: every pair is seen twice, each time adding to atom 1 only.

            localData[LOCAL_ID] = data1;
            SYNC_WARPS;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+j;
                real3 delta = make_real3(localData[atom2].x-pos1.x, localData[atom2].y-pos1.y, localData[atom2].z-pos1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    int atom2g = y*TILE_SIZE+j;
                    if (atom1 < NUM_ATOMS && atom2g < NUM_ATOMS && atom1 != atom2g) {
#if PAIR_KIND == 2
                        computePair(delta, r2, data1.q, data1.rg, d1, &localData[atom2], true, &field, &force, &energy);
#else
                        computePair(delta, r2, data1.q, data1.rg, d1, &localData[atom2], true, &field);
#endif
                    }
#ifdef USE_CUTOFF
                }
#endif
                SYNC_WARPS;
            }
        }
        else {
            // An off-diagonal tile.

            const unsigned int localAtomIndex = LOCAL_ID;
            unsigned int j = y*TILE_SIZE + tgx;
            loadAtom(&localData[localAtomIndex], posq[j], j, charge, rgauss, dipole);
            SYNC_WARPS;
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+tj;
                real3 delta = make_real3(localData[atom2].x-pos1.x, localData[atom2].y-pos1.y, localData[atom2].z-pos1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    int atom2g = y*TILE_SIZE+tj;
                    if (atom1 < NUM_ATOMS && atom2g < NUM_ATOMS) {
#if PAIR_KIND == 2
                        computePair(delta, r2, data1.q, data1.rg, d1, &localData[atom2], false, &field, &force, &energy);
#else
                        computePair(delta, r2, data1.q, data1.rg, d1, &localData[atom2], false, &field);
#endif
                    }
#ifdef USE_CUTOFF
                }
#endif
                tj = (tj + 1) & (TILE_SIZE - 1);
                SYNC_WARPS;
            }
        }

        // Write results.

        unsigned int offset1 = x*TILE_SIZE + tgx;
        ATOMIC_ADD(&fieldBuffer[offset1], (mm_ulong) realToFixedPoint(field.x));
        ATOMIC_ADD(&fieldBuffer[offset1+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(field.y));
        ATOMIC_ADD(&fieldBuffer[offset1+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(field.z));
#if PAIR_KIND == 2
        ATOMIC_ADD(&forceBuffer[offset1], (mm_ulong) realToFixedPoint(force.x));
        ATOMIC_ADD(&forceBuffer[offset1+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
        ATOMIC_ADD(&forceBuffer[offset1+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
#endif
        if (x != y) {
            unsigned int offset2 = y*TILE_SIZE + tgx;
            ATOMIC_ADD(&fieldBuffer[offset2], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].ex));
            ATOMIC_ADD(&fieldBuffer[offset2+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].ey));
            ATOMIC_ADD(&fieldBuffer[offset2+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].ez));
#if PAIR_KIND == 2
            ATOMIC_ADD(&forceBuffer[offset2], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].fx));
            ATOMIC_ADD(&forceBuffer[offset2+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].fy));
            ATOMIC_ADD(&forceBuffer[offset2+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].fz));
#endif
        }
    }

    // Second loop: tiles without exclusions, from the neighbor list (cutoff) or all of them (no cutoff).

#ifdef USE_CUTOFF
    unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int tile = (int) (warp*(numTiles > maxTiles ? NUM_BLOCKS*((mm_long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
    int end = (int) ((warp+1)*(numTiles > maxTiles ? NUM_BLOCKS*((mm_long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
#else
    int tile = (int) (warp*(mm_long)numTiles/totalWarps);
    int end = (int) ((warp+1)*(mm_long)numTiles/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    LOCAL int atomIndices[THREAD_BLOCK_SIZE];
    LOCAL volatile int skipTiles[THREAD_BLOCK_SIZE];
    skipTiles[LOCAL_ID] = -1;

    while (tile < end) {
        real3 field = make_real3(0);
#if PAIR_KIND == 2
        real3 force = make_real3(0);
#endif
        bool includeTile = true;
        int x, y;
        bool singlePeriodicCopy = false;
#ifdef USE_CUTOFF
        x = tiles[tile];
        real4 blockSizeX = blockSize[x];
        singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= CUTOFF &&
                              0.5f*periodicBoxSize.y-blockSizeX.y >= CUTOFF &&
                              0.5f*periodicBoxSize.z-blockSizeX.z >= CUTOFF);
#else
        y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*tile));
        x = (tile-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (tile-y*NUM_BLOCKS+y*(y+1)/2);
        }

        // Skip the tiles with exclusions, which were processed above.

        SYNC_WARPS;
        while (skipTiles[tbx+TILE_SIZE-1] < tile) {
            SYNC_WARPS;
            if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                int2 tile = exclusionTiles[skipBase+tgx];
                skipTiles[LOCAL_ID] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
            }
            else
                skipTiles[LOCAL_ID] = end;
            skipBase += TILE_SIZE;
            currentSkipIndex = tbx;
            SYNC_WARPS;
        }
        while (skipTiles[currentSkipIndex] < tile)
            currentSkipIndex++;
        includeTile = (skipTiles[currentSkipIndex] != tile);
#endif
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;
            real4 pos1 = posq[atom1];
            AtomData data1;
            loadAtom(&data1, pos1, atom1, charge, rgauss, dipole);
            real3 d1 = make_real3(data1.dx, data1.dy, data1.dz);
            const unsigned int localAtomIndex = LOCAL_ID;
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[tile*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[LOCAL_ID] = j;
            if (j < PADDED_NUM_ATOMS)
                loadAtom(&localData[localAtomIndex], posq[j], j, charge, rgauss, dipole);
            SYNC_WARPS;
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is large enough that all atoms can be translated into one periodic copy.

                real4 blockCenterX = blockCenter[x];
                APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
                APPLY_PERIODIC_TO_POS_WITH_CENTER(localData[LOCAL_ID], blockCenterX)
                SYNC_WARPS;
                unsigned int tj = tgx;
                for (unsigned int j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
                    real3 delta = make_real3(localData[atom2].x-pos1.x, localData[atom2].y-pos1.y, localData[atom2].z-pos1.z);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (r2 < CUTOFF_SQUARED) {
                        int atom2g = atomIndices[tbx+tj];
                        if (atom1 < NUM_ATOMS && atom2g < NUM_ATOMS) {
#if PAIR_KIND == 2
                            computePair(delta, r2, data1.q, data1.rg, d1, &localData[atom2], false, &field, &force, &energy);
#else
                            computePair(delta, r2, data1.q, data1.rg, d1, &localData[atom2], false, &field);
#endif
                        }
                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);
                    SYNC_WARPS;
                }
            }
            else
#endif
            {
                unsigned int tj = tgx;
                for (unsigned int j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
                    real3 delta = make_real3(localData[atom2].x-pos1.x, localData[atom2].y-pos1.y, localData[atom2].z-pos1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < CUTOFF_SQUARED) {
#endif
                        int atom2g = atomIndices[tbx+tj];
                        if (atom1 < NUM_ATOMS && atom2g < NUM_ATOMS) {
#if PAIR_KIND == 2
                            computePair(delta, r2, data1.q, data1.rg, d1, &localData[atom2], false, &field, &force, &energy);
#else
                            computePair(delta, r2, data1.q, data1.rg, d1, &localData[atom2], false, &field);
#endif
                        }
#ifdef USE_CUTOFF
                    }
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);
                    SYNC_WARPS;
                }
            }

            // Write results.

            ATOMIC_ADD(&fieldBuffer[atom1], (mm_ulong) realToFixedPoint(field.x));
            ATOMIC_ADD(&fieldBuffer[atom1+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(field.y));
            ATOMIC_ADD(&fieldBuffer[atom1+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(field.z));
#if PAIR_KIND == 2
            ATOMIC_ADD(&forceBuffer[atom1], (mm_ulong) realToFixedPoint(force.x));
            ATOMIC_ADD(&forceBuffer[atom1+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
            ATOMIC_ADD(&forceBuffer[atom1+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
#endif
            unsigned int atom2 = atomIndices[LOCAL_ID];
            if (atom2 < PADDED_NUM_ATOMS) {
                ATOMIC_ADD(&fieldBuffer[atom2], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].ex));
                ATOMIC_ADD(&fieldBuffer[atom2+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].ey));
                ATOMIC_ADD(&fieldBuffer[atom2+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].ez));
#if PAIR_KIND == 2
                ATOMIC_ADD(&forceBuffer[atom2], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].fx));
                ATOMIC_ADD(&forceBuffer[atom2+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].fy));
                ATOMIC_ADD(&forceBuffer[atom2+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].fz));
#endif
            }
        }
        tile++;
    }
#if PAIR_KIND == 2
    energyBuffer[GLOBAL_ID] += energy;
#endif
}

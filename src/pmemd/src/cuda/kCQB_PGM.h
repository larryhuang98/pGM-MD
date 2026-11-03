#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kCQB_PGM.h - PGM version of kCQB.h (charge-to-grid template)
//
// Copied from kCQB.h, modified for PGM dipole contributions.
// Differences from kCQB.h:
//   1. Reads fractional dipoles (pPGMFracDipX/Y/Z) in addition to charges
//   2. Computes B-spline derivatives (dtheta) in addition to theta
//   3. Grid scatter includes dipole contributions:
//      q_grid += charge*Tx*Ty*Tz + dip_x*dTx*Ty*Tz + dip_y*Tx*dTy*Tz + dip_z*Tx*Ty*dTz
//   4. No TI/AFE masking
//
// Fortran reference: pme_slab_fill_grid_PGM.i
//   k_term  = theta_z * charge + dtheta_z * dip_z
//   k_term1 = theta_z * dip_x
//   k_term2 = theta_z * dip_y
//   j_term  = k_term * theta_y + k_term2 * dtheta_y
//   j_term1 = k_term1 * theta_y
//   q(i,j,k) += theta_x * j_term + dtheta_x * j_term1
//---------------------------------------------------------------------------------------------

#define BATCHSIZE        30
#define BATCH_BITS        5
#define BATCH_BITS_MASK  31
{
  // Coordinates and charges: [ x1..x30, y1..y30, z1..z30, q1..q30 ]
  __shared__ volatile PMEFloat crdq[4*BATCHSIZE];

  // Fractional dipoles: [ dip_x_1..dip_x_30, dip_y_1..dip_y_30, dip_z_1..dip_z_30 ]
  __shared__ volatile PMEFloat dipoles[3*BATCHSIZE];

  // Starting mesh indices: [ ix1..ix30, iy1..iy30, iz1..iz30 ]
  __shared__ volatile int sAtom_ixyz[3*BATCHSIZE];

  // B-spline values (theta) and derivatives (dtheta), NOT pre-multiplied by charge
#if (PME_ORDER == 4)
  __shared__ volatile PMEFloat sAtom_txyz[12*BATCHSIZE];
  __shared__ volatile PMEFloat sAtom_dtxyz[12*BATCHSIZE];
#elif (PME_ORDER == 5)
  __shared__ volatile PMEFloat sAtom_txyz[15*BATCHSIZE];
  __shared__ volatile PMEFloat sAtom_dtxyz[15*BATCHSIZE];
#elif (PME_ORDER == 6)
  __shared__ volatile PMEFloat sAtom_txyz[18*BATCHSIZE];
  __shared__ volatile PMEFloat sAtom_dtxyz[18*BATCHSIZE];
#endif

  // Thread indexing
  const unsigned int tgx = threadIdx.x & GRID_BITS_MASK;
  const unsigned int ndim = threadIdx.x / BATCHSIZE;
  const unsigned int nfftval = (ndim == 0)*cSim.nfft1 + (ndim == 1)*cSim.nfft2 +
                               (ndim == 2)*cSim.nfft3;
  const unsigned int nmap = BATCHSIZE / (P2M_THREADS_PER_BLOCK >> GRID_BITS);

  // Grid offset constants
  const PMEFloat ONEHALF    = 0.5;
  const PMEFloat ONETHIRD   = (1.0 / 3.0);
  const PMEFloat ONE        = 1.0;
  const PMEFloat THREE      = 3.0;
#if (PME_ORDER == 4)
  const unsigned int iOffsetX = tgx & 0x03;
  const unsigned int iOffsetY = (tgx & 0x0f) >> 2;
  const unsigned int iOffsetZ = tgx >> 4;
#elif (PME_ORDER == 5)
  const PMEFloat ONEQUARTER = 0.25;
  const PMEFloat TWO        = 2.0;
  const PMEFloat FOUR       = 4.0;
  const unsigned int iOffsetX = tgx % 5;
  const unsigned int iOffsetY = tgx / 5;
  const unsigned int iOffsetZ = 0;
#elif (PME_ORDER == 6)
  const PMEFloat ONEQUARTER = 0.25;
  const PMEFloat ONEFIFTH   = 0.2;
  const PMEFloat TWO        = 2.0;
  const PMEFloat FOUR       = 4.0;
  const PMEFloat FIVE       = 5.0;
  const unsigned int iOffsetX = tgx % 6;
  const unsigned int iOffsetY = tgx / 6;
  const unsigned int iOffsetZ = 0;
  const unsigned int jOffsetX = (tgx & 0x03) + 2;
  const unsigned int jOffsetY = 5;
  const unsigned int jOffsetZ = tgx >> 2;
#endif

  // Iterate through atoms in batches
  int pos = blockIdx.x * BATCHSIZE;

  while (pos < cSim.atoms) {
    int atomid = (threadIdx.x & BATCH_BITS_MASK);
    int pos1 = pos + atomid;

    // Read coordinates, charges, AND fractional dipoles
    // P2M_THREADS_PER_BLOCK=96, so 3 segments of 32.
    // Piggyback dipole reads onto existing segments.
    if ((pos1 < cSim.atoms) && (atomid < BATCHSIZE)) {
      int segment = (threadIdx.x >> BATCH_BITS);
      if (segment == 0) {
        PMEFloat charge = cSim.pPGMPermChg[pos1];
        PMEFloat fx = cSim.pFractX[pos1];
        crdq[              atomid] = fx;
        crdq[3*BATCHSIZE + atomid] = charge;
        dipoles[atomid] = cSim.pPGMFracDipX[pos1];
      }
      else if (segment == 1) {
        PMEFloat fy = cSim.pFractY[pos1];
        crdq[  BATCHSIZE + atomid] = fy;
        dipoles[BATCHSIZE + atomid] = cSim.pPGMFracDipY[pos1];
      }
      else if (segment == 2) {
        PMEFloat fz = cSim.pFractZ[pos1];
        crdq[2*BATCHSIZE + atomid] = fz;
        dipoles[2*BATCHSIZE + atomid] = cSim.pPGMFracDipZ[pos1];
      }
    }
    __syncthreads();

    // Compute B-spline coefficients AND derivatives.
    // Uses 3*BATCHSIZE = 90 threads (30 per dimension).
    if (threadIdx.x < 3*BATCHSIZE) {
      PMEFloat fx = crdq[threadIdx.x];
      int ix = int(fx);
      fx -= ix;
      ix -= cSim.orderMinusOne;
      ix += (ix < 0) * nfftval;

      PMEFloat tx0, tx1, tx2, tx3;

      // Order 2 B-spline
      tx0 = ONE - fx;
      tx1 = fx;

      // Order 3 B-spline
      tx2 = ONEHALF * fx * tx1;
      tx0 = ONEHALF * (ONE - fx) * tx0;
      tx1 = ONE - tx0 - tx2;

#if (PME_ORDER == 4)
      // Save B-spline derivatives BEFORE order-4 step.
      // dtheta[k] = M3[k-1] - M3[k] (following kPGS.h sign convention)
      PMEFloat dtx0 = -tx0;
      PMEFloat dtx1 = tx0 - tx1;
      PMEFloat dtx2 = tx1 - tx2;
      PMEFloat dtx3 = tx2;

      // Order 4 B-spline
      tx3 = ONETHIRD * fx * tx2;
      tx2 = ONETHIRD * ((fx + ONE) * tx1 + (THREE - fx) * tx2);
      tx0 = ONETHIRD *  (ONE - fx) * tx0;
      tx1 = ONE - tx0 - tx2 - tx3;

#elif (PME_ORDER == 5)
      // Order 4 B-spline
      tx3 = ONETHIRD * fx * tx2;
      tx2 = ONETHIRD * ((fx + ONE) * tx1 + (THREE - fx) * tx2);
      tx0 = ONETHIRD *  (ONE - fx) * tx0;
      tx1 = ONE - tx0 - tx2 - tx3;

      // Save B-spline derivatives (differences of order-4 M-splines)
      PMEFloat dtx0 = -tx0;
      PMEFloat dtx1 = tx0 - tx1;
      PMEFloat dtx2 = tx1 - tx2;
      PMEFloat dtx3 = tx2 - tx3;
      PMEFloat dtx4 = tx3;

      // Order 5 B-spline
      PMEFloat tx4;
      tx4 = ONEQUARTER * fx * tx3;
      tx3 = ONEQUARTER * ((fx + ONE)   * tx2 + (FOUR  - fx) * tx3);
      tx2 = ONEQUARTER * ((fx + TWO)   * tx1 + (THREE - fx) * tx2);
      tx1 = ONEQUARTER * ((fx + THREE) * tx0 + (TWO   - fx) * tx1);
      tx0 = ONEQUARTER * (ONE - fx) * tx0;

#elif (PME_ORDER == 6)
      // Order 4 B-spline
      tx3 = ONETHIRD * fx * tx2;
      tx2 = ONETHIRD * ((fx + ONE) * tx1 + (THREE - fx) * tx2);
      tx0 = ONETHIRD *  (ONE - fx) * tx0;
      tx1 = ONE - tx0 - tx2 - tx3;

      // Order 5 B-spline
      PMEFloat tx4;
      tx4 = ONEQUARTER * fx * tx3;
      tx3 = ONEQUARTER * ((fx + ONE)   * tx2 + (FOUR  - fx) * tx3);
      tx2 = ONEQUARTER * ((fx + TWO)   * tx1 + (THREE - fx) * tx2);
      tx1 = ONEQUARTER * ((fx + THREE) * tx0 + (TWO   - fx) * tx1);
      tx0 = ONEQUARTER * (ONE - fx) * tx0;

      // Save B-spline derivatives (differences of order-5 M-splines)
      PMEFloat dtx0 = -tx0;
      PMEFloat dtx1 = tx0 - tx1;
      PMEFloat dtx2 = tx1 - tx2;
      PMEFloat dtx3 = tx2 - tx3;
      PMEFloat dtx4 = tx3 - tx4;
      PMEFloat dtx5 = tx4;

      // Order 6 B-spline
      PMEFloat tx5;
      tx5 = ONEFIFTH * fx * tx4;
      tx4 = ONEFIFTH * ((fx + ONE)   * tx3 + (FIVE  - fx) * tx4);
      tx3 = ONEFIFTH * ((fx + TWO)   * tx2 + (FOUR  - fx) * tx3);
      tx2 = ONEFIFTH * ((fx + THREE) * tx1 + (THREE - fx) * tx2);
      tx1 = ONEFIFTH * ((fx + FOUR)  * tx0 + (TWO   - fx) * tx1);
      tx0 = ONEFIFTH * (ONE - fx) * tx0;
#endif

      // Store grid index
      sAtom_ixyz[threadIdx.x] = ix;

      // Store theta and dtheta (NO charge pre-multiplication for PGM)
      int ptstore = cSim.pmeOrder * threadIdx.x;
      sAtom_txyz[ptstore    ] = tx0;
      sAtom_txyz[ptstore + 1] = tx1;
      sAtom_txyz[ptstore + 2] = tx2;
      sAtom_txyz[ptstore + 3] = tx3;
#if (PME_ORDER == 5)
      sAtom_txyz[ptstore + 4] = tx4;
#elif (PME_ORDER == 6)
      sAtom_txyz[ptstore + 4] = tx4;
      sAtom_txyz[ptstore + 5] = tx5;
#endif

      sAtom_dtxyz[ptstore    ] = dtx0;
      sAtom_dtxyz[ptstore + 1] = dtx1;
      sAtom_dtxyz[ptstore + 2] = dtx2;
      sAtom_dtxyz[ptstore + 3] = dtx3;
#if (PME_ORDER == 5)
      sAtom_dtxyz[ptstore + 4] = dtx4;
#elif (PME_ORDER == 6)
      sAtom_dtxyz[ptstore + 4] = dtx4;
      sAtom_dtxyz[ptstore + 5] = dtx5;
#endif
    }
    __syncthreads();

    // Map atoms onto the mesh. Each group of 32 threads handles one atom.
    pos1 = (threadIdx.x >> GRID_BITS) * nmap;
    int lastAtom = min(pos1 + nmap, cSim.atoms - pos);
    int iOffsetXld = (cSim.pmeOrder *                 pos1) + iOffsetX;
    int iOffsetYld = (cSim.pmeOrder * (pos1 +   BATCHSIZE)) + iOffsetY;
    int iOffsetZld = (cSim.pmeOrder * (pos1 + 2*BATCHSIZE)) + iOffsetZ;
#if (PME_ORDER == 6)
    int jOffsetXld = (cSim.pmeOrder *                 pos1) + jOffsetX;
    int jOffsetYld = (cSim.pmeOrder * (pos1 +   BATCHSIZE)) + jOffsetY;
    int jOffsetZld = (cSim.pmeOrder * (pos1 + 2*BATCHSIZE)) + jOffsetZ;
#endif

    while (pos1 < lastAtom) {

      // Read charge and fractional dipoles for this atom
      PMEFloat charge = crdq[3*BATCHSIZE + pos1];
      PMEFloat dip_x  = dipoles[              pos1];
      PMEFloat dip_y  = dipoles[  BATCHSIZE + pos1];
      PMEFloat dip_z  = dipoles[2*BATCHSIZE + pos1];

      // Read theta and dtheta for x and y dimensions
      PMEFloat theta_x  = sAtom_txyz[iOffsetXld];
      PMEFloat theta_y  = sAtom_txyz[iOffsetYld];
      PMEFloat dtheta_x = sAtom_dtxyz[iOffsetXld];
      PMEFloat dtheta_y = sAtom_dtxyz[iOffsetYld];

      // Grid position calculation (same as standard PME)
      int ix = sAtom_ixyz[              pos1] + iOffsetX;
      int iy = sAtom_ixyz[  BATCHSIZE + pos1] + iOffsetY;
      int iz = sAtom_ixyz[2*BATCHSIZE + pos1] + iOffsetZ;

      // Ensure coordinates stay in bounds
      ix -= (ix >= cSim.nfft1) * cSim.nfft1;
      iy -= (iy >= cSim.nfft2) * cSim.nfft2;

#if (PME_ORDER == 4)
      // Grid position in 4x4 tiled layout
      int gposxy = ((ix & 0x3) + (iy & 0x3)*4) +
                   (((ix >> 2) << 4) + ((iy >> 2) << 2)*cSim.nfft1)*cSim.nfft3;

      int level;
      for (level = 0; level < 4; level += 2) {
        iz += level;
        iz -= (iz >= cSim.nfft3) * cSim.nfft3;
        int gposxyz = gposxy + (iz << 4);

        // Read theta_z and dtheta_z for this z-level
        PMEFloat theta_z  = sAtom_txyz[iOffsetZld + level];
        PMEFloat dtheta_z = sAtom_dtxyz[iOffsetZld + level];

        // PGM grid fill:
        // value = charge*Tx*Ty*Tz + dip_x*dTx*Ty*Tz + dip_y*Tx*dTy*Tz + dip_z*Tx*Ty*dTz
        PMEFloat value_f = theta_y * theta_z * (charge * theta_x + dip_x * dtheta_x)
                         + theta_x * theta_z * (dip_y * dtheta_y)
                         + theta_x * theta_y * (dip_z * dtheta_z);

#ifdef use_DPFP
        unsigned long long int value = llitoulli(llrint((double)value_f * LATTICESCALE));
        atomicAdd((unsigned long long int*)&cSim.plliXYZ_q[gposxyz], value);
#else
        int value = __float2int_rn(value_f * LATTICESCALEF);
        atomicAdd(&cSim.plliXYZ_q[gposxyz], value);
#endif
      }

#elif (PME_ORDER == 5)
      // Grid position in row-major layout with 4-wide x-groups (same as order 6)
      // 25 of 32 threads active: 5 x-values * 5 y-values, 5 z-levels in loop
      if (tgx < 25) {
        int gposxy = (ix & 0x3) + (iy << 2) + ((ix >> 2) << 2)*cSim.nfft2;

        int level;
        for (level = 0; level < 5; level++) {
          if (level > 0) iz++;
          iz -= (iz >= cSim.nfft3) * cSim.nfft3;
          int gposxyz = gposxy + iz*cSim.nfft1xnfft2;

          // Read theta_z and dtheta_z for this z-level
          PMEFloat theta_z  = sAtom_txyz[iOffsetZld + level];
          PMEFloat dtheta_z = sAtom_dtxyz[iOffsetZld + level];

          // PGM grid fill:
          // value = charge*Tx*Ty*Tz + dip_x*dTx*Ty*Tz + dip_y*Tx*dTy*Tz + dip_z*Tx*Ty*dTz
          PMEFloat value_f = theta_y * theta_z * (charge * theta_x + dip_x * dtheta_x)
                           + theta_x * theta_z * (dip_y * dtheta_y)
                           + theta_x * theta_y * (dip_z * dtheta_z);

#ifdef use_DPFP
          unsigned long long int value = llitoulli(llrint((double)value_f * LATTICESCALE));
          atomicAdd((unsigned long long int*)&cSim.plliXYZ_q[gposxyz], value);
#else
          int value = __float2int_rn(value_f * LATTICESCALEF);
          atomicAdd(&cSim.plliXYZ_q[gposxyz], value);
#endif
        }
      }

#elif (PME_ORDER == 6)
      // Grid position in order-6 layout
      int gposxy = (ix & 0x3) + (iy << 2) + ((ix >> 2) << 2)*cSim.nfft2;

      int level;
      for (level = 0; level < 6; level++) {
        if (level > 0) iz++;
        iz -= (iz >= cSim.nfft3) * cSim.nfft3;
        int gposxyz = gposxy + iz*cSim.nfft1xnfft2;

        // Read theta_z and dtheta_z for this z-level
        PMEFloat theta_z  = sAtom_txyz[iOffsetZld + level];
        PMEFloat dtheta_z = sAtom_dtxyz[iOffsetZld + level];

        // PGM grid fill:
        // value = charge*Tx*Ty*Tz + dip_x*dTx*Ty*Tz + dip_y*Tx*dTy*Tz + dip_z*Tx*Ty*dTz
        PMEFloat value_f = theta_y * theta_z * (charge * theta_x + dip_x * dtheta_x)
                         + theta_x * theta_z * (dip_y * dtheta_y)
                         + theta_x * theta_y * (dip_z * dtheta_z);

#ifdef use_DPFP
        unsigned long long int value = llitoulli(llrint((double)value_f * LATTICESCALE));
        atomicAdd((unsigned long long int*)&cSim.plliXYZ_q[gposxyz], value);
#else
        int value = __float2int_rn(value_f * LATTICESCALEF);
        atomicAdd(&cSim.plliXYZ_q[gposxyz], value);
#endif
      }

      // Sixth-order interpolation has left out 24 pieces of the stencil.
      // Fill those in now.
      if (tgx < 24) {
        ix = sAtom_ixyz[              pos1] + jOffsetX;
        iy = sAtom_ixyz[  BATCHSIZE + pos1] + jOffsetY;
        iz = sAtom_ixyz[2*BATCHSIZE + pos1] + jOffsetZ;
        ix -= (ix >= cSim.nfft1) * cSim.nfft1;
        iy -= (iy >= cSim.nfft2) * cSim.nfft2;
        iz -= (iz >= cSim.nfft3) * cSim.nfft3;

        PMEFloat theta_xj  = sAtom_txyz[jOffsetXld];
        PMEFloat theta_yj  = sAtom_txyz[jOffsetYld];
        PMEFloat theta_zj  = sAtom_txyz[jOffsetZld];
        PMEFloat dtheta_xj = sAtom_dtxyz[jOffsetXld];
        PMEFloat dtheta_yj = sAtom_dtxyz[jOffsetYld];
        PMEFloat dtheta_zj = sAtom_dtxyz[jOffsetZld];

        int gpos = (ix & 0x3) + (iy << 2) + ((ix >> 2) << 2)*cSim.nfft2 +
                   iz*cSim.nfft1xnfft2;

        PMEFloat value_f = theta_yj * theta_zj * (charge * theta_xj + dip_x * dtheta_xj)
                         + theta_xj * theta_zj * (dip_y * dtheta_yj)
                         + theta_xj * theta_yj * (dip_z * dtheta_zj);

#ifdef use_DPFP
        unsigned long long int value = llitoulli(llrint((double)value_f * LATTICESCALE));
        atomicAdd((unsigned long long int*)&cSim.plliXYZ_q[gpos], value);
#else
        int value = __float2int_rn(value_f * LATTICESCALEF);
        atomicAdd(&cSim.plliXYZ_q[gpos], value);
#endif
      }
      jOffsetXld += cSim.pmeOrder;
      jOffsetYld += cSim.pmeOrder;
      jOffsetZld += cSim.pmeOrder;
#endif

      // Advance to next atom
      iOffsetXld += cSim.pmeOrder;
      iOffsetYld += cSim.pmeOrder;
      iOffsetZld += cSim.pmeOrder;
      pos1++;
    }

    // Advance to the next batch
    pos += gridDim.x * BATCHSIZE;
    __syncthreads();
  }
}
#undef BATCHSIZE
#undef BATCH_BITS
#undef BATCH_BITS_MASK

#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kPBGS_PGM.h - PGM version of kPBGS.h (batched gradient sum template)
//
// Copied from kPBGS.h, modified for PGM:
//   1. Extracts electric FIELD (not force) from the convolved grid
//   2. No charge multiplication - field = nfft * recipf * grad_sum
//   3. Accumulates to pPGMFieldAccumulator (not pNBForceAccumulator)
//   4. No TI/AFE, no virial - keep simple
//   5. Supports order 5 and 6 (PME_ORDER == 5 or 6)
//
// The electric field at atom i from the reciprocal grid is:
//   E_frac(i) = sum_{grid near i} dtheta * theta * theta * Q_convolved
//   E_cart(i) = nfft * recipf * E_frac
//
// This is the SAME computation as the standard PME force, but WITHOUT
// the charge multiplication (force = charge * field).
//---------------------------------------------------------------------------------------------
#ifdef use_DPFP
#  define BATCHSIZE        32
#  define BATCH_BITS        5
#  define BATCH_BITS_MASK  31
#else
#  define BATCHSIZE        32
#  define BATCH_BITS        5
#  define BATCH_BITS_MASK  31
#endif
{
  // Field accumulators in shared memory (instead of force accumulators in kPBGS.h).
  // Each of 4 warps contributes partial fields via atomicAdd.
  __shared__ volatile PMEAccumulator fld[3*BATCHSIZE];

  // Starting mesh indices
  __shared__ volatile int sAtom_ixyz[3*BATCHSIZE];

  // B-spline coefficients and derivatives
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
#ifndef PGM_FIELD_ONLY
#if (PME_ORDER == 4)
  __shared__ volatile PMEFloat sAtom_d2txyz[12*BATCHSIZE];
#elif (PME_ORDER == 5)
  __shared__ volatile PMEFloat sAtom_d2txyz[15*BATCHSIZE];
#elif (PME_ORDER == 6)
  __shared__ volatile PMEFloat sAtom_d2txyz[18*BATCHSIZE];
#endif
  __shared__ volatile PMEAccumulator frc[3*BATCHSIZE];
  __shared__ volatile PMEAccumulator phi[BATCHSIZE];  // phi_000 for energy
#endif

  // Warp-level indexing for the grid interpolation stage
  const int warpIdx = threadIdx.x >> GRID_BITS;
#if (PME_ORDER == 4)
  const int zOffset = (warpIdx & 3);
#elif (PME_ORDER == 5)
  const int zOffset = (warpIdx & 3);
#elif (PME_ORDER == 6)
  const int zOffset = (warpIdx & 3) + ((warpIdx & 3) >= 2);
  const int yOffset = 3*(warpIdx & 1);
#endif

  // Thread identity for B-spline computation phase (first 96 of 128 threads)
  const int ndim = threadIdx.x / BATCHSIZE;
  const int bchidx = (threadIdx.x & BATCH_BITS_MASK);
  const int nfftval = (ndim == 0)*cSim.nfft1 + (ndim == 1)*cSim.nfft2 + (ndim == 2)*cSim.nfft3;

  // Constants for B-spline calculations
  const PMEFloat ONEHALF    = 0.5;
  const PMEFloat ONETHIRD   = (1.0 / 3.0);
  const PMEFloat ONE        = 1.0;
  const PMEFloat THREE      = 3.0;
#if (PME_ORDER == 5)
  const PMEFloat ONEQUARTER = 0.25;
  const PMEFloat TWO        = 2.0;
  const PMEFloat FOUR       = 4.0;
#elif (PME_ORDER == 6)
  const PMEFloat ONEQUARTER = 0.25;
  const PMEFloat ONEFIFTH   = 0.2;
  const PMEFloat TWO        = 2.0;
  const PMEFloat FOUR       = 4.0;
  const PMEFloat FIVE       = 5.0;
#endif

  // Iterate through the simulation cell
  int pos = blockIdx.x * BATCHSIZE;

  while (pos < cSim.atoms) {
    int pos1 = pos + bchidx;

    // Initialize field accumulators to zero (last 32 of 128 threads)
    if (pos1 < cSim.atoms && threadIdx.x >= 3*BATCHSIZE) {
      fld[threadIdx.x - 3*BATCHSIZE] = 0;
      fld[threadIdx.x - 2*BATCHSIZE] = 0;
      fld[threadIdx.x -   BATCHSIZE] = 0;
#ifndef PGM_FIELD_ONLY
      frc[threadIdx.x - 3*BATCHSIZE] = 0;
      frc[threadIdx.x - 2*BATCHSIZE] = 0;
      frc[threadIdx.x -   BATCHSIZE] = 0;
      // Clear phi_000 (first 32 threads of the last warp)
      if (threadIdx.x < 4*BATCHSIZE) {
        phi[threadIdx.x - 3*BATCHSIZE] = 0;
      }
#endif
    }

    // Compute B-spline coefficients (first 96 of 128 threads)
    if (threadIdx.x < 3*BATCHSIZE) {
      PMEFloat fx;
      if (pos1 < cSim.atoms) {
        int segment = (threadIdx.x >> BATCH_BITS);
        if (segment == 0) {
          fx = cSim.pFractX[pos1];
        }
        else if (segment == 1) {
          fx = cSim.pFractY[pos1];
        }
        else if (segment == 2) {
          fx = cSim.pFractZ[pos1];
        }
      }
      int ix = int(fx);
      int ptstore = (ndim * BATCHSIZE * cSim.pmeOrder) + bchidx;
      fx -= ix;
      ix -= cSim.orderMinusOne;
      ix += (ix < 0) * nfftval;
      PMEFloat tx0, tx1, tx2, tx3;
      tx0 = ONE - fx;
      tx1 = fx;
      tx2 = ONEHALF * fx * tx1;
      tx0 = ONEHALF * (ONE - fx) * tx0;
      tx1 = ONE - tx0 - tx2;
#if (PME_ORDER == 4)
      sAtom_dtxyz[ptstore              ] = -tx0;
      sAtom_dtxyz[ptstore +   BATCHSIZE] = tx0 - tx1;
      sAtom_dtxyz[ptstore + 2*BATCHSIZE] = tx1 - tx2;
      sAtom_dtxyz[ptstore + 3*BATCHSIZE] = tx2;
#endif
#if (PME_ORDER == 5) && !defined(PGM_FIELD_ONLY)
      // Save order-3 M-splines for d2theta (2nd derivative of order-5 B-spline)
      PMEFloat t3_0 = tx0, t3_1 = tx1, t3_2 = tx2;
#endif
      tx3 = ONETHIRD * fx * tx2;
      tx2 = ONETHIRD * ((fx + ONE) * tx1 + (THREE - fx) * tx2);
      tx0 = ONETHIRD *  (ONE - fx) * tx0;
      tx1 = ONE - tx0 - tx2 - tx3;
#if (PME_ORDER == 5)
      // dtheta = 1st differences of order-4 M-splines (order-5 1st derivative)
      sAtom_dtxyz[ptstore              ] = -tx0;
      sAtom_dtxyz[ptstore +   BATCHSIZE] = tx0 - tx1;
      sAtom_dtxyz[ptstore + 2*BATCHSIZE] = tx1 - tx2;
      sAtom_dtxyz[ptstore + 3*BATCHSIZE] = tx2 - tx3;
      sAtom_dtxyz[ptstore + 4*BATCHSIZE] = tx3;
#ifndef PGM_FIELD_ONLY
      // d2theta = 2nd differences of order-3 M-splines (order-5 2nd derivative)
      sAtom_d2txyz[ptstore              ] = t3_0;
      sAtom_d2txyz[ptstore +   BATCHSIZE] = t3_1 - TWO*t3_0;
      sAtom_d2txyz[ptstore + 2*BATCHSIZE] = t3_0 - TWO*t3_1 + t3_2;
      sAtom_d2txyz[ptstore + 3*BATCHSIZE] = t3_1 - TWO*t3_2;
      sAtom_d2txyz[ptstore + 4*BATCHSIZE] = t3_2;
#endif
      // Order 5 M-spline recursion
      PMEFloat tx4;
      tx4 = ONEQUARTER * fx * tx3;
      tx3 = ONEQUARTER * ((fx + ONE)   * tx2 + (FOUR  - fx) * tx3);
      tx2 = ONEQUARTER * ((fx + TWO)   * tx1 + (THREE - fx) * tx2);
      tx1 = ONEQUARTER * ((fx + THREE) * tx0 + (TWO   - fx) * tx1);
      tx0 = ONEQUARTER * (ONE - fx) * tx0;
#endif
#if (PME_ORDER == 6)
#ifndef PGM_FIELD_ONLY
      // Save order-4 values for d2theta (2nd derivative of order-6 B-spline)
      PMEFloat t4_0 = tx0, t4_1 = tx1, t4_2 = tx2, t4_3 = tx3;
#endif
      PMEFloat tx4, tx5;
      tx4 = ONEQUARTER * fx * tx3;
      tx3 = ONEQUARTER * ((fx + ONE)   * tx2 + (FOUR  - fx) * tx3);
      tx2 = ONEQUARTER * ((fx + TWO)   * tx1 + (THREE - fx) * tx2);
      tx1 = ONEQUARTER * ((fx + THREE) * tx0 + (TWO   - fx) * tx1);
      tx0 = ONEQUARTER * (ONE - fx) * tx0;
      sAtom_dtxyz[ptstore              ] = -tx0;
      sAtom_dtxyz[ptstore +   BATCHSIZE] = tx0 - tx1;
      sAtom_dtxyz[ptstore + 2*BATCHSIZE] = tx1 - tx2;
      sAtom_dtxyz[ptstore + 3*BATCHSIZE] = tx2 - tx3;
      sAtom_dtxyz[ptstore + 4*BATCHSIZE] = tx3 - tx4;
      sAtom_dtxyz[ptstore + 5*BATCHSIZE] = tx4;
#ifndef PGM_FIELD_ONLY
      // d2theta = 2nd differences of order-4 values (see fill_bspline_2_3d in bspline.F90)
      sAtom_d2txyz[ptstore              ] = t4_0;
      sAtom_d2txyz[ptstore +   BATCHSIZE] = t4_1 - TWO*t4_0;
      sAtom_d2txyz[ptstore + 2*BATCHSIZE] = t4_2 - TWO*t4_1 + t4_0;
      sAtom_d2txyz[ptstore + 3*BATCHSIZE] = t4_3 - TWO*t4_2 + t4_1;
      sAtom_d2txyz[ptstore + 4*BATCHSIZE] = t4_2 - TWO*t4_3;
      sAtom_d2txyz[ptstore + 5*BATCHSIZE] = t4_3;
#endif
      tx5 = ONEFIFTH * fx * tx4;
      tx4 = ONEFIFTH * ((fx + ONE)   * tx3 + (FIVE  - fx) * tx4);
      tx3 = ONEFIFTH * ((fx + TWO)   * tx2 + (FOUR  - fx) * tx3);
      tx2 = ONEFIFTH * ((fx + THREE) * tx1 + (THREE - fx) * tx2);
      tx1 = ONEFIFTH * ((fx + FOUR)  * tx0 + (TWO   - fx) * tx1);
      tx0 = ONEFIFTH * (ONE - fx) * tx0;
#endif
      sAtom_ixyz[threadIdx.x] = ix;
      sAtom_txyz[ptstore              ] = tx0;
      sAtom_txyz[ptstore +   BATCHSIZE] = tx1;
      sAtom_txyz[ptstore + 2*BATCHSIZE] = tx2;
      sAtom_txyz[ptstore + 3*BATCHSIZE] = tx3;
#if (PME_ORDER == 5)
      sAtom_txyz[ptstore + 4*BATCHSIZE] = tx4;
#elif (PME_ORDER == 6)
      sAtom_txyz[ptstore + 4*BATCHSIZE] = tx4;
      sAtom_txyz[ptstore + 5*BATCHSIZE] = tx5;
#endif
    }
    __syncthreads();

    // Interpolate fields for a group of atoms. All 128 threads participate.
    // 4 warps × 32 threads: each warp handles the same 32 atoms but different z-slabs.
    int workidx = ((threadIdx.x >> (GRID_BITS + 2)) << GRID_BITS) +
                 (threadIdx.x & GRID_BITS_MASK);
    if (pos + workidx < cSim.atoms) {

      // Calculate the starting mesh point
      int iMeshStart = sAtom_ixyz[              workidx];
#if (PME_ORDER == 4)
      int jMeshStart = sAtom_ixyz[  BATCHSIZE + workidx];
      int kMeshLevel = sAtom_ixyz[2*BATCHSIZE + workidx] + zOffset;
#elif (PME_ORDER == 5)
      int jMeshStart = sAtom_ixyz[  BATCHSIZE + workidx];
      int kMeshLevel = sAtom_ixyz[2*BATCHSIZE + workidx] + zOffset;
#elif (PME_ORDER == 6)
      int jMeshStart = sAtom_ixyz[  BATCHSIZE + workidx];  // NO yOffset here!
      int kMeshLevel = sAtom_ixyz[2*BATCHSIZE + workidx] + zOffset;
#endif
      kMeshLevel -= (kMeshLevel >= cSim.nfft3) * cSim.nfft3;
      PMEFloat fx = (PMEFloat)0.0;
      PMEFloat fy = (PMEFloat)0.0;
      PMEFloat fz = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
      PMEFloat f200 = (PMEFloat)0.0, f020 = (PMEFloat)0.0, f002 = (PMEFloat)0.0;
      PMEFloat f110 = (PMEFloat)0.0, f101 = (PMEFloat)0.0, f011 = (PMEFloat)0.0;
      PMEFloat phi000 = (PMEFloat)0.0;  // electrostatic potential (no nfft scaling)
#endif
      int j;
#if (PME_ORDER == 4)
      for (j = 0; j < 4; j++) {
#elif (PME_ORDER == 5)
      for (j = 0; j < 5; j++) {
#elif (PME_ORDER == 6)
      for (j = yOffset; j < 6; j++) {
#endif
        int iidx = iMeshStart;
        int jidx = jMeshStart + j;
        jidx -= (jidx >= cSim.nfft2) * cSim.nfft2;
        int kidx = (kMeshLevel * cSim.nfft1xnfft2) + (jidx * cSim.nfft1);
        int k;
        PMEFloat xsum = (PMEFloat)0.0;
        PMEFloat dxsum = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
        PMEFloat d2xsum = (PMEFloat)0.0;
#endif
#if (PME_ORDER == 4)
#pragma unroll 4
#elif (PME_ORDER == 5)
#pragma unroll 5
#elif (PME_ORDER == 6)
#pragma unroll 6
#endif
        for (k = 0; k < PME_ORDER; k++) {
#ifdef use_DPFP
          int2 i2term = tex1Dfetch<int2>(cSim.texXYZ_q, kidx + iidx);
          PMEFloat qterm = __hiloint2double(i2term.y, i2term.x);
#else
          PMEFloat qterm = tex1Dfetch<float>(cSim.texXYZ_q, kidx + iidx);
#endif
          xsum  +=  sAtom_txyz[k*BATCHSIZE + workidx] * qterm;
          dxsum += sAtom_dtxyz[k*BATCHSIZE + workidx] * qterm;
#ifndef PGM_FIELD_ONLY
          d2xsum += sAtom_d2txyz[k*BATCHSIZE + workidx] * qterm;
#endif
          iidx++;
          iidx -= (iidx >= cSim.nfft1) * cSim.nfft1;
        }

        // Fold in y and z B-spline coefficients
        jidx = (cSim.pmeOrder + j)*BATCHSIZE + workidx;
        kidx = (2*cSim.pmeOrder + zOffset)*BATCHSIZE + workidx;
        // PGM FIELD_ONLY: use += (positive gradient), matching CPU double-negation
        // convention: CPU does f1 -= ...; then f1 = -f1*nfft (net positive).
        fx += dxsum * sAtom_txyz[jidx]  * sAtom_txyz[kidx];
        fy += xsum  * sAtom_dtxyz[jidx] * sAtom_txyz[kidx];
        fz += xsum  * sAtom_txyz[jidx]  * sAtom_dtxyz[kidx];
#ifndef PGM_FIELD_ONLY
        {
          PMEFloat ty = sAtom_txyz[jidx];
          PMEFloat tz = sAtom_txyz[kidx];
          PMEFloat dty = sAtom_dtxyz[jidx];
          PMEFloat dtz = sAtom_dtxyz[kidx];
          f200 += d2xsum * ty * tz;
          f020 += xsum * sAtom_d2txyz[jidx] * tz;
          f002 += xsum * ty * sAtom_d2txyz[kidx];
          f110 += dxsum * dty * tz;
          f101 += dxsum * ty * dtz;
          f011 += xsum * dty * dtz;
          // phi_000: theta_x * theta_y * theta_z * Q (no derivatives, no nfft)
          phi000 += xsum * ty * tz;
        }
#endif
#if (PME_ORDER == 4)
      }
#elif (PME_ORDER == 5)
      }

      // Second pass: z=4 distributed across warps for load balance
      // warp 0 → y0,y1; warp 1 → y2; warp 2 → y3; warp 3 → y4
      {
        int kMeshLevel4 = sAtom_ixyz[2*BATCHSIZE + workidx] + 4;
        kMeshLevel4 -= (kMeshLevel4 >= cSim.nfft3) * cSim.nfft3;
        int y5start = warpIdx + (warpIdx > 0);
        int y5end   = warpIdx + 2;
        for (j = y5start; j < y5end; j++) {
          int iidx = iMeshStart;
          int jidx = jMeshStart + j;
          jidx -= (jidx >= cSim.nfft2) * cSim.nfft2;
          int kidx = (kMeshLevel4 * cSim.nfft1xnfft2) + (jidx * cSim.nfft1);
          int k;
          PMEFloat xsum = (PMEFloat)0.0;
          PMEFloat dxsum = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
          PMEFloat d2xsum = (PMEFloat)0.0;
#endif
#pragma unroll 5
          for (k = 0; k < 5; k++) {
#ifdef use_DPFP
            int2 i2term = tex1Dfetch<int2>(cSim.texXYZ_q, kidx + iidx);
            PMEFloat qterm = __hiloint2double(i2term.y, i2term.x);
#else
            PMEFloat qterm = tex1Dfetch<float>(cSim.texXYZ_q, kidx + iidx);
#endif
            xsum  +=  sAtom_txyz[k*BATCHSIZE + workidx] * qterm;
            dxsum += sAtom_dtxyz[k*BATCHSIZE + workidx] * qterm;
#ifndef PGM_FIELD_ONLY
            d2xsum += sAtom_d2txyz[k*BATCHSIZE + workidx] * qterm;
#endif
            iidx++;
            iidx -= (iidx >= cSim.nfft1) * cSim.nfft1;
          }

          // Fold in y and z B-spline coefficients (z=4 for second pass)
          jidx = (cSim.pmeOrder + j)*BATCHSIZE + workidx;
          kidx = (2*cSim.pmeOrder + 4)*BATCHSIZE + workidx;
          fx += dxsum * sAtom_txyz[jidx]  * sAtom_txyz[kidx];
          fy += xsum  * sAtom_dtxyz[jidx] * sAtom_txyz[kidx];
          fz += xsum  * sAtom_txyz[jidx]  * sAtom_dtxyz[kidx];
#ifndef PGM_FIELD_ONLY
          {
            PMEFloat ty = sAtom_txyz[jidx];
            PMEFloat tz = sAtom_txyz[kidx];
            PMEFloat dty = sAtom_dtxyz[jidx];
            PMEFloat dtz = sAtom_dtxyz[kidx];
            f200 += d2xsum * ty * tz;
            f020 += xsum * sAtom_d2txyz[jidx] * tz;
            f002 += xsum * ty * sAtom_d2txyz[kidx];
            f110 += dxsum * dty * tz;
            f101 += dxsum * ty * dtz;
            f011 += xsum * dty * dtz;
            phi000 += xsum * ty * tz;
          }
#endif
        }
      }
#elif (PME_ORDER == 6)
      }

      // Second pass for the remaining 1.5 z-slabs (order 6)
      kMeshLevel++;
      kMeshLevel -= (kMeshLevel >= cSim.nfft3) * cSim.nfft3;
      for (j = 0; j < yOffset+3; j++) {
        int iidx = iMeshStart;
        int jidx = jMeshStart + j;
        jidx -= (jidx >= cSim.nfft2) * cSim.nfft2;
        int kidx = (kMeshLevel * cSim.nfft1xnfft2) + (jidx * cSim.nfft1);
        int k;
        PMEFloat xsum = (PMEFloat)0.0;
        PMEFloat dxsum = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
        PMEFloat d2xsum = (PMEFloat)0.0;
#endif
#pragma unroll 6
        for (k = 0; k < 6; k++) {
#ifdef use_DPFP
          int2 i2term = tex1Dfetch<int2>(cSim.texXYZ_q, kidx + iidx);
          PMEFloat qterm = __hiloint2double(i2term.y, i2term.x);
#else
          PMEFloat qterm = tex1Dfetch<float>(cSim.texXYZ_q, kidx + iidx);
#endif
          xsum  +=  sAtom_txyz[k*BATCHSIZE + workidx] * qterm;
          dxsum += sAtom_dtxyz[k*BATCHSIZE + workidx] * qterm;
#ifndef PGM_FIELD_ONLY
          d2xsum += sAtom_d2txyz[k*BATCHSIZE + workidx] * qterm;
#endif
          iidx++;
          iidx -= (iidx >= cSim.nfft1) * cSim.nfft1;
        }

        // Fold in y and z B-spline coefficients (note: zOffset+1 for second pass)
        jidx = (cSim.pmeOrder + j)*BATCHSIZE + workidx;
        kidx = (2*cSim.pmeOrder + zOffset + 1)*BATCHSIZE + workidx;
        fx += dxsum * sAtom_txyz[jidx]  * sAtom_txyz[kidx];
        fy += xsum  * sAtom_dtxyz[jidx] * sAtom_txyz[kidx];
        fz += xsum  * sAtom_txyz[jidx]  * sAtom_dtxyz[kidx];
#ifndef PGM_FIELD_ONLY
        {
          PMEFloat ty = sAtom_txyz[jidx];
          PMEFloat tz = sAtom_txyz[kidx];
          PMEFloat dty = sAtom_dtxyz[jidx];
          PMEFloat dtz = sAtom_dtxyz[kidx];
          f200 += d2xsum * ty * tz;
          f020 += xsum * sAtom_d2txyz[jidx] * tz;
          f002 += xsum * ty * sAtom_d2txyz[kidx];
          f110 += dxsum * dty * tz;
          f101 += dxsum * ty * dtz;
          f011 += xsum * dty * dtz;
          // phi_000: theta_x * theta_y * theta_z * Q (no derivatives, no nfft)
          phi000 += xsum * ty * tz;
        }
#endif
      }
#endif

      // Transform fractional gradient to Cartesian electric field.
      // PGM: field = nfft * recipf * grad_sum (no charge multiplication!)
      fz = cSim.recipf[2][0] * fx + cSim.recipf[2][1] * fy + cSim.recipf[2][2] * fz;
      fy = cSim.recipf[1][0] * fx + cSim.recipf[1][1] * fy;
      fx = cSim.recipf[0][0] * fx;

#ifndef PGM_FIELD_ONLY
      // Force: F = -(q*gradphi_cart + mu_tot · H_cart)
      // Using v/w approach: v = R^T*mu, w = H_frac*v, F_hess = -R*w
      // R[i][j] = nfft_{i+1} * recipf[i][j]
      PMEFloat qi_val = cSim.pPGMPermChg[pos + workidx];
      PMEFloat2 dxy = cSim.pPDipXYSP[pos + workidx];
      PMEFloat mu_x = dxy.x, mu_y = dxy.y;
      PMEFloat mu_z = cSim.pPDipZSP[pos + workidx];

      PMEFloat nf1 = (PMEFloat)cSim.nfft1;
      PMEFloat nf2 = (PMEFloat)cSim.nfft2;
      PMEFloat nf3 = (PMEFloat)cSim.nfft3;

      // Cartesian gradphi (scaled by nfft)
      PMEFloat gx = nf1 * fx;
      PMEFloat gy = nf2 * fy;
      PMEFloat gz = nf3 * fz;

      // v = R^T * mu_cart
      PMEFloat r00 = cSim.recipf[0][0];
      PMEFloat r10 = cSim.recipf[1][0], r11 = cSim.recipf[1][1];
      PMEFloat r20 = cSim.recipf[2][0], r21 = cSim.recipf[2][1], r22 = cSim.recipf[2][2];
      PMEFloat v0 = nf1*r00*mu_x + nf2*r10*mu_y + nf3*r20*mu_z;
      PMEFloat v1 =                 nf2*r11*mu_y + nf3*r21*mu_z;
      PMEFloat v2 =                                nf3*r22*mu_z;

      // w = H_frac * v
      PMEFloat w0 = f200*v0 + f110*v1 + f101*v2;
      PMEFloat w1 = f110*v0 + f020*v1 + f011*v2;
      PMEFloat w2 = f101*v0 + f011*v1 + f002*v2;

      // F_hess = -R * w
      PMEFloat fhx = -(nf1*r00*w0);
      PMEFloat fhy = -(nf2*(r10*w0 + r11*w1));
      PMEFloat fhz = -(nf3*(r20*w0 + r21*w1 + r22*w2));

      // Total force = -(q*gradphi_cart) + F_hess
      PMEFloat force_x = -(qi_val*gx) + fhx;
      PMEFloat force_y = -(qi_val*gy) + fhy;
      PMEFloat force_z = -(qi_val*gz) + fhz;
#endif

#ifdef use_DPFP
      fx *= cSim.nfft1 * FORCESCALE;
      fy *= cSim.nfft2 * FORCESCALE;
      fz *= cSim.nfft3 * FORCESCALE;
      atomicAdd((unsigned long long int*)&fld[              workidx], llrint(fx));
      atomicAdd((unsigned long long int*)&fld[  BATCHSIZE + workidx], llrint(fy));
      atomicAdd((unsigned long long int*)&fld[2*BATCHSIZE + workidx], llrint(fz));
#ifndef PGM_FIELD_ONLY
      atomicAdd((unsigned long long int*)&frc[              workidx], llrint(force_x * FORCESCALE));
      atomicAdd((unsigned long long int*)&frc[  BATCHSIZE + workidx], llrint(force_y * FORCESCALE));
      atomicAdd((unsigned long long int*)&frc[2*BATCHSIZE + workidx], llrint(force_z * FORCESCALE));
      // phi_000 for energy (no nfft scaling, just raw grid interpolation)
      atomicAdd((unsigned long long int*)&phi[workidx], llrint(phi000 * FORCESCALE));
#endif
#else
      fx *= cSim.nfft1 * FORCESCALEF;
      fy *= cSim.nfft2 * FORCESCALEF;
      fz *= cSim.nfft3 * FORCESCALEF;
      atomicAdd((unsigned long long int*)&fld[              workidx], fast_llrintf(fx));
      atomicAdd((unsigned long long int*)&fld[  BATCHSIZE + workidx], fast_llrintf(fy));
      atomicAdd((unsigned long long int*)&fld[2*BATCHSIZE + workidx], fast_llrintf(fz));
#ifndef PGM_FIELD_ONLY
      atomicAdd((unsigned long long int*)&frc[              workidx], fast_llrintf(force_x * FORCESCALEF));
      atomicAdd((unsigned long long int*)&frc[  BATCHSIZE + workidx], fast_llrintf(force_y * FORCESCALEF));
      atomicAdd((unsigned long long int*)&frc[2*BATCHSIZE + workidx], fast_llrintf(force_z * FORCESCALEF));
      // phi_000 for energy (no nfft scaling, just raw grid interpolation)
      atomicAdd((unsigned long long int*)&phi[workidx], fast_llrintf(phi000 * FORCESCALEF));
#endif
#endif
    }

    // Write accumulated fields to PGM field accumulators in global memory
    __syncthreads();
    if (pos1 < cSim.atoms) {
      if (threadIdx.x < BATCHSIZE) {
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldXAccumulator[pos1],
                  (unsigned long long int)fld[threadIdx.x]);
      }
      else if (threadIdx.x < 2*BATCHSIZE) {
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldYAccumulator[pos1],
                  (unsigned long long int)fld[threadIdx.x]);
      }
      else if (threadIdx.x < 3*BATCHSIZE) {
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldZAccumulator[pos1],
                  (unsigned long long int)fld[threadIdx.x]);
      }
#ifndef PGM_FIELD_ONLY
      if (threadIdx.x < BATCHSIZE) {
        atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[pos1],
                  (unsigned long long int)frc[threadIdx.x]);
        // phi_000 (only first 32 threads write it)
        atomicAdd((unsigned long long int*)&cSim.pPGMPotentialAccumulator[pos1],
                  (unsigned long long int)phi[threadIdx.x]);
      }
      else if (threadIdx.x < 2*BATCHSIZE) {
        atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[pos1],
                  (unsigned long long int)frc[threadIdx.x]);
      }
      else if (threadIdx.x < 3*BATCHSIZE) {
        atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[pos1],
                  (unsigned long long int)frc[threadIdx.x]);
      }
#endif
    }

    // Advance to the next batch
    pos += gridDim.x * BATCHSIZE;
    __syncthreads();
  }
}
#undef BATCHSIZE
#undef BATCH_BITS_MASK
#undef BATCH_BITS

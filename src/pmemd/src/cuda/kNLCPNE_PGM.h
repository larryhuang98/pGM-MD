#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kNLCPNE_PGM.h - PGM direct-space kernel template (field + optional force)
//
// Copied from kNLCPNE.h NL traversal pattern with PGM physics.
// Uses compile-time PME_ATOMS_PER_WARP (must be #defined before #include).
//
// PGM direct: NO exclusions for electrostatics, standard 12-6 LJ VDW (with exclusions), NO TI/PHMD.
// Charge-charge, charge-dipole, dipole-dipole with Gaussian damping.
//
// Two modes:
//   PGM_FIELD_ONLY defined: compute gradphi(3) only (for SCF iterations)
//   PGM_FIELD_ONLY NOT defined: compute gradphi(3) + force(3) (post-SCF)
//     Force = -(q*gradphi + mu_tot · Hessian_phi), needs B3 + Rn_5 order-3
//     Reaction force on j = -F_i (Newton's 3rd law)
//
// #defines required: PME_ATOMS_PER_WARP (32, 16, or 8)
// #defines optional: PME_IS_ORTHOGONAL, PGM_FIELD_ONLY,
//                    PGM_BFACTOR_WRITE, PGM_BFACTOR_READ
//---------------------------------------------------------------------------------------------

// Fast exponential for B-factor derivative: DPFP keeps full precision,
// SPFP uses hardware __expf (~3x faster than expf, 2 ulp accuracy).
// Two exp() calls per pair dominate the inner-loop cost — this alone gives ~20% speedup.
#undef PGM_EXP
#ifdef use_DPFP
#define PGM_EXP(x) exp(x)
#else
#define PGM_EXP(x) __expf(x)
#endif

// PGMPairVDW (LJ or GVDW pair term) is defined in kPGMPairVDW_PGM.h, included at file scope.



{
#ifdef PGM_DEBUG_PAIR_COUNTS
  // Pair counters (defined in kCalculatePMENonbondEnergy_PGM.cu)
  extern __device__ unsigned long long int dPGMHomePairCount;
  extern __device__ unsigned long long int dPGMNeighPairCount;
  extern __device__ unsigned long long int dPGMNLEntryCount;
#endif

  const PMEFloat delta = (PMEFloat)1.0e-5;
  const int THREADS_PER_BLOCK = PMENONBONDFORCES_THREADS_PER_BLOCK;

  struct NLWarp {
    NLEntry nlEntry;
    unsigned int pos;
    bool bHomeCell;
  };

  __shared__ unsigned int sNLEntries;
  __shared__ unsigned int sNext[GRID];
  __shared__ unsigned int sRecv[GRID];
#if (PME_ATOMS_PER_WARP <= 16)
  __shared__ unsigned int sStart[GRID];
  __shared__ unsigned int sRecvStart[GRID];
#endif
  __shared__ volatile NLWarp sWarp[THREADS_PER_BLOCK / GRID];

  if (threadIdx.x == 0) {
    sNLEntries = *(cSim.pNLEntries);
  }
  if (threadIdx.x < GRID) {
    unsigned int offset = cSim.NLAtomsPerWarp * (threadIdx.x >> cSim.NLAtomsPerWarpBits);
    sNext[threadIdx.x] = ((threadIdx.x + 1) & cSim.NLAtomsPerWarpBitsMask) + offset;
    sRecv[threadIdx.x] = threadIdx.x - 1 +
                         cSim.NLAtomsPerWarp*((threadIdx.x &
                                               cSim.NLAtomsPerWarpBitsMask) == 0);
  }
#if (PME_ATOMS_PER_WARP == 16)
  if (threadIdx.x < GRID) {
    unsigned int basenum = ((threadIdx.x + 1) & 15) + 20*(threadIdx.x >= 16);
    sStart[threadIdx.x] = basenum - 16*(basenum >> 5);
    basenum = threadIdx.x + 16*((threadIdx.x & 15) == 0) + 12*(threadIdx.x >= 16) - 1;
    sRecvStart[threadIdx.x] = basenum - 16*(basenum >> 5);
  }
#elif (PME_ATOMS_PER_WARP == 8)
  __syncthreads();
  if (threadIdx.x < GRID) {
    int j = sNext[threadIdx.x];
    int jrec = sRecv[threadIdx.x];
    int ffstep = threadIdx.x >> 3;
    while (ffstep > 0) {
      j = sNext[j];
      jrec = sRecv[jrec];
      ffstep--;
    }
    sStart[threadIdx.x] = j;
    sRecvStart[threadIdx.x] = jrec;
  }
#endif
  __syncthreads();

  unsigned int tgx = threadIdx.x & (GRID - 1);
  volatile NLWarp* psWarp = &sWarp[threadIdx.x / GRID];
  if (tgx == 0) {
    psWarp->pos = ((blockIdx.x * blockDim.x) + threadIdx.x) / GRID;
  }
  __syncthreads();

#ifndef PGM_FIELD_ONLY
  const PMEFloat sixth = (PMEFloat)0.16666666666667;
  const PMEFloat twelvth = (PMEFloat)0.083333333333333;
  PMEForceAccumulator evdw = (PMEForceAccumulator)0;
#ifdef PME_VIRIAL
  PMEForceAccumulator vir_11 = (PMEForceAccumulator)0;
  PMEForceAccumulator vir_22 = (PMEForceAccumulator)0;
  PMEForceAccumulator vir_33 = (PMEForceAccumulator)0;
#endif
#endif

  while (psWarp->pos < sNLEntries) {

#ifdef PGM_DEBUG_PAIR_COUNTS
    if (tgx == 0) atomicAdd(&dPGMNLEntryCount, 1ULL);
#endif

    if (tgx < 4) {
      psWarp->nlEntry.array[tgx] = cSim.pNLEntry[psWarp->pos].array[tgx];
    }
    __SYNCWARP(0xFFFFFFFF);
    if (tgx == 0) {
      psWarp->bHomeCell = psWarp->nlEntry.NL.ymax & 0x1;
      psWarp->nlEntry.NL.ymax >>= NLENTRY_YMAX_SHIFT;
    }
    __SYNCWARP(0xFFFFFFFF);

#if defined(PGM_BFACTOR_WRITE) || defined(PGM_BFACTOR_READ) || defined(PGM_BFACTOR_WRITE16) || defined(PGM_BFACTOR_READ16)
    int bfactor_base = cSim.pPGMBFactorOffset[psWarp->pos];
#endif

    // Load i atom data
    PMEFloat dip_ix, dip_iy, dip_iz;
    PMEFloat fld_ix = (PMEFloat)0.0;
    PMEFloat fld_iy = (PMEFloat)0.0;
    PMEFloat fld_iz = (PMEFloat)0.0;
#ifndef PGM_BFACTOR_READ
    PMEFloat xi, yi, zi, qi;
    PMEFloat rg_i;
#endif
#ifndef PGM_FIELD_ONLY
    PMEFloat frc_ix = (PMEFloat)0.0;
    PMEFloat frc_iy = (PMEFloat)0.0;
    PMEFloat frc_iz = (PMEFloat)0.0;
    PMEFloat phi_i = (PMEFloat)0.0;  // phi_000 accumulator for energy
#endif

    unsigned int iindex = psWarp->nlEntry.NL.ypos + (tgx & cSim.NLAtomsPerWarpBitsMask);

#ifdef PGM_BFACTOR_READ
    // T2 cache-read: only need i-atom dipoles
    if (iindex < psWarp->nlEntry.NL.ymax) {
      PMEFloat2 dipxy = cSim.pPDipXYSP[iindex];
      dip_ix = dipxy.x;
      dip_iy = dipxy.y;
      dip_iz = cSim.pPDipZSP[iindex];
    }
    else {
      dip_ix = (PMEFloat)0.0;
      dip_iy = (PMEFloat)0.0;
      dip_iz = (PMEFloat)0.0;
    }
#else
    if (iindex < psWarp->nlEntry.NL.ymax) {
      PMEFloat2 xy = cSim.pAtomXYSP[iindex];
      xi = xy.x;
      yi = xy.y;
      zi = cSim.pAtomZSP[iindex];
      qi = cSim.pPGMPermChg[iindex];
      PMEFloat2 dipxy = cSim.pPDipXYSP[iindex];
      dip_ix = dipxy.x;
      dip_iy = dipxy.y;
      dip_iz = cSim.pPDipZSP[iindex];
      rg_i = (PMEFloat)cSim.pPGMRGauss[iindex];
    }
    else {
      xi = (PMEFloat)10000.0 * iindex;
      yi = (PMEFloat)10000.0 * iindex;
      zi = (PMEFloat)10000.0 * iindex;
      qi = (PMEFloat)0.0;
      dip_ix = (PMEFloat)0.0;
      dip_iy = (PMEFloat)0.0;
      dip_iz = (PMEFloat)0.0;
      rg_i = (PMEFloat)1.0;
    }

#ifndef PME_IS_ORTHOGONAL
    xi = cSim.ucellf[0][0]*xi + cSim.ucellf[0][1]*yi + cSim.ucellf[0][2]*zi;
    yi =                        cSim.ucellf[1][1]*yi + cSim.ucellf[1][2]*zi;
    zi =                                               cSim.ucellf[2][2]*zi;
#endif
#endif // PGM_BFACTOR_READ

#ifndef PGM_FIELD_ONLY
    // VDW: Load LJID from standard charge+LJID buffer
    unsigned int LJIDi = 0;
    if (iindex < psWarp->nlEntry.NL.ymax) {
      PMEFloat2 qljid = cSim.pAtomChargeSPLJID[iindex];
#ifdef use_DPFP
      LJIDi = __double_as_longlong(qljid.y);
#else
      LJIDi = __float_as_uint(qljid.y);
#endif
    }
#endif

    // ---- Home cell tile ----
    if (psWarp->bHomeCell) {
#ifndef PGM_FIELD_ONLY
      // Read exclusion mask for VDW
      unsigned int exclusion = cSim.pNLAtomList[psWarp->nlEntry.NL.offset +
                               (tgx & cSim.NLAtomsPerWarpBitsMask)];
#endif
      __SYNCWARP(0xFFFFFFFF);
      if (tgx == 0)
        psWarp->nlEntry.NL.offset += cSim.NLAtomsPerWarp;
      __SYNCWARP(0xFFFFFFFF);
#ifndef PGM_FIELD_ONLY
#if (PME_ATOMS_PER_WARP == 16)
      exclusion = __SHFL_UP(0xFFFFFFFF, exclusion, 16);
      exclusion >>= 4*(tgx >> cSim.NLAtomsPerWarpBits);
#elif (PME_ATOMS_PER_WARP == 8)
      exclusion = __SHFL_UP(0xFFFFFFFF, exclusion, 16);
      exclusion = __SHFL_UP(0xFFFFFFFF, exclusion, 8);
      exclusion >>= (tgx >> cSim.NLAtomsPerWarpBits);
#endif
      unsigned int LJIDj = LJIDi;
      LJIDi *= cSim.LJTypes;
      PMEFloat TLevdw = (PMEFloat)0.0;
#ifdef PME_VIRIAL
      PMEFloat TLvir_11 = (PMEFloat)0.0;
      PMEFloat TLvir_22 = (PMEFloat)0.0;
      PMEFloat TLvir_33 = (PMEFloat)0.0;
#endif
#endif

      // Set up initial j and jrec positions
#if (PME_ATOMS_PER_WARP < 32)
      int j = sStart[tgx];
      int jrec = sRecvStart[tgx];
#else
      int j = sNext[tgx];
      int jrec = sRecv[tgx];
#endif

#if (PME_ATOMS_PER_WARP == 32)
      const int iftlim = 16;
#elif (PME_ATOMS_PER_WARP == 16)
      const int iftlim = 4;
#elif (PME_ATOMS_PER_WARP == 8)
      const int iftlim = 1;
#endif

      for (int ift = 0; ift < iftlim; ift++) {
        // Init field temporaries (outside divergent block)
        PMEFloat e_ix = (PMEFloat)0.0, e_iy = (PMEFloat)0.0, e_iz = (PMEFloat)0.0;
        PMEFloat rjx = (PMEFloat)0.0, rjy = (PMEFloat)0.0, rjz = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
        PMEFloat fx_ij = (PMEFloat)0.0, fy_ij = (PMEFloat)0.0, fz_ij = (PMEFloat)0.0;
        PMEFloat phi_000_i = (PMEFloat)0.0;  // phi_000 at i from j
        PMEFloat phi_000_rj = (PMEFloat)0.0; // phi_000 at j from i (for shuffle)
        PMEFloat evdw_pair = (PMEFloat)0.0;
#ifdef PME_VIRIAL
        PMEFloat xij_v = (PMEFloat)0.0, yij_v = (PMEFloat)0.0, zij_v = (PMEFloat)0.0;
#endif
#endif

#ifdef PGM_BFACTOR_READ
        // ---- BD cache-read path: reconstruct field from {nB1, nB2, dx, dy, dz} ----
        // 5 loads instead of 6 (T2), trades 17% bandwidth for negligible extra FLOPs.
        // Field = nB1*mu + nB2*(r·mu)*r  where nB1=-B1, nB2=-B2, r=(dx,dy,dz)
        // All __SHFL calls MUST be outside divergent branches (all threads participate)
        PMEFloat djx   = __SHFL(0xFFFFFFFF, dip_ix, j);
        PMEFloat djy   = __SHFL(0xFFFFFFFF, dip_iy, j);
        PMEFloat djz   = __SHFL(0xFFFFFFFF, dip_iz, j);
        {
          int pair_idx = bfactor_base + ift * 32 + tgx;
          PMEFloat nB1 = cSim.pPGMBD_nB1[pair_idx];
          PMEFloat nB2 = cSim.pPGMBD_nB2[pair_idx];
          PMEFloat dx  = cSim.pPGMBD_dx[pair_idx];
          PMEFloat dy  = cSim.pPGMBD_dy[pair_idx];
          PMEFloat dz  = cSim.pPGMBD_dz[pair_idx];

          // Field on i from j's dipole: E_i = nB1*mu_j + nB2*(r·mu_j)*r
          PMEFloat dot_rj = dx * djx + dy * djy + dz * djz;
          PMEFloat bd_j = nB2 * dot_rj;
          e_ix = nB1 * djx + bd_j * dx;
          e_iy = nB1 * djy + bd_j * dy;
          e_iz = nB1 * djz + bd_j * dz;

          // Reaction field on j from i's dipole (same {nB1,nB2,r} — tensor is symmetric)
          PMEFloat dot_ri = dx * dip_ix + dy * dip_iy + dz * dip_iz;
          PMEFloat bd_i = nB2 * dot_ri;
          rjx = nB1 * dip_ix + bd_i * dx;
          rjy = nB1 * dip_iy + bd_i * dy;
          rjz = nB1 * dip_iz + bd_i * dz;
        }
#else
        // All __SHFL calls MUST be outside divergent branches (all threads must participate)
        PMEFloat xj_s  = __SHFL(0xFFFFFFFF, xi, j);
        PMEFloat yj_s  = __SHFL(0xFFFFFFFF, yi, j);
        PMEFloat zj_s  = __SHFL(0xFFFFFFFF, zi, j);
        PMEFloat qj    = __SHFL(0xFFFFFFFF, qi, j);
        PMEFloat djx   = __SHFL(0xFFFFFFFF, dip_ix, j);
        PMEFloat djy   = __SHFL(0xFFFFFFFF, dip_iy, j);
        PMEFloat djz   = __SHFL(0xFFFFFFFF, dip_iz, j);
        PMEFloat rg_j  = __SHFL(0xFFFFFFFF, rg_i, j);
#ifndef PGM_FIELD_ONLY
        unsigned int shLJIDj = __SHFL(0xFFFFFFFF, LJIDj, j);
#endif

        PMEFloat xij = xi - xj_s;
        PMEFloat yij = yi - yj_s;
        PMEFloat zij = zi - zj_s;
        PMEFloat r2  = xij*xij + yij*yij + zij*zij;
#if defined(PME_VIRIAL) && !defined(PGM_FIELD_ONLY)
        xij_v = xij; yij_v = yij; zij_v = zij;
#endif
        // ---- Standard computation path (also used for cache-write) ----
#ifdef PGM_SHORT_RANGE
        if (r2 < cSim.scfLocalCut2 && r2 > delta) {
#else
        if (r2 < cSim.cut2 && r2 > delta) {
#endif
#ifdef PGM_DEBUG_PAIR_COUNTS
          atomicAdd(&dPGMHomePairCount, 1ULL);
#endif
#ifdef PGM_BFACTOR_READ16
          PMEFloat B1, B2;
          {
            int pair_idx = bfactor_base + ift * 32 + tgx;
            half2 hv = *reinterpret_cast<const half2*>(&cSim.pPGMBD_h2[pair_idx]);
            float2 fv = __half22float2(hv);
            B1 = -fv.x * (PMEFloat)(1.0/4096.0);
            B2 = -fv.y * (PMEFloat)(1.0/4096.0);
          }
#else
          PMEFloat rinv  = rsqrt(r2);
          PMEFloat r     = r2 * rinv;
          PMEFloat r2inv = rinv * rinv;

          // Ewald B-factors (PGM uses erf, not erfc, for direct-space Ewald part)
#ifdef use_DPFP
          PMEFloat erf_ew = (PMEFloat)1.0 - erfc(cSim.ew_coeffSP * r);
#else
          PMEFloat erf_ew = (PMEFloat)1.0 - fasterfc(r);
#endif
          PMEFloat B0_ew = erf_ew * rinv;
          PMEFloat fact  = -cSim.negTwoEw_coeffRsqrtPI * PGM_EXP(-cSim.ew_coeff2 * r2);
          PMEFloat B1_ew = (B0_ew - fact) * r2inv;
          fact = (PMEFloat)2.0 * cSim.ew_coeff2 * fact;
          PMEFloat B2_ew = ((PMEFloat)3.0 * B1_ew - fact) * r2inv;

          // Gaussian B-factors (erf(beta*r)/r)
          PMEFloat betaij = rsqrt(rg_i + rg_j);
          PMEFloat y = betaij * r;
#ifdef use_DPFP
          PMEFloat erfc_g = erfc(y);
#else
          PMEFloat erfc_g = erfcf(y);
#endif
          PMEFloat B0_g = ((PMEFloat)1.0 - erfc_g) * rinv;   // erf(beta*r)/r
          PMEFloat fact_g = (PMEFloat)M_2_SQRTPI * betaij * PGM_EXP(-y * y);
          PMEFloat B1_g = (B0_g - fact_g) * r2inv;
          fact_g = (PMEFloat)2.0 * betaij * betaij * fact_g;
          PMEFloat B2_g = ((PMEFloat)3.0 * B1_g - fact_g) * r2inv;

#ifndef PGM_FIELD_ONLY
          // B3 extension for force (Hessian of potential)
          fact = (PMEFloat)2.0 * cSim.ew_coeff2 * fact;
          PMEFloat B3_ew = ((PMEFloat)5.0 * B2_ew - fact) * r2inv;
          fact_g = (PMEFloat)2.0 * betaij * betaij * fact_g;
          PMEFloat B3_g = ((PMEFloat)5.0 * B2_g - fact_g) * r2inv;
#endif

          // Combined: direct-space PGM = Ewald(erf) - Gaussian(erf)
          PMEFloat B1 = B1_ew - B1_g;
          PMEFloat B2 = B2_g - B2_ew;
#endif // PGM_BFACTOR_READ16
#ifndef PGM_FIELD_ONLY
          PMEFloat B0 = B0_ew - B0_g;  // for phi_000 (potential)
          PMEFloat B3 = B3_ew - B3_g;
#endif

#ifdef PGM_BFACTOR_WRITE
          // Store BD cache {-B1, -B2, dx, dy, dz} for SCF iterations
          // Field = nB1*mu + nB2*(r·mu)*r (5 loads, 17% less BW than T2)
          {
            int pair_idx = bfactor_base + ift * 32 + tgx;
            cSim.pPGMBD_nB1[pair_idx] = -B1;
            cSim.pPGMBD_nB2[pair_idx] = -B2;
            cSim.pPGMBD_dx[pair_idx]  = xij;
            cSim.pPGMBD_dy[pair_idx]  = yij;
            cSim.pPGMBD_dz[pair_idx]  = zij;
          }
#endif
#ifdef PGM_BFACTOR_WRITE16
          {
            int pair_idx = bfactor_base + ift * 32 + tgx;
            half2 hv = __floats2half2_rn(-B1 * (PMEFloat)4096.0, -B2 * (PMEFloat)4096.0);
            *reinterpret_cast<half2*>(&cSim.pPGMBD_h2[pair_idx]) = hv;
          }
#endif

          // gradphi on i from j (= -E_field, stored in accumulators for chain-rule)
          // gradphi_x = B1*xij*qj - B1*djx - B2*(dj·rij)*xij  [rij = pos_i - pos_j]
          PMEFloat mu_dot_r = djx * xij + djy * yij + djz * zij;
          e_ix = qj * B1 * xij - B1 * djx - B2 * mu_dot_r * xij;
          e_iy = qj * B1 * yij - B1 * djy - B2 * mu_dot_r * yij;
          e_iz = qj * B1 * zij - B1 * djz - B2 * mu_dot_r * zij;

          // Reaction gradphi on j from i (to be shuffled back)
          PMEFloat mu_dot_r_i = dip_ix*(-xij) + dip_iy*(-yij) + dip_iz*(-zij);
          rjx = qi*B1*(-xij) - B1*dip_ix - B2*mu_dot_r_i*(-xij);
          rjy = qi*B1*(-yij) - B1*dip_iy - B2*mu_dot_r_i*(-yij);
          rjz = qi*B1*(-zij) - B1*dip_iz - B2*mu_dot_r_i*(-zij);

#ifndef PGM_FIELD_ONLY
          // phi_000 at i from j: -(B0_ew - B0_g)*qj - B1*(dj·xij)
          // Matches CPU: B(0)*qj + B(1)*del·dj where B(0)=B0_g-B0_ew=-B0, del=-xij
          phi_000_i = -B0 * qj - B1 * mu_dot_r;

          // phi_000 at j from i: -(B0_ew - B0_g)*qi - B1*(di·(-xij))
          phi_000_rj = -B0 * qi - B1 * mu_dot_r_i;

          // Hessian of phi at i from j (6 components)
          // Uses Rn_5 elements in GPU convention (del = -xij for CPU, but squared terms same)
          PMEFloat dx2 = xij * xij, dy2 = yij * yij, dz2 = zij * zij;
          PMEFloat delxd = B2 + dx2 * B3;
          PMEFloat delyd = B2 + dy2 * B3;
          PMEFloat delzd = B2 + dz2 * B3;

          PMEFloat phi_200 = (B1 + dx2*B2)*qj - (PMEFloat)2.0*B2*xij*djx - delxd*mu_dot_r;
          PMEFloat phi_020 = (B1 + dy2*B2)*qj - (PMEFloat)2.0*B2*yij*djy - delyd*mu_dot_r;
          PMEFloat phi_002 = (B1 + dz2*B2)*qj - (PMEFloat)2.0*B2*zij*djz - delzd*mu_dot_r;
          PMEFloat phi_110 = xij*yij*B2*qj - B2*(yij*djx + xij*djy) - xij*yij*B3*mu_dot_r;
          PMEFloat phi_101 = xij*zij*B2*qj - B2*(zij*djx + xij*djz) - xij*zij*B3*mu_dot_r;
          PMEFloat phi_011 = yij*zij*B2*qj - B2*(zij*djy + yij*djz) - yij*zij*B3*mu_dot_r;

          // Force on i from j: F = -(q*gradphi + mu_tot · Hessian)
          // e_ix = phi(100) = gradphi_x; dip_ix = mu_tot_ix
          fx_ij = -(qi*e_ix + dip_ix*phi_200 + dip_iy*phi_110 + dip_iz*phi_101);
          fy_ij = -(qi*e_iy + dip_ix*phi_110 + dip_iy*phi_020 + dip_iz*phi_011);
          fz_ij = -(qi*e_iz + dip_ix*phi_101 + dip_iy*phi_011 + dip_iz*phi_002);

          // VDW: standard 12-6 LJ (uses exclusion mask from NL tiles)
          // Pre-divide by coulomb_const because Fortran multiplies ALL forces
          // by coulomb_const after download (but VDW is already in kcal/mol/A).
          if (!(exclusion & 0x1)) {
            unsigned int ljindex = LJIDi + shLJIDj;
            PMEFloat2 ljterm = cSim.pLJTerm[ljindex];
            PMEFloat e_vdw, df_vdw;
            PGMPairVDW(ljterm, r2inv, rinv, betaij, y, (PMEFloat)1.0 - erfc_g, e_vdw, df_vdw);
            df_vdw *= cSim.invCoulombConst;
            fx_ij += df_vdw * xij;
            fy_ij += df_vdw * yij;
            fz_ij += df_vdw * zij;
            evdw_pair = e_vdw;
          }
#endif
        }
#ifdef PGM_BFACTOR_WRITE16
        else {
          int pair_idx = bfactor_base + ift * 32 + tgx;
          cSim.pPGMBD_h2[pair_idx] = 0u;
        }
#endif
#ifdef PGM_BFACTOR_WRITE
        else {
          // Out-of-cutoff: store zeros so BD cache-read sees zero contribution
          int pair_idx = bfactor_base + ift * 32 + tgx;
          cSim.pPGMBD_nB1[pair_idx] = (PMEFloat)0.0;
          cSim.pPGMBD_nB2[pair_idx] = (PMEFloat)0.0;
          cSim.pPGMBD_dx[pair_idx]  = (PMEFloat)0.0;
          cSim.pPGMBD_dy[pair_idx]  = (PMEFloat)0.0;
          cSim.pPGMBD_dz[pair_idx]  = (PMEFloat)0.0;
        }
#endif
#endif // !PGM_BFACTOR_READ

        // Halve at last iteration to avoid double-counting symmetric pairs
        // (matches standard kernel's df *= 0.5 pattern)
#if (PME_ATOMS_PER_WARP == 32)
        if (ift == iftlim - 1) {
#elif (PME_ATOMS_PER_WARP == 16)
        if (ift == iftlim - 1 && tgx >= 16) {
#elif (PME_ATOMS_PER_WARP == 8)
        if (tgx >= 24) {
#endif
          e_ix *= (PMEFloat)0.5; e_iy *= (PMEFloat)0.5; e_iz *= (PMEFloat)0.5;
          rjx  *= (PMEFloat)0.5; rjy  *= (PMEFloat)0.5; rjz  *= (PMEFloat)0.5;
#ifndef PGM_FIELD_ONLY
          fx_ij *= (PMEFloat)0.5; fy_ij *= (PMEFloat)0.5; fz_ij *= (PMEFloat)0.5;
          phi_000_i *= (PMEFloat)0.5; phi_000_rj *= (PMEFloat)0.5;
          evdw_pair *= (PMEFloat)0.5;
#endif
        }
#ifndef PGM_FIELD_ONLY
        TLevdw += evdw_pair;
#ifdef PME_VIRIAL
        TLvir_11 -= xij_v * fx_ij;
        TLvir_22 -= yij_v * fy_ij;
        TLvir_33 -= zij_v * fz_ij;
#endif
#endif

        // Accumulate forward gradphi on i
        fld_ix += e_ix; fld_iy += e_iy; fld_iz += e_iz;

        // Shuffle reaction gradphi back (ALL threads must participate)
        fld_ix += __SHFL(0xFFFFFFFF, rjx, jrec);
        fld_iy += __SHFL(0xFFFFFFFF, rjy, jrec);
        fld_iz += __SHFL(0xFFFFFFFF, rjz, jrec);

#ifndef PGM_FIELD_ONLY
        // Accumulate force on i
        frc_ix += fx_ij; frc_iy += fy_ij; frc_iz += fz_ij;

        // Shuffle reaction force to j: F_j = -F_i (Newton's 3rd law)
        frc_ix -= __SHFL(0xFFFFFFFF, fx_ij, jrec);
        frc_iy -= __SHFL(0xFFFFFFFF, fy_ij, jrec);
        frc_iz -= __SHFL(0xFFFFFFFF, fz_ij, jrec);

        // Accumulate phi_000: forward + shuffle reaction back
        phi_i += phi_000_i;
        phi_i += __SHFL(0xFFFFFFFF, phi_000_rj, jrec);
        exclusion >>= 1;
#endif

        j = sNext[j];
        jrec = sRecv[jrec];
      }

#ifndef PGM_FIELD_ONLY
      // Accumulate tile VDW energy
      evdw += llrint(TLevdw * ENERGYSCALE);
#ifdef PME_VIRIAL
#ifdef use_SPFP
      vir_11 += fast_llrintf(TLvir_11 * FORCESCALEF);
      vir_22 += fast_llrintf(TLvir_22 * FORCESCALEF);
      vir_33 += fast_llrintf(TLvir_33 * FORCESCALEF);
#else
      vir_11 += llrint(TLvir_11 * FORCESCALE);
      vir_22 += llrint(TLvir_22 * FORCESCALE);
      vir_33 += llrint(TLvir_33 * FORCESCALE);
#endif
#endif
#endif

      // Write home cell gradphi (field accumulator)
      if (iindex < psWarp->nlEntry.NL.ymax) {
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldXAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)fld_ix * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldYAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)fld_iy * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldZAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)fld_iz * FORCESCALE)));
#ifndef PGM_FIELD_ONLY
        // Write home cell force
        atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)frc_ix * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)frc_iy * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)frc_iz * FORCESCALE)));
        // Write home cell phi_000
        atomicAdd((unsigned long long int*)&cSim.pPGMPotentialAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)phi_i * FORCESCALE)));
#endif
      }
      fld_ix = (PMEFloat)0.0;
      fld_iy = (PMEFloat)0.0;
      fld_iz = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
      frc_ix = (PMEFloat)0.0;
      frc_iy = (PMEFloat)0.0;
      frc_iz = (PMEFloat)0.0;
      phi_i = (PMEFloat)0.0;
#endif
    }
#ifndef PGM_FIELD_ONLY
    else {
      LJIDi *= cSim.LJTypes;
    }
#endif

#if defined(PGM_BFACTOR_WRITE) || defined(PGM_BFACTOR_READ) || defined(PGM_BFACTOR_WRITE16) || defined(PGM_BFACTOR_READ16)
    // Advance bfactor_base past home cell pairs (if home cell was processed)
    if (psWarp->bHomeCell) {
#if (PME_ATOMS_PER_WARP == 32)
      bfactor_base += 16 * 32;
#elif (PME_ATOMS_PER_WARP == 16)
      bfactor_base += 4 * 32;
#elif (PME_ATOMS_PER_WARP == 8)
      bfactor_base += 1 * 32;
#endif
    }
#endif

    // ---- Neighbor tiles ----
    unsigned int xatoms = psWarp->nlEntry.NL.xatoms;
    int tx = 0;
    while (tx < (int)xatoms) {

      // Read j atom ID (packed: atom << NLATOM_CELL_SHIFT | cellType)
      unsigned int jid_raw = cSim.pNLAtomList[psWarp->nlEntry.NL.offset + tgx];
      __SYNCWARP(0xFFFFFFFF);
      if (tgx == 0)
        psWarp->nlEntry.NL.offset += GRID;  // skip atom IDs (GRID=32 entries)
      __SYNCWARP(0xFFFFFFFF);
#ifndef PGM_FIELD_ONLY
      // Read exclusion mask for VDW
      unsigned int exclusion = cSim.pNLAtomList[psWarp->nlEntry.NL.offset +
                               (tgx & cSim.NLAtomsPerWarpBitsMask)];
#endif
      __SYNCWARP(0xFFFFFFFF);
      if (tgx == 0)
        psWarp->nlEntry.NL.offset += cSim.NLAtomsPerWarp;
      __SYNCWARP(0xFFFFFFFF);
#ifndef PGM_FIELD_ONLY
#if (PME_ATOMS_PER_WARP < 32)
      exclusion >>= cSim.NLAtomsPerWarp * (tgx >> cSim.NLAtomsPerWarpBits);
#endif
#endif

      // Unpack j atom data
      unsigned int jindex = jid_raw >> NLATOM_CELL_SHIFT;
#ifndef PGM_BFACTOR_READ
      int cell = jid_raw & NLATOM_CELL_TYPE_MASK;
#endif

#ifdef PGM_BFACTOR_READ
      // T2 cache-read: only need j dipoles, skip positions/charges/r_gauss
      PMEFloat djx, djy, djz;
      if (tx + (int)tgx < (int)xatoms) {
        PMEFloat2 dipxy = cSim.pPDipXYSP[jindex];
        djx = dipxy.x;  djy = dipxy.y;
        djz = cSim.pPDipZSP[jindex];
      }
      else {
        djx = (PMEFloat)0.0;  djy = (PMEFloat)0.0;  djz = (PMEFloat)0.0;
      }
#else
      PMEFloat xj, yj, zj, qj_val, djx, djy, djz, rg_j_val;
#ifndef PGM_FIELD_ONLY
      unsigned int LJIDj_val = 0;
#endif
      if (tx + (int)tgx < (int)xatoms) {
        PMEFloat2 xy = cSim.pAtomXYSP[jindex];
        xj = xy.x;  yj = xy.y;
        zj = cSim.pAtomZSP[jindex];
        qj_val = cSim.pPGMPermChg[jindex];
        PMEFloat2 dipxy = cSim.pPDipXYSP[jindex];
        djx = dipxy.x;  djy = dipxy.y;
        djz = cSim.pPDipZSP[jindex];
        rg_j_val = (PMEFloat)cSim.pPGMRGauss[jindex];
#ifndef PGM_FIELD_ONLY
        {
          PMEFloat2 ljqid = cSim.pAtomChargeSPLJID[jindex];
#ifdef use_DPFP
          LJIDj_val = __double_as_longlong(ljqid.y);
#else
          LJIDj_val = __float_as_uint(ljqid.y);
#endif
        }
#endif
      }
      else {
        xj = (PMEFloat)-10000.0 * tgx;
        yj = (PMEFloat)-10000.0 * tgx;
        zj = (PMEFloat)-10000.0 * tgx;
        qj_val = (PMEFloat)0.0;
        djx = (PMEFloat)0.0;  djy = (PMEFloat)0.0;  djz = (PMEFloat)0.0;
        rg_j_val = (PMEFloat)1.0;
      }

      // Apply cell offset for PBC
      xj += cSim.cellOffset[cell][0];
      yj += cSim.cellOffset[cell][1];
      zj += cSim.cellOffset[cell][2];

#ifndef PME_IS_ORTHOGONAL
      xj = cSim.ucellf[0][0]*xj + cSim.ucellf[0][1]*yj + cSim.ucellf[0][2]*zj;
      yj =                         cSim.ucellf[1][1]*yj + cSim.ucellf[1][2]*zj;
      zj =                                                cSim.ucellf[2][2]*zj;
#endif
#endif // !PGM_BFACTOR_READ

      PMEFloat fld_jx = (PMEFloat)0.0;
      PMEFloat fld_jy = (PMEFloat)0.0;
      PMEFloat fld_jz = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
      PMEFloat frc_jx = (PMEFloat)0.0;
      PMEFloat frc_jy = (PMEFloat)0.0;
      PMEFloat frc_jz = (PMEFloat)0.0;
      PMEFloat phi_j = (PMEFloat)0.0;  // phi_000 accumulator for j atoms
      PMEFloat TLevdw = (PMEFloat)0.0;
#ifdef PME_VIRIAL
      PMEFloat TLvir_11 = (PMEFloat)0.0;
      PMEFloat TLvir_22 = (PMEFloat)0.0;
      PMEFloat TLvir_33 = (PMEFloat)0.0;
#endif
#endif

      int j = tgx;
      int jrec = tgx;
#if defined(PGM_BFACTOR_WRITE) || defined(PGM_BFACTOR_READ) || defined(PGM_BFACTOR_WRITE16) || defined(PGM_BFACTOR_READ16)
      int neigh_iter = 0;
#endif
#pragma unroll 2
      do {
        // Compute gradphi on j from i (init outside divergent branch for shuffle)
        PMEFloat ej_x = (PMEFloat)0.0;
        PMEFloat ej_y = (PMEFloat)0.0;
        PMEFloat ej_z = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
        PMEFloat fx_ij = (PMEFloat)0.0;
        PMEFloat fy_ij = (PMEFloat)0.0;
        PMEFloat fz_ij = (PMEFloat)0.0;
        PMEFloat phi_000_ij = (PMEFloat)0.0;  // phi_000 at j from i (for shuffle)
#endif

#ifdef PGM_BFACTOR_READ
        // ---- BD cache-read path: reconstruct field from {nB1, nB2, dx, dy, dz} ----
        PMEFloat d_jx  = __SHFL(0xFFFFFFFF, djx, j);
        PMEFloat d_jy  = __SHFL(0xFFFFFFFF, djy, j);
        PMEFloat d_jz  = __SHFL(0xFFFFFFFF, djz, j);
        {
          int pair_idx = bfactor_base + neigh_iter * 32 + tgx;
          PMEFloat nB1 = cSim.pPGMBD_nB1[pair_idx];
          PMEFloat nB2 = cSim.pPGMBD_nB2[pair_idx];
          PMEFloat dx  = cSim.pPGMBD_dx[pair_idx];
          PMEFloat dy  = cSim.pPGMBD_dy[pair_idx];
          PMEFloat dz  = cSim.pPGMBD_dz[pair_idx];

          // Field on i from j's dipole: E_i = nB1*mu_j + nB2*(r·mu_j)*r
          PMEFloat dot_rj = dx * d_jx + dy * d_jy + dz * d_jz;
          PMEFloat bd_j = nB2 * dot_rj;
          fld_ix += nB1 * d_jx + bd_j * dx;
          fld_iy += nB1 * d_jy + bd_j * dy;
          fld_iz += nB1 * d_jz + bd_j * dz;

          // Reaction field on j from i's dipole (same tensor, symmetric)
          PMEFloat dot_ri = dx * dip_ix + dy * dip_iy + dz * dip_iz;
          PMEFloat bd_i = nB2 * dot_ri;
          ej_x = nB1 * dip_ix + bd_i * dx;
          ej_y = nB1 * dip_iy + bd_i * dy;
          ej_z = nB1 * dip_iz + bd_i * dz;
        }
#else
        // All __SHFL calls MUST be outside divergent branches
        PMEFloat xj_s  = __SHFL(0xFFFFFFFF, xj, j);
        PMEFloat yj_s  = __SHFL(0xFFFFFFFF, yj, j);
        PMEFloat zj_s  = __SHFL(0xFFFFFFFF, zj, j);
        PMEFloat qj    = __SHFL(0xFFFFFFFF, qj_val, j);
        PMEFloat d_jx  = __SHFL(0xFFFFFFFF, djx, j);
        PMEFloat d_jy  = __SHFL(0xFFFFFFFF, djy, j);
        PMEFloat d_jz  = __SHFL(0xFFFFFFFF, djz, j);
        PMEFloat rg_j  = __SHFL(0xFFFFFFFF, rg_j_val, j);
#ifndef PGM_FIELD_ONLY
        unsigned int shLJIDj = __SHFL(0xFFFFFFFF, LJIDj_val, j);
#endif

        PMEFloat xij = xi - xj_s;
        PMEFloat yij = yi - yj_s;
        PMEFloat zij = zi - zj_s;
        PMEFloat r2  = xij*xij + yij*yij + zij*zij;
        // ---- Standard computation path (also used for cache-write) ----
#ifdef PGM_SHORT_RANGE
        if (r2 < cSim.scfLocalCut2 && r2 > delta) {
#else
        if (r2 < cSim.cut2 && r2 > delta) {
#endif
#ifdef PGM_DEBUG_PAIR_COUNTS
          atomicAdd(&dPGMNeighPairCount, 1ULL);
#endif
#ifdef PGM_BFACTOR_READ16
          PMEFloat B1, B2;
          {
            int pair_idx = bfactor_base + neigh_iter * 32 + tgx;
            half2 hv = *reinterpret_cast<const half2*>(&cSim.pPGMBD_h2[pair_idx]);
            float2 fv = __half22float2(hv);
            B1 = -fv.x * (PMEFloat)(1.0/4096.0);
            B2 = -fv.y * (PMEFloat)(1.0/4096.0);
          }
#else
          PMEFloat rinv  = rsqrt(r2);
          PMEFloat r     = r2 * rinv;
          PMEFloat r2inv = rinv * rinv;

          // Ewald B-factors (PGM uses erf, not erfc)
#ifdef use_DPFP
          PMEFloat erf_ew = (PMEFloat)1.0 - erfc(cSim.ew_coeffSP * r);
#else
          PMEFloat erf_ew = (PMEFloat)1.0 - fasterfc(r);
#endif
          PMEFloat B0_ew = erf_ew * rinv;
          PMEFloat fact  = -cSim.negTwoEw_coeffRsqrtPI * PGM_EXP(-cSim.ew_coeff2 * r2);
          PMEFloat B1_ew = (B0_ew - fact) * r2inv;
          fact = (PMEFloat)2.0 * cSim.ew_coeff2 * fact;
          PMEFloat B2_ew = ((PMEFloat)3.0 * B1_ew - fact) * r2inv;

          // Gaussian B-factors (erf(beta*r)/r)
          PMEFloat betaij = rsqrt(rg_i + rg_j);
          PMEFloat y = betaij * r;
#ifdef use_DPFP
          PMEFloat erfc_g = erfc(y);
#else
          PMEFloat erfc_g = erfcf(y);
#endif
          PMEFloat B0_g = ((PMEFloat)1.0 - erfc_g) * rinv;
          PMEFloat fact_g = (PMEFloat)M_2_SQRTPI * betaij * PGM_EXP(-y * y);
          PMEFloat B1_g = (B0_g - fact_g) * r2inv;
          fact_g = (PMEFloat)2.0 * betaij * betaij * fact_g;
          PMEFloat B2_g = ((PMEFloat)3.0 * B1_g - fact_g) * r2inv;

#ifndef PGM_FIELD_ONLY
          // B3 extension for force
          fact = (PMEFloat)2.0 * cSim.ew_coeff2 * fact;
          PMEFloat B3_ew = ((PMEFloat)5.0 * B2_ew - fact) * r2inv;
          fact_g = (PMEFloat)2.0 * betaij * betaij * fact_g;
          PMEFloat B3_g = ((PMEFloat)5.0 * B2_g - fact_g) * r2inv;
#endif

          // Combined: direct-space PGM
          PMEFloat B1 = B1_ew - B1_g;
          PMEFloat B2 = B2_g - B2_ew;
#endif // PGM_BFACTOR_READ16
#ifndef PGM_FIELD_ONLY
          PMEFloat B0 = B0_ew - B0_g;  // for phi_000 (potential)
          PMEFloat B3 = B3_ew - B3_g;
#endif

#ifdef PGM_BFACTOR_WRITE
          // Store BD cache {-B1, -B2, dx, dy, dz} for SCF iterations
          {
            int pair_idx = bfactor_base + neigh_iter * 32 + tgx;
            cSim.pPGMBD_nB1[pair_idx] = -B1;
            cSim.pPGMBD_nB2[pair_idx] = -B2;
            cSim.pPGMBD_dx[pair_idx]  = xij;
            cSim.pPGMBD_dy[pair_idx]  = yij;
            cSim.pPGMBD_dz[pair_idx]  = zij;
          }
#endif
#ifdef PGM_BFACTOR_WRITE16
          {
            int pair_idx = bfactor_base + neigh_iter * 32 + tgx;
            half2 hv = __floats2half2_rn(-B1 * (PMEFloat)4096.0, -B2 * (PMEFloat)4096.0);
            *reinterpret_cast<half2*>(&cSim.pPGMBD_h2[pair_idx]) = hv;
          }
#endif

          // gradphi on i from j
          PMEFloat mu_dot_r = d_jx*xij + d_jy*yij + d_jz*zij;
          PMEFloat gp_ix = qj*B1*xij - B1*d_jx - B2*mu_dot_r*xij;
          PMEFloat gp_iy = qj*B1*yij - B1*d_jy - B2*mu_dot_r*yij;
          PMEFloat gp_iz = qj*B1*zij - B1*d_jz - B2*mu_dot_r*zij;
          fld_ix += gp_ix;
          fld_iy += gp_iy;
          fld_iz += gp_iz;

          // Reaction gradphi on j from i (stored in ej for shuffle routing)
          PMEFloat mu_dot_r_i = dip_ix*(-xij) + dip_iy*(-yij) + dip_iz*(-zij);
          ej_x = qi*B1*(-xij) - B1*dip_ix - B2*mu_dot_r_i*(-xij);
          ej_y = qi*B1*(-yij) - B1*dip_iy - B2*mu_dot_r_i*(-yij);
          ej_z = qi*B1*(-zij) - B1*dip_iz - B2*mu_dot_r_i*(-zij);

#ifndef PGM_FIELD_ONLY
          // phi_000 at i from j (for energy)
          phi_i += -B0 * qj - B1 * mu_dot_r;

          // phi_000 at j from i (for shuffle to j-owning thread)
          phi_000_ij = -B0 * qi - B1 * mu_dot_r_i;

          // Hessian of phi at i from j
          PMEFloat dx2 = xij * xij, dy2 = yij * yij, dz2 = zij * zij;
          PMEFloat delxd = B2 + dx2 * B3;
          PMEFloat delyd = B2 + dy2 * B3;
          PMEFloat delzd = B2 + dz2 * B3;

          PMEFloat phi_200 = (B1 + dx2*B2)*qj - (PMEFloat)2.0*B2*xij*d_jx - delxd*mu_dot_r;
          PMEFloat phi_020 = (B1 + dy2*B2)*qj - (PMEFloat)2.0*B2*yij*d_jy - delyd*mu_dot_r;
          PMEFloat phi_002 = (B1 + dz2*B2)*qj - (PMEFloat)2.0*B2*zij*d_jz - delzd*mu_dot_r;
          PMEFloat phi_110 = xij*yij*B2*qj - B2*(yij*d_jx + xij*d_jy) - xij*yij*B3*mu_dot_r;
          PMEFloat phi_101 = xij*zij*B2*qj - B2*(zij*d_jx + xij*d_jz) - xij*zij*B3*mu_dot_r;
          PMEFloat phi_011 = yij*zij*B2*qj - B2*(zij*d_jy + yij*d_jz) - yij*zij*B3*mu_dot_r;

          // Force on i from j: F = -(q*gradphi + mu_tot · Hessian)
          fx_ij = -(qi*gp_ix + dip_ix*phi_200 + dip_iy*phi_110 + dip_iz*phi_101);
          fy_ij = -(qi*gp_iy + dip_ix*phi_110 + dip_iy*phi_020 + dip_iz*phi_011);
          fz_ij = -(qi*gp_iz + dip_ix*phi_101 + dip_iy*phi_011 + dip_iz*phi_002);

          // VDW: standard 12-6 LJ (uses exclusion mask from NL tiles)
          // Pre-divide by coulomb_const (see home cell comment)
          if (!(exclusion & 0x1)) {
            unsigned int ljindex = LJIDi + shLJIDj;
            PMEFloat2 ljterm = cSim.pLJTerm[ljindex];
            PMEFloat e_vdw, df_vdw;
            PGMPairVDW(ljterm, r2inv, rinv, betaij, y, (PMEFloat)1.0 - erfc_g, e_vdw, df_vdw);
            df_vdw *= cSim.invCoulombConst;
            fx_ij += df_vdw * xij;
            fy_ij += df_vdw * yij;
            fz_ij += df_vdw * zij;
            TLevdw += e_vdw;
          }

          // Virial: -r_ij dot F_ij (before accumulation)
#ifdef PME_VIRIAL
          TLvir_11 -= xij * fx_ij;
          TLvir_22 -= yij * fy_ij;
          TLvir_33 -= zij * fz_ij;
#endif
          // Accumulate force on i
          frc_ix += fx_ij; frc_iy += fy_ij; frc_iz += fz_ij;
#endif
        }
#ifdef PGM_BFACTOR_WRITE16
        else {
          int pair_idx = bfactor_base + neigh_iter * 32 + tgx;
          cSim.pPGMBD_h2[pair_idx] = 0u;
        }
#endif
#ifdef PGM_BFACTOR_WRITE
        else {
          // Out-of-cutoff: store zeros so BD cache-read sees zero contribution
          int pair_idx = bfactor_base + neigh_iter * 32 + tgx;
          cSim.pPGMBD_nB1[pair_idx] = (PMEFloat)0.0;
          cSim.pPGMBD_nB2[pair_idx] = (PMEFloat)0.0;
          cSim.pPGMBD_dx[pair_idx]  = (PMEFloat)0.0;
          cSim.pPGMBD_dy[pair_idx]  = (PMEFloat)0.0;
          cSim.pPGMBD_dz[pair_idx]  = (PMEFloat)0.0;
        }
#endif
#endif // !PGM_BFACTOR_READ

#if defined(PGM_BFACTOR_WRITE) || defined(PGM_BFACTOR_READ) || defined(PGM_BFACTOR_WRITE16) || defined(PGM_BFACTOR_READ16)
        neigh_iter++;
#endif

        // Shuffle-route j-gradphi to the thread that owns j-atom (ALL threads participate)
        fld_jx += __SHFL(0xFFFFFFFF, ej_x, jrec);
        fld_jy += __SHFL(0xFFFFFFFF, ej_y, jrec);
        fld_jz += __SHFL(0xFFFFFFFF, ej_z, jrec);

#ifndef PGM_FIELD_ONLY
        // Shuffle reaction force to j: F_j = -F_i
        frc_jx -= __SHFL(0xFFFFFFFF, fx_ij, jrec);
        frc_jy -= __SHFL(0xFFFFFFFF, fy_ij, jrec);
        frc_jz -= __SHFL(0xFFFFFFFF, fz_ij, jrec);

        // Shuffle phi_000 to j
        phi_j += __SHFL(0xFFFFFFFF, phi_000_ij, jrec);
        exclusion >>= 1;
#endif

        j = sNext[j];
        jrec = sRecv[jrec];
      } while (j != tgx);

#ifndef PGM_FIELD_ONLY
      // Accumulate tile VDW energy
      evdw += llrint(TLevdw * ENERGYSCALE);
#ifdef PME_VIRIAL
#ifdef use_SPFP
      vir_11 += fast_llrintf(TLvir_11 * FORCESCALEF);
      vir_22 += fast_llrintf(TLvir_22 * FORCESCALEF);
      vir_33 += fast_llrintf(TLvir_33 * FORCESCALEF);
#else
      vir_11 += llrint(TLvir_11 * FORCESCALE);
      vir_22 += llrint(TLvir_22 * FORCESCALE);
      vir_33 += llrint(TLvir_33 * FORCESCALE);
#endif
#endif
#endif

      // Write i gradphi + force + phi_000
      if (iindex < psWarp->nlEntry.NL.ymax) {
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldXAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)fld_ix * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldYAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)fld_iy * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldZAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)fld_iz * FORCESCALE)));
#ifndef PGM_FIELD_ONLY
        atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)frc_ix * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)frc_iy * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)frc_iz * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pPGMPotentialAccumulator[iindex],
                  llitoulli(llrint((PMEDouble)phi_i * FORCESCALE)));
#endif
      }
      fld_ix = (PMEFloat)0.0;
      fld_iy = (PMEFloat)0.0;
      fld_iz = (PMEFloat)0.0;
#ifndef PGM_FIELD_ONLY
      frc_ix = (PMEFloat)0.0;
      frc_iy = (PMEFloat)0.0;
      frc_iz = (PMEFloat)0.0;
      phi_i = (PMEFloat)0.0;
#endif

      // Write j gradphi + force + phi_000
      if (tx + (int)tgx < (int)xatoms) {
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldXAccumulator[jindex],
                  llitoulli(llrint((PMEDouble)fld_jx * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldYAccumulator[jindex],
                  llitoulli(llrint((PMEDouble)fld_jy * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pPGMFieldZAccumulator[jindex],
                  llitoulli(llrint((PMEDouble)fld_jz * FORCESCALE)));
#ifndef PGM_FIELD_ONLY
        atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[jindex],
                  llitoulli(llrint((PMEDouble)frc_jx * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[jindex],
                  llitoulli(llrint((PMEDouble)frc_jy * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[jindex],
                  llitoulli(llrint((PMEDouble)frc_jz * FORCESCALE)));
        atomicAdd((unsigned long long int*)&cSim.pPGMPotentialAccumulator[jindex],
                  llitoulli(llrint((PMEDouble)phi_j * FORCESCALE)));
#endif
      }

#if defined(PGM_BFACTOR_WRITE) || defined(PGM_BFACTOR_READ) || defined(PGM_BFACTOR_WRITE16) || defined(PGM_BFACTOR_READ16)
      bfactor_base += PME_ATOMS_PER_WARP * 32;
#endif
      tx += GRID;
    }

    if (tgx == 0) {
      psWarp->pos += (gridDim.x * blockDim.x) / GRID;
    }
    __SYNCWARP(0xFFFFFFFF);
  }

#ifndef PGM_FIELD_ONLY
  // Warp-reduce and write VDW energy
  unsigned int rlev;
  for (rlev = 16; rlev > 0; rlev /= 2) {
    evdw += __SHFL(0xFFFFFFFF, evdw, tgx + rlev);
#ifdef PME_VIRIAL
    vir_11 += __SHFL(0xFFFFFFFF, vir_11, tgx + rlev);
    vir_22 += __SHFL(0xFFFFFFFF, vir_22, tgx + rlev);
    vir_33 += __SHFL(0xFFFFFFFF, vir_33, tgx + rlev);
#endif
  }
  if ((threadIdx.x & GRID_BITS_MASK) == 0) {
    atomicAdd(cSim.pEVDW, llitoulli(evdw));
#ifdef PME_VIRIAL
    atomicAdd(cSim.pVirial_11, llitoulli(vir_11));
    atomicAdd(cSim.pVirial_22, llitoulli(vir_22));
    atomicAdd(cSim.pVirial_33, llitoulli(vir_33));
#endif
  }
#endif
}

#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kPrecondCache_PGM.cu — Per-atom CSR short-range tensor cache for PCG preconditioner
//
// Phase 2 of the PCG implementation.  Replaces Phase 1's full-NL short-range field kernel
// with a compact per-atom CSR of pre-computed tensor elements.  The SpMV kernel only
// touches qualifying pairs (~18/atom at 4 Å cutoff), eliminating ~92% wasted NL traversal.
//
// Build phase (once per NL rebuild):
//   1. Count: iterate NL, atomicAdd per-atom counters for pairs within scfLocalCut²
//   2. Prefix sum: parallel scan to get CSR offsets
//   3. Fill: re-iterate NL, store {j, nB1, nB2, dx, dy, dz} at CSR positions
//
// SpMV (each inner CG iteration):
//   For each atom i, gather neighbor dipoles from CG vector, compute field.
//   Field = nB1*mu_j + nB2*(r·mu_j)*r  (same formula as BD cache-read path)
//---------------------------------------------------------------------------------------------

#include <cuda.h>
#include "gpu.h"
#include "ptxmacros.h"

#include "simulationConst.h"

CSIM_STO simulationConst cSim;

//---------------------------------------------------------------------------------------------
// SetkPrecondCache_PGMSim: copy simulation constants to device constant memory
//---------------------------------------------------------------------------------------------
void SetkPrecondCache_PGMSim(gpuContext gpu)
{
  cudaError_t status;
  status = cudaMemcpyToSymbol(cSim, &gpu->sim, sizeof(simulationConst));
  RTERROR(status, "SetkPrecondCache_PGMSim: cudaMemcpyToSymbol failed");
}

//---------------------------------------------------------------------------------------------
// kPGMShortRangeCSRBuild_kernel — Unified count/fill kernel for short-range pair CSR
//
// Template-style kernel (instantiated via #define) that traverses the NL and processes
// pairs within scfLocalCut².  Simplified from kNLCPNE_PGM.h: no field accumulation,
// no forces, no exclusions, no BD cache.
//
// MODE = 0: COUNT — atomicAdd to per-atom pair counters
// MODE = 1: FILL  — compute {nB1, nB2, dx, dy, dz} and store in CSR
//---------------------------------------------------------------------------------------------

// Count kernel: walks NL, counts qualifying pairs per atom
//
// Only threads 0..APW-1 in each warp have unique i-atoms.  Threads APW..31
// share the same iindex values, so they must NOT participate in counting.
// For neighbor tiles, all 32 j-atom slots are visited (not just APW).
template<int APW, bool ORTHO>
__global__ void
__launch_bounds__(128, 1)
kPGMShortRangeCount_kernel()
{
  __shared__ unsigned int sNLEntries;
  if (threadIdx.x == 0) {
    sNLEntries = *(cSim.pNLEntries);
  }
  __syncthreads();
  unsigned int nEntries = sNLEntries;

  const PMEFloat delta = (PMEFloat)1.0e-5;
  const int tgx = threadIdx.x & 31;
  unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  bool i_active = (tgx < APW);  // only first APW lanes own unique i-atoms

  for (unsigned int entry_idx = warp_id; entry_idx < nEntries;
       entry_idx += (gridDim.x * blockDim.x) / 32) {

    // Load NL entry: threads 0-3 each load one field into a scalar register,
    // then shuffle from the correct source lane.  (The old code stored into a
    // local NLEntry union and shuffled all fields from lane 0, but only lane 0
    // had array[0] valid — array[1..3] were uninitialized garbage in lane 0's
    // registers, causing the neighbor-tile crash.)
    unsigned int entry_val = 0;
    if (tgx < 4) entry_val = cSim.pNLEntry[entry_idx].array[tgx];
    __syncwarp(0xFFFFFFFF);

    unsigned int ypos     = __shfl_sync(0xFFFFFFFF, entry_val, 0);
    unsigned int ymax_raw = __shfl_sync(0xFFFFFFFF, entry_val, 1);
    unsigned int xatoms   = __shfl_sync(0xFFFFFFFF, entry_val, 2);
    unsigned int offset   = __shfl_sync(0xFFFFFFFF, entry_val, 3);
    bool bHome = ymax_raw & 0x1;
    unsigned int ymax = ymax_raw >> NLENTRY_YMAX_SHIFT;

    // Load i atom data (all threads load, but only tgx < APW participate)
    unsigned int iindex = ypos + (tgx & (APW - 1));
    PMEFloat xi = (PMEFloat)0.0, yi = (PMEFloat)0.0, zi = (PMEFloat)0.0;
    if (iindex < ymax) {
      PMEFloat2 xy = cSim.pAtomXYSP[iindex];
      xi = xy.x; yi = xy.y;
      zi = cSim.pAtomZSP[iindex];
      if (!ORTHO) {
        PMEFloat tx = cSim.ucellf[0][0]*xi + cSim.ucellf[0][1]*yi + cSim.ucellf[0][2]*zi;
        PMEFloat ty = cSim.ucellf[1][1]*yi + cSim.ucellf[1][2]*zi;
        PMEFloat tz = cSim.ucellf[2][2]*zi;
        xi = tx; yi = ty; zi = tz;
      }
    }

    // ---- Home cell pairs ----
    if (bHome) {
      offset += APW;  // skip exclusion mask
      // Uniform loop: ALL threads iterate the same range so __shfl_sync
      // is never divergent.  Only upper-triangle pairs (tgx < j) count.
      for (int j = 0; j < APW; j++) {
        PMEFloat xj = __shfl_sync(0xFFFFFFFF, xi, j);
        PMEFloat yj = __shfl_sync(0xFFFFFFFF, yi, j);
        PMEFloat zj = __shfl_sync(0xFFFFFFFF, zi, j);
        PMEFloat xij = xi - xj;
        PMEFloat yij = yi - yj;
        PMEFloat zij = zi - zj;
        PMEFloat r2 = xij*xij + yij*yij + zij*zij;

        unsigned int jindex = ypos + j;
        if ((int)tgx < j && r2 < cSim.scfLocalCut2 && r2 > delta &&
            i_active && iindex < ymax && jindex < ymax) {
          atomicAdd(&cSim.pPGMShortPairCount[iindex], 1);
          atomicAdd(&cSim.pPGMShortPairCount[jindex], 1);
        }
      }
    }

    // ---- Neighbor tile pairs ----
    int tx = 0;
    while (tx < (int)xatoms) {
      // All 32 threads load j-atom IDs (unconditional for valid slots,
      // matches reference kernel pattern)
      unsigned int jid_raw = cSim.pNLAtomList[offset + tgx];
      offset += GRID;    // atom IDs
      offset += APW;     // exclusion mask

      unsigned int jindex = jid_raw >> NLATOM_CELL_SHIFT;
      int cell = jid_raw & NLATOM_CELL_TYPE_MASK;

      PMEFloat xj, yj, zj;
      if (tx + (int)tgx < (int)xatoms) {
        PMEFloat2 xy = cSim.pAtomXYSP[jindex];
        xj = xy.x; yj = xy.y;
        zj = cSim.pAtomZSP[jindex];
      } else {
        xj = (PMEFloat)-10000.0 * (tgx + 1);
        yj = (PMEFloat)-10000.0 * (tgx + 1);
        zj = (PMEFloat)-10000.0 * (tgx + 1);
      }

      // Cell offset and non-ortho transform applied to ALL threads
      xj += cSim.cellOffset[cell][0];
      yj += cSim.cellOffset[cell][1];
      zj += cSim.cellOffset[cell][2];
      if (!ORTHO) {
        PMEFloat txj = cSim.ucellf[0][0]*xj + cSim.ucellf[0][1]*yj + cSim.ucellf[0][2]*zj;
        PMEFloat tyj = cSim.ucellf[1][1]*yj + cSim.ucellf[1][2]*zj;
        PMEFloat tzj = cSim.ucellf[2][2]*zj;
        xj = txj; yj = tyj; zj = tzj;
      }

      // Uniform loop over all 32 j-atom slots
      for (int j_lane = 0; j_lane < 32; j_lane++) {
        PMEFloat xj_s = __shfl_sync(0xFFFFFFFF, xj, j_lane);
        PMEFloat yj_s = __shfl_sync(0xFFFFFFFF, yj, j_lane);
        PMEFloat zj_s = __shfl_sync(0xFFFFFFFF, zj, j_lane);
        unsigned int jidx = __shfl_sync(0xFFFFFFFF, jindex, j_lane);

        PMEFloat xij = xi - xj_s;
        PMEFloat yij = yi - yj_s;
        PMEFloat zij = zi - zj_s;
        PMEFloat r2 = xij*xij + yij*yij + zij*zij;

        bool j_valid = (tx + j_lane) < (int)xatoms;
        if (r2 < cSim.scfLocalCut2 && r2 > delta &&
            i_active && iindex < ymax && j_valid) {
          atomicAdd(&cSim.pPGMShortPairCount[iindex], 1);
          atomicAdd(&cSim.pPGMShortPairCount[jidx], 1);
        }
      }

      tx += GRID;
    }
  }
}


// Fill kernel: walks NL, computes and stores tensor data for qualifying pairs
//
// Same structure as count kernel: only threads 0..APW-1 have unique i-atoms.
// Home cell uses upper-triangle iteration; neighbor tiles visit all 32 j slots.
template<int APW, bool ORTHO>
__global__ void
__launch_bounds__(128, 1)
kPGMShortRangeFill_kernel()
{
  __shared__ unsigned int sNLEntries;
  if (threadIdx.x == 0) {
    sNLEntries = *(cSim.pNLEntries);
  }
  __syncthreads();
  unsigned int nEntries = sNLEntries;

  const PMEFloat delta = (PMEFloat)1.0e-5;
  const int tgx = threadIdx.x & 31;
  unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  bool i_active = (tgx < APW);

  for (unsigned int entry_idx = warp_id; entry_idx < nEntries;
       entry_idx += (gridDim.x * blockDim.x) / 32) {

    // Load NL entry (same fix as count kernel — shuffle from correct lane)
    unsigned int entry_val = 0;
    if (tgx < 4) entry_val = cSim.pNLEntry[entry_idx].array[tgx];
    __syncwarp(0xFFFFFFFF);

    unsigned int ypos     = __shfl_sync(0xFFFFFFFF, entry_val, 0);
    unsigned int ymax_raw = __shfl_sync(0xFFFFFFFF, entry_val, 1);
    unsigned int xatoms   = __shfl_sync(0xFFFFFFFF, entry_val, 2);
    unsigned int offset   = __shfl_sync(0xFFFFFFFF, entry_val, 3);
    bool bHome = ymax_raw & 0x1;
    unsigned int ymax = ymax_raw >> NLENTRY_YMAX_SHIFT;

    // Load i atom data
    unsigned int iindex = ypos + (tgx & (APW - 1));
    PMEFloat xi = (PMEFloat)0.0, yi = (PMEFloat)0.0, zi = (PMEFloat)0.0;
    PMEFloat rg_i = (PMEFloat)1.0;
    if (iindex < ymax) {
      PMEFloat2 xy = cSim.pAtomXYSP[iindex];
      xi = xy.x; yi = xy.y;
      zi = cSim.pAtomZSP[iindex];
      rg_i = (PMEFloat)cSim.pPGMRGauss[iindex];
      if (!ORTHO) {
        PMEFloat tx = cSim.ucellf[0][0]*xi + cSim.ucellf[0][1]*yi + cSim.ucellf[0][2]*zi;
        PMEFloat ty = cSim.ucellf[1][1]*yi + cSim.ucellf[1][2]*zi;
        PMEFloat tz = cSim.ucellf[2][2]*zi;
        xi = tx; yi = ty; zi = tz;
      }
    }

    // ---- Home cell pairs ----
    if (bHome) {
      offset += APW;  // skip exclusion mask
      // Uniform loop: ALL threads iterate same range for non-divergent shuffles
      for (int j = 0; j < APW; j++) {
        PMEFloat xj = __shfl_sync(0xFFFFFFFF, xi, j);
        PMEFloat yj = __shfl_sync(0xFFFFFFFF, yi, j);
        PMEFloat zj = __shfl_sync(0xFFFFFFFF, zi, j);
        PMEFloat rg_j = __shfl_sync(0xFFFFFFFF, rg_i, j);
        PMEFloat xij = xi - xj;
        PMEFloat yij = yi - yj;
        PMEFloat zij = zi - zj;
        PMEFloat r2 = xij*xij + yij*yij + zij*zij;

        unsigned int jindex = ypos + j;
        if ((int)tgx < j && r2 < cSim.scfLocalCut2 && r2 > delta &&
            i_active && iindex < ymax && jindex < ymax) {
          PMEFloat rinv  = rsqrt(r2);
          PMEFloat r     = r2 * rinv;
          PMEFloat r2inv = rinv * rinv;

#ifdef use_DPFP
          PMEFloat erf_ew = (PMEFloat)1.0 - erfc(cSim.ew_coeffSP * r);
#else
          PMEFloat erf_ew = (PMEFloat)1.0 - erfcf(cSim.ew_coeffSP * r);
#endif
          PMEFloat B0_ew = erf_ew * rinv;
          PMEFloat fact  = -cSim.negTwoEw_coeffRsqrtPI * exp(-cSim.ew_coeff2 * r2);
          PMEFloat B1_ew = (B0_ew - fact) * r2inv;
          fact = (PMEFloat)2.0 * cSim.ew_coeff2 * fact;
          PMEFloat B2_ew = ((PMEFloat)3.0 * B1_ew - fact) * r2inv;

          PMEFloat betaij = rsqrt(rg_i + rg_j);
          PMEFloat y = betaij * r;
#ifdef use_DPFP
          PMEFloat erfc_g = erfc(y);
#else
          PMEFloat erfc_g = erfcf(y);
#endif
          PMEFloat B0_g = ((PMEFloat)1.0 - erfc_g) * rinv;
          PMEFloat fact_g = (PMEFloat)M_2_SQRTPI * betaij * exp(-y * y);
          PMEFloat B1_g = (B0_g - fact_g) * r2inv;
          fact_g = (PMEFloat)2.0 * betaij * betaij * fact_g;
          PMEFloat B2_g = ((PMEFloat)3.0 * B1_g - fact_g) * r2inv;

          PMEFloat nB1 = -(B1_ew - B1_g);
          PMEFloat nB2 = -(B2_g - B2_ew);

          // Store pair for atom i
          int pos_i = atomicAdd(&cSim.pPGMShortPairCount[iindex], 1);
          cSim.pPGMShortPairAtom[pos_i] = jindex;
          cSim.pPGMShortPairNB1[pos_i]  = nB1;
          cSim.pPGMShortPairNB2[pos_i]  = nB2;
          cSim.pPGMShortPairDX[pos_i]   = xij;
          cSim.pPGMShortPairDY[pos_i]   = yij;
          cSim.pPGMShortPairDZ[pos_i]   = zij;

          // Store reverse pair for atom j
          int pos_j = atomicAdd(&cSim.pPGMShortPairCount[jindex], 1);
          cSim.pPGMShortPairAtom[pos_j] = iindex;
          cSim.pPGMShortPairNB1[pos_j]  = nB1;
          cSim.pPGMShortPairNB2[pos_j]  = nB2;
          cSim.pPGMShortPairDX[pos_j]   = -xij;
          cSim.pPGMShortPairDY[pos_j]   = -yij;
          cSim.pPGMShortPairDZ[pos_j]   = -zij;
        }
      }
    }

    // ---- Neighbor tile pairs ----
    int tx = 0;
    while (tx < (int)xatoms) {
      // All 32 threads load j-atom IDs unconditionally
      unsigned int jid_raw = cSim.pNLAtomList[offset + tgx];
      offset += GRID;    // atom IDs
      offset += APW;     // exclusion mask

      unsigned int jindex = jid_raw >> NLATOM_CELL_SHIFT;
      int cell = jid_raw & NLATOM_CELL_TYPE_MASK;

      PMEFloat xj, yj, zj;
      PMEFloat rg_j_val = (PMEFloat)1.0;
      if (tx + (int)tgx < (int)xatoms) {
        PMEFloat2 xy = cSim.pAtomXYSP[jindex];
        xj = xy.x; yj = xy.y;
        zj = cSim.pAtomZSP[jindex];
        rg_j_val = (PMEFloat)cSim.pPGMRGauss[jindex];
      } else {
        xj = (PMEFloat)-10000.0 * (tgx + 1);
        yj = (PMEFloat)-10000.0 * (tgx + 1);
        zj = (PMEFloat)-10000.0 * (tgx + 1);
      }

      // Cell offset and non-ortho transform applied to ALL threads
      xj += cSim.cellOffset[cell][0];
      yj += cSim.cellOffset[cell][1];
      zj += cSim.cellOffset[cell][2];
      if (!ORTHO) {
        PMEFloat txj = cSim.ucellf[0][0]*xj + cSim.ucellf[0][1]*yj + cSim.ucellf[0][2]*zj;
        PMEFloat tyj = cSim.ucellf[1][1]*yj + cSim.ucellf[1][2]*zj;
        PMEFloat tzj = cSim.ucellf[2][2]*zj;
        xj = txj; yj = tyj; zj = tzj;
      }

      // Iterate over all 32 j-atom slots
      for (int j_lane = 0; j_lane < 32; j_lane++) {
        PMEFloat xj_s  = __shfl_sync(0xFFFFFFFF, xj, j_lane);
        PMEFloat yj_s  = __shfl_sync(0xFFFFFFFF, yj, j_lane);
        PMEFloat zj_s  = __shfl_sync(0xFFFFFFFF, zj, j_lane);
        PMEFloat rg_j  = __shfl_sync(0xFFFFFFFF, rg_j_val, j_lane);
        unsigned int jidx = __shfl_sync(0xFFFFFFFF, jindex, j_lane);

        PMEFloat xij = xi - xj_s;
        PMEFloat yij = yi - yj_s;
        PMEFloat zij = zi - zj_s;
        PMEFloat r2 = xij*xij + yij*yij + zij*zij;

        bool j_valid = (tx + j_lane) < (int)xatoms;
        if (r2 < cSim.scfLocalCut2 && r2 > delta &&
            i_active && iindex < ymax && j_valid) {
          PMEFloat rinv  = rsqrt(r2);
          PMEFloat r     = r2 * rinv;
          PMEFloat r2inv = rinv * rinv;

#ifdef use_DPFP
          PMEFloat erf_ew = (PMEFloat)1.0 - erfc(cSim.ew_coeffSP * r);
#else
          PMEFloat erf_ew = (PMEFloat)1.0 - erfcf(cSim.ew_coeffSP * r);
#endif
          PMEFloat B0_ew = erf_ew * rinv;
          PMEFloat fact  = -cSim.negTwoEw_coeffRsqrtPI * exp(-cSim.ew_coeff2 * r2);
          PMEFloat B1_ew = (B0_ew - fact) * r2inv;
          fact = (PMEFloat)2.0 * cSim.ew_coeff2 * fact;
          PMEFloat B2_ew = ((PMEFloat)3.0 * B1_ew - fact) * r2inv;

          PMEFloat betaij = rsqrt(rg_i + rg_j);
          PMEFloat y_g = betaij * r;
#ifdef use_DPFP
          PMEFloat erfc_g = erfc(y_g);
#else
          PMEFloat erfc_g = erfcf(y_g);
#endif
          PMEFloat B0_g = ((PMEFloat)1.0 - erfc_g) * rinv;
          PMEFloat fact_g = (PMEFloat)M_2_SQRTPI * betaij * exp(-y_g * y_g);
          PMEFloat B1_g = (B0_g - fact_g) * r2inv;
          fact_g = (PMEFloat)2.0 * betaij * betaij * fact_g;
          PMEFloat B2_g = ((PMEFloat)3.0 * B1_g - fact_g) * r2inv;

          PMEFloat nB1 = -(B1_ew - B1_g);
          PMEFloat nB2 = -(B2_g - B2_ew);

          // Store pair for atom i
          int pos_i = atomicAdd(&cSim.pPGMShortPairCount[iindex], 1);
          cSim.pPGMShortPairAtom[pos_i] = jidx;
          cSim.pPGMShortPairNB1[pos_i]  = nB1;
          cSim.pPGMShortPairNB2[pos_i]  = nB2;
          cSim.pPGMShortPairDX[pos_i]   = xij;
          cSim.pPGMShortPairDY[pos_i]   = yij;
          cSim.pPGMShortPairDZ[pos_i]   = zij;

          // Store reverse pair for atom j
          int pos_j = atomicAdd(&cSim.pPGMShortPairCount[jidx], 1);
          cSim.pPGMShortPairAtom[pos_j] = iindex;
          cSim.pPGMShortPairNB1[pos_j]  = nB1;
          cSim.pPGMShortPairNB2[pos_j]  = nB2;
          cSim.pPGMShortPairDX[pos_j]   = -xij;
          cSim.pPGMShortPairDY[pos_j]   = -yij;
          cSim.pPGMShortPairDZ[pos_j]   = -zij;
        }
      }

      tx += GRID;
    }
  }
}


//---------------------------------------------------------------------------------------------
// kPGMShortRangePrefixSum_kernel — Parallel prefix sum on per-atom pair counts
//
// Converts pPGMShortPairCount[0..atoms-1] → pPGMShortPairOffset[0..atoms]
// Uses same Hillis-Steele pattern as kPGMPrefixSumParallel in kNLRemap_PGM.cu.
// Handles up to 256 * chunk atoms.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 1)
kPGMShortRangePrefixSum_kernel()
{
  __shared__ int partials[256];
  unsigned int atoms = cSim.atoms;
  if (atoms == 0) return;

  unsigned int chunk = (atoms + 255) / 256;
  unsigned int myStart = threadIdx.x * chunk;
  unsigned int myEnd = myStart + chunk;
  if (myEnd > atoms) myEnd = atoms;

  // Phase 1: thread-local sum
  int localSum = 0;
  for (unsigned int i = myStart; i < myEnd; i++) {
    localSum += cSim.pPGMShortPairCount[i];
  }
  partials[threadIdx.x] = localSum;
  __syncthreads();

  // Phase 2: inclusive prefix scan of 256 partial sums (Hillis-Steele)
  for (int stride = 1; stride < 256; stride *= 2) {
    int val = (threadIdx.x >= (unsigned int)stride) ? partials[threadIdx.x - stride] : 0;
    __syncthreads();
    partials[threadIdx.x] += val;
    __syncthreads();
  }

  // Phase 3: convert counts to exclusive prefix sums → offsets
  int prefix = (threadIdx.x > 0) ? partials[threadIdx.x - 1] : 0;

  int running = prefix;
  for (unsigned int i = myStart; i < myEnd; i++) {
    int count = cSim.pPGMShortPairCount[i];
    cSim.pPGMShortPairOffset[i] = running;
    running += count;
  }

  // Last thread writes totalPairs
  if (threadIdx.x == 255) {
    cSim.pPGMShortPairOffset[atoms] = partials[255];
  }
}


//---------------------------------------------------------------------------------------------
// kPGMShortRangeCopyOffsets_kernel — Copy CSR offsets to fill pointers (count array)
//
// After prefix sum, pPGMShortPairOffset holds final CSR offsets.
// Copy offset[i] to pPGMShortPairCount[i] so fill kernel can use atomicAdd
// from the correct starting position.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMShortRangeCopyOffsets_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  while (pos < cSim.atoms) {
    cSim.pPGMShortPairCount[pos] = cSim.pPGMShortPairOffset[pos];
    pos += blockDim.x * gridDim.x;
  }
}


//---------------------------------------------------------------------------------------------
// kPGMShortRangeSpMV_kernel — Per-atom CSR SpMV for preconditioner field
//
// For each atom i, gathers neighbor dipoles from the inner CG vector,
// computes field via E = nB1*mu_j + nB2*(r·mu_j)*r, and stores result
// directly to fixed-point field accumulators.
//
// src: inner CG vector (pPGMPCGpu), layout [3*pos + d] (AoS)
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMShortRangeSpMV_kernel(const double* __restrict__ src)
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < cSim.atoms) {
    int start = cSim.pPGMShortPairOffset[pos];
    int end   = cSim.pPGMShortPairOffset[pos + 1];

    double field_x = 0.0;
    double field_y = 0.0;
    double field_z = 0.0;

    for (int k = start; k < end; k++) {
      int j = cSim.pPGMShortPairAtom[k];
      double nB1 = (double)cSim.pPGMShortPairNB1[k];
      double nB2 = (double)cSim.pPGMShortPairNB2[k];
      double dx  = (double)cSim.pPGMShortPairDX[k];
      double dy  = (double)cSim.pPGMShortPairDY[k];
      double dz  = (double)cSim.pPGMShortPairDZ[k];

      double mu_jx = src[3*j + 0];
      double mu_jy = src[3*j + 1];
      double mu_jz = src[3*j + 2];

      double dot_rj = dx * mu_jx + dy * mu_jy + dz * mu_jz;
      double bd = nB2 * dot_rj;

      field_x += nB1 * mu_jx + bd * dx;
      field_y += nB1 * mu_jy + bd * dy;
      field_z += nB1 * mu_jz + bd * dz;
    }

    // Direct store to fixed-point accumulators (no atomic needed — 1 thread per atom)
    cSim.pPGMFieldXAccumulator[pos] = llrint(field_x * FORCESCALE);
    cSim.pPGMFieldYAccumulator[pos] = llrint(field_y * FORCESCALE);
    cSim.pPGMFieldZAccumulator[pos] = llrint(field_z * FORCESCALE);
    cSim.pPGMPotentialAccumulator[pos] = (PMEAccumulator)0;

    pos += blockDim.x * gridDim.x;
  }

  // Zero padding for accumulator alignment
  pos = blockIdx.x * blockDim.x + threadIdx.x + cSim.atoms;
  while (pos < cSim.stride) {
    cSim.pPGMFieldXAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMFieldYAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMFieldZAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMPotentialAccumulator[pos] = (PMEAccumulator)0;
    pos += blockDim.x * gridDim.x;
  }
}



//---------------------------------------------------------------------------------------------
// kPGMShortRangeSpMVWarp_kernel — LPA lanes cooperate on one atom's CSR row (coalesced
// CSR reads, LPA independent gathers in flight per atom, shuffle reduction).  Replaces the
// one-thread-per-atom kernel above which was latency bound (~100 us at 98k atoms).
//---------------------------------------------------------------------------------------------
template<int LPA>
__global__ void
__launch_bounds__(128, 8)
kPGMShortRangeSpMVWarp_kernel(const double* __restrict__ src)
{
  const int sub  = threadIdx.x & (LPA - 1);
  unsigned int atom = (blockIdx.x * blockDim.x + threadIdx.x) / LPA;
  const unsigned int astride = (gridDim.x * blockDim.x) / LPA;
  const int*    __restrict__ pAtom = cSim.pPGMShortPairAtom;
  const PMEFloat* __restrict__ pB1 = cSim.pPGMShortPairNB1;
  const PMEFloat* __restrict__ pB2 = cSim.pPGMShortPairNB2;
  const PMEFloat* __restrict__ pDX = cSim.pPGMShortPairDX;
  const PMEFloat* __restrict__ pDY = cSim.pPGMShortPairDY;
  const PMEFloat* __restrict__ pDZ = cSim.pPGMShortPairDZ;
  while (atom < cSim.stride) {
    double fx = 0.0, fy = 0.0, fz = 0.0;
    if (atom < cSim.atoms) {
      const int start = cSim.pPGMShortPairOffset[atom];
      const int end   = cSim.pPGMShortPairOffset[atom + 1];
      for (int k = start + sub; k < end; k += LPA) {
        const int j = pAtom[k];
        const double nB1 = (double)pB1[k];
        const double nB2 = (double)pB2[k];
        const double dx  = (double)pDX[k];
        const double dy  = (double)pDY[k];
        const double dz  = (double)pDZ[k];
        const double mx = src[3*j + 0];
        const double my = src[3*j + 1];
        const double mz = src[3*j + 2];
        const double bd = nB2 * (dx*mx + dy*my + dz*mz);
        fx += nB1 * mx + bd * dx;
        fy += nB1 * my + bd * dy;
        fz += nB1 * mz + bd * dz;
      }
    }
#pragma unroll
    for (int off = LPA / 2; off > 0; off >>= 1) {
      fx += __shfl_xor_sync(0xffffffff, fx, off);
      fy += __shfl_xor_sync(0xffffffff, fy, off);
      fz += __shfl_xor_sync(0xffffffff, fz, off);
    }
    if (sub == 0) {
      cSim.pPGMFieldXAccumulator[atom] = llrint(fx * FORCESCALE);
      cSim.pPGMFieldYAccumulator[atom] = llrint(fy * FORCESCALE);
      cSim.pPGMFieldZAccumulator[atom] = llrint(fz * FORCESCALE);
      cSim.pPGMPotentialAccumulator[atom] = (PMEAccumulator)0;
    }
    atom += astride;
  }
}

//---------------------------------------------------------------------------------------------
// C wrapper functions
//---------------------------------------------------------------------------------------------

extern "C" void kPGMBuildShortRangeCSR(gpuContext gpu)
{
  int nbBlocks = gpu->PMENonbondBlocks;
  int apw = gpu->sim.NLAtomsPerWarp;
  int blocks_atom = (gpu->sim.atoms + 255) / 256;

  // Step 1: Zero per-atom count array and offset array
  cudaMemset(gpu->sim.pPGMShortPairCount, 0, gpu->sim.stride * sizeof(int));
  cudaMemset(gpu->sim.pPGMShortPairOffset, 0, (gpu->sim.atoms + 1) * sizeof(int));

  // Step 2: Count pass — count qualifying pairs per atom
  if (apw == 32) {
    if (gpu->sim.is_orthog)
      kPGMShortRangeCount_kernel<32, true><<<nbBlocks, 128>>>();
    else
      kPGMShortRangeCount_kernel<32, false><<<nbBlocks, 128>>>();
  } else if (apw == 16) {
    if (gpu->sim.is_orthog)
      kPGMShortRangeCount_kernel<16, true><<<nbBlocks, 128>>>();
    else
      kPGMShortRangeCount_kernel<16, false><<<nbBlocks, 128>>>();
  } else {
    if (gpu->sim.is_orthog)
      kPGMShortRangeCount_kernel<8, true><<<nbBlocks, 128>>>();
    else
      kPGMShortRangeCount_kernel<8, false><<<nbBlocks, 128>>>();
  }
  LAUNCHERROR("kPGMShortRangeCount");

  // Step 3: Prefix sum → CSR offsets
  kPGMShortRangePrefixSum_kernel<<<1, 256>>>();
  LAUNCHERROR("kPGMShortRangePrefixSum");

  // Step 4: Read total pairs from device
  int totalPairs;
  cudaMemcpy(&totalPairs, &gpu->sim.pPGMShortPairOffset[gpu->sim.atoms],
             sizeof(int), cudaMemcpyDeviceToHost);

  gpu->sim.pgmShortPairTotalPairs = totalPairs;

  // Step 5: Allocate if needed (grow by 1.5×)
  if (totalPairs > gpu->pgmShortPairMaxPairs) {
    int allocSize = totalPairs * 3 / 2;
    delete gpu->pbPGMShortPairAtom;
    delete gpu->pbPGMShortPairCache;
    gpu->pbPGMShortPairAtom  = new GpuBuffer<int>(allocSize);
    gpu->pbPGMShortPairCache = new GpuBuffer<PMEFloat>(5 * allocSize);
    gpu->sim.pPGMShortPairAtom = gpu->pbPGMShortPairAtom->_pDevData;
    gpu->sim.pPGMShortPairNB1  = gpu->pbPGMShortPairCache->_pDevData + 0 * allocSize;
    gpu->sim.pPGMShortPairNB2  = gpu->pbPGMShortPairCache->_pDevData + 1 * allocSize;
    gpu->sim.pPGMShortPairDX   = gpu->pbPGMShortPairCache->_pDevData + 2 * allocSize;
    gpu->sim.pPGMShortPairDY   = gpu->pbPGMShortPairCache->_pDevData + 3 * allocSize;
    gpu->sim.pPGMShortPairDZ   = gpu->pbPGMShortPairCache->_pDevData + 4 * allocSize;
    gpu->pgmShortPairMaxPairs = allocSize;
  }

  // Step 6: Copy offsets to count array (fill pointers start at offsets)
  kPGMShortRangeCopyOffsets_kernel<<<blocks_atom, 256>>>();
  LAUNCHERROR("kPGMShortRangeCopyOffsets");

  // Step 7: Update cSim before fill kernel
  SetkPrecondCache_PGMSim(gpu);

  // Step 8: Fill pass — compute and store tensor data
  if (apw == 32) {
    if (gpu->sim.is_orthog)
      kPGMShortRangeFill_kernel<32, true><<<nbBlocks, 128>>>();
    else
      kPGMShortRangeFill_kernel<32, false><<<nbBlocks, 128>>>();
  } else if (apw == 16) {
    if (gpu->sim.is_orthog)
      kPGMShortRangeFill_kernel<16, true><<<nbBlocks, 128>>>();
    else
      kPGMShortRangeFill_kernel<16, false><<<nbBlocks, 128>>>();
  } else {
    if (gpu->sim.is_orthog)
      kPGMShortRangeFill_kernel<8, true><<<nbBlocks, 128>>>();
    else
      kPGMShortRangeFill_kernel<8, false><<<nbBlocks, 128>>>();
  }
  LAUNCHERROR("kPGMShortRangeFill");
}

extern "C" void kPGMShortRangeSpMV(gpuContext gpu, double* src_dev)
{
  int blocks __attribute__((unused)) = (gpu->sim.stride + 127) / 128;
  cudaStream_t s = gpu->pgmStream;
  // NOTE: cSim is already current from NL rebuild (SetkPrecondCache_PGMSim called
  // in kPGMBuildShortRangeCSR).  Removed per-call cudaMemcpyToSymbol that was
  // causing unnecessary GPU sync every inner CG iteration.
#ifndef PGM_SPMV_WARP
  kPGMShortRangeSpMV_kernel<<<blocks, 128, 0, s>>>(src_dev);
  LAUNCHERROR("kPGMShortRangeSpMV");
#else
  {
    const int LPA = 8;
    int wblocks = (gpu->sim.stride * LPA + 127) / 128;
    kPGMShortRangeSpMVWarp_kernel<LPA><<<wblocks, 128, 0, s>>>(src_dev);
    LAUNCHERROR("kPGMShortRangeSpMVWarp");
  }
#endif
}

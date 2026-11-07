#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kSCF_PGM.cu - GPU-resident CG solver helper kernels for PGM induced dipoles
//
// All CG algebra (init, axpy, dot products, convergence) runs on GPU.
// The only CPU-GPU transfer per iteration is ONE convergence scalar.
//
// CG vectors are interleaved xyz: [x0,y0,z0, x1,y1,z1, ...], size 3*atoms.
//---------------------------------------------------------------------------------------------

#include <cuda.h>
#include "gpu.h"
#include "ptxmacros.h"

#include "simulationConst.h"
CSIM_STO simulationConst cSim;

void SetkSCF_PGMSim(gpuContext gpu)
{
  cudaError_t status;
  status = cudaMemcpyToSymbol(cSim, &gpu->sim, sizeof(simulationConst));
  RTERROR(status, "SetkSCF_PGMSim: cudaMemcpyToSymbol failed");
}

//---------------------------------------------------------------------------------------------
// kPGMInitSCF_kernel: Compute b = -alpha * E_perm from field accumulators
//
// Also computes partial sums for bnorm = mean(|b|) stored in partials buffer.
// Field accumulators contain the permanent field in fixed-point format.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMInitSCF_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_bsum = 0.0;
  const double fscale_inv = 1.0 / (double)(1LL << 40);

  while (pos < cSim.atoms) {
    double alpha_i = cSim.pPGMPolarizability[pos];

    double ex = (double)cSim.pPGMFieldXAccumulator[pos] * fscale_inv;
    double ey = (double)cSim.pPGMFieldYAccumulator[pos] * fscale_inv;
    double ez = (double)cSim.pPGMFieldZAccumulator[pos] * fscale_inv;

    double bx = -alpha_i * ex;
    double by = -alpha_i * ey;
    double bz = -alpha_i * ez;

    cSim.pPGMCGb[3*pos + 0] = bx;
    cSim.pPGMCGb[3*pos + 1] = by;
    cSim.pPGMCGb[3*pos + 2] = bz;

    local_bsum += fabs(bx) + fabs(by) + fabs(bz);

    pos += blockDim.x * gridDim.x;
  }

  // Warp reduction for bnorm partial sum
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_bsum += __shfl_down_sync(0xffffffff, local_bsum, offset);
  }

  // Block reduction via shared memory
  __shared__ double sBsum[4];  // 128 threads / 32 = 4 warps
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;

  if (lane == 0) {
    sBsum[warpId] = local_bsum;
  }
  __syncthreads();

  if (threadIdx.x < 4) {
    local_bsum = sBsum[threadIdx.x];
  } else {
    local_bsum = 0.0;
  }

  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      local_bsum += __shfl_down_sync(0xffffffff, local_bsum, offset);
    }
  }

  // Thread 0 writes partial sum for this block
  if (threadIdx.x == 0) {
    cSim.pPGMCGPartials[blockIdx.x] = local_bsum;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMReduceBnorm_kernel: Final reduction of bnorm partial sums → scalars[3]
// Launch with 1 block, nblocks threads (or 128 threads with loop)
//---------------------------------------------------------------------------------------------
__global__ void
kPGMReduceBnorm_kernel(int nblocks, int n3)
{
  double sum = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    sum += cSim.pPGMCGPartials[i];
  }

  // Warp reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  __shared__ double sSum[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;

  if (lane == 0) sSum[warpId] = sum;
  __syncthreads();

  if (threadIdx.x < 4) sum = sSum[threadIdx.x]; else sum = 0.0;

  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }

  if (threadIdx.x == 0) {
    // scalars[3] = bnorm = mean(|b|)
    cSim.pPGMCGScalars[3] = sum / (double)n3;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMCopyDipToBuffers_kernel: Copy CG vector → dipole input buffers for field kernels
//
// src is a CG vector (interleaved xyz). Copies to:
//   pPDipXYSP, pPDipZSP (SP for direct kernel)
//   pPGMFracDipX/Y/Z (fractional for reciprocal grid fill)
//   pPermDipX/Y/Z (DP for any kernel that reads these)
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMCopyDipToBuffers_kernel(double* src)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= cSim.atoms) return;

  double dx = src[3*idx + 0];
  double dy = src[3*idx + 1];
  double dz = src[3*idx + 2];

  // Double precision permanent dipole buffers
  cSim.pPermDipX[idx] = dx;
  cSim.pPermDipY[idx] = dy;
  cSim.pPermDipZ[idx] = dz;

  // Single precision packed
  PMEFloat2 pdxy;
  pdxy.x = (PMEFloat)dx;
  pdxy.y = (PMEFloat)dy;
  cSim.pPDipXYSP[idx] = pdxy;
  cSim.pPDipZSP[idx]  = (PMEFloat)dz;

  // Fractional dipoles: frac_k = nfft_k * sum_j recipf[j][k] * cart_j
  cSim.pPGMFracDipX[idx] = (PMEFloat)(cSim.nfft1 *
      (cSim.recipf[0][0]*dx + cSim.recipf[1][0]*dy + cSim.recipf[2][0]*dz));
  cSim.pPGMFracDipY[idx] = (PMEFloat)(cSim.nfft2 *
      (cSim.recipf[1][1]*dy + cSim.recipf[2][1]*dz));
  cSim.pPGMFracDipZ[idx] = (PMEFloat)(cSim.nfft3 *
      (cSim.recipf[2][2]*dz));
}

//---------------------------------------------------------------------------------------------
// kPGMComputeAp_kernel: Extract field from accumulators, compute z = src + alpha*T(src)
//
// After field kernels run, accumulators contain T(src).
// z[i] = src[i] + alpha[i] * field[i]  (A*src in the CG linear system)
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMComputeAp_kernel(double* src, double* dst)
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const double fscale_inv = 1.0 / (double)(1LL << 40);

  while (pos < cSim.atoms) {
    double alpha_i = cSim.pPGMPolarizability[pos];

    double fx = (double)cSim.pPGMFieldXAccumulator[pos] * fscale_inv;
    double fy = (double)cSim.pPGMFieldYAccumulator[pos] * fscale_inv;
    double fz = (double)cSim.pPGMFieldZAccumulator[pos] * fscale_inv;

    dst[3*pos + 0] = src[3*pos + 0] + alpha_i * fx;
    dst[3*pos + 1] = src[3*pos + 1] + alpha_i * fy;
    dst[3*pos + 2] = src[3*pos + 2] + alpha_i * fz;

    pos += blockDim.x * gridDim.x;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMInitResidual_kernel: r = b - z; p = M^{-1}*r; partial rdotz = dot(r, M^{-1}*r)
//   PCG: preconditioner M^{-1} = diag(1/(1 + alpha_i * selfFieldPrefac))
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMInitResidual_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rdotz = 0.0;

  while (pos < cSim.atoms) {
    double minv = cSim.pPGMPrecondInv[pos];
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      double ri = cSim.pPGMCGb[idx] - cSim.pPGMCGz[idx];
      cSim.pPGMCGr[idx] = ri;
      cSim.pPGMCGp[idx] = minv * ri;
      local_rdotz += ri * (minv * ri);
    }
    pos += blockDim.x * gridDim.x;
  }

  // Warp reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_rdotz += __shfl_down_sync(0xffffffff, local_rdotz, offset);
  }

  __shared__ double sRdotz[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;

  if (lane == 0) sRdotz[warpId] = local_rdotz;
  __syncthreads();

  if (threadIdx.x < 4) local_rdotz = sRdotz[threadIdx.x]; else local_rdotz = 0.0;

  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      local_rdotz += __shfl_down_sync(0xffffffff, local_rdotz, offset);
    }
  }

  if (threadIdx.x == 0) {
    cSim.pPGMCGPartials[blockIdx.x] = local_rdotz;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMReduceRdotr_kernel: Final reduction of rdotz partial sums → scalars[0]
//---------------------------------------------------------------------------------------------
__global__ void
kPGMReduceRdotr_kernel(int nblocks)
{
  double sum = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    sum += cSim.pPGMCGPartials[i];
  }

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  __shared__ double sSum[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;
  if (lane == 0) sSum[warpId] = sum;
  __syncthreads();
  if (threadIdx.x < 4) sum = sSum[threadIdx.x]; else sum = 0.0;
  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }

  if (threadIdx.x == 0) {
    cSim.pPGMCGScalars[0] = sum;  // rdotz (PCG: dot(r, M^{-1}*r))
  }
}

//---------------------------------------------------------------------------------------------
// kPGMDotPZ_kernel: Compute partial sums of dot(p, z) → partials buffer
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMDotPZ_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_pdotz = 0.0;

  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      local_pdotz += cSim.pPGMCGp[idx] * cSim.pPGMCGz[idx];
    }
    pos += blockDim.x * gridDim.x;
  }

  // Warp reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_pdotz += __shfl_down_sync(0xffffffff, local_pdotz, offset);
  }

  __shared__ double sVal[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;
  if (lane == 0) sVal[warpId] = local_pdotz;
  __syncthreads();
  if (threadIdx.x < 4) local_pdotz = sVal[threadIdx.x]; else local_pdotz = 0.0;
  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      local_pdotz += __shfl_down_sync(0xffffffff, local_pdotz, offset);
    }
  }

  if (threadIdx.x == 0) {
    cSim.pPGMCGPartials[blockIdx.x] = local_pdotz;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMReducePdotZ_kernel: Final reduction → scalars[1] = pdotz
//---------------------------------------------------------------------------------------------
__global__ void
kPGMReducePdotZ_kernel(int nblocks)
{
  double sum = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    sum += cSim.pPGMCGPartials[i];
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  __shared__ double sSum[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;
  if (lane == 0) sSum[warpId] = sum;
  __syncthreads();
  if (threadIdx.x < 4) sum = sSum[threadIdx.x]; else sum = 0.0;
  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }
  if (threadIdx.x == 0) {
    cSim.pPGMCGScalars[1] = sum;  // pdotz
    cSim.pPGMCGScalars[6] = cSim.pPGMCGScalars[0] / sum;  // alpha_cg = rdotz/pdotz (PCG)
  }
}

//---------------------------------------------------------------------------------------------
// kPGMCGUpdate_kernel: x += alpha_cg * p; r -= alpha_cg * z
//   Also computes partial rdotz_new = dot(r_new, M^{-1}*r_new) and rmax.
//   PCG: uses preconditioner M^{-1} for dot product.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMCGUpdate_kernel()
{
  double alpha_cg = cSim.pPGMCGScalars[6];  // computed by kPGMReducePdotZ
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rdotz = 0.0;
  double local_rmax  = 0.0;

  while (pos < cSim.atoms) {
    double minv = cSim.pPGMPrecondInv[pos];
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      cSim.pPGMCGx[idx] += alpha_cg * cSim.pPGMCGp[idx];
      double ri = cSim.pPGMCGr[idx] - alpha_cg * cSim.pPGMCGz[idx];
      cSim.pPGMCGr[idx] = ri;
      local_rdotz += ri * (minv * ri);
      double ari = fabs(ri);
      if (ari > local_rmax) local_rmax = ari;
    }
    pos += blockDim.x * gridDim.x;
  }

  // Warp reduction for rdotz (sum) and rmax (max)
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_rdotz += __shfl_down_sync(0xffffffff, local_rdotz, offset);
    double other_rmax = __shfl_down_sync(0xffffffff, local_rmax, offset);
    if (other_rmax > local_rmax) local_rmax = other_rmax;
  }

  __shared__ double sRdotz[4];
  __shared__ double sRmax[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;

  if (lane == 0) {
    sRdotz[warpId] = local_rdotz;
    sRmax[warpId]  = local_rmax;
  }
  __syncthreads();

  if (threadIdx.x < 4) {
    local_rdotz = sRdotz[threadIdx.x];
    local_rmax  = sRmax[threadIdx.x];
  } else {
    local_rdotz = 0.0;
    local_rmax  = 0.0;
  }

  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      local_rdotz += __shfl_down_sync(0xffffffff, local_rdotz, offset);
      double other_rmax = __shfl_down_sync(0xffffffff, local_rmax, offset);
      if (other_rmax > local_rmax) local_rmax = other_rmax;
    }
  }

  if (threadIdx.x == 0) {
    // Store rdotz partial in first half, rmax partial in second half
    cSim.pPGMCGPartials[blockIdx.x] = local_rdotz;
    cSim.pPGMCGPartials[cSim.pgmCGnblocks + blockIdx.x] = local_rmax;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMReduceUpdate_kernel: Final reduction of rdotz_new → scalars[0], rmax → scalars[2]
//   Also computes scalars[4] = rmax / bnorm (convergence metric, ready for CPU read)
//   PCG: beta = rdotz_new / rdotz_old
//---------------------------------------------------------------------------------------------
__global__ void
kPGMReduceUpdate_kernel(int nblocks)
{
  double sum_rdotz = 0.0;
  double max_rmax  = 0.0;

  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    sum_rdotz += cSim.pPGMCGPartials[i];
    double rm = cSim.pPGMCGPartials[nblocks + i];
    if (rm > max_rmax) max_rmax = rm;
  }

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum_rdotz += __shfl_down_sync(0xffffffff, sum_rdotz, offset);
    double other = __shfl_down_sync(0xffffffff, max_rmax, offset);
    if (other > max_rmax) max_rmax = other;
  }

  __shared__ double sSum[4], sMax[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;
  if (lane == 0) { sSum[warpId] = sum_rdotz; sMax[warpId] = max_rmax; }
  __syncthreads();
  if (threadIdx.x < 4) { sum_rdotz = sSum[threadIdx.x]; max_rmax = sMax[threadIdx.x]; }
  else { sum_rdotz = 0.0; max_rmax = 0.0; }
  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      sum_rdotz += __shfl_down_sync(0xffffffff, sum_rdotz, offset);
      double other = __shfl_down_sync(0xffffffff, max_rmax, offset);
      if (other > max_rmax) max_rmax = other;
    }
  }

  if (threadIdx.x == 0) {
    double old_rdotz = cSim.pPGMCGScalars[0];
    cSim.pPGMCGScalars[0] = sum_rdotz;  // rdotz_new (overwrites old)
    cSim.pPGMCGScalars[2] = max_rmax;   // rmax
    double bnorm = cSim.pPGMCGScalars[3];
    cSim.pPGMCGScalars[4] = max_rmax / bnorm;  // convergence metric
    // scalars[5] = beta = rdotz_new / rdotz_old (PCG)
    cSim.pPGMCGScalars[5] = (old_rdotz > 0.0) ? sum_rdotz / old_rdotz : 0.0;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMCGUpdateP_kernel: p = M^{-1}*r + beta * p (PCG)
//   beta is read from scalars[5] (computed by kPGMReduceUpdate_kernel)
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMCGUpdateP_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double beta = cSim.pPGMCGScalars[5];

  while (pos < cSim.atoms) {
    double minv = cSim.pPGMPrecondInv[pos];
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      cSim.pPGMCGp[idx] = minv * cSim.pPGMCGr[idx] + beta * cSim.pPGMCGp[idx];
    }
    pos += blockDim.x * gridDim.x;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMStoreInducedDipoles_kernel: Copy converged x → induced dipole buffers
//   Also computes total dipoles = perm + induced, and their SP/frac copies.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMStoreInducedDipoles_kernel()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= cSim.atoms) return;

  double ix = cSim.pPGMCGx[3*idx + 0];
  double iy = cSim.pPGMCGx[3*idx + 1];
  double iz = cSim.pPGMCGx[3*idx + 2];

  // Store induced dipoles
  cSim.pIndDipX[idx] = ix;
  cSim.pIndDipY[idx] = iy;
  cSim.pIndDipZ[idx] = iz;

  PMEFloat2 idxy;
  idxy.x = (PMEFloat)ix;
  idxy.y = (PMEFloat)iy;
  cSim.pIDipXYSP[idx] = idxy;
  cSim.pIDipZSP[idx]  = (PMEFloat)iz;

  // Compute total dipoles = perm + induced
  double tx = cSim.pPGMSavedPermDipX[idx] + ix;
  double ty = cSim.pPGMSavedPermDipY[idx] + iy;
  double tz = cSim.pPGMSavedPermDipZ[idx] + iz;

  cSim.pTotDipX[idx] = tx;
  cSim.pTotDipY[idx] = ty;
  cSim.pTotDipZ[idx] = tz;

  PMEFloat2 tdxy;
  tdxy.x = (PMEFloat)tx;
  tdxy.y = (PMEFloat)ty;
  cSim.pTDipXYSP[idx] = tdxy;
  cSim.pTDipZSP[idx]  = (PMEFloat)tz;

  // Fractional total dipoles
  cSim.pPGMFracTotDipX[idx] = (PMEFloat)(cSim.nfft1 *
      (cSim.recipf[0][0]*tx + cSim.recipf[1][0]*ty + cSim.recipf[2][0]*tz));
  cSim.pPGMFracTotDipY[idx] = (PMEFloat)(cSim.nfft2 *
      (cSim.recipf[1][1]*ty + cSim.recipf[2][1]*tz));
  cSim.pPGMFracTotDipZ[idx] = (PMEFloat)(cSim.nfft3 *
      (cSim.recipf[2][2]*tz));
}

//---------------------------------------------------------------------------------------------
// kPGMFusedCopyDipClearSelf_kernel: Fused CopyDipToBuffers + ClearAndSelfField
//
// Combines two per-atom kernels into one pass, saving 1 kernel launch per CG iteration:
//   1. Reads CG vector → writes to SP dipole buffers + fractional dipole buffers
//   2. Computes self-field → directly stores to field accumulators (no atomicAdd)
//   3. Zeroes potential accumulator (charges are zero during SCF)
//   4. Zeroes padding slots in accumulators
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMFusedCopyDipClearSelf_kernel(double* src)
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Self-field prefactor (constant across all atoms)
  PMEFloat ew  = cSim.ew_coeffSP;
  PMEFloat ew3 = ew * ew * ew;
  PMEFloat selfFieldPrefac = (PMEFloat)(-4.0 / 3.0) * ew3 * (PMEFloat)M_2_SQRTPI *
                             (PMEFloat)0.5;

  while (pos < cSim.atoms) {
    // Read dipole from CG vector
    double dx = src[3*pos + 0];
    double dy = src[3*pos + 1];
    double dz = src[3*pos + 2];

    // Double precision permanent dipole buffers
    cSim.pPermDipX[pos] = dx;
    cSim.pPermDipY[pos] = dy;
    cSim.pPermDipZ[pos] = dz;

    // Single precision packed dipole buffers
    PMEFloat2 pdxy;
    pdxy.x = (PMEFloat)dx;
    pdxy.y = (PMEFloat)dy;
    cSim.pPDipXYSP[pos] = pdxy;
    cSim.pPDipZSP[pos]  = (PMEFloat)dz;

    // Fractional dipoles for reciprocal grid fill
    cSim.pPGMFracDipX[pos] = (PMEFloat)(cSim.nfft1 *
        (cSim.recipf[0][0]*dx + cSim.recipf[1][0]*dy + cSim.recipf[2][0]*dz));
    cSim.pPGMFracDipY[pos] = (PMEFloat)(cSim.nfft2 *
        (cSim.recipf[1][1]*dy + cSim.recipf[2][1]*dz));
    cSim.pPGMFracDipZ[pos] = (PMEFloat)(cSim.nfft3 *
        (cSim.recipf[2][2]*dz));

    // Self-field: directly store to accumulators (clear + self in one write)
    cSim.pPGMFieldXAccumulator[pos] = llrint((PMEDouble)(selfFieldPrefac * pdxy.x) * FORCESCALE);
    cSim.pPGMFieldYAccumulator[pos] = llrint((PMEDouble)(selfFieldPrefac * pdxy.y) * FORCESCALE);
    cSim.pPGMFieldZAccumulator[pos] = llrint((PMEDouble)(selfFieldPrefac * (PMEFloat)dz) * FORCESCALE);

    // Phi_000 = 0 (charges are zero during SCF)
    cSim.pPGMPotentialAccumulator[pos] = (PMEAccumulator)0;

    pos += blockDim.x * gridDim.x;
  }

  // Zero out padding (stride > atoms) to prevent stale accumulator values
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
// kPGMFusedComputeApDotPZ_kernel: Fused ComputeAp + DotPZ partial sums
//
// Combines z = src + alpha*field with partial dot(p, z) in one pass, saving 1 launch.
// Partials written to pPGMCGPartials[] for subsequent reduction by kPGMReducePdotZ.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMFusedComputeApDotPZ_kernel(double* src, double* dst)
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_pdotz = 0.0;
  const double fscale_inv = 1.0 / (double)(1LL << 40);

  while (pos < cSim.atoms) {
    double alpha_i = cSim.pPGMPolarizability[pos];

    double fx = (double)cSim.pPGMFieldXAccumulator[pos] * fscale_inv;
    double fy = (double)cSim.pPGMFieldYAccumulator[pos] * fscale_inv;
    double fz = (double)cSim.pPGMFieldZAccumulator[pos] * fscale_inv;

    double sx = src[3*pos + 0];
    double sy = src[3*pos + 1];
    double sz = src[3*pos + 2];

    double zx = sx + alpha_i * fx;
    double zy = sy + alpha_i * fy;
    double zz = sz + alpha_i * fz;

    dst[3*pos + 0] = zx;
    dst[3*pos + 1] = zy;
    dst[3*pos + 2] = zz;

    // Accumulate dot(p, z) — src IS p in the CG loop
    local_pdotz += sx * zx + sy * zy + sz * zz;

    pos += blockDim.x * gridDim.x;
  }

  // Warp reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_pdotz += __shfl_down_sync(0xffffffff, local_pdotz, offset);
  }

  __shared__ double sVal[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;
  if (lane == 0) sVal[warpId] = local_pdotz;
  __syncthreads();
  if (threadIdx.x < 4) local_pdotz = sVal[threadIdx.x]; else local_pdotz = 0.0;
  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      local_pdotz += __shfl_down_sync(0xffffffff, local_pdotz, offset);
    }
  }

  if (threadIdx.x == 0) {
    cSim.pPGMCGPartials[blockIdx.x] = local_pdotz;
  }
}

//---------------------------------------------------------------------------------------------
// Host dispatch functions
//---------------------------------------------------------------------------------------------

extern "C" void kPGMFusedCopyDipClearSelf(gpuContext gpu, double* src_dev)
{
  int blocks = (gpu->sim.stride + 127) / 128;
  kPGMFusedCopyDipClearSelf_kernel<<<blocks, 128, 0, gpu->pgmStream>>>(src_dev);
  LAUNCHERROR("kPGMFusedCopyDipClearSelf");
}

extern "C" void kPGMFusedComputeApDotPZ(gpuContext gpu, double* src_dev, double* dst_dev)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMFusedComputeApDotPZ_kernel<<<blocks, 128, 0, gpu->pgmStream>>>(src_dev, dst_dev);
  LAUNCHERROR("kPGMFusedComputeApDotPZ");

  kPGMReducePdotZ_kernel<<<1, 128, 0, gpu->pgmStream>>>(blocks);
  LAUNCHERROR("kPGMReducePdotZ");
}

extern "C" void kPGMInitSCF(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMInitSCF_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMInitSCF");

  // Use N_total for bnorm to match CPU convention (pGM_induced.F90:722).
  int n3 = 3 * gpu->sim.atoms;
  kPGMReduceBnorm_kernel<<<1, 128>>>(blocks, n3);
  LAUNCHERROR("kPGMReduceBnorm");
}

extern "C" void kPGMCopyDipToBuffers(gpuContext gpu, double* src_dev)
{
  int nblocks = (gpu->sim.atoms + 255) / 256;
  kPGMCopyDipToBuffers_kernel<<<nblocks, 256, 0, gpu->pgmStream>>>(src_dev);
  LAUNCHERROR("kPGMCopyDipToBuffers");
}

extern "C" void kPGMComputeAp(gpuContext gpu, double* src_dev, double* dst_dev)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMComputeAp_kernel<<<blocks, 128, 0, gpu->pgmStream>>>(src_dev, dst_dev);
  LAUNCHERROR("kPGMComputeAp");
}

extern "C" void kPGMInitResidual(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMInitResidual_kernel<<<blocks, 128, 0, gpu->pgmStream>>>();
  LAUNCHERROR("kPGMInitResidual");

  kPGMReduceRdotr_kernel<<<1, 128, 0, gpu->pgmStream>>>(blocks);
  LAUNCHERROR("kPGMReduceRdotr");
}

extern "C" void kPGMDotPZ(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMDotPZ_kernel<<<blocks, 128, 0, gpu->pgmStream>>>();
  LAUNCHERROR("kPGMDotPZ");

  kPGMReducePdotZ_kernel<<<1, 128, 0, gpu->pgmStream>>>(blocks);
  LAUNCHERROR("kPGMReducePdotZ");
}

extern "C" void kPGMCGUpdate(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMCGUpdate_kernel<<<blocks, 128, 0, gpu->pgmStream>>>();
  LAUNCHERROR("kPGMCGUpdate");

  kPGMReduceUpdate_kernel<<<1, 128, 0, gpu->pgmStream>>>(blocks);
  LAUNCHERROR("kPGMReduceUpdate");
}

extern "C" void kPGMCGUpdateP(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMCGUpdateP_kernel<<<blocks, 128, 0, gpu->pgmStream>>>();
  LAUNCHERROR("kPGMCGUpdateP");
}

extern "C" void kPGMStoreInducedDipoles(gpuContext gpu)
{
  int nblocks = (gpu->sim.atoms + 255) / 256;
  kPGMStoreInducedDipoles_kernel<<<nblocks, 256>>>();
  LAUNCHERROR("kPGMStoreInducedDipoles");
}

//---------------------------------------------------------------------------------------------
// kPGMComputeResidual_kernel: Like kPGMInitResidual but also computes rmax.
//
// Computes r = b - z, p = M^{-1}*r, and partial sums for both rdotz and rmax.
// Used by operator-split SCF to check convergence at outer iteration restarts.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMComputeResidual_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rdotz = 0.0;
  double local_rmax  = 0.0;

  while (pos < cSim.atoms) {
    double minv = cSim.pPGMPrecondInv[pos];
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      double ri = cSim.pPGMCGb[idx] - cSim.pPGMCGz[idx];
      cSim.pPGMCGr[idx] = ri;
      cSim.pPGMCGp[idx] = minv * ri;
      local_rdotz += ri * (minv * ri);
      double ari = fabs(ri);
      if (ari > local_rmax) local_rmax = ari;
    }
    pos += blockDim.x * gridDim.x;
  }

  // Warp reduction for rdotz (sum) and rmax (max)
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_rdotz += __shfl_down_sync(0xffffffff, local_rdotz, offset);
    double other_rmax = __shfl_down_sync(0xffffffff, local_rmax, offset);
    if (other_rmax > local_rmax) local_rmax = other_rmax;
  }

  __shared__ double sRdotz[4];
  __shared__ double sRmax[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;

  if (lane == 0) {
    sRdotz[warpId] = local_rdotz;
    sRmax[warpId]  = local_rmax;
  }
  __syncthreads();

  if (threadIdx.x < 4) {
    local_rdotz = sRdotz[threadIdx.x];
    local_rmax  = sRmax[threadIdx.x];
  } else {
    local_rdotz = 0.0;
    local_rmax  = 0.0;
  }

  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      local_rdotz += __shfl_down_sync(0xffffffff, local_rdotz, offset);
      double other_rmax = __shfl_down_sync(0xffffffff, local_rmax, offset);
      if (other_rmax > local_rmax) local_rmax = other_rmax;
    }
  }

  if (threadIdx.x == 0) {
    cSim.pPGMCGPartials[blockIdx.x] = local_rdotz;
    cSim.pPGMCGPartials[cSim.pgmCGnblocks + blockIdx.x] = local_rmax;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMReduceResidual_kernel: Reduce rdotz + rmax → scalars[0,2,4]
//   scalars[0] = rdotz
//   scalars[2] = rmax
//   scalars[4] = rmax / bnorm (convergence metric, same as standard CG)
//---------------------------------------------------------------------------------------------
__global__ void
kPGMReduceResidual_kernel(int nblocks)
{
  double sum_rdotr = 0.0;
  double max_rmax  = 0.0;

  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    sum_rdotr += cSim.pPGMCGPartials[i];
    double rm = cSim.pPGMCGPartials[nblocks + i];
    if (rm > max_rmax) max_rmax = rm;
  }

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum_rdotr += __shfl_down_sync(0xffffffff, sum_rdotr, offset);
    double other = __shfl_down_sync(0xffffffff, max_rmax, offset);
    if (other > max_rmax) max_rmax = other;
  }

  __shared__ double sSum[4], sMax[4];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;
  if (lane == 0) { sSum[warpId] = sum_rdotr; sMax[warpId] = max_rmax; }
  __syncthreads();
  if (threadIdx.x < 4) { sum_rdotr = sSum[threadIdx.x]; max_rmax = sMax[threadIdx.x]; }
  else { sum_rdotr = 0.0; max_rmax = 0.0; }
  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      sum_rdotr += __shfl_down_sync(0xffffffff, sum_rdotr, offset);
      double other = __shfl_down_sync(0xffffffff, max_rmax, offset);
      if (other > max_rmax) max_rmax = other;
    }
  }

  if (threadIdx.x == 0) {
    cSim.pPGMCGScalars[0] = sum_rdotr;  // rdotr
    cSim.pPGMCGScalars[2] = max_rmax;   // rmax
    double bnorm = cSim.pPGMCGScalars[3];
    cSim.pPGMCGScalars[4] = max_rmax / bnorm;  // convergence metric
  }
}

//---------------------------------------------------------------------------------------------
// Host dispatch: kPGMComputeResidual
//---------------------------------------------------------------------------------------------
extern "C" void kPGMComputeResidual(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMComputeResidual_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMComputeResidual");

  kPGMReduceResidual_kernel<<<1, 128>>>(blocks);
  LAUNCHERROR("kPGMReduceResidual");
}

//---------------------------------------------------------------------------------------------
// kPGMExtrapolateLS_kernel: Least-squares quadratic extrapolation of induced dipoles
//
// Uses LS quadratic fit through K equispaced history points to predict SCF initial guess.
// K=2: linear (c=[2,-1]), K=3: exact quadratic (c=[3,-3,1]),
// K>=4: overdetermined LS quadratic (lower noise amplification → fewer CG iterations).
//
// Coefficients derived from c = X*(X'X)^{-1}*x_pred for LS quadratic at equispaced points:
//   K=4: [2.25, -0.75, -1.25, 0.75]       amplification=5.0
//   K=5: [1.8, 0.0, -0.8, -0.6, 0.6]      amplification=3.8
//   K=6: [1.5, 0.3, -0.4, -0.6, -0.3, 0.5] amplification=3.6
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMExtrapolateLS_kernel(double* __restrict__ dst,
                         const double* __restrict__ h0,
                         const double* __restrict__ h1,
                         const double* __restrict__ h2,
                         const double* __restrict__ h3,
                         const double* __restrict__ h4,
                         const double* __restrict__ h5,
                         int K, int n3,
                         const double* __restrict__ base, int cubic)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n3) return;

  double val;
  if (K == 4 && cubic) {          // exact cubic through 4 equispaced points
    val = 4.0*h0[idx] - 6.0*h1[idx] + 4.0*h2[idx] - h3[idx];
    dst[idx] = (base ? base[idx] : 0.0) + val;
    return;
  }
  if (K == 5 && cubic) {          // exact quartic through 5 equispaced points
    val = 5.0*h0[idx] - 10.0*h1[idx] + 10.0*h2[idx] - 5.0*h3[idx] + h4[idx];
    dst[idx] = (base ? base[idx] : 0.0) + val;
    return;
  }
  switch (K) {
    case 1:  // copy
      val = h0[idx];
      break;
    case 2:  // Linear: c=[2, -1]
      val = 2.0*h0[idx] - h1[idx];
      break;
    case 3:  // Exact quadratic: c=[3, -3, 1]
      val = 3.0*h0[idx] - 3.0*h1[idx] + h2[idx];
      break;
    case 4:  // LS quadratic: c=[2.25, -0.75, -1.25, 0.75]
      val = 2.25*h0[idx] - 0.75*h1[idx] - 1.25*h2[idx] + 0.75*h3[idx];
      break;
    case 5:  // LS quadratic: c=[1.8, 0.0, -0.8, -0.6, 0.6]
      val = 1.8*h0[idx] - 0.8*h2[idx] - 0.6*h3[idx] + 0.6*h4[idx];
      break;
    default: // K>=6, LS quadratic: c=[1.5, 0.3, -0.4, -0.6, -0.3, 0.5]
      val = 1.5*h0[idx] + 0.3*h1[idx] - 0.4*h2[idx]
          - 0.6*h3[idx] - 0.3*h4[idx] + 0.5*h5[idx];
      break;
  }
  dst[idx] = (base ? base[idx] : 0.0) + val;
}

// dst = x - b   (history of the many-body correction Delta = mu - alpha*E_perm)
__global__ void
__launch_bounds__(256, 4)
kPGMVecSub_kernel(double* __restrict__ dst, const double* __restrict__ x,
                  const double* __restrict__ b, int n3)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n3) dst[idx] = x[idx] - b[idx];
}
extern "C" void kPGMVecSub(gpuContext gpu, double* dst, const double* x, const double* b)
{
  int n3 = 3 * gpu->sim.atoms;
  kPGMVecSub_kernel<<<(n3 + 255) / 256, 256>>>(dst, x, b, n3);
  LAUNCHERROR("kPGMVecSub");
}

//---------------------------------------------------------------------------------------------
// Host dispatch: kPGMExtrapolateLS
//---------------------------------------------------------------------------------------------
extern "C" void kPGMExtrapolateLS(gpuContext gpu, double* dst,
                                  double* h0, double* h1, double* h2,
                                  double* h3, double* h4, double* h5,
                                  int K)
{
  int n3 = 3 * gpu->sim.atoms;
  int nblocks = (n3 + 255) / 256;
  kPGMExtrapolateLS_kernel<<<nblocks, 256>>>(dst, h0, h1, h2, h3, h4, h5, K, n3, NULL, 0);
  LAUNCHERROR("kPGMExtrapolateLS");
}
// Variant with an additive base vector (x0 = base + extrapolation) and optional exact cubic for K=4.
extern "C" void kPGMExtrapolateBase(gpuContext gpu, double* dst, const double* base,
                                    double* h0, double* h1, double* h2, double* h3,
                                    int K, int cubic)
{
  int n3 = 3 * gpu->sim.atoms;
  int nblocks = (n3 + 255) / 256;
  kPGMExtrapolateLS_kernel<<<nblocks, 256>>>(dst, h0, h1, h2, h3, NULL, NULL, K, n3, base, cubic);
  LAUNCHERROR("kPGMExtrapolateBase");
}
extern "C" void kPGMExtrapolate5(gpuContext gpu, double* dst, const double* base,
                                 double* h0, double* h1, double* h2, double* h3, double* h4)
{
  int n3 = 3 * gpu->sim.atoms;
  int nblocks = (n3 + 255) / 256;
  kPGMExtrapolateLS_kernel<<<nblocks, 256>>>(dst, h0, h1, h2, h3, h4, NULL, 5, n3, base, 1);
  LAUNCHERROR("kPGMExtrapolate5");
}

//---------------------------------------------------------------------------------------------
// kPGMAddScaledDip_kernel: perm-dipole buffers += scale * v (double, SP and fractional copies)
//   Used by the fused permanent-field + initial-residual sweep: with dipoles mu_perm + x0 the
//   field accumulator returns gradphi(q, mu_perm) + T x0 in one sweep (linearity).
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMAddScaledDip_kernel(const double* __restrict__ v, double scale)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= cSim.atoms) return;
  double dx = cSim.pPermDipX[idx] + scale * v[3*idx + 0];
  double dy = cSim.pPermDipY[idx] + scale * v[3*idx + 1];
  double dz = cSim.pPermDipZ[idx] + scale * v[3*idx + 2];
  cSim.pPermDipX[idx] = dx; cSim.pPermDipY[idx] = dy; cSim.pPermDipZ[idx] = dz;
  PMEFloat2 pdxy; pdxy.x = (PMEFloat)dx; pdxy.y = (PMEFloat)dy;
  cSim.pPDipXYSP[idx] = pdxy; cSim.pPDipZSP[idx] = (PMEFloat)dz;
  cSim.pPGMFracDipX[idx] = (PMEFloat)(cSim.nfft1 *
      (cSim.recipf[0][0]*dx + cSim.recipf[1][0]*dy + cSim.recipf[2][0]*dz));
  cSim.pPGMFracDipY[idx] = (PMEFloat)(cSim.nfft2 *
      (cSim.recipf[1][1]*dy + cSim.recipf[2][1]*dz));
  cSim.pPGMFracDipZ[idx] = (PMEFloat)(cSim.nfft3 * (cSim.recipf[2][2]*dz));
}
extern "C" void kPGMAddScaledDip(gpuContext gpu, const double* v, double scale)
{
  int nblocks = (gpu->sim.atoms + 255) / 256;
  kPGMAddScaledDip_kernel<<<nblocks, 256>>>(v, scale);
  LAUNCHERROR("kPGMAddScaledDip");
}
// dst = base + sum_j c_j h_j : least-squares quartic extrapolation through 8 equispaced points
// (noise amplification sum|c| = 11.2 vs 15 for the exact cubic, truncation order 5)
__constant__ double cPGM_LS8[8] = { 3.1250000000, -2.2321428572, -1.3392857143, 0.8035714286, 1.4464285715, 0.0892857143, -1.5178571429, 0.6250000000 };
__global__ void
__launch_bounds__(256, 4)
kPGMExtrapolate8_kernel(double* __restrict__ dst, const double* __restrict__ base,
                        const double* h0, const double* h1, const double* h2, const double* h3,
                        const double* h4, const double* h5, const double* h6, const double* h7, int n3)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n3) return;
  double v = cPGM_LS8[0]*h0[idx] + cPGM_LS8[1]*h1[idx] + cPGM_LS8[2]*h2[idx] + cPGM_LS8[3]*h3[idx]
           + cPGM_LS8[4]*h4[idx] + cPGM_LS8[5]*h5[idx] + cPGM_LS8[6]*h6[idx] + cPGM_LS8[7]*h7[idx];
  dst[idx] = (base ? base[idx] : 0.0) + v;
}
extern "C" void kPGMExtrapolate8(gpuContext gpu, double* dst, const double* base, double** H)
{
  int n3 = 3 * gpu->sim.atoms;
  kPGMExtrapolate8_kernel<<<(n3 + 255) / 256, 256>>>(dst, base, H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], n3);
  LAUNCHERROR("kPGMExtrapolate8");
}
// Extended-Lagrangian auxiliary-dipole update (Niklasson dissipative Verlet):
//   dst = 2 n0 - n1 + kappa*(mu - n0) + alpha * sum_{j=0..K} c_j n_j      (n_j = H[j], j = 0 newest)
__constant__ double cPGM_XLC[8];
__global__ void
__launch_bounds__(256, 4)
kPGMXLUpdate_kernel(double* __restrict__ dst, const double* __restrict__ mu,
                    const double* h0, const double* h1, const double* h2, const double* h3,
                    const double* h4, const double* h5, int K, double kappa, double alpha, int n3)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n3) return;
  double n0 = h0[i], n1 = h1[i];
  double diss = cPGM_XLC[0]*n0 + cPGM_XLC[1]*n1;
  if (K >= 2) diss += cPGM_XLC[2]*h2[i];
  if (K >= 3) diss += cPGM_XLC[3]*h3[i];
  if (K >= 4) diss += cPGM_XLC[4]*h4[i];
  if (K >= 5) diss += cPGM_XLC[5]*h5[i];
  dst[i] = 2.0*n0 - n1 + kappa*(mu[i] - n0) + alpha*diss;
}
extern "C" void kPGMXLUpdate(gpuContext gpu, double* dst, const double* mu, double** H, int K, double kappa, double alpha, const double* c)
{
  static bool cset = false;
  if (!cset) { cudaMemcpyToSymbol(cPGM_XLC, c, 6*sizeof(double)); cset = true; }
  int n3 = 3 * gpu->sim.atoms;
  kPGMXLUpdate_kernel<<<(n3 + 255) / 256, 256>>>(dst, mu, H[0], H[1], H[2], H[3], H[4], H[5], K, kappa, alpha, n3);
  LAUNCHERROR("kPGMXLUpdate");
}
// x += w * r  (peek / Richardson step on the final CG residual)
__global__ void
__launch_bounds__(256, 4)
kPGMAxpy_kernel(double* __restrict__ x, const double* __restrict__ r, double w, int n3)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n3) x[idx] += w * r[idx];
}
extern "C" void kPGMAxpy(gpuContext gpu, double* x, const double* r, double w)
{
  int n3 = 3 * gpu->sim.atoms;
  kPGMAxpy_kernel<<<(n3 + 255) / 256, 256>>>(x, r, w, n3);
  LAUNCHERROR("kPGMAxpy");
}

//=============================================================================
// PCG (Preconditioned CG with local preconditioner) kernels
//=============================================================================

//---------------------------------------------------------------------------------------------
// Standard warp+block reduction helper (sum). Used by many PCG kernels.
//---------------------------------------------------------------------------------------------
#define PCG_REDUCE_SUM(val, sArr)                                              \
  for (int _off = warpSize/2; _off > 0; _off >>= 1)                           \
    val += __shfl_down_sync(0xffffffff, val, _off);                            \
  { int _lane = threadIdx.x & 31; int _wid = threadIdx.x >> 5;                \
    if (_lane == 0) sArr[_wid] = val;                                          \
    __syncthreads();                                                           \
    if (threadIdx.x < 4) val = sArr[threadIdx.x]; else val = 0.0;             \
    if (_wid == 0) { for (int _off = 2; _off > 0; _off >>= 1)                 \
      val += __shfl_down_sync(0xffffffff, val, _off); } }

#define PCG_REDUCE_MAX(val, sArr)                                              \
  for (int _off = warpSize/2; _off > 0; _off >>= 1)                           \
  { double _o = __shfl_down_sync(0xffffffff, val, _off);                       \
    if (_o > val) val = _o; }                                                  \
  { int _lane = threadIdx.x & 31; int _wid = threadIdx.x >> 5;                \
    if (_lane == 0) sArr[_wid] = val;                                          \
    __syncthreads();                                                           \
    if (threadIdx.x < 4) val = sArr[threadIdx.x]; else val = 0.0;             \
    if (_wid == 0) { for (int _off = 2; _off > 0; _off >>= 1)                 \
    { double _o = __shfl_down_sync(0xffffffff, val, _off);                     \
      if (_o > val) val = _o; } } }

//---------------------------------------------------------------------------------------------
// kPGMPrecondInit_kernel: Initialize inner CG for local preconditioner
//   xu (zpc) = 0, ru = r (outer residual), pu = r
//   Compute partial rdotr = dot(ru, ru) → partials
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPrecondInit_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rdotr = 0.0;

  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      double ri = cSim.pPGMCGr[idx];
      cSim.pPGMPCGzpc[idx] = 0.0;
      cSim.pPGMPCGru[idx]  = ri;
      cSim.pPGMPCGpu[idx]  = ri;
      local_rdotr += ri * ri;
    }
    pos += blockDim.x * gridDim.x;
  }

  __shared__ double sR[4];
  PCG_REDUCE_SUM(local_rdotr, sR);
  if (threadIdx.x == 0)
    cSim.pPGMCGPartials[blockIdx.x] = local_rdotr;
}

__global__ void
kPGMPrecondReduceRdotrInit_kernel(int nblocks)
{
  double sum = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x)
    sum += cSim.pPGMCGPartials[i];
  __shared__ double sR[4];
  PCG_REDUCE_SUM(sum, sR);
  if (threadIdx.x == 0)
    cSim.pPGMCGScalars[8] = sum;  // rdotr_inner
}

extern "C" void kPGMPrecondInit(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  cudaStream_t s = gpu->pgmStream;
  kPGMPrecondInit_kernel<<<blocks, 128, 0, s>>>();
  LAUNCHERROR("kPGMPrecondInit");
  kPGMPrecondReduceRdotrInit_kernel<<<1, 128, 0, s>>>(blocks);
  LAUNCHERROR("kPGMPrecondReduceRdotrInit");
}

//---------------------------------------------------------------------------------------------
// kPGMPrecondCopyDipClear_kernel: Copy inner pu → dipole buffers, clear field accumulators
//   NO self-field (preconditioner uses only pairwise short-range tensor)
//   NO fractional dipoles (no reciprocal space in preconditioner)
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPrecondCopyDipClear_kernel(double* src)
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < cSim.atoms) {
    double dx = src[3*pos + 0];
    double dy = src[3*pos + 1];
    double dz = src[3*pos + 2];

    cSim.pPermDipX[pos] = dx;
    cSim.pPermDipY[pos] = dy;
    cSim.pPermDipZ[pos] = dz;

    PMEFloat2 pdxy;
    pdxy.x = (PMEFloat)dx;
    pdxy.y = (PMEFloat)dy;
    cSim.pPDipXYSP[pos] = pdxy;
    cSim.pPDipZSP[pos]  = (PMEFloat)dz;

    // Clear field accumulators (NO self-field)
    cSim.pPGMFieldXAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMFieldYAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMFieldZAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMPotentialAccumulator[pos] = (PMEAccumulator)0;

    pos += blockDim.x * gridDim.x;
  }

  // Zero padding
  pos = blockIdx.x * blockDim.x + threadIdx.x + cSim.atoms;
  while (pos < cSim.stride) {
    cSim.pPGMFieldXAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMFieldYAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMFieldZAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMPotentialAccumulator[pos] = (PMEAccumulator)0;
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMPrecondCopyDipClear(gpuContext gpu, double* src_dev)
{
  int blocks = (gpu->sim.stride + 127) / 128;
  cudaStream_t s = gpu->pgmStream;
  kPGMPrecondCopyDipClear_kernel<<<blocks, 128, 0, s>>>(src_dev);
  LAUNCHERROR("kPGMPrecondCopyDipClear");
}

//---------------------------------------------------------------------------------------------
// kPGMPrecondFusedApDotPZ_kernel: Inner CG matvec + dot product
//   zu = pu + alpha_i * field  (A_short * pu, reuses pPGMCGz buffer)
//   Partial pdotz = dot(pu, zu)
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPrecondFusedApDotPZ_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_pdotz = 0.0;
  const double fscale_inv = 1.0 / (double)(1LL << 40);

  while (pos < cSim.atoms) {
    double alpha_i = cSim.pPGMPolarizability[pos];

    double fx = (double)cSim.pPGMFieldXAccumulator[pos] * fscale_inv;
    double fy = (double)cSim.pPGMFieldYAccumulator[pos] * fscale_inv;
    double fz = (double)cSim.pPGMFieldZAccumulator[pos] * fscale_inv;

    double px = cSim.pPGMPCGpu[3*pos + 0];
    double py = cSim.pPGMPCGpu[3*pos + 1];
    double pz = cSim.pPGMPCGpu[3*pos + 2];

    double zx = px + alpha_i * fx;
    double zy = py + alpha_i * fy;
    double zz = pz + alpha_i * fz;

    cSim.pPGMCGz[3*pos + 0] = zx;
    cSim.pPGMCGz[3*pos + 1] = zy;
    cSim.pPGMCGz[3*pos + 2] = zz;

    local_pdotz += px*zx + py*zy + pz*zz;

    pos += blockDim.x * gridDim.x;
  }

  __shared__ double sV[4];
  PCG_REDUCE_SUM(local_pdotz, sV);
  if (threadIdx.x == 0)
    cSim.pPGMCGPartials[blockIdx.x] = local_pdotz;
}

__global__ void
kPGMPrecondReduceAlpha_kernel(int nblocks)
{
  double sum = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x)
    sum += cSim.pPGMCGPartials[i];
  __shared__ double sR[4];
  PCG_REDUCE_SUM(sum, sR);
  if (threadIdx.x == 0) {
    cSim.pPGMCGScalars[9]  = sum;                                // pdotz_inner
    cSim.pPGMCGScalars[10] = cSim.pPGMCGScalars[8] / sum;       // alpha_cg_inner
  }
}

extern "C" void kPGMPrecondFusedApDotPZ(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  cudaStream_t s = gpu->pgmStream;
  kPGMPrecondFusedApDotPZ_kernel<<<blocks, 128, 0, s>>>();
  LAUNCHERROR("kPGMPrecondFusedApDotPZ");
  kPGMPrecondReduceAlpha_kernel<<<1, 128, 0, s>>>(blocks);
  LAUNCHERROR("kPGMPrecondReduceAlpha");
}

//---------------------------------------------------------------------------------------------
// kPGMPrecondUpdate_kernel: Inner CG x/r update + partial rdotr
//   xu += alpha_cg * pu; ru -= alpha_cg * zu
//   Partial rdotr_new = dot(ru, ru)
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPrecondUpdate_kernel()
{
  double alpha_cg = cSim.pPGMCGScalars[10];
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rdotr = 0.0;

  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      cSim.pPGMPCGzpc[idx] += alpha_cg * cSim.pPGMPCGpu[idx];
      double ri = cSim.pPGMPCGru[idx] - alpha_cg * cSim.pPGMCGz[idx];
      cSim.pPGMPCGru[idx] = ri;
      local_rdotr += ri * ri;
    }
    pos += blockDim.x * gridDim.x;
  }

  __shared__ double sR[4];
  PCG_REDUCE_SUM(local_rdotr, sR);
  if (threadIdx.x == 0)
    cSim.pPGMCGPartials[blockIdx.x] = local_rdotr;
}

__global__ void
kPGMPrecondReduceRdotrBeta_kernel(int nblocks)
{
  double sum = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x)
    sum += cSim.pPGMCGPartials[i];
  __shared__ double sR[4];
  PCG_REDUCE_SUM(sum, sR);
  if (threadIdx.x == 0) {
    double old_rdotr = cSim.pPGMCGScalars[8];
    cSim.pPGMCGScalars[8]  = sum;                                     // rdotr = rdotr_new
    cSim.pPGMCGScalars[11] = (old_rdotr > 0.0) ? sum / old_rdotr : 0.0;  // beta_inner
  }
}

extern "C" void kPGMPrecondUpdate(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  cudaStream_t s = gpu->pgmStream;
  kPGMPrecondUpdate_kernel<<<blocks, 128, 0, s>>>();
  LAUNCHERROR("kPGMPrecondUpdate");
  kPGMPrecondReduceRdotrBeta_kernel<<<1, 128, 0, s>>>(blocks);
  LAUNCHERROR("kPGMPrecondReduceRdotrBeta");
}

//---------------------------------------------------------------------------------------------
// kPGMPrecondUpdateP_kernel: Inner CG direction update
//   pu = ru + beta_inner * pu
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPrecondUpdateP_kernel()
{
  double beta = cSim.pPGMCGScalars[11];
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      cSim.pPGMPCGpu[idx] = cSim.pPGMPCGru[idx] + beta * cSim.pPGMPCGpu[idx];
    }
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMPrecondUpdateP(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  cudaStream_t s = gpu->pgmStream;
  kPGMPrecondUpdateP_kernel<<<blocks, 128, 0, s>>>();
  LAUNCHERROR("kPGMPrecondUpdateP");
}

//---------------------------------------------------------------------------------------------
// kPGMPrecondPeek_kernel: SOR peek step at end of preconditioner
//   xu += ru * alpha_i
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPrecondPeek_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < cSim.atoms) {
    double alpha_i = cSim.pPGMPolarizability[pos];
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      cSim.pPGMPCGzpc[idx] += cSim.pPGMPCGru[idx] * alpha_i;
    }
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMPrecondPeek(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  cudaStream_t s = gpu->pgmStream;
  kPGMPrecondPeek_kernel<<<blocks, 128, 0, s>>>();
  LAUNCHERROR("kPGMPrecondPeek");
}

//---------------------------------------------------------------------------------------------
// Outer PCG kernels (replace diagonal-preconditioner variants)
//---------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------
// kPGMPCGInitResidual_kernel: r = b - z (no inline preconditioner)
//   Stores r in pPGMCGr. No reduction needed here.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPCGInitResidual_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      cSim.pPGMCGr[idx] = cSim.pPGMCGb[idx] - cSim.pPGMCGz[idx];
    }
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMPCGInitResidual(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMPCGInitResidual_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMPCGInitResidual");
}

//---------------------------------------------------------------------------------------------
// kPGMPCGSetPFromZpc_kernel: p = zpc; partial rdotz = dot(r, zpc)
//   Called after preconditioner to set up the search direction and rdotz.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPCGSetPFromZpc_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rdotz = 0.0;

  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      double zpc_d = cSim.pPGMPCGzpc[idx];
      cSim.pPGMCGp[idx] = zpc_d;
      local_rdotz += cSim.pPGMCGr[idx] * zpc_d;
    }
    pos += blockDim.x * gridDim.x;
  }

  __shared__ double sR[4];
  PCG_REDUCE_SUM(local_rdotz, sR);
  if (threadIdx.x == 0)
    cSim.pPGMCGPartials[blockIdx.x] = local_rdotz;
}

extern "C" void kPGMPCGSetPFromZpc(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMPCGSetPFromZpc_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMPCGSetPFromZpc");

  // Reduce rdotz → scalars[0]
  kPGMReduceRdotr_kernel<<<1, 128>>>(blocks);
  LAUNCHERROR("kPGMPCGReduceRdotzInit");
}

//---------------------------------------------------------------------------------------------
// kPGMPCGUpdate_kernel: x += alpha*p; r -= alpha*z; compute partial rmax
//   Does NOT compute rdotz (that comes after preconditioner application).
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPCGUpdate_kernel()
{
  double alpha_cg = cSim.pPGMCGScalars[6];
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rmax = 0.0;

  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      cSim.pPGMCGx[idx] += alpha_cg * cSim.pPGMCGp[idx];
      double ri = cSim.pPGMCGr[idx] - alpha_cg * cSim.pPGMCGz[idx];
      cSim.pPGMCGr[idx] = ri;
      double ari = fabs(ri);
      if (ari > local_rmax) local_rmax = ari;
    }
    pos += blockDim.x * gridDim.x;
  }

  __shared__ double sM[4];
  PCG_REDUCE_MAX(local_rmax, sM);
  if (threadIdx.x == 0)
    cSim.pPGMCGPartials[blockIdx.x] = local_rmax;
}

__global__ void
kPGMPCGReduceRmax_kernel(int nblocks)
{
  double max_rmax = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
    double rm = cSim.pPGMCGPartials[i];
    if (rm > max_rmax) max_rmax = rm;
  }
  __shared__ double sM[4];
  PCG_REDUCE_MAX(max_rmax, sM);
  if (threadIdx.x == 0) {
    cSim.pPGMCGScalars[2] = max_rmax;
    cSim.pPGMCGScalars[4] = max_rmax / cSim.pPGMCGScalars[3];  // rmax/bnorm
  }
}

extern "C" void kPGMPCGUpdate(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMPCGUpdate_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMPCGUpdate");
  kPGMPCGReduceRmax_kernel<<<1, 128>>>(blocks);
  LAUNCHERROR("kPGMPCGReduceRmax");
}

//---------------------------------------------------------------------------------------------
// kPGMPCGDotRZpc_kernel: rdotz_new = dot(r, zpc)
//   Called after preconditioner application in the main loop.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPCGDotRZpc_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rdotz = 0.0;

  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      local_rdotz += cSim.pPGMCGr[idx] * cSim.pPGMPCGzpc[idx];
    }
    pos += blockDim.x * gridDim.x;
  }

  __shared__ double sR[4];
  PCG_REDUCE_SUM(local_rdotz, sR);
  if (threadIdx.x == 0)
    cSim.pPGMCGPartials[blockIdx.x] = local_rdotz;
}

__global__ void
kPGMPCGReduceRdotzBeta_kernel(int nblocks)
{
  double sum = 0.0;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x)
    sum += cSim.pPGMCGPartials[i];
  __shared__ double sR[4];
  PCG_REDUCE_SUM(sum, sR);
  if (threadIdx.x == 0) {
    double old_rdotz = cSim.pPGMCGScalars[0];
    cSim.pPGMCGScalars[0] = sum;                                      // rdotz = rdotz_new
    cSim.pPGMCGScalars[5] = (old_rdotz > 0.0) ? sum / old_rdotz : 0.0;  // beta
  }
}

extern "C" void kPGMPCGDotRZpc(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMPCGDotRZpc_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMPCGDotRZpc");
  kPGMPCGReduceRdotzBeta_kernel<<<1, 128>>>(blocks);
  LAUNCHERROR("kPGMPCGReduceRdotzBeta");
}

//---------------------------------------------------------------------------------------------
// kPGMPCGUpdateP_kernel: p = zpc + beta * p
//   Uses preconditioned residual instead of diagonal M^{-1}*r
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMPCGUpdateP_kernel()
{
  double beta = cSim.pPGMCGScalars[5];
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < cSim.atoms) {
    for (int d = 0; d < 3; d++) {
      int idx = 3*pos + d;
      cSim.pPGMCGp[idx] = cSim.pPGMPCGzpc[idx] + beta * cSim.pPGMCGp[idx];
    }
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMPCGUpdateP(gpuContext gpu)
{
  int blocks = gpu->sim.pgmCGnblocks;
  kPGMPCGUpdateP_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMPCGUpdateP");
}

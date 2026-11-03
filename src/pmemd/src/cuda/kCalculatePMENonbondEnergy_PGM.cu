#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kCalculatePMENonbondEnergy_PGM.cu - PGM direct-space electric field kernel
//
// Copied from kCalculatePMENonbondEnergy.cu pattern.
// Computes direct-space electric FIELD (charge-charge, charge-dipole, dipole-dipole)
// with Gaussian damping. Uses existing pmemd neighbor list.
//
// This is for the permanent field step (Step 3 of PGM workflow).
// Forces are computed later after SCF convergence.
//---------------------------------------------------------------------------------------------

// Debug: uncomment to disable VDW force (keep VDW energy) for EEL-only force comparison
// #define PGM_DEBUG_NO_VDW_FORCE

#include <cuda.h>
#include <cuda_fp16.h>
#ifndef PGM_FIELD_MINBLOCKS
#define PGM_FIELD_MINBLOCKS 8
#endif
#ifndef PGM_FORCE_MINBLOCKS
#define PGM_FORCE_MINBLOCKS 5
#endif
#define GPU_CPP
#include "gputypes.h"
#include "gpu.h"
#undef GPU_CPP
#include "ptxmacros.h"

#include "simulationConst.h"
CSIM_STO simulationConst cSim;

#ifdef PGM_DEBUG_PAIR_COUNTS
// Pair counters for diagnostics
__device__ unsigned long long int dPGMHomePairCount;
__device__ unsigned long long int dPGMNeighPairCount;
__device__ unsigned long long int dPGMNLEntryCount;
#endif

//---------------------------------------------------------------------------------------------
// SetkCalculatePMENonbondEnergy_PGMSim: copy constants to device
//---------------------------------------------------------------------------------------------
void SetkCalculatePMENonbondEnergy_PGMSim(gpuContext gpu)
{
  cudaError_t status;
  status = cudaMemcpyToSymbol(cSim, &gpu->sim, sizeof(simulationConst));
  RTERROR(status, "SetkCalculatePMENonbondEnergy_PGMSim: cudaMemcpyToSymbol failed");
}

#ifndef use_DPFP
//---------------------------------------------------------------------------------------------
// fasterfc for Ewald: reuse the same polynomial approximation
//---------------------------------------------------------------------------------------------
struct cudaERFC
{
  PMEFloat c0, c1, c2, c3, c4, c5, c6, c7, c8, c9;
};
static __constant__ cudaERFC cERFC;

void SetkCalculatePMENonbondEnergy_PGMERFC(double ewcoeff)
{
  cudaERFC c;
  c.c0 = (PMEFloat)(-1.6488499458192755E-006 * pow(ewcoeff, 10.0));
  c.c1 = (PMEFloat)( 2.9524665006554534E-005 * pow(ewcoeff,  9.0));
  c.c2 = (PMEFloat)(-2.3341951153749626E-004 * pow(ewcoeff,  8.0));
  c.c3 = (PMEFloat)( 1.0424943374047289E-003 * pow(ewcoeff,  7.0));
  c.c4 = (PMEFloat)(-2.5501426008983853E-003 * pow(ewcoeff,  6.0));
  c.c5 = (PMEFloat)( 3.1979939710877236E-004 * pow(ewcoeff,  5.0));
  c.c6 = (PMEFloat)( 2.7605379075746249E-002 * pow(ewcoeff,  4.0));
  c.c7 = (PMEFloat)(-1.4827402067461906E-001 * pow(ewcoeff,  3.0));
  c.c8 = (PMEFloat)(-9.1844764013203406E-001 * ewcoeff * ewcoeff);
  c.c9 = (PMEFloat)(-1.6279070384382459E+000 * ewcoeff);
  cudaError_t status;
  status = cudaMemcpyToSymbol(cERFC, &c, sizeof(cudaERFC));
  RTERROR(status, "SetkCalculatePMENonbondEnergy_PGMERFC: cudaMemcpyToSymbol failed");
}

static __forceinline__ __device__ float __internal_fmad(float a, float b, float c)
{
  return __fmaf_rn(a, b, c);
}

static __forceinline__ __device__ PMEFloat fasterfc(PMEFloat a)
{
  PMEFloat t, x = a;
  t = cERFC.c0;
  t = __internal_fmad(t, x, cERFC.c1);
  t = __internal_fmad(t, x, cERFC.c2);
  t = __internal_fmad(t, x, cERFC.c3);
  t = __internal_fmad(t, x, cERFC.c4);
  t = __internal_fmad(t, x, cERFC.c5);
  t = __internal_fmad(t, x, cERFC.c6);
  t = __internal_fmad(t, x, cERFC.c7);
  t = __internal_fmad(t, x, cERFC.c8);
  t = __internal_fmad(t, x, cERFC.c9);
  t = t * x;
  return exp2f(t);
}
#endif // !use_DPFP

//---------------------------------------------------------------------------------------------
// Field-only kernel instantiations (for SCF iterations)
// PME_ATOMS_PER_WARP = 32, 16, 8 × ortho/non-ortho
//---------------------------------------------------------------------------------------------
#define PGM_FIELD_ONLY

// ===== PME_ATOMS_PER_WARP = 32 =====
#define PME_ATOMS_PER_WARP (32)

#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectField32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL

__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectField32_kernel()
#include "kNLCPNE_PGM.h"

#undef PME_ATOMS_PER_WARP

// ===== PME_ATOMS_PER_WARP = 16 =====
#define PME_ATOMS_PER_WARP (16)

#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectField16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL

__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectField16_kernel()
#include "kNLCPNE_PGM.h"

#undef PME_ATOMS_PER_WARP

// ===== PME_ATOMS_PER_WARP = 8 =====
#define PME_ATOMS_PER_WARP (8)

#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectField8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL

__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectField8_kernel()
#include "kNLCPNE_PGM.h"

#undef PME_ATOMS_PER_WARP
#undef PGM_FIELD_ONLY

//---------------------------------------------------------------------------------------------
// Short-range field-only kernel instantiations (for PCG preconditioner inner CG)
// Uses scfLocalCut2 instead of cut2 for distance check.
// PME_ATOMS_PER_WARP = 32, 16, 8 × ortho/non-ortho
//---------------------------------------------------------------------------------------------
#define PGM_FIELD_ONLY
#define PGM_SHORT_RANGE

#define PME_ATOMS_PER_WARP (32)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldShort32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldShort32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (16)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldShort16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldShort16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (8)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldShort8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldShort8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#undef PGM_SHORT_RANGE
#undef PGM_FIELD_ONLY

//---------------------------------------------------------------------------------------------
// Cache-WRITE field-only kernel instantiations (permanent field: compute + store B1/B2)
// PME_ATOMS_PER_WARP = 32, 16, 8 × ortho/non-ortho
//---------------------------------------------------------------------------------------------
#define PGM_FIELD_ONLY
#define PGM_BFACTOR_WRITE

#define PME_ATOMS_PER_WARP (32)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheWrite32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheWrite32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (16)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheWrite16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheWrite16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (8)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheWrite8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheWrite8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#undef PGM_BFACTOR_WRITE
#undef PGM_FIELD_ONLY

//---------------------------------------------------------------------------------------------
// Cache-READ field-only kernel instantiations (SCF matvec: read B1/B2, skip erfc/exp)
// PME_ATOMS_PER_WARP = 32, 16, 8 × ortho/non-ortho
//---------------------------------------------------------------------------------------------
#define PGM_FIELD_ONLY
#define PGM_BFACTOR_READ

#define PME_ATOMS_PER_WARP (32)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheRead32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheRead32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (16)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheRead16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheRead16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (8)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheRead8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheRead8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#undef PGM_BFACTOR_READ
#undef PGM_FIELD_ONLY

//---------------------------------------------------------------------------------------------
// Cache-WRITE field-only kernel instantiations (permanent field: compute + store B1/B2)
// PME_ATOMS_PER_WARP = 32, 16, 8 × ortho/non-ortho
//---------------------------------------------------------------------------------------------
#define PGM_FIELD_ONLY
#define PGM_BFACTOR_WRITE16

#define PME_ATOMS_PER_WARP (32)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheWriteH32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheWriteH32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (16)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheWriteH16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheWriteH16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (8)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheWriteH8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheWriteH8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#undef PGM_BFACTOR_WRITE16
#undef PGM_FIELD_ONLY

//---------------------------------------------------------------------------------------------
// Cache-READ field-only kernel instantiations (SCF matvec: read B1/B2, skip erfc/exp)
// PME_ATOMS_PER_WARP = 32, 16, 8 × ortho/non-ortho
//---------------------------------------------------------------------------------------------
#define PGM_FIELD_ONLY
#define PGM_BFACTOR_READ16

#define PME_ATOMS_PER_WARP (32)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheReadH32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheReadH32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (16)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheReadH16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheReadH16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#define PME_ATOMS_PER_WARP (8)
#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEOrthoPGMDirectFieldCacheReadH8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FIELD_MINBLOCKS)
kCalcPMEPGMDirectFieldCacheReadH8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_ATOMS_PER_WARP

#undef PGM_BFACTOR_READ16
#undef PGM_FIELD_ONLY

//---------------------------------------------------------------------------------------------
// Force+field kernel instantiations (post-SCF: force + gradphi for chain-rule)
// PME_ATOMS_PER_WARP = 32, 16, 8 × ortho/non-ortho
//---------------------------------------------------------------------------------------------

// ===== PME_ATOMS_PER_WARP = 32 =====
#define PME_ATOMS_PER_WARP (32)

#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FORCE_MINBLOCKS)
kCalcPMEOrthoPGMDirectForce32_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL

__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FORCE_MINBLOCKS)
kCalcPMEPGMDirectForce32_kernel()
#include "kNLCPNE_PGM.h"

#undef PME_ATOMS_PER_WARP

// ===== PME_ATOMS_PER_WARP = 16 =====
#define PME_ATOMS_PER_WARP (16)

#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FORCE_MINBLOCKS)
kCalcPMEOrthoPGMDirectForce16_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL

__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FORCE_MINBLOCKS)
kCalcPMEPGMDirectForce16_kernel()
#include "kNLCPNE_PGM.h"

#undef PME_ATOMS_PER_WARP

// ===== PME_ATOMS_PER_WARP = 8 =====
#define PME_ATOMS_PER_WARP (8)

#define PME_IS_ORTHOGONAL
__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FORCE_MINBLOCKS)
kCalcPMEOrthoPGMDirectForce8_kernel()
#include "kNLCPNE_PGM.h"
#undef PME_IS_ORTHOGONAL

__global__ void
__launch_bounds__(PMENONBONDFORCES_THREADS_PER_BLOCK, PGM_FORCE_MINBLOCKS)
kCalcPMEPGMDirectForce8_kernel()
#include "kNLCPNE_PGM.h"

#undef PME_ATOMS_PER_WARP

//---------------------------------------------------------------------------------------------
// Force+field+virial kernel instantiations (for NPT)
// PME_ATOMS_PER_WARP = 32, 16, 8 × ortho/non-ortho
//---------------------------------------------------------------------------------------------
// Host dispatch functions
//---------------------------------------------------------------------------------------------
#ifdef PGM_DEBUG_PAIR_COUNTS
extern "C" void kPGMResetPairCounters()
{
  unsigned long long int zero = 0;
  cudaMemcpyToSymbol(dPGMHomePairCount,  &zero, sizeof(zero));
  cudaMemcpyToSymbol(dPGMNeighPairCount, &zero, sizeof(zero));
  cudaMemcpyToSymbol(dPGMNLEntryCount,   &zero, sizeof(zero));
}

extern "C" void kPGMReadPairCounters(unsigned long long int* home,
                                     unsigned long long int* neigh,
                                     unsigned long long int* nlEntries)
{
  cudaMemcpyFromSymbol(home,      dPGMHomePairCount,  sizeof(*home));
  cudaMemcpyFromSymbol(neigh,     dPGMNeighPairCount, sizeof(*neigh));
  cudaMemcpyFromSymbol(nlEntries, dPGMNLEntryCount,   sizeof(*nlEntries));
}
#endif

extern "C" void kCalculatePMEDirectField_PGM(gpuContext gpu)
{
  int nbBlocks = gpu->PMENonbondBlocks;
  int frcThreads = gpu->PMENonbondForcesThreadsPerBlock;
  cudaStream_t s = gpu->pgmStream;

  if (gpu->sim.NLAtomsPerWarp == 32) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectField32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectField32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else if (gpu->sim.NLAtomsPerWarp == 16) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectField16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectField16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectField8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectField8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  LAUNCHERROR("kCalculatePMEDirectField_PGM");
}

extern "C" void kCalculatePMEDirectForce_PGM(gpuContext gpu)
{
  int nbBlocks = gpu->PMENonbondBlocks;
  int frcThreads = gpu->PMENonbondForcesThreadsPerBlock;
  cudaStream_t s = gpu->pgmStream;

  if (gpu->sim.NLAtomsPerWarp == 32) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectForce32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectForce32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else if (gpu->sim.NLAtomsPerWarp == 16) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectForce16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectForce16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectForce8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectForce8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  LAUNCHERROR("kCalculatePMEDirectForce_PGM");
}

extern "C" void kCalculatePMEDirectFieldCacheWrite_PGM(gpuContext gpu)
{
  int nbBlocks = gpu->PMENonbondBlocks;
  int frcThreads = gpu->PMENonbondForcesThreadsPerBlock;
  cudaStream_t s = gpu->pgmStream;

  if (gpu->sim.NLAtomsPerWarp == 32) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheWrite32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheWrite32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else if (gpu->sim.NLAtomsPerWarp == 16) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheWrite16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheWrite16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheWrite8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheWrite8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  LAUNCHERROR("kCalculatePMEDirectFieldCacheWrite_PGM");
}

extern "C" void kCalculatePMEDirectFieldCacheRead_PGM(gpuContext gpu)
{
  int nbBlocks = gpu->PMENonbondBlocks;
  int frcThreads = gpu->PMENonbondForcesThreadsPerBlock;
  cudaStream_t s = gpu->pgmStream;

  if (gpu->sim.NLAtomsPerWarp == 32) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheRead32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheRead32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else if (gpu->sim.NLAtomsPerWarp == 16) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheRead16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheRead16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheRead8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheRead8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  LAUNCHERROR("kCalculatePMEDirectFieldCacheRead_PGM");
}

extern "C" void kCalculatePMEDirectFieldCacheWriteH_PGM(gpuContext gpu)
{
  int nbBlocks = gpu->PMENonbondBlocks;
  int frcThreads = gpu->PMENonbondForcesThreadsPerBlock;
  cudaStream_t s = gpu->pgmStream;

  if (gpu->sim.NLAtomsPerWarp == 32) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheWriteH32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheWriteH32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else if (gpu->sim.NLAtomsPerWarp == 16) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheWriteH16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheWriteH16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheWriteH8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheWriteH8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  LAUNCHERROR("kCalculatePMEDirectFieldCacheWriteH_PGM");
}

extern "C" void kCalculatePMEDirectFieldCacheReadH_PGM(gpuContext gpu)
{
  int nbBlocks = gpu->PMENonbondBlocks;
  int frcThreads = gpu->PMENonbondForcesThreadsPerBlock;
  cudaStream_t s = gpu->pgmStream;

  if (gpu->sim.NLAtomsPerWarp == 32) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheReadH32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheReadH32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else if (gpu->sim.NLAtomsPerWarp == 16) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheReadH16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheReadH16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldCacheReadH8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldCacheReadH8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  LAUNCHERROR("kCalculatePMEDirectFieldCacheReadH_PGM");
}

extern "C" void kCalculatePMEDirectFieldShort_PGM(gpuContext gpu)
{
  int nbBlocks = gpu->PMENonbondBlocks;
  int frcThreads = gpu->PMENonbondForcesThreadsPerBlock;
  cudaStream_t s = gpu->pgmStream;

  if (gpu->sim.NLAtomsPerWarp == 32) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldShort32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldShort32_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else if (gpu->sim.NLAtomsPerWarp == 16) {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldShort16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldShort16_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  else {
    if (gpu->sim.is_orthog) {
      kCalcPMEOrthoPGMDirectFieldShort8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    } else {
      kCalcPMEPGMDirectFieldShort8_kernel<<<nbBlocks, frcThreads, 0, s>>>();
    }
  }
  LAUNCHERROR("kCalculatePMEDirectFieldShort_PGM");
}


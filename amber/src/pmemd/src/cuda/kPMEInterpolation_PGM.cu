#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kPMEInterpolation_PGM.cu - PGM reciprocal space kernel dispatch
//
// Copied from kPMEInterpolation.cu pattern, simplified for PGM:
//   - Order-5 and order-6 (no order-4)
//   - No TI/AFE variants
//   - No virial
//   - Grid fill includes dipole contributions (kCQB_PGM.h)
//   - Grad sum extracts electric FIELD, not force (kPBGS_PGM.h)
//
// The standard PME grid operations (clear, reduce, FFT, scalar sum) are REUSED
// from the existing infrastructure. Only the fill and grad sum need PGM versions.
//---------------------------------------------------------------------------------------------

#include <cuda.h>
#include "gpu.h"
#include "ptxmacros.h"

#include "simulationConst.h"
CSIM_STO simulationConst cSim;

//---------------------------------------------------------------------------------------------
// SetkPMEInterpolation_PGMSim: copy constants to device constant memory
//---------------------------------------------------------------------------------------------
void SetkPMEInterpolation_PGMSim(gpuContext gpu)
{
  cudaError_t status;
  status = cudaMemcpyToSymbol(cSim, &gpu->sim, sizeof(simulationConst));
  RTERROR(status, "SetkPMEInterpolation_PGMSim: cudaMemcpyToSymbol failed");
}

//---------------------------------------------------------------------------------------------
// Grid fill kernel: charge + dipole scatter to single grid (order 5)
//---------------------------------------------------------------------------------------------
#define PME_ORDER 5
__global__ void
__launch_bounds__(P2M_THREADS_PER_BLOCK, P2M_BLOCKS_MULTIPLIER)
kPMEFillQMeshBuffer555_PGM_kernel()
#include "kCQB_PGM.h"
#undef PME_ORDER

//---------------------------------------------------------------------------------------------
// Grid fill kernel: charge + dipole scatter to single grid (order 6)
//---------------------------------------------------------------------------------------------
#define PME_ORDER 6
__global__ void
__launch_bounds__(P2M_THREADS_PER_BLOCK, P2M_BLOCKS_MULTIPLIER)
kPMEFillQMeshBuffer666_PGM_kernel()
#include "kCQB_PGM.h"
#undef PME_ORDER

//---------------------------------------------------------------------------------------------
// Gradient sum kernel (field-only): extract electric field from convolved grid (order 5)
//---------------------------------------------------------------------------------------------
#define PGM_FIELD_ONLY
#define PME_ORDER 5
__global__ void
__launch_bounds__(128, 4)
kPMEGradientSum555_PGM_kernel()
#include "kPBGS_PGM.h"
#undef PME_ORDER
#undef PGM_FIELD_ONLY

//---------------------------------------------------------------------------------------------
// Gradient sum kernel (field-only): extract electric field from convolved grid (order 6)
// Uses the batched gradient sum (kPBGS_PGM.h) which supports order 5 and 6.
// 128 threads per block, 32 atoms per batch, 4 warps per atom.
//---------------------------------------------------------------------------------------------
#define PGM_FIELD_ONLY
#define PME_ORDER 6
__global__ void
__launch_bounds__(128, 4)
kPMEGradientSum666_PGM_kernel()
#include "kPBGS_PGM.h"
#undef PME_ORDER
#undef PGM_FIELD_ONLY

//---------------------------------------------------------------------------------------------
// Gradient sum kernel (force mode): extract field + compute force (order 5)
//---------------------------------------------------------------------------------------------
#define PME_ORDER 5
__global__ void
__launch_bounds__(128, 4)
kPMEGradientSumForce555_PGM_kernel()
#include "kPBGS_PGM.h"
#undef PME_ORDER

//---------------------------------------------------------------------------------------------
// Gradient sum kernel (force mode): extract field + compute force (order 6)
// Same as field-only but additionally computes:
//   d2theta Hessian, frac→cart transform, F = -(q*gradphi + mu·Hessian)
//---------------------------------------------------------------------------------------------
#define PME_ORDER 6
__global__ void
__launch_bounds__(128, 4)
kPMEGradientSumForce666_PGM_kernel()
#include "kPBGS_PGM.h"
#undef PME_ORDER

//---------------------------------------------------------------------------------------------
// kPMEFillChargeGridBuffer_PGM: host function to fill the charge grid with PGM
//                                charge + dipole contributions (order 5 or 6)
//---------------------------------------------------------------------------------------------
extern "C" void kPMEFillChargeGridBuffer_PGM(gpuContext gpu)
{
  int nblocks = P2M_BLOCKS_MULTIPLIER * gpu->blocks;
  if (gpu->sim.pmeOrder == 5) {
    kPMEFillQMeshBuffer555_PGM_kernel<<<nblocks, P2M_THREADS_PER_BLOCK, 0, gpu->pgmStream>>>();
  } else {
    kPMEFillQMeshBuffer666_PGM_kernel<<<nblocks, P2M_THREADS_PER_BLOCK, 0, gpu->pgmStream>>>();
  }
  LAUNCHERROR("kPMEFillChargeGridBuffer_PGM");
}

//---------------------------------------------------------------------------------------------
// kPMEGradSum_PGM: host function to extract electric field from the convolved grid
//                   (mesh --> particle interpolation for field, not force)
//                   Uses 128-thread batched gradient sum for order 5 or 6.
//---------------------------------------------------------------------------------------------
extern "C" void kPMEGradSum_PGM(gpuContext gpu)
{
  int nblocks = (gpu->sim.atoms + 31) / 32;  // BATCHSIZE=32
  if (gpu->sim.pmeOrder == 5) {
    kPMEGradientSum555_PGM_kernel<<<nblocks, 128, 0, gpu->pgmStream>>>();
  } else {
    kPMEGradientSum666_PGM_kernel<<<nblocks, 128, 0, gpu->pgmStream>>>();
  }
  LAUNCHERROR("kPGMGradSum_PGM");
}

extern "C" void kPMEGradSumForce_PGM(gpuContext gpu)
{
  int nblocks = (gpu->sim.atoms + 31) / 32;  // BATCHSIZE=32
  if (gpu->sim.pmeOrder == 5) {
    kPMEGradientSumForce555_PGM_kernel<<<nblocks, 128, 0, gpu->pgmStream>>>();
  } else {
    kPMEGradientSumForce666_PGM_kernel<<<nblocks, 128, 0, gpu->pgmStream>>>();
  }
  LAUNCHERROR("kPMEGradSumForce_PGM");
}


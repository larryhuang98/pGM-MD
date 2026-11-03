#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kSelf_PGM.cu - PGM self-energy and self-field kernel
//
// No PME equivalent - this is unique to PGM.
//
// Self-energy (Gaussian Ewald):
//   U_self_chg = -ew_coeff/sqrt(pi) * sum_i(q_i^2)
//   U_self_dip = -(2/3)*ew_coeff^3/sqrt(pi) * sum_i(|mu_i|^2)
//
// Self-field (added to field accumulators):
//   E_self_i = -(4/3)*ew_coeff^3/sqrt(pi) * mu_i
//
// Note: B-factors depend ONLY on ew_coeff, NOT on r_gauss.
//---------------------------------------------------------------------------------------------

#include <cuda.h>
#include "gpu.h"
#include "ptxmacros.h"

#include "simulationConst.h"

CSIM_STO simulationConst cSim;

//---------------------------------------------------------------------------------------------
// SetkSelf_PGMSim: copy simulation constants to device constant memory
//---------------------------------------------------------------------------------------------
void SetkSelf_PGMSim(gpuContext gpu)
{
  cudaError_t status;
  status = cudaMemcpyToSymbol(cSim, &gpu->sim, sizeof(simulationConst));
  RTERROR(status, "SetkSelf_PGMSim: cudaMemcpyToSymbol failed");
}

//---------------------------------------------------------------------------------------------
// kPGMSelfField_kernel: compute self-field contribution for each atom
//
// E_self_i = -(4/3)*ew_coeff^3/sqrt(pi) * mu_i
//
// Accumulated into field accumulators (fixed-point).
// One thread per atom.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMSelfField_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < cSim.atoms) {

    // Read permanent dipole (SP packed format)
    PMEFloat2 dipXY = cSim.pPDipXYSP[pos];
    PMEFloat  dipZ  = cSim.pPDipZSP[pos];

    // Self-field prefactor: -(4/3) * ew_coeff^3 / sqrt(pi)
    // ew_coeff is already in cSim.ew_coeffSP
    PMEFloat ew  = cSim.ew_coeffSP;
    PMEFloat ew3 = ew * ew * ew;
    PMEFloat selfFieldPrefac = (PMEFloat)(-4.0 / 3.0) * ew3 * (PMEFloat)M_2_SQRTPI *
                               (PMEFloat)0.5;
    // Note: M_2_SQRTPI = 2/sqrt(pi), so 0.5*M_2_SQRTPI = 1/sqrt(pi)

    PMEFloat fx = selfFieldPrefac * dipXY.x;
    PMEFloat fy = selfFieldPrefac * dipXY.y;
    PMEFloat fz = selfFieldPrefac * dipZ;

    // Accumulate to field accumulators (fixed-point)
    long long int ifx = llrint((PMEDouble)fx * FORCESCALE);
    long long int ify = llrint((PMEDouble)fy * FORCESCALE);
    long long int ifz = llrint((PMEDouble)fz * FORCESCALE);

    atomicAdd((unsigned long long int*)&cSim.pPGMFieldXAccumulator[pos], llitoulli(ifx));
    atomicAdd((unsigned long long int*)&cSim.pPGMFieldYAccumulator[pos], llitoulli(ify));
    atomicAdd((unsigned long long int*)&cSim.pPGMFieldZAccumulator[pos], llitoulli(ifz));

    // Self phi_000: -(2*ew/sqrt(pi)) * q_i
    // (Self-potential from Ewald screening charge subtraction)
    PMEFloat qi = cSim.pPGMPermChg[pos];
    PMEFloat phi000 = (PMEFloat)(-1.0) * (PMEFloat)M_2_SQRTPI * ew * qi;
    // Note: M_2_SQRTPI = 2/sqrt(pi), so -M_2_SQRTPI*ew = -2*ew/sqrt(pi)
    long long int iphi = llrint((PMEDouble)phi000 * FORCESCALE);
    atomicAdd((unsigned long long int*)&cSim.pPGMPotentialAccumulator[pos], llitoulli(iphi));

    pos += blockDim.x * gridDim.x;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMSelfEnergy_kernel: compute self-energy (charge + dipole)
//
// U_self_chg = -ew_coeff/sqrt(pi) * sum_i(q_i^2)
// U_self_dip = -(2/3)*ew_coeff^3/sqrt(pi) * sum_i(|mu_i|^2)
//
// Uses warp-level reduction, then one atomicAdd per block to energy buffer.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMSelfEnergy_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  PMEDouble selfEneChg = (PMEDouble)0.0;
  PMEDouble selfEneDip = (PMEDouble)0.0;

  PMEFloat ew  = cSim.ew_coeffSP;
  PMEFloat ew3 = ew * ew * ew;

  while (pos < cSim.atoms) {

    // Charge self-energy: -ew_coeff/sqrt(pi) * q_i^2
    PMEFloat qi = cSim.pPGMPermChg[pos];
    selfEneChg += (PMEDouble)(qi * qi);

    // Dipole self-energy: -(2/3)*ew^3/sqrt(pi) * |mu_i|^2
    PMEFloat2 dipXY = cSim.pPDipXYSP[pos];
    PMEFloat  dipZ  = cSim.pPDipZSP[pos];
    PMEDouble mu2 = (PMEDouble)(dipXY.x * dipXY.x + dipXY.y * dipXY.y + dipZ * dipZ);
    selfEneDip += mu2;

    pos += blockDim.x * gridDim.x;
  }

  // Warp reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    selfEneChg += __shfl_down_sync(0xffffffff, selfEneChg, offset);
    selfEneDip += __shfl_down_sync(0xffffffff, selfEneDip, offset);
  }

  // Block reduction via shared memory
  __shared__ PMEDouble sEneChg[4];  // 128 threads / 32 = 4 warps
  __shared__ PMEDouble sEneDip[4];

  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;

  if (lane == 0) {
    sEneChg[warpId] = selfEneChg;
    sEneDip[warpId] = selfEneDip;
  }
  __syncthreads();

  // Final reduction by first warp
  if (threadIdx.x < 4) {
    selfEneChg = sEneChg[threadIdx.x];
    selfEneDip = sEneDip[threadIdx.x];
  } else {
    selfEneChg = (PMEDouble)0.0;
    selfEneDip = (PMEDouble)0.0;
  }

  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      selfEneChg += __shfl_down_sync(0xffffffff, selfEneChg, offset);
      selfEneDip += __shfl_down_sync(0xffffffff, selfEneDip, offset);
    }
  }

  // Thread 0 writes to energy buffer
  if (threadIdx.x == 0) {
    // Apply prefactors:
    //   chg: -ew_coeff / sqrt(pi)
    //   dip: -(2/3) * ew_coeff^3 / sqrt(pi)
    PMEDouble oneOverSqrtPi = (PMEDouble)M_2_SQRTPI * (PMEDouble)0.5; // 1/sqrt(pi)
    PMEDouble eneChg = -((PMEDouble)ew) * oneOverSqrtPi * selfEneChg;
    PMEDouble eneDip = -((PMEDouble)2.0 / (PMEDouble)3.0) *
                        (PMEDouble)ew3 * oneOverSqrtPi * selfEneDip;

    // Accumulate into EEL term (electrostatic energy)
    unsigned long long int iEneChg = llitoulli(llrint(eneChg * ENERGYSCALE));
    unsigned long long int iEneDip = llitoulli(llrint(eneDip * ENERGYSCALE));
    atomicAdd(cSim.pEELT, iEneChg);
    atomicAdd(cSim.pEELT, iEneDip);
  }
}

//---------------------------------------------------------------------------------------------
// kPGMClearFieldAccumulators_kernel: zero the field accumulators before each step
//---------------------------------------------------------------------------------------------
__global__ void
kPGMClearFieldAccumulators_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  while (pos < cSim.atoms) {
    cSim.pPGMFieldXAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMFieldYAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMFieldZAccumulator[pos] = (PMEAccumulator)0;
    cSim.pPGMPotentialAccumulator[pos] = (PMEAccumulator)0;
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMClearFieldAccumulators(gpuContext gpu)
{
  int blocks = (gpu->sim.atoms + 127) / 128;
  kPGMClearFieldAccumulators_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMClearFieldAccumulators");
}

extern "C" void kPGMSelfField(gpuContext gpu)
{
  int blocks = (gpu->sim.atoms + 127) / 128;
  kPGMSelfField_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMSelfField");
}

//---------------------------------------------------------------------------------------------
// kPGMClearAndSelfField_kernel: fused clear + self-field (no atomicAdd needed)
//
// Directly writes self-field values to accumulators (they were zero, so store = clear+add).
// Saves one kernel launch per matvec call (~8 per step in SCF).
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMClearAndSelfField_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Self-field prefactor (constant across all atoms, hoisted out of loop)
  PMEFloat ew  = cSim.ew_coeffSP;
  PMEFloat ew3 = ew * ew * ew;
  PMEFloat selfFieldPrefac = (PMEFloat)(-4.0 / 3.0) * ew3 * (PMEFloat)M_2_SQRTPI *
                             (PMEFloat)0.5;

  while (pos < cSim.atoms) {
    // Self-field: -(4/3) * ew^3 / sqrt(pi) * mu_i
    PMEFloat2 dipXY = cSim.pPDipXYSP[pos];
    PMEFloat  dipZ  = cSim.pPDipZSP[pos];

    // Direct store (no atomicAdd needed since we're the only writer)
    cSim.pPGMFieldXAccumulator[pos] = llrint((PMEDouble)(selfFieldPrefac * dipXY.x) * FORCESCALE);
    cSim.pPGMFieldYAccumulator[pos] = llrint((PMEDouble)(selfFieldPrefac * dipXY.y) * FORCESCALE);
    cSim.pPGMFieldZAccumulator[pos] = llrint((PMEDouble)(selfFieldPrefac * dipZ)    * FORCESCALE);

    // Self phi_000: -(2*ew/sqrt(pi)) * q_i
    PMEFloat qi = cSim.pPGMPermChg[pos];
    PMEFloat phi000 = (PMEFloat)(-1.0) * (PMEFloat)M_2_SQRTPI * ew * qi;
    cSim.pPGMPotentialAccumulator[pos] = llrint((PMEDouble)phi000 * FORCESCALE);

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

extern "C" void kPGMClearAndSelfField(gpuContext gpu)
{
  int blocks = (gpu->sim.stride + 127) / 128;
  kPGMClearAndSelfField_kernel<<<blocks, 128, 0, gpu->pgmStream>>>();
  LAUNCHERROR("kPGMClearAndSelfField");
}

//---------------------------------------------------------------------------------------------
// kPGMSaveAndZeroCharges_kernel: fused save charges/dipoles + zero charges for SCF
//
// Saves: pPGMPermChg → pPGMSavedPermChg, pAtomChargeSP → pPGMSavedAtomChgSP,
//        pPermDipX/Y/Z → pPGMSavedPermDipX/Y/Z
// Zeros: pPGMPermChg, pAtomChargeSP
// Replaces 5 cudaMemcpy D2D + 2 cudaMemset = 7 kernel launches → 1.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 4)
kPGMSaveAndZeroCharges_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  while (pos < cSim.stride) {
    // Save and zero charges
    cSim.pPGMSavedPermChg[pos]    = cSim.pPGMPermChg[pos];
    cSim.pPGMPermChg[pos]         = (double)0.0;
    cSim.pPGMSavedAtomChgSP[pos]  = cSim.pAtomChargeSP[pos];
    cSim.pAtomChargeSP[pos]       = (PMEFloat)0.0;
    // Save perm dipoles
    cSim.pPGMSavedPermDipX[pos]   = cSim.pPermDipX[pos];
    cSim.pPGMSavedPermDipY[pos]   = cSim.pPermDipY[pos];
    cSim.pPGMSavedPermDipZ[pos]   = cSim.pPermDipZ[pos];
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMSaveAndZeroCharges(gpuContext gpu)
{
  int blocks = (gpu->sim.stride + 127) / 128;
  kPGMSaveAndZeroCharges_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMSaveAndZeroCharges");
}

//---------------------------------------------------------------------------------------------
// kPGMRestoreAndCopyTotalDip_kernel: fused restore charges/dipoles + copy total dips
//
// Restores: pPGMSavedPermChg → pPGMPermChg, pPGMSavedAtomChgSP → pAtomChargeSP,
//           pPGMSavedPermDipX/Y/Z → pPermDipX/Y/Z
// Copies:   pTDipXYSP → pPDipXYSP, pTDipZSP → pPDipZSP,
//           pPGMFracTotDipX/Y/Z → pPGMFracDipX/Y/Z
// Replaces 5 + 5 = 10 cudaMemcpy D2D → 1 kernel launch.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 4)
kPGMRestoreAndCopyTotalDip_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  while (pos < cSim.stride) {
    // Restore charges
    cSim.pPGMPermChg[pos]     = cSim.pPGMSavedPermChg[pos];
    cSim.pAtomChargeSP[pos]   = cSim.pPGMSavedAtomChgSP[pos];
    // Restore perm dipoles
    cSim.pPermDipX[pos]       = cSim.pPGMSavedPermDipX[pos];
    cSim.pPermDipY[pos]       = cSim.pPGMSavedPermDipY[pos];
    cSim.pPermDipZ[pos]       = cSim.pPGMSavedPermDipZ[pos];
    // Copy total dipoles to active buffers
    cSim.pPDipXYSP[pos]       = cSim.pTDipXYSP[pos];
    cSim.pPDipZSP[pos]        = cSim.pTDipZSP[pos];
    cSim.pPGMFracDipX[pos]    = cSim.pPGMFracTotDipX[pos];
    cSim.pPGMFracDipY[pos]    = cSim.pPGMFracTotDipY[pos];
    cSim.pPGMFracDipZ[pos]    = cSim.pPGMFracTotDipZ[pos];
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMRestoreAndCopyTotalDip(gpuContext gpu)
{
  int blocks = (gpu->sim.stride + 127) / 128;
  kPGMRestoreAndCopyTotalDip_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMRestoreAndCopyTotalDip");
}

extern "C" void kPGMSelfEnergy(gpuContext gpu)
{
  int blocks = (gpu->sim.atoms + 127) / 128;
  kPGMSelfEnergy_kernel<<<blocks, 128>>>();
  LAUNCHERROR("kPGMSelfEnergy");
}

//---------------------------------------------------------------------------------------------
// kPGMComputeEnergy_kernel: compute total PGM electrostatic energy from accumulated phi
//
// Energy formula (matches CPU pGM_energy_force_virial):
//   ene_chg = 0.5 * sum(q_i * phi_000_accum[i])
//   ene_dip = 0.5 * sum(perm_dip_i · gradphi_accum[i])
//
// phi_000_accum and gradphi_accum contain contributions from self + direct + recip.
// gradphi is stored in pPGMFieldX/Y/ZAccumulator (same sign convention as CPU phi(2:4)).
// perm_dip is read from pPermDipX/Y/Z (DP buffers, NOT overwritten by total dip copy).
//
// Writes total energy (ene_chg + ene_dip) * coulomb_const to pEELT (in kcal/mol).
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 8)
kPGMComputeEnergy_kernel(double coulomb_const)
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  PMEDouble localEne = (PMEDouble)0.0;
  PMEDouble inv_fscale = (PMEDouble)1.0 / (PMEDouble)FORCESCALE;

  while (pos < cSim.atoms) {
    // Read charge
    PMEDouble qi = cSim.pPGMPermChg[pos];

    // Read permanent dipoles (DP, in spatial order, NOT overwritten by total dip)
    PMEDouble pdx = cSim.pPermDipX[pos];
    PMEDouble pdy = cSim.pPermDipY[pos];
    PMEDouble pdz = cSim.pPermDipZ[pos];

    // Read phi_000 accumulator
    PMEDouble phi000 = (PMEDouble)cSim.pPGMPotentialAccumulator[pos] * inv_fscale;

    // Read gradphi accumulators
    PMEDouble gx = (PMEDouble)cSim.pPGMFieldXAccumulator[pos] * inv_fscale;
    PMEDouble gy = (PMEDouble)cSim.pPGMFieldYAccumulator[pos] * inv_fscale;
    PMEDouble gz = (PMEDouble)cSim.pPGMFieldZAccumulator[pos] * inv_fscale;

    // Energy: 0.5 * (q*phi_000 + perm_dip · gradphi) * coulomb_const
    localEne += (PMEDouble)0.5 * (qi * phi000 + pdx * gx + pdy * gy + pdz * gz) * coulomb_const;

    pos += blockDim.x * gridDim.x;
  }

  // Warp reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    localEne += __shfl_down_sync(0xffffffff, localEne, offset);
  }

  // Block reduction via shared memory
  __shared__ PMEDouble sEne[4];  // 128 threads / 32 = 4 warps
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;

  if (lane == 0) {
    sEne[warpId] = localEne;
  }
  __syncthreads();

  if (threadIdx.x < 4) {
    localEne = sEne[threadIdx.x];
  } else {
    localEne = (PMEDouble)0.0;
  }

  if (warpId == 0) {
    for (int offset = 2; offset > 0; offset >>= 1) {
      localEne += __shfl_down_sync(0xffffffff, localEne, offset);
    }
  }

  // Thread 0 writes to energy buffer
  if (threadIdx.x == 0) {
    unsigned long long int iEne = llitoulli(llrint(localEne * ENERGYSCALE));
    atomicAdd(cSim.pEELT, iEne);
  }
}

extern "C" void kPGMComputeEnergy(gpuContext gpu, double coulomb_const)
{
  int blocks = (gpu->sim.atoms + 127) / 128;
  kPGMComputeEnergy_kernel<<<blocks, 128, 0, gpu->pgmStream>>>(coulomb_const);
  LAUNCHERROR("kPGMComputeEnergy");
}

//---------------------------------------------------------------------------------------------
// kPGMScaleForces_kernel: multiply all force accumulator entries by coulomb_const.
//
// PGM electrostatic forces are computed in raw e^2/A^2 units (charges not pre-scaled).
// The GPU integrator (kUpdate) reads forces directly from pForceAccumulator, so we must
// scale them to kcal/mol/A BEFORE integration.  VDW forces are pre-divided by coulomb_const
// in kNLCPNE_PGM.h, so they also come out correct after this multiplication.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(128, 4)
kPGMScaleForces_kernel(double coulomb_const)
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  while (pos < cSim.stride) {
    long long int fx = cSim.pNBForceXAccumulator[pos];
    long long int fy = cSim.pNBForceYAccumulator[pos];
    long long int fz = cSim.pNBForceZAccumulator[pos];
    cSim.pNBForceXAccumulator[pos] = llrint((double)fx * coulomb_const);
    cSim.pNBForceYAccumulator[pos] = llrint((double)fy * coulomb_const);
    cSim.pNBForceZAccumulator[pos] = llrint((double)fz * coulomb_const);
    pos += blockDim.x * gridDim.x;
  }
}

extern "C" void kPGMScaleForces(gpuContext gpu, double coulomb_const)
{
  int blocks = (gpu->sim.stride + 127) / 128;
  kPGMScaleForces_kernel<<<blocks, 128, 0, gpu->pgmStream>>>(coulomb_const);
  LAUNCHERROR("kPGMScaleForces");
}


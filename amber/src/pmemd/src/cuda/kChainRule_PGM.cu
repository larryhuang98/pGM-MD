#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kChainRule_PGM.cu - Chain-rule force from covalent dipole direction changes
//
// No PME equivalent - this is unique to PGM.
//
// When atom i moves, the directions of its covalent bond vectors change,
// which changes the global-frame dipole computed from local (scalar) bond dipoles.
// This creates a "torque-like" force on both atoms of each bond.
//
// For each atom i, for each covalent bond (i -> j_atm):
//   del = pos(j_atm) - pos(i)  (minimum-image PBC)
//   cov_dipj = bond_dip_mag(i->j_atm) / |del|
//   cov_dipk = bond_dip_mag(j_atm->i) / |del|   (reverse bond)
//   proj_i  = (del . gradphi(i)) / |del|^2
//   proj_j  = (del . gradphi(j)) / |del|^2
//
//   F_chain(i) += cov_dipj * (gradphi(i) - del * proj_i)
//               - cov_dipk * (gradphi(j) - del * proj_j)
//
// The term (gradphi - del * proj) is the component of gradphi perpendicular
// to the bond vector, which drives the rotation of the bond dipole direction.
//
// CSR bond topology is in ORIGINAL atom order.
// Field accumulators, force accumulators, and coordinates are in IMAGE (NL) order.
// Uses pImageAtom (image->orig) and pImageAtomLookup (orig->image) to translate.
//
// Reads: pPGMFieldAccumulator (total gradphi from direct + recip + self)
// Reads: pImageX/Y/Z (coordinates, double precision)
// Reads: pPGMCovalentPtr/Atm/Dip (CSR bond topology, ORIGINAL order)
// Writes: pNBForceAccumulator (standard force buffers)
//
// Ref: pGM_energy_force_virial() in pGM_interface.F90, lines 1256-1305
//---------------------------------------------------------------------------------------------

#include <cuda.h>
#include "gpu.h"
#include "ptxmacros.h"

#include "simulationConst.h"
CSIM_STO simulationConst cSim;

//---------------------------------------------------------------------------------------------
// SetkChainRule_PGMSim: copy simulation constants to device constant memory
//---------------------------------------------------------------------------------------------
void SetkChainRule_PGMSim(gpuContext gpu)
{
  cudaError_t status;
  status = cudaMemcpyToSymbol(cSim, &gpu->sim, sizeof(simulationConst));
  RTERROR(status, "SetkChainRule_PGMSim: cudaMemcpyToSymbol failed");
}

//---------------------------------------------------------------------------------------------
// kPGMChainRuleForce_kernel: compute chain-rule force from covalent dipole direction changes
//
// One thread per atom (image order). Each atom loops over its covalent bonds
// (CSR format in original order, typically 1-4 bonds per atom).
// For each bond, performs a short linear search in the bonded atom's CSR to find the
// reverse bond magnitude.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMChainRuleForce_kernel()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // image-order slot
  if (idx >= cSim.atoms) return;

  // Map image slot to original atom index for CSR access
  int orig = cSim.pImageAtom[idx];

  // Read total gradphi at this atom from field accumulators (fixed-point -> double)
  double fscale_inv = 1.0 / (double)(1LL << 40);
  double gradphi_ix = (double)cSim.pPGMFieldXAccumulator[idx] * fscale_inv;
  double gradphi_iy = (double)cSim.pPGMFieldYAccumulator[idx] * fscale_inv;
  double gradphi_iz = (double)cSim.pPGMFieldZAccumulator[idx] * fscale_inv;

  // This atom's coordinates (double precision, image order)
  double xI = cSim.pImageX[idx];
  double yI = cSim.pImageY[idx];
  double zI = cSim.pImageZ[idx];

  // CSR range for bonds of this atom (CSR in original order)
  int start = cSim.pPGMCovalentPtr[orig];
  int end   = cSim.pPGMCovalentPtr[orig + 1];

  double force_x = 0.0;
  double force_y = 0.0;
  double force_z = 0.0;

  for (int jbond = start; jbond < end; jbond++) {
    int j_orig = cSim.pPGMCovalentAtm[jbond];          // bonded atom, original order
    int j_slot = cSim.pImageAtomLookup[j_orig];         // bonded atom, image slot
    double dip_mag_j = cSim.pPGMCovalentDip[jbond];

    // Bond vector: del = pos(j) - pos(i), using image-order coordinates
    double xJ = cSim.pImageX[j_slot];
    double yJ = cSim.pImageY[j_slot];
    double zJ = cSim.pImageZ[j_slot];

    double delx = xJ - xI;
    double dely = yJ - yI;
    double delz = zJ - zI;

    // Minimum-image PBC: frac -> round -> cart (general non-orthogonal)
    double dfx = delx*cSim.recip[0][0] + dely*cSim.recip[1][0] + delz*cSim.recip[2][0];
    double dfy =                          dely*cSim.recip[1][1] + delz*cSim.recip[2][1];
    double dfz =                                                   delz*cSim.recip[2][2];
    dfx -= round(dfx);
    dfy -= round(dfy);
    dfz -= round(dfz);
    delx = cSim.ucellf[0][0]*dfx + cSim.ucellf[0][1]*dfy + cSim.ucellf[0][2]*dfz;
    dely =                          cSim.ucellf[1][1]*dfy + cSim.ucellf[1][2]*dfz;
    delz =                                                   cSim.ucellf[2][2]*dfz;

    double dij2 = delx*delx + dely*dely + delz*delz;
    double distance = sqrt(dij2);

    // Find reverse bond: j_orig -> orig (linear search in j_orig's CSR, typically 1-4 entries)
    int k_start = cSim.pPGMCovalentPtr[j_orig];
    int k_end   = cSim.pPGMCovalentPtr[j_orig + 1];
    double dip_mag_k = 0.0;
    for (int kbond = k_start; kbond < k_end; kbond++) {
      if (cSim.pPGMCovalentAtm[kbond] == orig) {   // compare original indices
        dip_mag_k = cSim.pPGMCovalentDip[kbond];
        break;
      }
    }

    // Read gradphi at bonded atom j (image-order field accumulator)
    double gradphi_jx = (double)cSim.pPGMFieldXAccumulator[j_slot] * fscale_inv;
    double gradphi_jy = (double)cSim.pPGMFieldYAccumulator[j_slot] * fscale_inv;
    double gradphi_jz = (double)cSim.pPGMFieldZAccumulator[j_slot] * fscale_inv;

    // Projected dot products (del . gradphi / |del|^2)
    double proj_i = (delx*gradphi_ix + dely*gradphi_iy + delz*gradphi_iz) / dij2;
    double proj_j = (delx*gradphi_jx + dely*gradphi_jy + delz*gradphi_jz) / dij2;

    // Scaled dipole magnitudes (bond_dip / |del|)
    double cov_dipj = dip_mag_j / distance;
    double cov_dipk = dip_mag_k / distance;

    // Chain-rule force on atom i from this bond:
    //   cov_dipj * perp(gradphi_i) - cov_dipk * perp(gradphi_j)
    // where perp(v) = v - del * (del.v / |del|^2) projects out the parallel component
    force_x += cov_dipj * (gradphi_ix - delx * proj_i)
             - cov_dipk * (gradphi_jx - delx * proj_j);
    force_y += cov_dipj * (gradphi_iy - dely * proj_i)
             - cov_dipk * (gradphi_jy - dely * proj_j);
    force_z += cov_dipj * (gradphi_iz - delz * proj_i)
             - cov_dipk * (gradphi_jz - delz * proj_j);
  }

  // Write force to standard accumulator (fixed-point atomicAdd, image-order slot)
  if (force_x != 0.0 || force_y != 0.0 || force_z != 0.0) {
    long long int ifx = llrint(force_x * (double)(1LL << 40));
    long long int ify = llrint(force_y * (double)(1LL << 40));
    long long int ifz = llrint(force_z * (double)(1LL << 40));
    atomicAdd((unsigned long long int*)&cSim.pNBForceXAccumulator[idx],
              llitoulli(ifx));
    atomicAdd((unsigned long long int*)&cSim.pNBForceYAccumulator[idx],
              llitoulli(ify));
    atomicAdd((unsigned long long int*)&cSim.pNBForceZAccumulator[idx],
              llitoulli(ifz));
  }
}

//---------------------------------------------------------------------------------------------
// Host dispatch function
//---------------------------------------------------------------------------------------------
extern "C" void kPGMChainRuleForce(gpuContext gpu)
{
  int nblocks = (gpu->sim.atoms + 255) / 256;
  kPGMChainRuleForce_kernel<<<nblocks, 256, 0, gpu->pgmStream>>>();
  LAUNCHERROR("kPGMChainRuleForce");
}

//---------------------------------------------------------------------------------------------
// kPGMNB14GVDW_kernel: Gaussian VDW for 1-4 pairs (igvdw = 1).
//
// With GVDW the 1-4 Lennard-Jones coefficients of the AMBER bonded work units are zeroed in
// pgm_gpu_nb14_gvdw_setup_ (pgm_gpu.cpp) and the 1-4 VDW is computed here, with the same
// functional form as the GVDW direct-space pairs (kNLCPNE_PGM.h, PGMPairVDW):
//   U = A exp(-b y^2) - C6 F(y) / r^6,   y = kappa r   (Slater form: A exp(-b y)).
// One thread per 1-4 pair.  Pair atoms are in ORIGINAL order; coef = {A, C6, kappa} per pair.
// Forces go to the bonded force accumulator, energy to the 1-4 VDW energy slot.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMNB14GVDW_kernel(int npairs, const int* __restrict__ ij, const double* __restrict__ coef)
{
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  double e = 0.0;
  if (p < npairs) {
    int i_slot = cSim.pImageAtomLookup[ij[2 * p]];
    int j_slot = cSim.pImageAtomLookup[ij[2 * p + 1]];
    double A = coef[3 * p], C6 = coef[3 * p + 1], kap = coef[3 * p + 2];
    double delx = cSim.pImageX[j_slot] - cSim.pImageX[i_slot];
    double dely = cSim.pImageY[j_slot] - cSim.pImageY[i_slot];
    double delz = cSim.pImageZ[j_slot] - cSim.pImageZ[i_slot];
    double dfx = delx*cSim.recip[0][0] + dely*cSim.recip[1][0] + delz*cSim.recip[2][0];
    double dfy =                          dely*cSim.recip[1][1] + delz*cSim.recip[2][1];
    double dfz =                                                   delz*cSim.recip[2][2];
    dfx -= round(dfx);
    dfy -= round(dfy);
    dfz -= round(dfz);
    delx = cSim.ucellf[0][0]*dfx + cSim.ucellf[0][1]*dfy + cSim.ucellf[0][2]*dfz;
    dely =                          cSim.ucellf[1][1]*dfy + cSim.ucellf[1][2]*dfz;
    delz =                                                   cSim.ucellf[2][2]*dfz;
    double r2 = delx*delx + dely*dely + delz*delz;
    double rinv = rsqrt(r2);
    double r2inv = rinv * rinv;
    double r6 = r2inv * r2inv * r2inv;
    double y = kap / rinv, y2 = y * y;
    const double isp = 0.56418958354775628695;          // 1/sqrt(pi)
    double ey  = exp(-y2);
    double gx  = (4.0 / 3.0) * isp * y * y2 * ey;
    double ax  = erf(y) - 2.0 * isp * y * ey - gx;
    double fx  = ax * ax + 0.5 * gx * gx;
    double apx = (8.0 / 3.0) * isp * y2 * y2 * ey;
    double gpx = 4.0 * isp * y2 * (1.0 - (2.0 / 3.0) * y2) * ey;
    double fpx = 2.0 * ax * apx + gx * gpx;
    double b = (double)cSim.pgmBRepScale, erep, df;
    if (cSim.pgmGVDWRepForm == 0) {
      erep = A * exp(-b * y2);
      df   = 2.0 * b * kap * kap * erep;
    }
    else {
      erep = A * exp(-b * y);
      df   = b * kap * erep * rinv;
    }
    if (y > 0.01) {
      e   = erep - C6 * fx * r6;
      df += C6 * (kap * fpx * rinv - 6.0 * fx * r2inv) * r6;
    }
    else {
      e   = erep - C6 * (8.0 / (9.0 * 3.14159265358979323846)) * kap * kap * kap * kap * kap * kap;
    }
    // df = -(dU/dr)/r; force on j is +df*del, on i is -df*del
    long long int ifx = llrint(df * delx * FORCESCALE);
    long long int ify = llrint(df * dely * FORCESCALE);
    long long int ifz = llrint(df * delz * FORCESCALE);
    atomicAdd((unsigned long long int*)&cSim.pForceXAccumulator[j_slot], llitoulli(ifx));
    atomicAdd((unsigned long long int*)&cSim.pForceYAccumulator[j_slot], llitoulli(ify));
    atomicAdd((unsigned long long int*)&cSim.pForceZAccumulator[j_slot], llitoulli(ifz));
    atomicAdd((unsigned long long int*)&cSim.pForceXAccumulator[i_slot], llitoulli(-ifx));
    atomicAdd((unsigned long long int*)&cSim.pForceYAccumulator[i_slot], llitoulli(-ify));
    atomicAdd((unsigned long long int*)&cSim.pForceZAccumulator[i_slot], llitoulli(-ifz));
  }
  // block reduction of the energy
  __shared__ double se[256];
  se[threadIdx.x] = e;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) se[threadIdx.x] += se[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0 && se[0] != 0.0) {
    atomicAdd(cSim.pENB14, llitoulli(llrint(se[0] * ENERGYSCALE)));
  }
}

extern "C" void kPGMNB14GVDW(gpuContext gpu, int npairs, const int* ij, const double* coef)
{
  if (npairs <= 0) return;
  int nblocks = (npairs + 255) / 256;
  kPGMNB14GVDW_kernel<<<nblocks, 256, 0, gpu->pgmStream>>>(npairs, ij, coef);
  LAUNCHERROR("kPGMNB14GVDW");
}

#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kLocalToGlobal_PGM.cu - Convert local-frame bond dipoles to global Cartesian dipoles
//
// PGM permanent dipoles are defined as scalar magnitudes along covalent bonds.
// Each timestep, this kernel computes the global-frame 3D dipole at each atom:
//
//   mu(i) = sum_{bonds j of atom i} covalent_dip[j] * unit_vec(pos[other[j]] - pos[i])
//
// Uses the CSR (Compressed Sparse Row) bond topology:
//   pPGMCovalentPtr[atoms+1] - row pointers
//   pPGMCovalentAtm[ncovalent] - bonded atom indices
//   pPGMCovalentDip[ncovalent] - bond dipole magnitudes
//
// CSR is in ORIGINAL atom order. Coordinates (pImageX/Y/Z) and output dipole
// buffers are in IMAGE (NL-sorted) order. The kernel uses pImageAtomLookup
// to map original atom indices to image slots for coordinate reads and writes.
//
// Bond vectors use minimum-image PBC (recip + round + ucellf) for non-orthogonal cells.
// Output goes to pPermDipX/Y/Z (double) and pPDipXYSP/pPDipZSP (single).
//
// Also computes fractional dipoles for reciprocal grid fill:
//   frac_dip_k = nfft_k * sum_i recipf[i][k] * cart_dip_i  (recipf^T convention)
//---------------------------------------------------------------------------------------------

#include <cuda.h>
#include "gpu.h"
#include "ptxmacros.h"

#include "simulationConst.h"
CSIM_STO simulationConst cSim;

//---------------------------------------------------------------------------------------------
void SetkLocalToGlobal_PGMSim(gpuContext gpu)
{
  cudaError_t status;
  status = cudaMemcpyToSymbol(cSim, &gpu->sim, sizeof(simulationConst));
  RTERROR(status, "SetkLocalToGlobal_PGMSim: cudaMemcpyToSymbol failed");
}

//---------------------------------------------------------------------------------------------
// Kernel: one thread per atom, compute global dipole from CSR bond topology
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kLocalToGlobal_PGM_kernel()
{
  int orig = blockIdx.x * blockDim.x + threadIdx.x;  // original atom index
  if (orig >= cSim.atoms) return;

  // Map original atom to image slot for coordinate and output access
  int slot = cSim.pImageAtomLookup[orig];

  // Load this atom's absolute Cartesian coordinates from pImageX/Y/Z (image order)
  double xI = cSim.pImageX[slot];
  double yI = cSim.pImageY[slot];
  double zI = cSim.pImageZ[slot];

  // CSR range for bonds of this atom (CSR is in original order)
  int start = cSim.pPGMCovalentPtr[orig];
  int end   = cSim.pPGMCovalentPtr[orig + 1];

  double dipx = 0.0;
  double dipy = 0.0;
  double dipz = 0.0;

  for (int jbond = start; jbond < end; jbond++) {
    int j_orig = cSim.pPGMCovalentAtm[jbond];        // bonded atom in original order
    int j_slot = cSim.pImageAtomLookup[j_orig];       // bonded atom's image slot
    double dip_mag = cSim.pPGMCovalentDip[jbond];

    // Load bonded atom absolute Cartesian coordinates (image order)
    double xJ = cSim.pImageX[j_slot];
    double yJ = cSim.pImageY[j_slot];
    double zJ = cSim.pImageZ[j_slot];

    // Bond vector in Cartesian
    double dx = xJ - xI;
    double dy = yJ - yI;
    double dz = zJ - zI;

    // Minimum-image PBC: convert to fractional, round, convert back
    // Uses kReImageCoord convention: frac_k = sum_i recip[i][k] * cart_i
    // (recip is lower-triangular in C: recip[i][k] non-zero for i >= k)
#ifndef PME_IS_ORTHOGONAL
    double dfx = dx*cSim.recip[0][0] + dy*cSim.recip[1][0] + dz*cSim.recip[2][0];
    double dfy =                        dy*cSim.recip[1][1] + dz*cSim.recip[2][1];
    double dfz =                                               dz*cSim.recip[2][2];
    dfx -= round(dfx);
    dfy -= round(dfy);
    dfz -= round(dfz);
    dx = cSim.ucellf[0][0]*dfx + cSim.ucellf[0][1]*dfy + cSim.ucellf[0][2]*dfz;
    dy =                         cSim.ucellf[1][1]*dfy + cSim.ucellf[1][2]*dfz;
    dz =                                                 cSim.ucellf[2][2]*dfz;
#else
    dx -= round(dx * cSim.recip[0][0]) * cSim.a;
    dy -= round(dy * cSim.recip[1][1]) * cSim.b;
    dz -= round(dz * cSim.recip[2][2]) * cSim.c;
#endif

    // Unit vector
    double rinv = rsqrt(dx*dx + dy*dy + dz*dz);
    dx *= rinv;
    dy *= rinv;
    dz *= rinv;

    // Accumulate dipole contribution
    dipx += dip_mag * dx;
    dipy += dip_mag * dy;
    dipz += dip_mag * dz;
  }

  // Store global-frame permanent dipole (double precision) at image slot
  cSim.pPermDipX[slot] = dipx;
  cSim.pPermDipY[slot] = dipy;
  cSim.pPermDipZ[slot] = dipz;

  // Store single-precision copy for direct-space kernel
  PMEFloat2 pdxy;
  pdxy.x = (PMEFloat)dipx;
  pdxy.y = (PMEFloat)dipy;
  cSim.pPDipXYSP[slot] = pdxy;
  cSim.pPDipZSP[slot]  = (PMEFloat)dipz;

  // Compute fractional dipole for reciprocal grid fill:
  // frac_dip_k = nfft_k * sum_i recipf[i][k] * cart_dip_i  (recipf^T convention)
  // recipf is lower-triangular: recipf[i][k]=0 for i<k
  // So: frac_x uses full column 0, frac_y uses column 1 (rows 1,2), frac_z uses [2][2] only
  PMEFloat fdx = (PMEFloat)(cSim.nfft1 * (cSim.recipf[0][0]*dipx +
                                            cSim.recipf[1][0]*dipy +
                                            cSim.recipf[2][0]*dipz));
  PMEFloat fdy = (PMEFloat)(cSim.nfft2 * (cSim.recipf[1][1]*dipy +
                                            cSim.recipf[2][1]*dipz));
  PMEFloat fdz = (PMEFloat)(cSim.nfft3 * (cSim.recipf[2][2]*dipz));
  cSim.pPGMFracDipX[slot] = fdx;
  cSim.pPGMFracDipY[slot] = fdy;
  cSim.pPGMFracDipZ[slot] = fdz;
}

//---------------------------------------------------------------------------------------------
// Host dispatch
//---------------------------------------------------------------------------------------------
extern "C" void kLocalToGlobal_PGM(gpuContext gpu)
{
  int nblocks = (gpu->sim.atoms + 255) / 256;
  kLocalToGlobal_PGM_kernel<<<nblocks, 256>>>();
  LAUNCHERROR("kLocalToGlobal_PGM");
}

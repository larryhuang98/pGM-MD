#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// kNLRemap_PGM.cu - GPU-side NL remap for PGM data
//
// Replaces CPU scatter loops that re-ordered PGM prmtop data (charges, r_gauss,
// polarizabilities, preconditioner) from original atom order to NL image order
// after each neighbor list rebuild. Also computes BD cache prefix-sum
// offsets entirely on GPU with parallel scan (no single-thread bottleneck).
//---------------------------------------------------------------------------------------------

#include <cuda.h>
#include "gpu.h"
#include "ptxmacros.h"

#include "simulationConst.h"

CSIM_STO simulationConst cSim;

//---------------------------------------------------------------------------------------------
// SetkNLRemap_PGMSim: copy simulation constants to device constant memory
//---------------------------------------------------------------------------------------------
void SetkNLRemap_PGMSim(gpuContext gpu)
{
  cudaError_t status;
  status = cudaMemcpyToSymbol(cSim, &gpu->sim, sizeof(simulationConst));
  RTERROR(status, "SetkNLRemap_PGMSim: cudaMemcpyToSymbol failed");
}

//---------------------------------------------------------------------------------------------
// kPGMNLRemap_kernel: scatter original-order PGM data to NL image order
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMNLRemap_kernel()
{
  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Precompute self-field prefactor: -(4/3) * ew^3 / sqrt(pi)
  double ew  = (double)cSim.ew_coeff;
  double ew3 = ew * ew * ew;
  double selfFieldPrefac = -(4.0/3.0) * ew3 / sqrt(M_PI);

  while (pos < cSim.atoms) {
    unsigned int i1 = cSim.pImageAtomLookup[pos];

    double q     = cSim.pPGMOrigChg[pos];
    double rg    = cSim.pPGMOrigRGauss[pos];
    double alpha = cSim.pPGMOrigPol[pos];

    cSim.pPGMPermChg[i1]        = q;
    cSim.pAtomChargeSP[i1]      = (PMEFloat)q;
    cSim.pPGMRGauss[i1]         = rg;
    cSim.pPGMPolarizability[i1] = alpha;

    double diag = 1.0 + alpha * selfFieldPrefac;
    cSim.pPGMPrecondInv[i1] = 1.0 / diag;

    pos += blockDim.x * gridDim.x;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMComputeT2PairCounts_kernel: compute per-NL-entry pair counts for BD cache
//
// Reads nEntries from pNLEntries on device (no host parameter needed).
// Writes pair count for each NL entry into pPGMBFactorOffset[i+1].
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 1)
kPGMComputeT2PairCounts_kernel(int apw, int iftlim)
{
  __shared__ unsigned int sNEntries;
  if (threadIdx.x == 0) {
    sNEntries = *(cSim.pNLEntries);
  }
  __syncthreads();
  unsigned int nEntries = sNEntries;

  unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
  while (pos < nEntries) {
    NLEntry entry = cSim.pNLEntry[pos];

    bool bHome = entry.NL.ymax & 0x1;
    unsigned int xatoms = entry.NL.xatoms;
    int home_pairs = bHome ? (iftlim * 32) : 0;
    int neigh_tiles = (xatoms + 31) / 32;
    int neigh_pairs = neigh_tiles * apw * 32;

    // Write count to offset[pos+1] (prefix-summed next)
    cSim.pPGMBFactorOffset[pos + 1] = home_pairs + neigh_pairs;

    pos += blockDim.x * gridDim.x;
  }
}

//---------------------------------------------------------------------------------------------
// kPGMPrefixSumParallel_kernel: parallel exclusive prefix-sum of BD pair counts
//
// Reads nEntries from device memory.  Uses 256 threads with shared-memory scan.
// Handles up to 256 * 256 = 65536 NL entries.
//
// Converts pair counts in pPGMBFactorOffset[1..n] to cumulative offsets [0..n]
// where offset[0]=0 and offset[n]=totalPairs.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 1)
kPGMPrefixSumParallel_kernel()
{
  __shared__ int partials[256];
  __shared__ unsigned int sNEntries;

  if (threadIdx.x == 0) {
    sNEntries = *(cSim.pNLEntries);
  }
  __syncthreads();
  unsigned int nEntries = sNEntries;
  if (nEntries == 0) return;

  // Each thread handles a contiguous chunk of entries
  unsigned int chunk = (nEntries + 255) / 256;
  unsigned int myStart = threadIdx.x * chunk + 1;  // counts are at [1..nEntries]
  unsigned int myEnd = myStart + chunk;
  if (myEnd > nEntries + 1) myEnd = nEntries + 1;

  // Phase 1: Thread-local sum of pair counts in my chunk
  int localSum = 0;
  for (unsigned int i = myStart; i < myEnd; i++) {
    localSum += cSim.pPGMBFactorOffset[i];
  }
  partials[threadIdx.x] = localSum;
  __syncthreads();

  // Phase 2: Inclusive prefix scan of 256 partial sums (Hillis-Steele)
  for (int stride = 1; stride < 256; stride *= 2) {
    int val = (threadIdx.x >= (unsigned int)stride) ? partials[threadIdx.x - stride] : 0;
    __syncthreads();
    partials[threadIdx.x] += val;
    __syncthreads();
  }
  // partials[t] = sum of all chunks [0..t] (inclusive)

  // Phase 3: Convert counts to exclusive prefix sums
  int prefix = (threadIdx.x > 0) ? partials[threadIdx.x - 1] : 0;

  if (threadIdx.x == 0) {
    cSim.pPGMBFactorOffset[0] = 0;
  }

  int running = prefix;
  for (unsigned int i = myStart; i < myEnd; i++) {
    int count = cSim.pPGMBFactorOffset[i];
    running += count;
    cSim.pPGMBFactorOffset[i] = running;  // inclusive: offset[i] = start of entry i+1
  }
  // offset[nEntries] = totalPairs (written by last active thread)
}

//---------------------------------------------------------------------------------------------
// kPGMRemapDipHistory_kernel: remap CG dipole vector from old to new image ordering
//
// For each original atom i, copies 3 doubles (xyz) from old image slot to new image slot.
// Used to preserve dipole history across NL rebuilds so extrapolation stays valid.
//---------------------------------------------------------------------------------------------
__global__ void
__launch_bounds__(256, 4)
kPGMRemapDipHistory_kernel(double* __restrict__ dst,
                           const double* __restrict__ src,
                           const unsigned int* __restrict__ oldLookup,
                           const unsigned int* __restrict__ newLookup,
                           int atoms)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < atoms) {
    unsigned int old_slot = oldLookup[i];
    unsigned int new_slot = newLookup[i];
    dst[3*new_slot + 0] = src[3*old_slot + 0];
    dst[3*new_slot + 1] = src[3*old_slot + 1];
    dst[3*new_slot + 2] = src[3*old_slot + 2];
    i += blockDim.x * gridDim.x;
  }
}

//---------------------------------------------------------------------------------------------
// Wrapper functions
//---------------------------------------------------------------------------------------------
void kPGMNLRemap(gpuContext gpu)
{
  int blocks = (gpu->sim.atoms + 255) / 256;
  kPGMNLRemap_kernel<<<blocks, 256>>>();
  LAUNCHERROR("kPGMNLRemap_kernel");
}

void kPGMRemapDipHistory(gpuContext gpu, double* dst, const double* src,
                         const unsigned int* oldLookup)
{
  int blocks = (gpu->sim.atoms + 255) / 256;
  kPGMRemapDipHistory_kernel<<<blocks, 256>>>(dst, src, oldLookup,
      gpu->sim.pImageAtomLookup, gpu->sim.atoms);
  LAUNCHERROR("kPGMRemapDipHistory");
}

void kPGMComputeT2Offsets(gpuContext gpu)
{
  int apw = gpu->sim.NLAtomsPerWarp;
  int iftlim = (apw == 32) ? 16 : (apw == 16) ? 4 : 1;

  // Fixed grid size: kernels read nEntries from device, threads beyond nEntries skip
  kPGMComputeT2PairCounts_kernel<<<128, 256>>>(apw, iftlim);
  LAUNCHERROR("kPGMComputeT2PairCounts_kernel");

  kPGMPrefixSumParallel_kernel<<<1, 256>>>();
  LAUNCHERROR("kPGMPrefixSumParallel_kernel");
}

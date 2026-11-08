// kCPULS_PGM.cu (2026-10-08)
// Device kernels for the GPU port of the CPU pGM induced-dipole predictor
// (pGM_induced.F90: extrap_ind_dipole / least_square_extrap, dipole_scf_init=3, step 2).
// Raw-pointer kernels only (no cSim), so this file compiles on its own with nvcc.
// All reductions use a fixed launch shape and a fixed summation order -> bitwise deterministic.
#include <cuda_runtime.h>
#include <stdio.h>

#define CPULS_THREADS 256
#define CPULS_BLOCKS  120          // fixed -> deterministic partial order

__global__ void cpuls_k_scale_pol(double* __restrict__ B, const double* __restrict__ b,
                                  const double* __restrict__ pol, int n3)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n3; i += blockDim.x * gridDim.x)
    B[i] = pol[i / 3] * b[i];
}

// 5 dot products for the 2x2 normal equations: R0.R0, R0.R1, R1.R1, R0.T, R1.T
__global__ void cpuls_k_dots5(const double* __restrict__ R0, const double* __restrict__ R1,
                              const double* __restrict__ T, int n3, double* __restrict__ part)
{
  __shared__ double s[5][CPULS_THREADS];
  double a00 = 0.0, a01 = 0.0, a11 = 0.0, b0 = 0.0, b1 = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n3; i += blockDim.x * gridDim.x) {
    double r0 = R0[i], r1 = R1[i], t = T[i];
    a00 += r0 * r0; a01 += r0 * r1; a11 += r1 * r1; b0 += r0 * t; b1 += r1 * t;
  }
  s[0][threadIdx.x] = a00; s[1][threadIdx.x] = a01; s[2][threadIdx.x] = a11;
  s[3][threadIdx.x] = b0;  s[4][threadIdx.x] = b1;
  __syncthreads();
  for (int off = CPULS_THREADS / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off)
      for (int k = 0; k < 5; k++) s[k][threadIdx.x] += s[k][threadIdx.x + off];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int k = 0; k < 5; k++) part[blockIdx.x * 5 + k] = s[k][0];
}

// one thread: sum partials in fixed order, then exactly the CPU's least_square_extrap CG (2 iterations)
__global__ void cpuls_k_solve(const double* __restrict__ part, int nblk, double* __restrict__ c)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  double v[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int j = 0; j < nblk; j++) for (int k = 0; k < 5; k++) v[k] += part[j * 5 + k];
  double M[2][2] = {{v[0], v[1]}, {v[1], v[2]}};
  double r[2] = {v[3], v[4]}, p[2] = {v[3], v[4]}, x[2] = {0.0, 0.0}, Mp[2];
  for (int it = 0; it < 2; it++) {
    Mp[0] = M[0][0] * p[0] + M[0][1] * p[1];
    Mp[1] = M[1][0] * p[0] + M[1][1] * p[1];
    double pMp = Mp[0] * p[0] + Mp[1] * p[1];
    double rr  = r[0] * r[0] + r[1] * r[1];
    double alpha = (fabs(pMp) > 1.0e-50) ? rr / pMp : 0.0;
    x[0] += alpha * p[0]; x[1] += alpha * p[1];
    r[0] -= alpha * Mp[0]; r[1] -= alpha * Mp[1];
    double rr2 = r[0] * r[0] + r[1] * r[1];
    double beta = (rr > 1.0e-50) ? rr2 / rr : 0.0;
    p[0] = r[0] + beta * p[0]; p[1] = r[1] + beta * p[1];
  }
  c[0] = x[0]; c[1] = x[1];
}

// dst = (accumulate ? dst : 0) + c0*R0 + c1*R1     (c on the device)
__global__ void cpuls_k_comb(double* __restrict__ dst, const double* __restrict__ R0,
                             const double* __restrict__ R1, const double* __restrict__ c,
                             int n3, int accumulate)
{
  double c0 = c[0], c1 = c[1];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n3; i += blockDim.x * gridDim.x) {
    double v = c0 * R0[i] + c1 * R1[i];
    dst[i] = accumulate ? dst[i] + v : v;
  }
}

__global__ void cpuls_k_sub(double* __restrict__ d, const double* __restrict__ a,
                            const double* __restrict__ b, int n3)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n3; i += blockDim.x * gridDim.x)
    d[i] = a[i] - b[i];
}

static int cpuls_grid(int n3) { int g = (n3 + CPULS_THREADS - 1) / CPULS_THREADS; return g < 1024 ? g : 1024; }

extern "C" void cpuls_dev_scale_pol(double* B, const double* b, const double* pol, int n3)
{ cpuls_k_scale_pol<<<cpuls_grid(n3), CPULS_THREADS>>>(B, b, pol, n3); }

// c[0..1] (device) = LS coefficients fitting T with {R0,R1}; part = device scratch of >= 5*CPULS_BLOCKS doubles
extern "C" void cpuls_dev_ls(const double* R0, const double* R1, const double* T, int n3, double* part, double* c)
{
  cpuls_k_dots5<<<CPULS_BLOCKS, CPULS_THREADS>>>(R0, R1, T, n3, part);
  cpuls_k_solve<<<1, 1>>>(part, CPULS_BLOCKS, c);
}

extern "C" void cpuls_dev_comb(double* dst, const double* R0, const double* R1, const double* c, int n3, int accumulate)
{ cpuls_k_comb<<<cpuls_grid(n3), CPULS_THREADS>>>(dst, R0, R1, c, n3, accumulate); }

extern "C" void cpuls_dev_sub(double* d, const double* a, const double* b, int n3)
{ cpuls_k_sub<<<cpuls_grid(n3), CPULS_THREADS>>>(d, a, b, n3); }

extern "C" int cpuls_dev_scratch_size() { return 5 * CPULS_BLOCKS + 2; }

// ---- energy-drift diagnostic: per-step dH = -sum_i r_i.(x_i - xprev_i)/alpha_i  (internal AMBER units -> kcal/mol)
__global__ void cpuls_k_power(const double* __restrict__ r, const double* __restrict__ x,
                              double* __restrict__ xprev, const double* __restrict__ pol,
                              int n3, double* __restrict__ part)
{
  __shared__ double s[CPULS_THREADS];
  double a = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n3; i += blockDim.x * gridDim.x) {
    double xi = x[i];
    a += r[i] * (xi - xprev[i]) / pol[i / 3];
    xprev[i] = xi;
  }
  s[threadIdx.x] = a; __syncthreads();
  for (int off = CPULS_THREADS / 2; off > 0; off >>= 1) { if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off]; __syncthreads(); }
  if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}
__global__ void cpuls_k_copy(double* __restrict__ d, const double* __restrict__ s, int n3)
{ for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n3; i += blockDim.x * gridDim.x) d[i] = s[i]; }

// returns sum (host); xprev updated to x. If skip, only copies.
extern "C" double cpuls_dev_power(const double* r, const double* x, double* xprev, const double* pol, int n3, double* part, int skip)
{
  if (skip) { cpuls_k_copy<<<cpuls_grid(n3), CPULS_THREADS>>>(xprev, x, n3); return 0.0; }
  cpuls_k_power<<<CPULS_BLOCKS, CPULS_THREADS>>>(r, x, xprev, pol, n3, part);
  double h[CPULS_BLOCKS]; cudaMemcpy(h, part, CPULS_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);
  double s = 0.0; for (int j = 0; j < CPULS_BLOCKS; j++) s += h[j];
  return -s;
}

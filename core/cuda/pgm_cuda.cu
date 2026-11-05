/* pgm_cuda: pGM electrostatics on the GPU (see pgm_cuda.h).  The formulas are those of pgm_core.cpp and
 * of the GROMACS module (src/gromacs/pgm/pgm.cpp), whose layout (splines on grid points base - j,
 * unnormalized FFTs, the influence function, the strain terms) this file follows one to one. */

#include "pgm_cuda.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pgm_math.h"

namespace pgmcuda {

namespace {

#define CU(call)                                                                                     \
    do {                                                                                             \
        cudaError_t e_ = (call);                                                                     \
        if (e_ != cudaSuccess)                                                                       \
            throw std::runtime_error(std::string("pgm_cuda: ") + #call + ": " + cudaGetErrorString(e_)); \
    } while (0)
#define CUFFT(call)                                                                                  \
    do {                                                                                             \
        cufftResult r_ = (call);                                                                     \
        if (r_ != CUFFT_SUCCESS)                                                                     \
            throw std::runtime_error(std::string("pgm_cuda: ") + #call + " failed " + std::to_string(r_)); \
    } while (0)

typedef unsigned long long ull;
#if CUDART_VERSION >= 12040   // conditional graph nodes: the whole evaluation, CG loop included, as one graph
#define PGM_CUDA_GRAPHS 1
typedef cudaGraphConditionalHandle CondHandle;
#else
#define PGM_CUDA_GRAPHS 0
typedef unsigned long long CondHandle;
#endif
constexpr double FIX = 4294967296.0;         // fixed-point scale of fields, forces and the PME grid
constexpr float FIXF = 4294967296.0f;
constexpr int MAX_ORDER = 12;
constexpr int NSERIES = 15;
constexpr float DAMP_CUT = 4.5f;             // erfc(4.5) = 2e-10: below float resolution

struct Box {
    float b[9];   // rows a, b, c
    float r[9];   // recip[k][a] = (H^-1)_{k a}
    float vol;    // volume
};

// The box of the current evaluation, read by the kernels (set before each evaluation: the kernel
// parameters stay the same from step to step, so the evaluation can be a CUDA graph)
__constant__ Box c_box;

__device__ inline void addFix(ull* p, float v) {
    atomicAdd(p, (ull) __float2ll_rn(v*FIXF));
}

/** Minimum image in a reduced triclinic box (the diagonal of the inverse box is 1/diagonal). */
__device__ inline void minImage(float3& v, const Box& bx) {
    float k = rintf(v.z*bx.r[8]);
    v.x -= k*bx.b[6]; v.y -= k*bx.b[7]; v.z -= k*bx.b[8];
    k = rintf(v.y*bx.r[4]);
    v.x -= k*bx.b[3]; v.y -= k*bx.b[4];
    k = rintf(v.x*bx.r[0]);
    v.x -= k*bx.b[0];
}

// ------------------------------------------------------------------ pair kernels (float, as OpenMM's)

/** B_0..B_3 of erf(a r)/r: series for (a r)^2 < 1, point Coulomb for a r > DAMP_CUT, closed form else. */
__device__ void erfKernelsF(float a, float r, float invR, float* B) {
    const float x = a*r, s = x*x, c0 = 1.1283791670955126f*a, a2 = 2*a*a;
    if (s < 1) {
        float t = 1, sum[4] = {0, 0, 0, 0};
        for (int m = 0; m < NSERIES; m++) {
            for (int k = 0; k < 4; k++)
                sum[k] += t/(2*m+2*k+1);
            t *= -s/(m+1);
        }
        float pw = 1;
        for (int k = 0; k < 4; k++) {
            B[k] = c0*pw*sum[k];
            pw *= a2;
        }
    }
    else if (x > DAMP_CUT) {
        const float ir2 = invR*invR;
        B[0] = invR;
        for (int k = 1; k < 4; k++)
            B[k] = (2*k-1)*B[k-1]*ir2;
    }
    else {
        const float c = c0*expf(-s), ir2 = invR*invR;
        B[0] = erff(x)*invR;
        float pw = 1;
        for (int k = 1; k < 4; k++) {
            B[k] = ((2*k-1)*B[k-1]-pw*c)*ir2;
            pw *= a2;
        }
    }
}

/** B_0..B_3 of erfc(a r)/r (all terms positive). */
__device__ void erfcKernelsF(float a, float r, float invR, float* B) {
    const float x = a*r, c = 1.1283791670955126f*a*expf(-x*x), ir2 = invR*invR, a2 = 2*a*a;
    B[0] = erfcf(x)*invR;
    float pw = 1;
    for (int k = 1; k < 4; k++) {
        B[k] = ((2*k-1)*B[k-1]+pw*c)*ir2;
        pw *= a2;
    }
}

/** G_n = B_n[erf(a r)/r] - B_n[erf(beta r)/r] (pgmcore::ewaldPairKernels, float). */
__device__ void pairKernels(float rgSum, float r, float invR, float beta, float* G) {
    const float a = rsqrtf(rgSum), x = a*r;
    float B[4];
    if (x >= 1) {
        erfcKernelsF(beta, r, invR, G);
        if (x <= DAMP_CUT) {
            erfcKernelsF(a, r, invR, B);
            for (int k = 0; k < 4; k++)
                G[k] -= B[k];
        }
    }
    else {
        erfKernelsF(a, r, invR, G);
        erfKernelsF(beta, r, invR, B);
        for (int k = 0; k < 4; k++)
            G[k] -= B[k];
    }
}

// ------------------------------------------------------------------ pairs: one warp per atom, full list
// Every pair is in the lists of both atoms (x = r_i + t.box - r_j, t packed per candidate), so each warp
// sums only its own atom's field / force: no atomics; pair energies and the strain are counted twice.

__device__ inline float3 unpackShift(int code, const Box& bx) {
    const int t0 = (code&15)-8, t1 = ((code>>4)&15)-8, t2 = ((code>>8)&15)-8;
    return make_float3(t0*bx.b[0]+t1*bx.b[3]+t2*bx.b[6], t0*bx.b[1]+t1*bx.b[4]+t2*bx.b[7],
                       t0*bx.b[2]+t1*bx.b[5]+t2*bx.b[8]);
}

/** The pairs of atom i within the cut-off, compacted (in candidate order) to [atomStart[i], pairEnd[i]):
 * the partner j and the kernels G_1, G_2 (G12, the field passes) and G_0, G_3 (G03), x = r_i - r_j the
 * minimum image (SHIFTS: x = r_i + t.box - r_j as given, the same for pairs within the cut-off). */
constexpr int SKIP_BIT = 1<<30;     // in a partner index: no combination-rule GVDW for this pair
constexpr int INDEX_MASK = SKIP_BIT-1;
constexpr int PRE_CAP = 48;   // short-range pairs per atom kept for the preconditioner

/** The pairs of atom i within rp = max(rc, rvdw), compacted (in candidate order) to [atomStart[i],
 * pairEnd[i]): the partner j (with SKIP_BIT) and the kernels G_1, G_2 (G12, the field passes) and G_0, G_3
 * (G03; zero beyond rc), x = r_i - r_j the minimum image (SHIFTS: x = r_i + t.box - r_j as given, the same
 * for pairs within the cut-off). */
template<bool SHIFTS>
__global__ void pairSetup(int n, const int* order, const int* atomStart, const int* candJ, const int* candT,
                          const float4* posq, const float* rg, float rc2, float rp2, float beta, int* PJ,
                          float2* G12, float2* G03, int* pairEnd) {
    const Box bx = c_box;
    const int w = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    if (w >= n)
        return;
    const int i = order ? order[w] : w;   // spatial order (cell list): neighbouring warps share partners
    const float4 pi = posq[i];
    const float rgi = rg[i];
    const unsigned below = (1u<<lane)-1;
    const int end = atomStart[i+1];
    int out = atomStart[i];
    for (int k0 = atomStart[i]; k0 < end; k0 += 32) {
        const int k = k0+lane;
        bool inside = false;
        int jraw = 0, j = 0;
        float3 d = make_float3(0, 0, 0);
        if (k < end) {
            jraw = candJ[k];
            j = jraw&INDEX_MASK;
            const float4 pj = posq[j];
            if (SHIFTS) {
                const float3 sv = unpackShift(candT[k], bx);
                d = make_float3(pi.x+sv.x-pj.x, pi.y+sv.y-pj.y, pi.z+sv.z-pj.z);
            }
            else {
                d = make_float3(pi.x-pj.x, pi.y-pj.y, pi.z-pj.z);
                minImage(d, bx);
            }
            inside = d.x*d.x+d.y*d.y+d.z*d.z < rp2;
        }
        const unsigned mask = __ballot_sync(0xffffffff, inside);
        if (inside) {
            const float r2 = d.x*d.x+d.y*d.y+d.z*d.z, r = sqrtf(r2);
            const int q = out+__popc(mask&below);
            PJ[q] = jraw;
            if (G12) {
                float g[4] = {0, 0, 0, 0};
                if (r2 < rc2)
                    pairKernels(rgi+rg[j], r, r > 0 ? 1/r : 0.0f, beta, g);
                G12[q] = make_float2(g[1], g[2]);
                if (G03)
                    G03[q] = make_float2(g[0], g[3]);
            }
        }
        out += __popc(mask);
    }
    if (lane == 0)
        pairEnd[i] = out;
}

/** Sets SKIP_BIT on the candidates (i, j) of the skip lists (exclusions, overridden pairs). */
__global__ void markSkips(int n, const int* atomStart, int* candJ, const int* skipStart, const int* skipJ) {
    const int i = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    if (i >= n || skipStart[i] == skipStart[i+1])
        return;
    for (int k = atomStart[i]+lane; k < atomStart[i+1]; k += 32) {
        const int j = candJ[k]&INDEX_MASK;
        bool skip = false;
        for (int e = skipStart[i]; e < skipStart[i+1]; e++)
            skip |= (skipJ[e] == j);
        candJ[k] = j|(skip ? SKIP_BIT : 0);
    }
}

// ------------------------------------------------------------------ GVDW (core/gvdw_kernel.h in float)

/** G(y) = F(y)/y^6 and H(y) = G'(y)/y (pgmcore::gvdwG). */
__device__ inline void gvdwGF(float y, float& G, float& H) {
    const float C0 = 0.28294212f;   // 8/(9 pi)
    if (y < 0.6f) {
        const float c[15] = {1.0f, -2.0f, 58.0f/25, -188.0f/105, 2222.0f/2205, -1532.0f/3465, 64244.0f/405405,
                             -162056.0f/3378375, 1587734.0f/126351225, -211108.0f/72747675,
                             11902564.0f/19860115275.0f, -611992.0f/5465775315.0f,
                             755028172.0f/39529267903125.0f, -213335384.0f/71152682225625.0f,
                             15291334888.0f/35078272337233125.0f};
        const float y2 = y*y;
        float g = 0, h = 0;
        for (int k = 14; k >= 0; k--)
            g = g*y2+c[k];
        for (int k = 14; k >= 1; k--)
            h = h*y2+2.0f*k*c[k];
        G = C0*g;
        H = C0*h;
        return;
    }
    const float isp = 0.56418958f;   // 1/sqrt(pi)
    const float y2 = y*y, e = expf(-y2);
    const float A = erff(y)-2.0f*isp*y*(1.0f+2.0f*y2/3.0f)*e;
    const float g = (4.0f/3.0f)*isp*y2*y*e;
    const float F = A*A+0.5f*g*g;
    const float Ap = (8.0f/3.0f)*isp*y2*y2*e;
    const float gp = 4.0f*isp*y2*(1.0f-2.0f*y2/3.0f)*e;
    const float Fp = 2.0f*A*Ap+g*gp;
    const float y6 = y2*y2*y2;
    G = F/y6;
    H = (Fp/y6-6.0f*F/(y6*y))/y;
}

/** The pair energy; dEdR = -(1/r) dU/dr (force on i = dEdR x) (pgmcore::gvdwPair). */
__device__ inline float gvdwPairF(float r, float bij, float A, float C6, float B, bool slater, float& dEdR) {
    const float y = bij*r, b2 = bij*bij, b6 = b2*b2*b2;
    float erep, drep;
    if (!slater) {
        erep = A*expf(-B*y*y);
        drep = -2.0f*B*b2*erep;
    }
    else {
        erep = A*expf(-B*y);
        drep = (r > 0 ? -B*bij*erep/r : 0.0f);
    }
    float G, H;
    gvdwGF(y, G, H);
    dEdR = -(drep-C6*b6*b2*H);
    return erep-C6*b6*G;
}

__device__ inline float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__device__ inline double warpSumD(double v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

/** Where the pair passes get the kernels G_0..G_3 of a pair: stored by pairSetup (G12, and G03), or computed
 * on the fly (from rg and beta; zero beyond rc): all of them without G12, those of the final pass (all)
 * without G03. */
struct KernelSource {
    const float2* G12;
    const float2* G03;
    const float* rg;
    float rc2, beta;
    __device__ float4 get(int k, float rgi, int j, const float3& v, bool all) const {
        if (G12 && (!all || G03)) {
            const float2 a = G12[k], b = all ? G03[k] : make_float2(0, 0);
            return make_float4(b.x, a.x, a.y, b.y);
        }
        const float r2 = v.x*v.x+v.y*v.y+v.z*v.z;
        float g[4] = {0, 0, 0, 0};
        if (r2 < rc2) {
            const float r = sqrtf(r2);
            pairKernels(rgi+rg[j], r, 1/r, beta, g);
        }
        return make_float4(g[0], g[1], g[2], g[3]);
    }
};

/** Direct-space field of charges (WITHQ) and dipoles dip at every atom, into fld (3n). */
template<bool WITHQ>
__global__ void fieldPass(int n, const int* order, const int* atomStart, const int* pairEnd, const int* PJ,
                          const float4* posq, const float4* dip, KernelSource ks, double* fld) {
    const Box bx = c_box;
    const int w = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    if (w >= n)
        return;
    const int i = order ? order[w] : w;   // spatial order (cell list): neighbouring warps share partners
    const float4 pi = posq[i];
    const float rgi = ks.rg[i];
    float ex = 0, ey = 0, ez = 0;
    for (int k = atomStart[i]+lane; k < pairEnd[i]; k += 32) {
        const int jraw = PJ[k], j = jraw&INDEX_MASK;
        const float4 pj = posq[j];
        float3 v = make_float3(pi.x-pj.x, pi.y-pj.y, pi.z-pj.z);
        minImage(v, bx);
        const float4 g = ks.get(k, rgi, j, v, false);
        const float x = v.x, y = v.y, z = v.z, qj = WITHQ ? pj.w : 0.0f;
        const float4 dj4 = dip[j];
        const float djx = dj4.x, djy = dj4.y, djz = dj4.z;
        const float dxj = djx*x+djy*y+djz*z;
        ex += qj*x*g.y-djx*g.y+dxj*x*g.z;
        ey += qj*y*g.y-djy*g.y+dxj*y*g.z;
        ez += qj*z*g.y-djz*g.y+dxj*z*g.z;
    }
    ex = warpSum(ex); ey = warpSum(ey); ez = warpSum(ez);
    if (lane == 0) {
        fld[3*i] = ex; fld[3*i+1] = ey; fld[3*i+2] = ez;
    }
}

/** Energy, forces, field and (STRAIN) strain of charges and dipoles d; acc[0] energy, acc[1..9] S (both
 * halves). */
template<bool STRAIN, bool GVDW>
__global__ void finalPass(int n, const int* order, const int* atomStart, const int* pairEnd, const int* PJ,
                          const float4* posq, const float4* dip, KernelSource ks, const float* rg,
                          const float4* gv, bool slater, float rvdw2, float invKe, double* fld, double* fpair,
                          double* acc) {
    const Box bx = c_box;
    const int w = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    if (w >= n)
        return;
    const int i = order ? order[w] : w;   // spatial order (cell list): neighbouring warps share partners
    const float4 pi = posq[i];
    const float qi = pi.w;
    const float4 di4 = dip[i];
    const float di[3] = {di4.x, di4.y, di4.z};
    float Ei[3] = {0, 0, 0}, Fi[3] = {0, 0, 0}, S[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, en = 0, eg = 0;
    float4 gvi = make_float4(0, 0, 0, 0);
    float rgi = 0;
    if (GVDW) {
        gvi = gv[i];
        rgi = rg[i];
    }
    for (int k = atomStart[i]+lane; k < pairEnd[i]; k += 32) {
        const int jraw = PJ[k], j = jraw&INDEX_MASK;
        const float4 pj = posq[j];
        float3 v = make_float3(pi.x-pj.x, pi.y-pj.y, pi.z-pj.z);
        minImage(v, bx);
        const float4 g = ks.get(k, rg[i], j, v, true);
        const float x[3] = {v.x, v.y, v.z}, qj = pj.w;
        const float4 dj4 = dip[j];
        const float dj[3] = {dj4.x, dj4.y, dj4.z};
        const float dix = di[0]*x[0]+di[1]*x[1]+di[2]*x[2], djx = dj[0]*x[0]+dj[1]*x[1]+dj[2]*x[2];
        const float didj = di[0]*dj[0]+di[1]*dj[1]+di[2]*dj[2];
        en += qi*qj*g.x+(qi*djx-qj*dix)*g.y+didj*g.y-dix*djx*g.z;
        const float sc = -qi*qj*g.y-(qi*djx-qj*dix)*g.z-didj*g.z+dix*djx*g.w;
        for (int c = 0; c < 3; c++) {
            const float dedx = sc*x[c]+(qi*dj[c]-qj*di[c])*g.y-(djx*di[c]+dix*dj[c])*g.z;
            if (STRAIN)
                for (int b = 0; b < 3; b++)
                    S[3*c+b] += dedx*x[b];
            Fi[c] -= dedx;
            Ei[c] += qj*x[c]*g.y-dj[c]*g.y+djx*x[c]*g.z;
        }
        // the combination-rule GVDW (forces and strain in units of ke, as the electrostatics)
        if (GVDW && !(jraw&SKIP_BIT)) {
            const float r2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
            const float4 gvj = gv[j];
            if (r2 < rvdw2 && (gvi.x*gvj.x != 0 || gvi.y*gvj.y != 0)) {
                float dEdR;
                eg += gvdwPairF(sqrtf(r2), rsqrtf(rgi+rg[j]), gvi.x*gvj.x, gvi.y*gvj.y, 0.5f*(gvi.z+gvj.z), slater,
                                dEdR);
                const float f = dEdR*invKe;
                for (int c = 0; c < 3; c++) {
                    Fi[c] += f*x[c];
                    if (STRAIN)
                        for (int b = 0; b < 3; b++)
                            S[3*c+b] -= f*x[c]*x[b];
                }
            }
        }
    }
    for (int c = 0; c < 3; c++) {
        Ei[c] = warpSum(Ei[c]);
        Fi[c] = warpSum(Fi[c]);
    }
    double sum[10];
    const int ns = STRAIN ? 10 : 1;
    sum[0] = warpSumD(0.5*(double) en);
    for (int c = 0; c < ns-1; c++)
        sum[1+c] = warpSumD(0.5*(double) S[c]);
    if (lane == 0) {
        for (int c = 0; c < 3; c++) {
            fld[3*i+c] = Ei[c];
            fpair[3*i+c] = Fi[c];
        }
        for (int c = 0; c < ns; c++)
            atomicAdd(acc+c, sum[c]);
    }
    if (GVDW) {
        const double sg = warpSumD(0.5*(double) eg);
        if (lane == 0)
            atomicAdd(acc+40, sg);
    }
}

// ------------------------------------------------------------------ own pair list: cell list, minimum image
// Cells of the fractional coordinates, at least the list radius wide (perpendicular), so every pair within
// it is in neighbouring cells; per atom (one warp) the full list of the atoms within the radius, in cell
// order (atoms sorted by cell, stably: deterministic); rebuilt when the atoms have moved more than the skin.

__device__ inline void cellOf(const float4& p, const Box& bx, const int* nc, int* c) {
    const float pp[3] = {p.x, p.y, p.z};
    for (int a = 0; a < 3; a++) {
        float s = pp[0]*bx.r[a]+pp[1]*bx.r[3+a]+pp[2]*bx.r[6+a];
        s -= floorf(s);
        c[a] = min((int) (s*nc[a]), nc[a]-1);
    }
}

__global__ void cellKeys(int n, const float4* posq, int3 ncv, int* key, int* atom) {
    const Box bx = c_box;
    const int nc[3] = {ncv.x, ncv.y, ncv.z};
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        int c[3];
        cellOf(posq[i], bx, nc, c);
        key[i] = (c[0]*nc[1]+c[1])*nc[2]+c[2];
        atom[i] = i;
    }
}

__global__ void cellRanges(int n, const int* sortedKey, int* cellStart, int* cellEnd) {
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < n; k += gridDim.x*blockDim.x) {
        const int c = sortedKey[k];
        if (k == 0 || sortedKey[k-1] != c)
            cellStart[c] = k;
        if (k == n-1 || sortedKey[k+1] != c)
            cellEnd[c] = k+1;
    }
}

constexpr int LIST_R = 2;           // cells rl / LIST_R wide, LIST_R cells on each side searched
constexpr int LIST_THREADS = 128;   // per block (one block per cell)
constexpr int LIST_ICHUNK = 256;    // i atoms of a cell per round
constexpr int LIST_JCHUNK = 64;     // j atoms in shared memory at a time

/** FILL = false: slabCount[s n + i] = the atoms within rl of i in slab s of its neighbour cells; FILL:
 * candJ from atomStart[i] + the counts of the slabs before.  One block per cell and slab (the neighbour
 * cells of one offset along a and b, blockIdx.y): the atoms of each neighbour cell are loaded once into shared
 * memory and tested against all atoms of the cell (a warp per i atom), so the positions are read once per
 * cell, not once per atom.  A list holds the neighbour cells in a fixed order, the atoms of a cell in the
 * sorted order: deterministic. */
template<bool FILL>
__global__ void listBuild(int n, const float4* posSorted, const int* sortedAtom, int3 ncv, const int* cellStart,
                          const int* cellEnd, float rl2, int* slabCount, const int* atomStart, int* candJ) {
    __shared__ float4 jpos[LIST_JCHUNK];
    __shared__ int jidx[LIST_JCHUNK];
    __shared__ int cntS[LIST_ICHUNK], iidx[LIST_ICHUNK], istart[LIST_ICHUNK];
    __shared__ float4 ipos[LIST_ICHUNK];
    __shared__ int nbrS[(2*LIST_R+1)*(2*LIST_R+1)*(2*LIST_R+1)], nbrE[(2*LIST_R+1)*(2*LIST_R+1)*(2*LIST_R+1)];
    const Box bx = c_box;
    const int cell = blockIdx.x, slab = blockIdx.y, lane = threadIdx.x%32, warp = threadIdx.x/32;
    const int nwarps = LIST_THREADS/32;
    const int cs = cellStart[cell], ce = cellEnd[cell], ni = ce-cs;
    if (ni == 0)
        return;
    const int nc[3] = {ncv.x, ncv.y, ncv.z};
    const int c[3] = {cell/(nc[1]*nc[2]), (cell/nc[2])%nc[1], cell%nc[2]};
    int lo[3], len[3];   // distinct neighbour offsets per dimension (fewer than 2R + 1 cells: each once)
    for (int a = 0; a < 3; a++) {
        lo[a] = (nc[a] >= 2*LIST_R+1 ? -LIST_R : 0);
        len[a] = (nc[a] >= 2*LIST_R+1 ? 2*LIST_R+1 : nc[a]);
    }
    if (slab >= len[0]*len[1])
        return;
    const int ncell = len[2];   // in this slab (offsets o0, o1 fixed)
    for (int idx = threadIdx.x; idx < ncell; idx += LIST_THREADS) {   // the neighbour cells' atom ranges
        const int o0 = lo[0]+slab/len[1], o1 = lo[1]+slab%len[1], o2 = lo[2]+idx;
        const int nbr = (((c[0]+o0+nc[0])%nc[0])*nc[1]+(c[1]+o1+nc[1])%nc[1])*nc[2]+(c[2]+o2+nc[2])%nc[2];
        nbrS[idx] = cellStart[nbr];
        nbrE[idx] = cellEnd[nbr];
    }
    const unsigned below = (1u<<lane)-1;
    for (int ib = 0; ib < ni; ib += LIST_ICHUNK) {
        const int nib = min(LIST_ICHUNK, ni-ib);
        for (int t = threadIdx.x; t < nib; t += LIST_THREADS) {
            cntS[t] = 0;
            iidx[t] = sortedAtom[cs+ib+t];
            ipos[t] = posSorted[cs+ib+t];
            if (FILL) {
                int o = atomStart[iidx[t]];
                for (int s = 0; s < slab; s++)
                    o += slabCount[(size_t) s*n+iidx[t]];
                istart[t] = o;
            }
        }
        __syncthreads();
        for (int idx = 0; idx < ncell; idx++) {
            const int js = nbrS[idx], je = nbrE[idx];
            for (int jb = js; jb < je; jb += LIST_JCHUNK) {
                const int nj = min(LIST_JCHUNK, je-jb);
                for (int t = threadIdx.x; t < nj; t += LIST_THREADS) {
                    jpos[t] = posSorted[jb+t];
                    jidx[t] = sortedAtom[jb+t];
                }
                __syncthreads();
                for (int il = warp; il < nib; il += nwarps) {
                    const int i = iidx[il];
                    const float4 pi = ipos[il];
                    int base = cntS[il];
                    for (int jc = 0; jc < nj; jc += 32) {
                        const int l = jc+lane;
                        bool ok = false;
                        if (l < nj && jidx[l] != i) {
                            float3 d = make_float3(pi.x-jpos[l].x, pi.y-jpos[l].y, pi.z-jpos[l].z);
                            minImage(d, bx);
                            ok = d.x*d.x+d.y*d.y+d.z*d.z < rl2;
                        }
                        const unsigned mask = __ballot_sync(0xffffffff, ok);
                        if (FILL && ok)
                            candJ[istart[il]+base+__popc(mask&below)] = jidx[l];
                        base += __popc(mask);
                    }
                    if (lane == 0)
                        cntS[il] = base;
                    __syncwarp();
                }
                __syncthreads();
            }
        }
        if (!FILL)
            for (int t = threadIdx.x; t < nib; t += LIST_THREADS)
                slabCount[(size_t) slab*n+iidx[t]] = cntS[t];
        __syncthreads();
    }
}

/** count[i] = sum over the slabs (count[n] = 0). */
__global__ void sumSlabs(int n, int slabs, const int* slabCount, int* count) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i <= n; i += gridDim.x*blockDim.x) {
        int c = 0;
        if (i < n)
            for (int s = 0; s < slabs; s++)
                c += slabCount[(size_t) s*n+i];
        count[i] = c;
    }
}

// ------------------------------------------------------------------ the list across a reordering
// newOf[i]: the new local index of old atom i (the same atoms in another order, as after GROMACS's pair
// search): the rows move, the partners are renamed; positions of the build and of the last step move too.

__global__ void remapCounts(int n, const int* newOf, const int* atomStart, int* count) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i <= n; i += gridDim.x*blockDim.x)
        count[i < n ? newOf[i] : n] = (i < n ? atomStart[i+1]-atomStart[i] : 0);
}

__global__ void remapRows(int n, const int* newOf, const int* atomStart, const int* candJ, const int* newStart,
                          int* newCand) {
    const int i = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    if (i >= n)
        return;
    const int o = newStart[newOf[i]]-atomStart[i];
    for (int k = atomStart[i]+lane; k < atomStart[i+1]; k += 32)
        newCand[k+o] = newOf[candJ[k]&INDEX_MASK];
}

__global__ void remapAtoms(int n, const int* newOf, const float4* a, const float4* b, float4* a2, float4* b2,
                           int* order) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        a2[newOf[i]] = a[i];
        b2[newOf[i]] = b[i];
        order[i] = newOf[order[i]];
    }
}

__global__ void packPositions4(int n, const float4* x, const float* q, float4* posq) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const float4 a = x[i];
        posq[i] = make_float4(a.x, a.y, a.z, q[i]);
    }
}

/** forceFixed += f (3n, interleaved) in the caller's layout. */
__global__ void addForcesFixed(int n, const double* f, long long* out, int stride, double scale) {
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < 3*n; k += gridDim.x*blockDim.x) {
        const int i = k/3, c = k%3;
        atomicAdd((ull*) (out+i+(size_t) c*stride), (ull) (long long) llrint(f[k]*scale));
    }
}

__global__ void packPositions(int n, const float* x, const float* q, float4* posq) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x)
        posq[i] = make_float4(x[3*i], x[3*i+1], x[3*i+2], q[i]);
}

__global__ void gatherPositions(int n, const int* order, const float4* posq, float4* out) {
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < n; k += gridDim.x*blockDim.x)
        out[k] = posq[order[k]];
}

/** out[0] = max |r_i - r_i(list build)|^2, out[1] = max |r_i - r_i(last evaluation)|^2 (minimum image,
 * float bits); then prev = r. */
__global__ void displacements(int n, const float4* posq, const float4* ref, float4* prev, unsigned* out) {
    const Box bx = c_box;
    float m = 0, ms = 0;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const float4 a = posq[i], b = ref[i], c = prev[i];
        float3 d = make_float3(a.x-b.x, a.y-b.y, a.z-b.z), e = make_float3(a.x-c.x, a.y-c.y, a.z-c.z);
        minImage(d, bx);
        minImage(e, bx);
        m = fmaxf(m, d.x*d.x+d.y*d.y+d.z*d.z);
        ms = fmaxf(ms, e.x*e.x+e.y*e.y+e.z*e.z);
        prev[i] = a;
    }
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
        ms = fmaxf(ms, __shfl_down_sync(0xffffffff, ms, o));
    }
    if (threadIdx.x%32 == 0) {
        atomicMax(out, __float_as_uint(m));
        atomicMax(out+1, __float_as_uint(ms));
    }
}

// ------------------------------------------------------------------ covalent dipoles

__global__ void permDipolesKernel(int n, const float4* posq, const int* covStart, const int* covPartner,
                                 const double* covStrength, double* perm) {
    const Box bx = c_box;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        double p[3] = {0, 0, 0};
        const float4 pi = posq[i];
        for (int m = covStart[i]; m < covStart[i+1]; m++) {
            const float4 pj = posq[covPartner[m]];
            float3 v = make_float3(pj.x-pi.x, pj.y-pi.y, pj.z-pi.z);
            minImage(v, bx);
            const double L = sqrt((double) v.x*v.x+(double) v.y*v.y+(double) v.z*v.z);
            p[0] += covStrength[m]*v.x/L;
            p[1] += covStrength[m]*v.y/L;
            p[2] += covStrength[m]*v.z/L;
        }
        perm[3*i] = p[0]; perm[3*i+1] = p[1]; perm[3*i+2] = p[2];
    }
}

/** dE/dr_partner = c (g - (g.u) u)/|v|, g = -E_atom; strain into acc[0..8]. */
__global__ void covalentForces(int ncov, const float4* posq, const int* covAtom, const int* covPartner,
                               const double* covStrength, const double* E, ull* frc, double* acc) {
    const Box bx = c_box;
    const int m = blockIdx.x*blockDim.x+threadIdx.x, lane = threadIdx.x%32;
    double S[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (m < ncov) {
        const int i = covAtom[m], j = covPartner[m];
        const float4 pi = posq[i], pj = posq[j];
        float3 vf = make_float3(pj.x-pi.x, pj.y-pi.y, pj.z-pi.z);
        minImage(vf, bx);
        const double v[3] = {vf.x, vf.y, vf.z};
        const double L = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        const double u[3] = {v[0]/L, v[1]/L, v[2]/L};
        const double g[3] = {-E[3*i], -E[3*i+1], -E[3*i+2]};
        const double gu = g[0]*u[0]+g[1]*u[1]+g[2]*u[2];
        for (int k = 0; k < 3; k++) {
            const double t = covStrength[m]*(g[k]-gu*u[k])/L;
            addFix(frc+3*j+k, (float) -t);
            addFix(frc+3*i+k, (float) t);
            for (int b = 0; b < 3; b++)
                S[3*k+b] = t*v[b];
        }
    }
    if (acc)
        for (int c = 0; c < 9; c++) {
            const double s = warpSumD(S[c]);
            if (lane == 0)
                atomicAdd(acc+c, s);
        }
}

// ------------------------------------------------------------------ PME

/** Splines of every atom: theta etc. on grid point (base - j) mod K, j = 0..p-1 (as pgm.cpp). */
template<int PC>   // the order at compile time (0: run time p)
__global__ void splines(int n, int pRun, const float4* posq, int K1, int K2, int K3, float* theta,
                        float* dtheta, float* d2theta, int* gidx) {
    const Box bx = c_box;
    constexpr int PM = (PC ? PC : MAX_ORDER);
    const int p = (PC ? PC : pRun);
    const int K[3] = {K1, K2, K3};
    for (int idx = blockIdx.x*blockDim.x+threadIdx.x; idx < 3*n; idx += gridDim.x*blockDim.x) {
        const int a = idx/n, i = idx%n;   // one thread per atom and dimension
        const float4 pos = posq[i];
        float M[PM+1][PM+1];
        float t = pos.x*bx.r[a]+pos.y*bx.r[3+a]+pos.z*bx.r[6+a];
        t -= floorf(t);
        const float wv = K[a]*t;
        int base = (int) floorf(wv);
        const float fr = wv-base;
        base = ((base%K[a])+K[a])%K[a];
#pragma unroll
        for (int k = 0; k <= PM; k++)
#pragma unroll
            for (int j = 0; j <= PM; j++)
                M[k][j] = 0;
        M[1][0] = 1;
#pragma unroll
        for (int k = 2; k <= PM; k++)
#pragma unroll
            for (int j = 0; j < k; j++)
                if (k <= p)
                    M[k][j] = ((fr+j)*(j < k-1 ? M[k-1][j] : 0.0f)+(k-fr-j)*(j > 0 ? M[k-1][j-1] : 0.0f))/(k-1);
        const size_t o = ((size_t) a*n+i)*p;
#pragma unroll
        for (int j = 0; j < PM; j++) {
            if (j >= p)
                break;
            const float m1 = (j < p-1 ? M[p-1][j] : 0.0f), m1b = (j > 0 ? M[p-1][j-1] : 0.0f);
            const float m2 = (j < p-2 ? M[p-2][j] : 0.0f), m2b = (j > 0 && j-1 < p-2 ? M[p-2][j-1] : 0.0f),
                        m2c = (j > 1 && j-2 < p-2 ? M[p-2][j-2] : 0.0f);
            theta[o+j] = M[p][j];
            dtheta[o+j] = m1-m1b;
            d2theta[o+j] = m2-2*m2b+m2c;
        }
        gidx[(size_t) a*n+i] = base;
    }
}

/** Spread q (WITHQ) and the fractional dipoles e_a = K_a d.recip[:,a] onto the fixed-point grid; one warp
 * per atom, the lanes over the p^3 points (consecutive lanes along z). */
__device__ inline int wrapDown(int g, int K) {   // g in (-K, K) to [0, K)
    return g < 0 ? g+K : g;
}

/** The spline values of one atom for a warp: with 3p <= 32 in registers (lane a p + j: dimension a,
 * point j) and read by shuffles, else from memory.  get() must be called by the whole warp. */
struct SplineLanes {
    const float *t, *u, *v;
    int p, n, i;
    bool lanes;
    float tl = 0, ul = 0, vl = 0;
    __device__ SplineLanes(int lane, int p_, int n_, int i_, const float* theta, const float* dtheta,
                           const float* d2theta)
        : t(theta), u(dtheta), v(d2theta), p(p_), n(n_), i(i_), lanes(3*p_ <= 32) {
        if (lanes && lane < 3*p) {
            const size_t o = ((size_t) (lane/p)*n+i)*p+lane%p;
            tl = t[o];
            ul = u[o];
            if (v)
                vl = v[o];
        }
    }
    __device__ void get(int j1, int j2, int j3, float* T, float* U, float* V) const {
        const int j[3] = {j1, j2, j3};
        for (int a = 0; a < 3; a++) {
            if (lanes) {
                T[a] = __shfl_sync(0xffffffff, tl, a*p+j[a]);
                U[a] = __shfl_sync(0xffffffff, ul, a*p+j[a]);
                if (v)
                    V[a] = __shfl_sync(0xffffffff, vl, a*p+j[a]);
            }
            else {
                const size_t o = ((size_t) a*n+i)*p+j[a];
                T[a] = t[o];
                U[a] = u[o];
                if (v)
                    V[a] = v[o];
            }
        }
    }
};

template<int PC, bool WITHQ>   // PC: the order at compile time (0: run time p)
__global__ void spread(int n, int pRun, const float4* posq, const float4* dip, int K1, int K2, int K3,
                       const float* theta, const float* dtheta, const int* gidx, ull* grid) {
    const Box bx = c_box;
    const int i = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    const int p = PC ? PC : pRun;
    if (i >= n)
        return;
    const float q = WITHQ ? posq[i].w : 0.0f;
    const float4 d4 = dip[i];
    const float d[3] = {d4.x, d4.y, d4.z};
    float e[3];
    const int K[3] = {K1, K2, K3};
    for (int a = 0; a < 3; a++)
        e[a] = K[a]*(d[0]*bx.r[a]+d[1]*bx.r[3+a]+d[2]*bx.r[6+a]);
    const int b1 = gidx[i], b2 = gidx[n+i], b3 = gidx[2*n+i];
    const int p2 = p*p, iters = (p2*p+31)/32;
    SplineLanes sl(lane, p, n, i, theta, dtheta, nullptr);
#pragma unroll
    for (int it = 0; it < iters; it++) {
        const int c0 = lane+32*it, c = (c0 < p2*p ? c0 : 0);
        const int j1 = c/p2, j2 = (c/p)%p, j3 = c%p;
        float T[3], U[3], V[3];
        sl.get(j1, j2, j3, T, U, V);
        if (c0 >= p2*p)
            continue;
        const int g1 = wrapDown(b1-j1, K1), g2 = wrapDown(b2-j2, K2), g3 = wrapDown(b3-j3, K3);
        const float w12 = T[0]*T[1];
        const float v = (q*w12+e[0]*U[0]*T[1]+e[1]*T[0]*U[1])*T[2]+e[2]*w12*U[2];
        addFix(grid+((size_t) g1*K2+g2)*K3+g3, v);
    }
}

/** out = in / FIX, in = 0 (the next spread finds it cleared: no memset node). */
__global__ void fixedToReal(size_t count, ull* in, float* out) {
    for (size_t k = blockIdx.x*(size_t) blockDim.x+threadIdx.x; k < count; k += (size_t) gridDim.x*blockDim.x) {
        out[k] = (float) ((double) (long long) in[k]/FIX);
        in[k] = 0;
    }
}

/** n16 16-byte words src -> dst (pinned host memory), the odd last double separately. */
__global__ void copyToHost(const uint4* src, uint4* dst, size_t n16, const double* s8, double* d8, int m) {
    for (size_t k = blockIdx.x*(size_t) blockDim.x+threadIdx.x; k < n16; k += (size_t) gridDim.x*blockDim.x)
        dst[k] = src[k];
    if (blockIdx.x == 0 && threadIdx.x == 0 && (m&1))
        d8[m-1] = s8[m-1];
}

/** Clears the small accumulators of an evaluation in one node (memset nodes cost more in a graph). */
__global__ void zeroSmall(unsigned* a, int na, double* b, int nb, double* c, int nc, unsigned* d, int nd) {
    for (int k = threadIdx.x; k < na; k += blockDim.x)
        a[k] = 0;
    for (int k = threadIdx.x; k < nb; k += blockDim.x)
        b[k] = 0;
    for (int k = threadIdx.x; k < nc; k += blockDim.x)
        c[k] = 0;
    for (int k = threadIdx.x; k < nd; k += blockDim.x)
        d[k] = 0;
}

/** F *= G (influence), with the energy (acc[0], MODE >= 1) and the k-space strain (acc[1..9], MODE 2). */
template<int MODE>
__global__ void convolution(int K1, int K2, int K3, float beta, const float* mod1,
                            const float* mod2, const float* mod3, cufftComplex* F, double* acc, int batch) {
    const Box bx = c_box;
    const int Kz = K3/2+1;
    const size_t total = (size_t) K1*K2*Kz;
    double en = 0, S[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float fac = 9.869604401089358f/(beta*beta);
    for (size_t idx = blockIdx.x*(size_t) blockDim.x+threadIdx.x; idx < total*batch;
         idx += (size_t) gridDim.x*blockDim.x) {
        const size_t loc = idx%total;   // (batch: grids one after the other, energies only for batch 1)
        const int k3 = loc%Kz, k2 = (loc/Kz)%K2, k1 = loc/((size_t) Kz*K2);
        const int m1 = (k1 <= K1/2 ? k1 : k1-K1), m2 = (k2 <= K2/2 ? k2 : k2-K2), m3 = k3;
        float g = 0, ms[3] = {0, 0, 0}, mm2 = 0;
        if (m1 != 0 || m2 != 0 || m3 != 0) {
            for (int d = 0; d < 3; d++)
                ms[d] = m1*bx.r[3*d]+m2*bx.r[3*d+1]+m3*bx.r[3*d+2];
            mm2 = ms[0]*ms[0]+ms[1]*ms[1]+ms[2]*ms[2];
            g = expf(-fac*mm2)/(3.14159265358979f*bx.vol*mm2)/(mod1[k1]*mod2[k2]*mod3[k3]);
        }
        cufftComplex f = F[idx];
        if (MODE >= 1 && g != 0) {
            const double w = (k3 == 0 || (K3%2 == 0 && k3 == K3/2) ? 1.0 : 2.0);
            const double ek = 0.5*w*g*((double) f.x*f.x+(double) f.y*f.y);
            en += ek;
            if (MODE == 2) {
                const double cm = 2.0*(fac/1.0+1.0/mm2);
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        S[3*a+b] += ek*((a == b ? -1.0 : 0.0)+cm*ms[a]*ms[b]);
            }
        }
        f.x *= g;
        f.y *= g;
        F[idx] = f;
    }
    if (MODE >= 1) {
        const int lane = threadIdx.x%32;
        en = warpSumD(en);
        if (lane == 0)
            atomicAdd(acc, en);
        if (MODE == 2) {
            for (int c = 0; c < 9; c++) {
                const double v = warpSumD(S[c]);
                if (lane == 0)
                    atomicAdd(acc+1+c, v);
            }
        }
    }
}

/** Reciprocal field E_rec at every atom from the potential phi (FINAL: also the forces Frec); one warp
 * per atom, the lanes over the p^3 points. */
template<int PC, bool FINAL>
__global__ void gather(int n, int pRun, const float4* posq, const float4* dip, int K1, int K2, int K3,
                       const float* theta, const float* dtheta, const float* d2theta, const int* gidx,
                       const float* phi, double* erec, double* frec) {
    const Box bx = c_box;
    const int i = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    const int p = PC ? PC : pRun;
    if (i >= n)
        return;
    const int K[3] = {K1, K2, K3};
    const int b1 = gidx[i], b2 = gidx[n+i], b3 = gidx[2*n+i];
    // f[0..2] = f1, f[3..8] = f2 00 11 22 01 02 12
    float f[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int p2 = p*p, iters = (p2*p+31)/32;
    SplineLanes sl(lane, p, n, i, theta, dtheta, FINAL ? d2theta : nullptr);
#pragma unroll
    for (int it = 0; it < iters; it++) {
        const int c0 = lane+32*it, c = (c0 < p2*p ? c0 : 0);
        const int j1 = c/p2, j2 = (c/p)%p, j3 = c%p;
        float T[3], U[3], V[3];
        sl.get(j1, j2, j3, T, U, V);
        if (c0 >= p2*p)
            continue;
        const int g1 = wrapDown(b1-j1, K1), g2 = wrapDown(b2-j2, K2), g3 = wrapDown(b3-j3, K3);
        const float ph = phi[((size_t) g1*K2+g2)*K3+g3];
        f[0] += U[0]*T[1]*T[2]*ph;
        f[1] += T[0]*U[1]*T[2]*ph;
        f[2] += T[0]*T[1]*U[2]*ph;
        if (FINAL) {
            f[3] += V[0]*T[1]*T[2]*ph;
            f[4] += T[0]*V[1]*T[2]*ph;
            f[5] += T[0]*T[1]*V[2]*ph;
            f[6] += U[0]*U[1]*T[2]*ph;
            f[7] += U[0]*T[1]*U[2]*ph;
            f[8] += T[0]*U[1]*U[2]*ph;
        }
    }
    for (int c = 0; c < (FINAL ? 9 : 3); c++)
        f[c] = warpSum(f[c]);
    if (lane != 0)
        return;
    for (int k = 0; k < 3; k++)
        erec[3*i+k] = -(K1*(double) bx.r[3*k]*f[0]+K2*(double) bx.r[3*k+1]*f[1]+K3*(double) bx.r[3*k+2]*f[2]);
    if (FINAL) {
        const float f2[3][3] = {{f[3], f[6], f[7]}, {f[6], f[4], f[8]}, {f[7], f[8], f[5]}};
        const float q = posq[i].w;
        const float4 d4 = dip[i];
        const float d[3] = {d4.x, d4.y, d4.z};
        float e[3];
        for (int a = 0; a < 3; a++)
            e[a] = K[a]*(d[0]*bx.r[a]+d[1]*bx.r[3+a]+d[2]*bx.r[6+a]);
        double F[3] = {0, 0, 0};
        for (int a = 0; a < 3; a++) {
            const double c = q*f[a]+e[0]*f2[a][0]+e[1]*f2[a][1]+e[2]*f2[a][2];
            for (int k = 0; k < 3; k++)
                F[k] -= K[a]*(double) bx.r[3*k+a]*c;
        }
        for (int k = 0; k < 3; k++)
            frec[3*i+k] = F[k];
    }
}

/** gather with one thread per atom, the order PC > 0 at compile time, the splines in registers.  On
 * sm_120 (CUDA 12.8) the FINAL instance is 2.3x faster than the other (7.6 vs 17.7 us, 4096 waters, order
 * 6; no spills in either), so the field gathers use it too, with frec as scratch. */
template<int PC, bool FINAL>
__global__ void gatherT(int n, int, const float4* posq, const float4* dip, int K1, int K2, int K3,
                        const float* theta, const float* dtheta, const float* d2theta, const int* gidx,
                        const float* phi, double* erec, double* frec) {
    const Box bx = c_box;
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n)
        return;
    constexpr int P = PC;
    const int K[3] = {K1, K2, K3};
    float T[3][P], U[3][P], V[3][P];
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
        for (int j = 0; j < P; j++) {
            const size_t o = ((size_t) a*n+i)*P+j;
            T[a][j] = theta[o];
            U[a][j] = dtheta[o];
            V[a][j] = FINAL ? d2theta[o] : 0.0f;
        }
    const int b1 = gidx[i], b2 = gidx[n+i], b3 = gidx[2*n+i];
    int g3[P];
#pragma unroll
    for (int j = 0; j < P; j++)
        g3[j] = wrapDown(b3-j, K3);
    float f[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int j1 = 0; j1 < P; j1++) {
        const int g1 = wrapDown(b1-j1, K1);
#pragma unroll
        for (int j2 = 0; j2 < P; j2++) {
            const float* row = phi+((size_t) g1*K2+wrapDown(b2-j2, K2))*K3;
            float s0 = 0, s1 = 0, s2 = 0;
#pragma unroll
            for (int j3 = 0; j3 < P; j3++) {
                const float ph = row[g3[j3]];
                s0 += T[2][j3]*ph;
                s1 += U[2][j3]*ph;
                if (FINAL)
                    s2 += V[2][j3]*ph;
            }
            f[0] += U[0][j1]*T[1][j2]*s0;
            f[1] += T[0][j1]*U[1][j2]*s0;
            f[2] += T[0][j1]*T[1][j2]*s1;
            if (FINAL) {
                f[3] += V[0][j1]*T[1][j2]*s0;
                f[4] += T[0][j1]*V[1][j2]*s0;
                f[5] += T[0][j1]*T[1][j2]*s2;
                f[6] += U[0][j1]*U[1][j2]*s0;
                f[7] += U[0][j1]*T[1][j2]*s1;
                f[8] += T[0][j1]*U[1][j2]*s1;
            }
        }
    }
    for (int k = 0; k < 3; k++)
        erec[3*i+k] = -(K1*(double) bx.r[3*k]*f[0]+K2*(double) bx.r[3*k+1]*f[1]+K3*(double) bx.r[3*k+2]*f[2]);
    if (FINAL) {
        const float f2[3][3] = {{f[3], f[6], f[7]}, {f[6], f[4], f[8]}, {f[7], f[8], f[5]}};
        const float q = posq[i].w;
        const float4 d4 = dip[i];
        const float d[3] = {d4.x, d4.y, d4.z};
        float e[3];
        for (int a = 0; a < 3; a++)
            e[a] = K[a]*(d[0]*bx.r[a]+d[1]*bx.r[3+a]+d[2]*bx.r[6+a]);
        double F[3] = {0, 0, 0};
        for (int a = 0; a < 3; a++) {
            const double c = q*f[a]+e[0]*f2[a][0]+e[1]*f2[a][1]+e[2]*f2[a][2];
            for (int k = 0; k < 3; k++)
                F[k] -= K[a]*(double) bx.r[3*k+a]*c;
        }
        for (int k = 0; k < 3; k++)
            frec[3*i+k] = F[k];
    }
}

/** The field gather only (erec = -grad phi at every atom), 8 lanes per atom, lane l < P doing the plane
 * j1 = l: more loads in flight than one thread per atom (gatherT). */
template<int PC>
__global__ void gatherField8(int n, int K1, int K2, int K3, const float* theta, const float* dtheta, const int* gidx,
                             const float* phi, double* erec) {
    const Box bx = c_box;
    const int t = blockIdx.x*blockDim.x+threadIdx.x;
    const int i = t>>3, l = t&7;
    constexpr int P = PC;
    float f0 = 0, f1 = 0, f2 = 0;
    if (i < n && l < P) {
        float T1[P], U1[P], T2[P], U2[P];
        int g3[P];
        const int b2 = gidx[n+i], b3 = gidx[2*n+i];
#pragma unroll
        for (int j = 0; j < P; j++) {
            T1[j] = theta[((size_t) n+i)*P+j];
            U1[j] = dtheta[((size_t) n+i)*P+j];
            T2[j] = theta[((size_t) 2*n+i)*P+j];
            U2[j] = dtheta[((size_t) 2*n+i)*P+j];
            g3[j] = wrapDown(b3-j, K3);
        }
        const float T0 = theta[(size_t) i*P+l], U0 = dtheta[(size_t) i*P+l];
        const int g1 = wrapDown(gidx[i]-l, K1);
#pragma unroll
        for (int j2 = 0; j2 < P; j2++) {
            const float* row = phi+((size_t) g1*K2+wrapDown(b2-j2, K2))*K3;
            float s0 = 0, s1 = 0;
#pragma unroll
            for (int j3 = 0; j3 < P; j3++) {
                const float ph = row[g3[j3]];
                s0 += T2[j3]*ph;
                s1 += U2[j3]*ph;
            }
            f0 += U0*T1[j2]*s0;
            f1 += T0*U1[j2]*s0;
            f2 += T0*T1[j2]*s1;
        }
    }
    for (int o = 4; o > 0; o >>= 1) {
        f0 += __shfl_xor_sync(0xffffffff, f0, o);
        f1 += __shfl_xor_sync(0xffffffff, f1, o);
        f2 += __shfl_xor_sync(0xffffffff, f2, o);
    }
    if (i < n && l == 0)
        for (int k = 0; k < 3; k++)
            erec[3*i+k] = -(K1*(double) bx.r[3*k]*f0+K2*(double) bx.r[3*k+1]*f1+K3*(double) bx.r[3*k+2]*f2);
}

/** pairSetup<false> and fieldPassDual in one pass over the candidates: the first pair pass of a step
 * (the field of the charges and permanent dipoles dip1 and of the guess dip2) while the pairs are found. */
__global__ void pairSetupDual(int n, const int* order, const int* atomStart, const int* candJ, const float4* posq,
                              const float* rg, float rc2, float rp2, float beta, const float4* dip1,
                              const float4* dip2, int* PJ, float2* G12, float2* G03, int* pairEnd, double* fld1,
                              double* fld2, float rpre2, int* srCount, int* srJ, float2* srG, unsigned* srOver) {
    const Box bx = c_box;
    const int w = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    if (w >= n)
        return;
    const int i = order ? order[w] : w;
    const float4 pi = posq[i];
    const float rgi = rg[i];
    const unsigned below = (1u<<lane)-1;
    const int end = atomStart[i+1];
    int out = atomStart[i], srOut = 0;
    float e1[3] = {0, 0, 0}, e2[3] = {0, 0, 0};
    for (int k0 = atomStart[i]; k0 < end; k0 += 32) {
        const int k = k0+lane;
        bool inside = false;
        int jraw = 0, j = 0;
        float4 pj = make_float4(0, 0, 0, 0);
        float3 d = make_float3(0, 0, 0);
        if (k < end) {
            jraw = candJ[k];
            j = jraw&INDEX_MASK;
            pj = posq[j];
            d = make_float3(pi.x-pj.x, pi.y-pj.y, pi.z-pj.z);
            minImage(d, bx);
            inside = d.x*d.x+d.y*d.y+d.z*d.z < rp2;
        }
        const bool shortPair = inside && d.x*d.x+d.y*d.y+d.z*d.z < rpre2;   // (the preconditioner)
        const unsigned mask = __ballot_sync(0xffffffff, inside), smask = __ballot_sync(0xffffffff, shortPair);
        if (inside) {
            const float r2 = d.x*d.x+d.y*d.y+d.z*d.z, r = sqrtf(r2);
            float g[4] = {0, 0, 0, 0};
            if (r2 < rc2)
                pairKernels(rgi+rg[j], r, r > 0 ? 1/r : 0.0f, beta, g);
            if (shortPair) {
                const int qs = srOut+__popc(smask&below);
                if (qs < PRE_CAP) {
                    srJ[(size_t) i*PRE_CAP+qs] = j;
                    srG[(size_t) i*PRE_CAP+qs] = make_float2(g[1], g[2]);
                }
                else
                    atomicOr(srOver, 1u);
            }
            const int q = out+__popc(mask&below);
            PJ[q] = jraw;
            G12[q] = make_float2(g[1], g[2]);
            if (G03)
                G03[q] = make_float2(g[0], g[3]);
            const float x[3] = {d.x, d.y, d.z};
            const float4 a = dip1[j], b = dip2[j];
            const float ax = a.x*x[0]+a.y*x[1]+a.z*x[2], bxd = b.x*x[0]+b.y*x[1]+b.z*x[2];
            const float da[3] = {a.x, a.y, a.z}, db[3] = {b.x, b.y, b.z};
            for (int c = 0; c < 3; c++) {
                e1[c] += pj.w*x[c]*g[1]-da[c]*g[1]+ax*x[c]*g[2];
                e2[c] += -db[c]*g[1]+bxd*x[c]*g[2];
            }
        }
        out += __popc(mask);
        srOut += __popc(smask);
    }
    for (int c = 0; c < 3; c++) {
        e1[c] = warpSum(e1[c]);
        e2[c] = warpSum(e2[c]);
    }
    if (lane == 0) {
        pairEnd[i] = out;
        if (srCount)
            srCount[i] = min(srOut, PRE_CAP);
        for (int c = 0; c < 3; c++) {
            fld1[3*i+c] = e1[c];
            fld2[3*i+c] = e2[c];
        }
    }
}

// ------------------------------------------------------------------ two fields in one pass
// The field of the charges and permanent dipoles (set 1) and of the initial guess (set 2), needed together
// at the start of every step: one pair pass, one spread into two grids, one batched FFT pair, one gather.

__global__ void fieldPassDual(int n, const int* order, const int* atomStart, const int* pairEnd, const int* PJ,
                              const float4* posq, const float4* dip1, const float4* dip2, KernelSource ks,
                              double* fld1, double* fld2) {
    const Box bx = c_box;
    const int w = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    if (w >= n)
        return;
    const int i = order ? order[w] : w;
    const float4 pi = posq[i];
    const float rgi = ks.rg[i];
    float e1[3] = {0, 0, 0}, e2[3] = {0, 0, 0};
    for (int k = atomStart[i]+lane; k < pairEnd[i]; k += 32) {
        const int j = PJ[k]&INDEX_MASK;
        const float4 pj = posq[j];
        float3 v = make_float3(pi.x-pj.x, pi.y-pj.y, pi.z-pj.z);
        minImage(v, bx);
        const float4 g4 = ks.get(k, rgi, j, v, false);
        const float2 g = make_float2(g4.y, g4.z);
        const float x[3] = {v.x, v.y, v.z};
        const float4 a = dip1[j], b = dip2[j];
        const float ax = a.x*x[0]+a.y*x[1]+a.z*x[2], bxd = b.x*x[0]+b.y*x[1]+b.z*x[2];
        const float da[3] = {a.x, a.y, a.z}, db[3] = {b.x, b.y, b.z};
        for (int c = 0; c < 3; c++) {
            e1[c] += pj.w*x[c]*g.x-da[c]*g.x+ax*x[c]*g.y;
            e2[c] += -db[c]*g.x+bxd*x[c]*g.y;
        }
    }
    for (int c = 0; c < 3; c++) {
        e1[c] = warpSum(e1[c]);
        e2[c] = warpSum(e2[c]);
    }
    if (lane == 0)
        for (int c = 0; c < 3; c++) {
            fld1[3*i+c] = e1[c];
            fld2[3*i+c] = e2[c];
        }
}

/** spread<PC, true> of (q, dip1) into grid and spread<PC, false> of dip2 into grid + K1 K2 K3. */
template<int PC>
__global__ void spreadDual(int n, const float4* posq, const float4* dip1, const float4* dip2, int K1, int K2, int K3,
                           const float* theta, const float* dtheta, const int* gidx, ull* grid) {
    const Box bx = c_box;
    const int i = (blockIdx.x*blockDim.x+threadIdx.x)/32, lane = threadIdx.x%32;
    constexpr int p = PC;
    if (i >= n)
        return;
    const float q = posq[i].w;
    const float4 a = dip1[i], b = dip2[i];
    const int K[3] = {K1, K2, K3};
    float e[3], f[3];
    for (int c = 0; c < 3; c++) {
        e[c] = K[c]*(a.x*bx.r[c]+a.y*bx.r[3+c]+a.z*bx.r[6+c]);
        f[c] = K[c]*(b.x*bx.r[c]+b.y*bx.r[3+c]+b.z*bx.r[6+c]);
    }
    const int b1 = gidx[i], b2 = gidx[n+i], b3 = gidx[2*n+i];
    const size_t second = (size_t) K1*K2*K3;
    constexpr int p2 = p*p, iters = (p2*p+31)/32;
    SplineLanes sl(lane, p, n, i, theta, dtheta, nullptr);
#pragma unroll
    for (int it = 0; it < iters; it++) {
        const int c0 = lane+32*it, c = (c0 < p2*p ? c0 : 0);
        const int j1 = c/p2, j2 = (c/p)%p, j3 = c%p;
        float T[3], U[3], V[3];
        sl.get(j1, j2, j3, T, U, V);
        if (c0 >= p2*p)
            continue;
        const int g1 = wrapDown(b1-j1, K1), g2 = wrapDown(b2-j2, K2), g3 = wrapDown(b3-j3, K3);
        const float w12 = T[0]*T[1], u1 = U[0]*T[1], u2 = T[0]*U[1];
        const float v1 = (q*w12+e[0]*u1+e[1]*u2)*T[2]+e[2]*w12*U[2];
        const float v2 = (f[0]*u1+f[1]*u2)*T[2]+f[2]*w12*U[2];
        ull* g = grid+((size_t) g1*K2+g2)*K3+g3;
        addFix(g, v1);
        addFix(g+second, v2);
    }
}

// ------------------------------------------------------------------ vectors

/** out = fixed field + reciprocal field + cself d (3n). */
/** n1 words src1 -> dst1, n2 words src2 -> dst2 (dst: pinned host memory, mapped). */
__global__ void copyWords(const unsigned* src1, unsigned* dst1, int n1, const unsigned* src2, unsigned* dst2, int n2) {
    for (int k = threadIdx.x; k < n1; k += blockDim.x)
        dst1[k] = src1[k];
    for (int k = threadIdx.x; k < n2; k += blockDim.x)
        dst2[k] = src2[k];
}

__global__ void combineField(int m, const double* fld, const double* erec, const double* d, double cself, double* out) {
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x)
        out[k] = fld[k]+erec[k]+cself*d[k];
}

/** out[i] = (a + b)[3i..3i+2] in float (b may be nullptr), m = 3n. */
__global__ void toFloat(int m, const double* a, const double* b, float4* out) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < m/3; i += gridDim.x*blockDim.x) {
        const double* x = a+3*i;
        if (b)
            out[i] = make_float4((float) (x[0]+b[3*i]), (float) (x[1]+b[3*i+1]), (float) (x[2]+b[3*i+2]), 0.0f);
        else
            out[i] = make_float4((float) x[0], (float) x[1], (float) x[2], 0.0f);
    }
}

__global__ void addVec(int m, const double* a, const double* b, double* out) {
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x)
        out[k] = a[k]+b[k];
}

__global__ void peekStep(int n, double peek, const double* alpha, const double* r, double* x) {
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < 3*n; k += gridDim.x*blockDim.x)
        x[k] += peek*alpha[k/3]*r[k];
}

/** Per evaluation, set by the host before it (device memory): the initial guess and the history slots. */
struct Control {
    double c[4];    // predictor coefficients of the history h steps back (c[0] == 0: x = alpha b)
    int head;       // the slot of the newest history entry
    int pushSlot;   // the slot the solution goes into (-1: none)
};

constexpr int NHIST = 5;   // history slots: 4 for the predictor, the fifth for this step's solution (so an
                           // evaluation can be repeated: it never writes what its guess reads)
struct HistPtrs {          // (3 nGlobal each, the global order)
    double* h[NHIST];
};

/** x = sum_h c_h hist_(head+h) (through gidx) or alpha b; polarizable atoms only. */
__global__ void initialGuess(int n, const Control* ctl, const int* gidx, HistPtrs hp, const double* alpha,
                             const double* b, double* x) {
    const Control c = *ctl;
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < 3*n; k += gridDim.x*blockDim.x) {
        const double a = alpha[k/3];
        if (c.c[0] == 0) {
            x[k] = a*b[k];
            continue;
        }
        const size_t g = 3*(size_t) gidx[k/3]+k%3;
        double v = 0;
        for (int h = 0; h < 4; h++)
            if (c.c[h] != 0)
                v += c.c[h]*hp.h[(c.head+h)%NHIST][g];
        x[k] = (a > 0 ? v : 0.0);
    }
}

__global__ void zeroHistorySlot(size_t m, const Control* ctl, HistPtrs hp) {
    const int slot = ctl->pushSlot;
    if (slot < 0)
        return;
    for (size_t k = blockIdx.x*(size_t) blockDim.x+threadIdx.x; k < m; k += (size_t) gridDim.x*blockDim.x)
        hp.h[slot][k] = 0;
}

__global__ void pushHistory(int n, const Control* ctl, const int* gidx, HistPtrs hp, const double* x) {
    const int slot = ctl->pushSlot;
    if (slot < 0)
        return;
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < 3*n; k += gridDim.x*blockDim.x)
        hp.h[slot][3*(size_t) gidx[k/3]+k%3] = x[k];
}

/** f = ke (covalent + pair + reciprocal forces); acc[29] |d|^2, acc[30] sum mu^2/(2 alpha), acc[31..39]
 * sum E_rec,a d_b. */
template<bool STRAIN>
__global__ void finalReduce(int n, double ke, ull* frc, const double* fpair, const double* frec,
                            const double* erec, const double* d, const double* mu, const double* alpha, double* f,
                            double* acc, double* fHost) {
    double d2 = 0, pol = 0, S[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        for (int k = 0; k < 3; k++) {
            const int c = 3*i+k;
            const double fc = ke*((double) (long long) frc[c]/FIX+fpair[c]+frec[c]);
            frc[c] = 0;   // cleared for the next evaluation
            f[c] = fc;
            if (fHost)   // pinned host memory (computeF / compute)
                fHost[c] = fc;
            d2 += d[c]*d[c];
            if (STRAIN)
                for (int b = 0; b < 3; b++)
                    S[3*k+b] += erec[c]*d[3*i+b];
        }
        if (alpha[i] > 0)
            pol += (mu[3*i]*mu[3*i]+mu[3*i+1]*mu[3*i+1]+mu[3*i+2]*mu[3*i+2])/(2*alpha[i]);
    }
    d2 = warpSumD(d2);
    pol = warpSumD(pol);
    if (STRAIN)
        for (int c = 0; c < 9; c++)
            S[c] = warpSumD(S[c]);
    if (threadIdx.x%32 == 0) {
        atomicAdd(acc+29, d2);
        atomicAdd(acc+30, pol);
        if (STRAIN)
            for (int c = 0; c < 9; c++)
                atomicAdd(acc+31+c, S[c]);
    }
}

// CG kernels over many blocks; scal: [0] r.z, [1] norm, [2] err, [3] step, [4] beta; accumulators
// [5] p.Ap, [6] num / norm sum, [7] r.z new, [8] max (as the bits of a non-negative double); [9] iterations
constexpr int CG_THREADS = 256;

__device__ void blockAdd(double v, double* target) {
    __shared__ double buf[CG_THREADS/32];
    v = warpSumD(v);
    if (threadIdx.x%32 == 0)
        buf[threadIdx.x/32] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        double s = 0;
        for (int w = 0; w < CG_THREADS/32; w++)
            s += buf[w];
        atomicAdd(target, s);
    }
    __syncthreads();
}

__device__ void blockMaxBits(double v, ull* target) {
    for (int o = 16; o > 0; o >>= 1)
        v = fmax(v, __shfl_down_sync(0xffffffff, v, o));
    if (threadIdx.x%32 == 0)
        atomicMax(target, (ull) __double_as_longlong(v));
}

/** The CG loop goes on while err > tol and fewer than maxIter iterations (scal[9]); in a graph the
 * condition of its while node. */
__device__ inline void cgContinue(const double* scal, double tol, int maxIter, CondHandle h, bool useHandle) {
#if PGM_CUDA_GRAPHS
    if (useHandle)
        cudaGraphSetConditional(h, (scal[2] > tol && scal[9] < maxIter) ? 1u : 0u);
#else
    (void) scal; (void) tol; (void) maxIter; (void) h; (void) useHandle;
#endif
}

/** True in one thread of the block that finishes last (its reductions are then all in scal); the counter
 * (scal[15]) is reset. */
__device__ inline bool lastBlock(double* scal) {
    __shared__ bool last;
    unsigned* counter = (unsigned*) (scal+15);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        last = (atomicAdd(counter, 1u) == gridDim.x-1);
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence();
        *counter = 0;
        return true;
    }
    return false;
}

/** r = b - (x/alpha - T x), z = alpha r, p = z; sums r.z, |alpha b|, max |z|; then (the last block) the
 * scalars and the loop condition.  b = fldB + erecB + cs perm and T x = fldT + erecT + cs x directly from
 * the field parts if fldB (fieldDual), else from b and Tx. */
__global__ void cgInitK(int m, const double* b, const double* Tx, const double* fldB, const double* erecB,
                        const double* perm, const double* fldT, const double* erecT, double cs,
                        const double* alpha, const double* x, double* r, double* z, double* p, double* scal,
                        double tol, int maxIter, CondHandle h, bool useHandle, float4* pf, bool pre, double cj) {
    double rz = 0, nrm = 0, err = 0;
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x) {
        const double a = alpha[k/3];
        const double bk = fldB ? fldB[k]+erecB[k]+cs*perm[k] : b[k];
        const double tk = fldB ? fldT[k]+erecT[k]+cs*x[k] : Tx[k];
        const double rr = (a > 0 ? bk-(x[k]/a-tk) : 0.0), zz = a/(1-a*cj)*rr;   // cj: the self term in Jacobi
        r[k] = rr;
        nrm += fabs(a*bk);
        err = fmax(err, fabs(a*rr));
        if (pre)   // z, p and r.z: precondK
            continue;
        z[k] = zz;
        p[k] = zz;
        if (pf)   // p for the next field pass (toFloat)
            ((float*) pf)[4*(k/3)+k%3] = (float) zz;
        rz += rr*zz;
    }
    if (pre) {
        blockAdd(nrm, scal+6);
        blockMaxBits(err, (ull*) (scal+8));
        return;
    }
    blockAdd(rz, scal+7);
    blockAdd(nrm, scal+6);
    blockMaxBits(err, (ull*) (scal+8));
    if (lastBlock(scal)) {
        scal[0] = scal[7];
        scal[1] = scal[6]/m+1e-300;
        scal[2] = __longlong_as_double((long long) *(ull*) (scal+8))/scal[1];
        scal[5] = scal[6] = scal[7] = 0;
        scal[9] = 0;
        scal[11] = (scal[2] <= tol ? 0.0 : 1e9);   // the first iteration that met the tolerance
        *(ull*) (scal+8) = 0;
        cgContinue(scal, tol, maxIter, h, useHandle);
    }
}

/** Ap = p/alpha - T p, T p = fld + erec + cs p (the field parts of field(p)); sum p.Ap. */
__global__ void cgApply(int m, const double* fld, const double* erec, double cs, const double* alpha,
                        const double* p, double* Ap, double* scal) {
    double pAp = 0;
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x) {
        const double a = alpha[k/3];
        const double v = (a > 0 ? p[k]/a-(fld[k]+erec[k]+cs*p[k]) : 0.0);
        Ap[k] = v;
        pAp += p[k]*v;
    }
    blockAdd(pAp, scal+5);
}

/** x += step p, r' = r - step Ap, z' = alpha r'; sums z'.(r'-r), r'.z', max |z'|. */
__global__ void cgUpdate(int m, const double* alpha, double* x, double* r, double* z, const double* p, const double* Ap,
                         double* scal, bool pre, double cj) {
    const double step = scal[0]/fmax(scal[5], 1e-300);
    double num = 0, rz = 0, err = 0;
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x) {
        const double a = alpha[k/3];
        x[k] += step*p[k];
        const double rn = r[k]-step*Ap[k], zn = a/(1-a*cj)*rn;
        err = fmax(err, fabs(a*rn));   // the tolerance is on |alpha r| with every preconditioner
        if (!pre) {
            num += zn*(rn-r[k]);
            rz += rn*zn;
            z[k] = zn;
        }
        r[k] = rn;
    }
    if (!pre) {
        blockAdd(num, scal+6);
        blockAdd(rz, scal+7);
    }
    blockMaxBits(err, (ull*) (scal+8));
}

/**
 * The short-range preconditioner (as Tinker): z = M r, M = D + D T_s D, D = diag(alpha), T_s the direct-space
 * dipole field tensor of the pairs within the preconditioner radius (the pair kernels of this step); a
 * truncated list somewhere (srOver): M = D everywhere.  INIT: p = z (and its floats), sum r.z, then (the last
 * block) the scalars and the loop condition as cgInitK; else sum r.z and z.(r_new - r_old) = -step z.Ap.
 */
template<bool INIT>
__global__ void precondK(int n, const float4* posq, const int* srCount, const int* srJ, const float2* srG,
                         const unsigned* srOver, const double* alpha, const double* r, const double* Ap, double* z,
                         double* p, float4* pf, double* scal, double tol, int maxIter, CondHandle h, bool useHandle) {
    const Box bx = c_box;
    const bool useT = (*srOver == 0);
    const double step = INIT ? 0.0 : scal[0]/fmax(scal[5], 1e-300);
    double rz = 0, num = 0;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const double ai = alpha[i];
        double e[3] = {0, 0, 0};
        if (useT && ai > 0) {
            const float4 pi = posq[i];
            const int cnt = srCount[i];
            for (int s = 0; s < cnt; s++) {
                const int j = srJ[(size_t) i*PRE_CAP+s];
                const double aj = alpha[j];
                if (aj == 0)
                    continue;
                const float2 g = srG[(size_t) i*PRE_CAP+s];
                const float4 pj = posq[j];
                float3 v = make_float3(pi.x-pj.x, pi.y-pj.y, pi.z-pj.z);
                minImage(v, bx);
                const double u[3] = {aj*r[3*j], aj*r[3*j+1], aj*r[3*j+2]}, x[3] = {v.x, v.y, v.z};
                const double ux = u[0]*x[0]+u[1]*x[1]+u[2]*x[2];
                for (int c = 0; c < 3; c++)
                    e[c] += -u[c]*g.x+ux*x[c]*g.y;
            }
        }
        for (int c = 0; c < 3; c++) {
            const int k = 3*i+c;
            const double zk = ai*(r[k]+e[c]);
            z[k] = zk;
            rz += r[k]*zk;
            if (INIT) {
                p[k] = zk;
                ((float*) pf)[4*i+c] = (float) zk;
            }
            else
                num -= step*zk*Ap[k];
        }
    }
    blockAdd(rz, scal+7);
    if (!INIT)
        blockAdd(num, scal+6);
    else if (lastBlock(scal)) {
        const int m = 3*n;
        scal[0] = scal[7];
        scal[1] = scal[6]/m+1e-300;
        scal[2] = __longlong_as_double((long long) *(ull*) (scal+8))/scal[1];
        scal[5] = scal[6] = scal[7] = 0;
        scal[9] = 0;
        scal[11] = (scal[2] <= tol ? 0.0 : 1e9);
        *(ull*) (scal+8) = 0;
        cgContinue(scal, tol, maxIter, h, useHandle);
    }
}

/** p = z + beta p, beta = num / r.z (Polak-Ribiere); then (the last block) the scalars of the next
 * iteration and the loop condition. */
__global__ void cgDirection(int m, const double* z, double* p, double* scal, double tol, int maxIter, CondHandle h,
                            bool useHandle, float4* pf) {
    const double beta = scal[6]/fmax(scal[0], 1e-300);
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x) {
        const double v = z[k]+beta*p[k];
        p[k] = v;
        if (pf)   // p for the next field pass (toFloat; w stays 0)
            ((float*) pf)[4*(k/3)+k%3] = (float) v;
    }
    if (lastBlock(scal)) {
        scal[3] = scal[0]/fmax(scal[5], 1e-300);
        scal[4] = scal[6]/fmax(scal[0], 1e-300);
        scal[0] = scal[7];
        scal[2] = __longlong_as_double((long long) *(ull*) (scal+8))/scal[1];
        scal[5] = scal[6] = scal[7] = 0;
        scal[9] += 1;
        if (scal[11] > 1e8 && scal[2] <= tol)
            scal[11] = scal[9];
        *(ull*) (scal+8) = 0;
        cgContinue(scal, tol, maxIter, h, useHandle);
    }
}

long g_allocations = 0;   // device (re)allocations so far: a graph is valid while this is unchanged

template<class T>
struct DevArray {
    T* ptr = nullptr;
    size_t size = 0;
    void resize(size_t n) {
        if (n <= size && ptr)
            return;
        if (ptr)
            cudaFree(ptr);
        CU(cudaMalloc(&ptr, std::max<size_t>(n, 1)*sizeof(T)));
        size = n;
        g_allocations++;
    }
    void upload(const T* h, size_t n) {   // synchronous
        resize(n);
        if (n) {
            CU(cudaMemcpy(ptr, h, n*sizeof(T), cudaMemcpyHostToDevice));
            CU(cudaStreamSynchronize(0));   // (pageable: the DMA may still run) before the engine stream reads it
        }
    }
    void download(T* h, size_t n) const {   // synchronous
        if (n)
            CU(cudaMemcpy(h, ptr, n*sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero(size_t n) {   // synchronous
        resize(n);
        CU(cudaMemset(ptr, 0, std::max<size_t>(n, 1)*sizeof(T)));
        CU(cudaDeviceSynchronize());
    }
    ~DevArray() {
        if (ptr)
            cudaFree(ptr);
    }
};

template<class T>
struct HostArray {   // pinned
    T* ptr = nullptr;
    size_t size = 0;
    void resize(size_t n) {
        if (n <= size && ptr)
            return;
        if (ptr)
            cudaFreeHost(ptr);
        CU(cudaMallocHost(&ptr, std::max<size_t>(n, 1)*sizeof(T)));
        size = n;
    }
    ~HostArray() {
        if (ptr)
            cudaFreeHost(ptr);
    }
};

double wallSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int blocksFor(size_t work, int threads) {
    return (int) std::min<size_t>((work+threads-1)/threads, 65535);
}

constexpr int NACC = 48;   // acc: [0] pair energy, [1..9] pair S, [10] k-space energy, [11..19] k-space S,
                           // [20..28] covalent S, [29] |d|^2, [30] polarization, [31..39] E_rec d, [40] GVDW

} // namespace

/* One evaluation is queued on the engine's stream in three parts: before the CG (pairs, splines, the
 * field of the charges and permanent dipoles, the initial guess and its field), the CG iterations, and
 * after (peek, history, final pass, forces and reductions, the copies to the host).  With CUDA >= 12.4
 * all of it is one CUDA graph, the CG loop a while node whose condition the CG kernels set, so a step
 * costs one graph launch and one synchronisation; the graph is captured again when device buffers were
 * reallocated.  Without graphs (older CUDA, PGM_CUDA_NO_GRAPH) the parts run directly and the host
 * checks the residual after every iteration. */
struct Engine::Impl {
    Settings s;
    int n = 0, nGlobal = 0, ncov = 0, ncand = 0;
    double q2 = 0, Qtot = 0;
    bool polarizable = false;
    std::vector<double> alpha;
    cudaStream_t st = nullptr;
    // device
    DevArray<float4> posq, posBuild, dipF, dipF2, posSorted;
    DevArray<float2> G12, G03;
    DevArray<float> rg, theta, dtheta, d2theta, grid, mod1, mod2, mod3;
    DevArray<double> alphaD, perm, b, x, r, z, p, Ap, T, erec, frec, dTot, scal, acc, Etot, fld, fpair, fOut, erec2, fld2;
    DevArray<int> PJ, atomStart, pairEnd, candJ, candT, covStart, covPartnerSorted, covAtom, covPartner, gidx, globalIdx;
    DevArray<double> covStrengthSorted, covStrength;
    DevArray<ull> frc, gridFix;
    DevArray<cufftComplex> cgrid;
    DevArray<Control> ctl;
    // GVDW (combination rule): per atom (sqrt A, sqrt C6, b, 0); the skipped partners per atom (CSR)
    bool gvdwOn = false, gvdwSlater = true, skipsMarked = false;
    double rvdw = 0;
    DevArray<float4> gvP;
    DevArray<int> skipStart, skipJ;
    cufftHandle planF = 0, planB = 0, planF2 = 0, planB2 = 0;   // single, batched (two grids)
    // the PME work on a second stream beside the pair kernels (fork / join; in a graph: parallel branches)
    cudaStream_t st2 = nullptr;
    cudaEvent_t evFork = nullptr, evJoin = nullptr;
    bool useFork = std::getenv("PGM_CUDA_NO_FORK") == nullptr;
    cudaStream_t ps() const { return useFork ? st2 : st; }
    void fork() {
        if (useFork) {
            CU(cudaEventRecord(evFork, st));
            CU(cudaStreamWaitEvent(st2, evFork, 0));
        }
    }
    // the pair pass after the spread of the PME branch (spreadFirst): with both started together the pair
    // pass takes the GPU and the spread waits for its blocks; after the spread the rest of the PME branch
    // (FFTs, convolution, gather: small grids) runs beside the pair pass
    bool spreadFirst = std::getenv("PGM_CUDA_SPREADFIRST") != nullptr;   // (2 percent slower, 4096 waters)
    cudaEvent_t evMid = nullptr;
    std::function<void()> midWork;
    void pairAfterSpread(const std::function<void()>& pairs) {   // (between fork and the PME call)
        if (useFork && spreadFirst)
            midWork = pairs;
        else
            pairs();
    }
    void runMid() {   // in the PME branch after its spread
        if (!midWork)
            return;
        CU(cudaEventRecord(evMid, st2));
        CU(cudaStreamWaitEvent(st, evMid, 0));
        std::function<void()> w;
        w.swap(midWork);
        w();
    }
    void join() {
        runMid();   // (a PME call without the spread hook)
        if (useFork) {
            CU(cudaEventRecord(evJoin, st2));
            CU(cudaStreamWaitEvent(st, evJoin, 0));
        }
    }
    // the results to the host written by a kernel (pinned memory, mapped) instead of copy nodes
    bool useExport = std::getenv("PGM_CUDA_NO_EXPORT") == nullptr;
    bool useGather8 = std::getenv("PGM_CUDA_GATHER8") != nullptr;   // gatherField8 for the field gathers (slower on sm_120)
    // the short-range preconditioner (PGM_CUDA_PRECOND = its radius [nm]; 0: Jacobi), with the fused pair setup
    double rpre = std::getenv("PGM_CUDA_PRECOND") ? std::atof(std::getenv("PGM_CUDA_PRECOND")) : 0.0;
    DevArray<int> srCount, srJ;
    DevArray<float2> srG;
    DevArray<unsigned> srOver;
    bool curPre = false;   // the CG being enqueued uses it (set by enqueueCgInit)
    // Jacobi with the self term of the diagonal, alpha / (1 - alpha c_self) (as pmemd), or alpha alone
    bool useJacobiSelf = std::getenv("PGM_CUDA_JACOBI_SELF") != nullptr;
    double jacobiSelf() const {
        return useJacobiSelf ? 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI) : 0.0;
    }
    bool preFor(bool dual) const {   // this evaluation: the fused pair setup fills the short-range list
        return rpre > 0 && dual && polarizable && !externalList && storeKernels;
    }
    // CG iterations unrolled in the graph before its while node: the fewest of the recent evaluations (the
    // while node launches its body with a latency of its own); curUnroll: those of the graph being captured
    bool useUnroll = std::getenv("PGM_CUDA_NO_UNROLL") == nullptr;
    int recentIt[16] = {}, numRecent = 0, recentPos = 0, curUnroll = 0, execUnroll[4] = {-1, -1, -1, -1};
    int unrollWanted() const {
        if (!useUnroll || numRecent < 4)
            return 0;
        int k = 1 << 30;
        for (int i = 0; i < std::min(numRecent, 16); i++)
            k = std::min(k, recentIt[i]);
        return std::max(0, std::min(k, std::min(s.maxIter, 8)));
    }
    // own pair list
    bool externalList = false, haveList = false;
    long numBuilds = 0;
    double buildBox[3][3] = {}, skinNow = 0;   // the box and the skin of the current list
    DevArray<int> cellKey, cellKeySorted, cellAtom, cellAtomSorted, cellStart, cellEnd, count, slabCount;
    DevArray<unsigned char> cubTemp;
    DevArray<unsigned> maxDisp;
    // predictor history (device, global order): slot (head + h) % NHIST is h steps back
    DevArray<double> hist[NHIST];
    int head = 0, numHist = 0;
    // host staging (pinned)
    HostArray<float4> posqH;
    HostArray<double> fH, accH, scalH;
    HostArray<unsigned> dispH;
    HostArray<Box> boxH;
    HostArray<Control> ctlH;
    // graphs: [strain + 2 dual]
    bool useGraphs = false;
    cudaGraphExec_t exec[4] = {nullptr, nullptr, nullptr, nullptr};
    long execAllocations[4] = {-1, -1, -1, -1};
    std::array<long long, 10> execKey[4];
    std::array<long long, 10> graphKey() const {   // the host values the graph's kernels take as parameters
        return {n, ncov, nGlobal, polarizable ? 1 : 0, externalList ? 1 : 0, gvdwOn ? 1 : 0, usePosF ? 1 : 0,
                (long long) devPos, (long long) devForce, devStride};
    }
    double pairRadius() const { return gvdwOn ? std::max(s.cutoff, rvdw) : s.cutoff; }
    void markSkipList();
    // results
    mutable std::vector<double> mu, permH;
    mutable bool muValid = false, permValid = false;
    int lastIterations = 0;
    double lastResidual = 0;
    double boxD[3][3], recipD[3][3], volume = 0;
    // PGM_CUDA_TIMING: host-side phases of compute()
    bool timing = std::getenv("PGM_CUDA_TIMING") != nullptr;
    double tPhase[4] = {0, 0, 0, 0};
    long numCalls = 0;

    void field(bool withQ, const double* dvec, double* out, bool inFloat = false);   // out == nullptr: leave fld + erec
    void fieldDual(bool pairsDone);   // b = field(q, perm), T = field(x), together
    bool dualOrder() const { return s.order == 4 || s.order == 5 || s.order == 6 || s.order == 8; }
    void pme(bool withQ, bool final, int energyMode, double* acc10);
    double boxChange() const;
    bool listStaleBefore() const;
    bool listStaleAfter() const;
    double lastD2 = 0, lastS2 = 0;   // the last step's displacements (squared): since the build, the step
    bool prevValid = false;          // posPrev holds the last evaluation's positions in this order
    long numRepeats = 0, numRemaps = 0;
    std::vector<int> giHost;   // the global indices of the local atoms (setAtoms)
    double builtRadius = 0;    // the pair radius the list was built for
    void remapList(const std::vector<int>& newOf);
    DevArray<int> rmNewOf, rmStart, rmCand;
    DevArray<float4> rmA, rmB;
    DevArray<float4> posPrev;
    void buildList();
    const int* order() const { return externalList ? nullptr : cellAtomSorted.ptr; }   // of the pair kernels
    // the kernels stored by pairSetup (default) or computed in every pass (PGM_CUDA_KERNELS_ON_THE_FLY: slower
    // with 4 or more passes per step on sm_120)
    bool storeKernels = std::getenv("PGM_CUDA_KERNELS_ON_THE_FLY") == nullptr;
    size_t gatherSmem = std::getenv("PGM_CUDA_GATHER_SMEM") ? std::atoi(std::getenv("PGM_CUDA_GATHER_SMEM")) : 0;
    // G_0, G_3 (the final pass only) stored too (PGM_CUDA_STORE_G03), else computed there
    bool storeG03 = std::getenv("PGM_CUDA_STORE_G03") != nullptr;
    KernelSource kernels() const {
        return KernelSource{storeKernels ? G12.ptr : nullptr, storeKernels && storeG03 ? G03.ptr : nullptr, rg.ptr,
                            (float) (s.cutoff*s.cutoff), (float) s.beta};
    }
    HistPtrs histPtrs() { return HistPtrs{{hist[0].ptr, hist[1].ptr, hist[2].ptr, hist[3].ptr, hist[4].ptr}}; }
    void enqueuePre(bool dual);
    void enqueueCgInit(CondHandle h, bool useHandle, bool dual);
    void enqueueCgIteration(CondHandle h, bool useHandle);
    void enqueuePost(bool strain);
    void captureGraph(bool strain, bool dual);
    bool tryCapture(bool strain, bool dual);
    Terms evaluate(const double* pos, const float* posF, const double box[3][3], bool newStep, double* f,
                   double* strain);
    // evaluate = begin (inputs, list, guess, the evaluation enqueued) + finish (wait, check, outputs)
    void begin(const double* pos, const float* posF, const double box[3][3], bool newStep, bool wantStrain);
    bool finish(double* f, double* strain, Terms& t, bool canRepeat = true);
    void settle() {   // completes an evaluation launched by launchDevice (its bookkeeping)
        if (inFlight) {
            Terms t;
            finish(nullptr, nullptr, t, false);
        }
    }
    bool inFlight = false;
    cudaEvent_t doneEvent = nullptr;
    Terms lastTerms;
    long numLate = 0;   // out-of-date lists found after the forces were used (launchDevice without finishDevice)
    bool retry = false;        // the last evaluation used an out-of-date list: the next repeats it (same guess)
    bool pendingPush = false;  // the evaluation in flight enters the predictor history
    int pendingIt = 0;
    double pendingErr = 0, tStamp[3] = {0, 0, 0}, tLaunch = 0;
    // positions straight from the caller's (registered) float array
    const float* regPtr = nullptr;
    size_t regBytes = 0;
    bool usePosF = false;
    DevArray<float> pos3, qF;
    bool registerPositions(const float* p);
    void releasePositions() {   // (a registered range the caller frees would break later copies from its pages)
        if (regPtr) {
            cudaStreamSynchronize(st);
            cudaHostUnregister(const_cast<float*>(regPtr));
            regPtr = nullptr;
            regBytes = 0;
            g_allocations++;   // the graph copy read from it
        }
    }
    void uploadPositions();
    // device input / output (computeDevice)
    const float4* devPos = nullptr;
    long long* devForce = nullptr;
    int devStride = 0;
    double devScale = 0;
    cudaEvent_t devEvent = nullptr;
    void releaseGraphs();
};

/** The box change since the list build (sum of the row changes) [nm]. */
double Engine::Impl::boxChange() const {
    double dH = 0;
    for (int a = 0; a < 3; a++) {
        double r2 = 0;
        for (int c = 0; c < 3; c++)
            r2 += (boxD[a][c]-buildBox[a][c])*(boxD[a][c]-buildBox[a][c]);
        dH += std::sqrt(r2);
    }
    return dH;
}

/* The list check runs inside the evaluation (displacements, the first kernel): before a step the host
 * predicts from the last step's displacement plus twice the last step's largest move whether the list will
 * still be good, and rebuilds if not; after the step it checks the actual displacement and, should it be
 * beyond the skin after all, rebuilds and repeats the evaluation (which the history layout allows). */
bool Engine::Impl::listStaleBefore() const {
    if (externalList)
        return false;
    if (!haveList)
        return true;
    return 2*(std::sqrt(lastD2)+2*std::sqrt(lastS2))+boxChange() > skinNow;
}

bool Engine::Impl::listStaleAfter() const {
    if (externalList)
        return false;
    float d2;
    std::memcpy(&d2, dispH.ptr, sizeof(float));
    return 2*std::sqrt((double) d2)+boxChange() > skinNow;
}

void Engine::Impl::buildList() {
    // plane spacings of the box: 1/|column a of the reciprocal box|; the list radius stays below half the
    // smallest (minimum image), the skin shrinks if needed (small boxes)
    double w[3], wmin = 1e300;
    for (int a = 0; a < 3; a++) {
        w[a] = 1.0/std::sqrt(recipD[0][a]*recipD[0][a]+recipD[1][a]*recipD[1][a]+recipD[2][a]*recipD[2][a]);
        wmin = std::min(wmin, w[a]);
    }
    skinNow = std::min(s.skin, 0.5*wmin*(1-1e-6)-pairRadius());
    if (skinNow < 0.01)
        throw std::runtime_error("pgm_cuda: the cut-off (+ 0.01 nm) beyond half the box (minimum image)");
    const double rl = pairRadius()+skinNow;
    // cells rl / 2 wide: (5/6)^3 = 0.58 of the volume searched with cells rl wide
    int nc[3];
    for (int a = 0; a < 3; a++)
        nc[a] = std::max(1, std::min((int) (LIST_R*w[a]/rl), 1024));
    const int ncells = nc[0]*nc[1]*nc[2];
    const int3 ncv = make_int3(nc[0], nc[1], nc[2]);
    cellKey.resize(n);
    cellKeySorted.resize(n);
    cellAtom.resize(n);
    cellAtomSorted.resize(n);
    cellStart.resize(ncells);
    cellEnd.resize(ncells);
    count.resize(n+1);
    atomStart.resize(n+1);
    CU(cudaMemsetAsync(cellStart.ptr, 0, ncells*sizeof(int), st));
    CU(cudaMemsetAsync(cellEnd.ptr, 0, ncells*sizeof(int), st));
    cellKeys<<<blocksFor(n, 256), 256, 0, st>>>(n, posq.ptr, ncv, cellKey.ptr, cellAtom.ptr);
    int bits = 1;
    while ((1<<bits) < ncells)
        bits++;
    size_t sortBytes = 0, scanBytes = 0;
    CU(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, cellKey.ptr, cellKeySorted.ptr, cellAtom.ptr,
                                       cellAtomSorted.ptr, n, 0, bits, st));
    CU(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, count.ptr, atomStart.ptr, n+1, st));
    cubTemp.resize(std::max(sortBytes, scanBytes));
    CU(cub::DeviceRadixSort::SortPairs(cubTemp.ptr, sortBytes, cellKey.ptr, cellKeySorted.ptr, cellAtom.ptr,
                                       cellAtomSorted.ptr, n, 0, bits, st));
    cellRanges<<<blocksFor(n, 256), 256, 0, st>>>(n, cellKeySorted.ptr, cellStart.ptr, cellEnd.ptr);
    posSorted.resize(n);
    gatherPositions<<<blocksFor(n, 256), 256, 0, st>>>(n, cellAtomSorted.ptr, posq.ptr, posSorted.ptr);
    const int slabs = std::min(nc[0], 2*LIST_R+1)*std::min(nc[1], 2*LIST_R+1);
    slabCount.resize((size_t) (2*LIST_R+1)*(2*LIST_R+1)*n);
    CU(cudaMemsetAsync(slabCount.ptr, 0, (size_t) slabs*n*sizeof(int), st));
    const dim3 lgrid(ncells, slabs);
    listBuild<false><<<lgrid, LIST_THREADS, 0, st>>>(n, posSorted.ptr, cellAtomSorted.ptr, ncv, cellStart.ptr,
                                                     cellEnd.ptr, (float) (rl*rl), slabCount.ptr, nullptr, nullptr);
    sumSlabs<<<blocksFor(n+1, 256), 256, 0, st>>>(n, slabs, slabCount.ptr, count.ptr);
    CU(cub::DeviceScan::ExclusiveSum(cubTemp.ptr, scanBytes, count.ptr, atomStart.ptr, n+1, st));
    CU(cudaMemcpyAsync(&ncand, atomStart.ptr+n, sizeof(int), cudaMemcpyDeviceToHost, st));
    CU(cudaStreamSynchronize(st));
    // room for growth (fewer reallocations: each one means a new graph)
    if (candJ.size < (size_t) ncand || PJ.size < (size_t) ncand) {
        const size_t cap = (size_t) (1.15*ncand)+1024;
        candJ.resize(cap);
        PJ.resize(cap);
        if (storeKernels)
            G12.resize(cap);
        if (storeKernels && storeG03)
            G03.resize(cap);
    }
    listBuild<true><<<lgrid, LIST_THREADS, 0, st>>>(n, posSorted.ptr, cellAtomSorted.ptr, ncv, cellStart.ptr,
                                                    cellEnd.ptr, (float) (rl*rl), slabCount.ptr, atomStart.ptr,
                                                    candJ.ptr);
    CU(cudaMemcpyAsync(posBuild.ptr, posq.ptr, n*sizeof(float4), cudaMemcpyDeviceToDevice, st));
    std::memcpy(buildBox, boxD, sizeof(buildBox));
    haveList = true;
    builtRadius = pairRadius();
    lastD2 = 0;
    skipsMarked = false;
    numBuilds++;
}

void Engine::Impl::markSkipList() {
    if (gvdwOn && skipStart.ptr)
        markSkips<<<(n+3)/4, 128, 0, st>>>(n, atomStart.ptr, candJ.ptr, skipStart.ptr, skipJ.ptr);
    skipsMarked = true;
}

void Engine::Impl::pme(bool withQ, bool final, int energyMode, double* accDev) {
    const int K1 = s.grid[0], K2 = s.grid[1], K3 = s.grid[2], P = s.order;
    const size_t gsize = (size_t) K1*K2*K3;
    // the order as a compile-time constant for the usual orders
#define PGM_ORDER_DISPATCH(CALL) \
    switch (P) {                 \
        case 4: CALL(4); break;  \
        case 5: CALL(5); break;  \
        case 6: CALL(6); break;  \
        case 8: CALL(8); break;  \
        default: CALL(0);        \
    }
#define PGM_SPREAD(PC)                                                                                          \
    if (withQ)                                                                                                  \
        spread<PC, true><<<(n+3)/4, 128, 0, ps()>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr, dtheta.ptr,  \
                                                  gidx.ptr, gridFix.ptr);                                       \
    else                                                                                                        \
        spread<PC, false><<<(n+3)/4, 128, 0, ps()>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr, dtheta.ptr, \
                                                   gidx.ptr, gridFix.ptr)
    PGM_ORDER_DISPATCH(PGM_SPREAD)
    runMid();
    fixedToReal<<<blocksFor(gsize, 256), 256, 0, ps()>>>(gsize, gridFix.ptr, grid.ptr);
    CUFFT(cufftExecR2C(planF, grid.ptr, cgrid.ptr));
    const size_t csize = (size_t) K1*K2*(K3/2+1);
    const float beta = (float) s.beta;
    if (final && energyMode == 2)
        convolution<2><<<blocksFor(csize, 256), 256, 0, ps()>>>(K1, K2, K3, beta, mod1.ptr, mod2.ptr, mod3.ptr,
                                                               cgrid.ptr, accDev, 1);
    else if (final && energyMode == 1)
        convolution<1><<<blocksFor(csize, 256), 256, 0, ps()>>>(K1, K2, K3, beta, mod1.ptr, mod2.ptr, mod3.ptr,
                                                               cgrid.ptr, accDev, 1);
    else
        convolution<0><<<blocksFor(csize, 256), 256, 0, ps()>>>(K1, K2, K3, beta, mod1.ptr, mod2.ptr, mod3.ptr,
                                                               cgrid.ptr, nullptr, 1);
    CUFFT(cufftExecC2R(planB, cgrid.ptr, grid.ptr));
    // the usual orders: one thread per atom (gatherT, FINAL also for the field gathers, see there)
#define PGM_GATHER(PC)                                                                                          \
    if (PC > 0 && !final && useGather8)                                                                         \
        gatherField8<PC ? PC : 4><<<blocksFor(8*(size_t) n, 128), 128, 0, ps()>>>(n, K1, K2, K3, theta.ptr,      \
                                                                                 dtheta.ptr, gidx.ptr, grid.ptr,  \
                                                                                 erec.ptr);                       \
    else if (PC > 0)                                                                                            \
        gatherT<PC ? PC : 4, true><<<(n+63)/64, 64, gatherSmem, ps()>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr,   \
                                                            dtheta.ptr, d2theta.ptr, gidx.ptr, grid.ptr,       \
                                                            erec.ptr, frec.ptr);                               \
    else if (final)                                                                                             \
        gather<PC, true><<<(n+3)/4, 128, 0, ps()>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr, dtheta.ptr,  \
                                                  d2theta.ptr, gidx.ptr, grid.ptr, erec.ptr, frec.ptr);         \
    else                                                                                                        \
        gather<PC, false><<<(n+3)/4, 128, 0, ps()>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr, dtheta.ptr, \
                                                   d2theta.ptr, gidx.ptr, grid.ptr, erec.ptr, nullptr)
    PGM_ORDER_DISPATCH(PGM_GATHER)
#undef PGM_GATHER
#undef PGM_SPREAD
#undef PGM_ORDER_DISPATCH
}

/** out = T d (+ the charges' field if withQ): direct + reciprocal + self, for the dipoles dvec (double). */
void Engine::Impl::field(bool withQ, const double* dvec, double* out, bool inFloat) {
    const int m = 3*n;
    if (!inFloat)
        toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, dvec, nullptr, dipF.ptr);
    const int warpsPerBlock = 4, threads = 32*warpsPerBlock;
    const int blocks = (n+warpsPerBlock-1)/warpsPerBlock;
    fork();
    pairAfterSpread([&] {
        if (withQ)
            fieldPass<true><<<blocks, threads, 0, st>>>(n, order(), atomStart.ptr, pairEnd.ptr, PJ.ptr, posq.ptr,
                                                        dipF.ptr, kernels(), fld.ptr);
        else
            fieldPass<false><<<blocks, threads, 0, st>>>(n, order(), atomStart.ptr, pairEnd.ptr, PJ.ptr, posq.ptr,
                                                         dipF.ptr, kernels(), fld.ptr);
    });
    pme(withQ, false, 0, nullptr);
    join();
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    if (out)
        combineField<<<blocksFor(m, 256), 256, 0, st>>>(m, fld.ptr, erec.ptr, dvec, cs, out);
}

void Engine::Impl::fieldDual(bool pairsDone) {
    const int m = 3*n, K1 = s.grid[0], K2 = s.grid[1], K3 = s.grid[2], P = s.order;
    const size_t gsize = (size_t) K1*K2*K3, csize = (size_t) K1*K2*(K3/2+1);
    if (!pairsDone) {   // (else the caller launched pairSetupDual, the pair part, after fork())
        toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, perm.ptr, nullptr, dipF.ptr);
        toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, x.ptr, nullptr, dipF2.ptr);
        const int warpsPerBlock = 4, threads = 32*warpsPerBlock, blocks = (n+warpsPerBlock-1)/warpsPerBlock;
        fork();
        pairAfterSpread([=] {
            fieldPassDual<<<blocks, threads, 0, st>>>(n, order(), atomStart.ptr, pairEnd.ptr, PJ.ptr, posq.ptr,
                                                      dipF.ptr, dipF2.ptr, kernels(), fld.ptr, fld2.ptr);
        });
    }
    const cudaStream_t q = ps();
#define PGM_DUAL(PC)                                                                                           \
    spreadDual<PC><<<(n+3)/4, 128, 0, q>>>(n, posq.ptr, dipF.ptr, dipF2.ptr, K1, K2, K3, theta.ptr, dtheta.ptr, \
                                            gidx.ptr, gridFix.ptr);                                            \
    runMid();                                                                                                  \
    fixedToReal<<<blocksFor(2*gsize, 256), 256, 0, q>>>(2*gsize, gridFix.ptr, grid.ptr);                      \
    CUFFT(cufftExecR2C(planF2, grid.ptr, cgrid.ptr));                                                          \
    convolution<0><<<blocksFor(2*csize, 256), 256, 0, q>>>(K1, K2, K3, (float) s.beta, mod1.ptr, mod2.ptr,     \
                                                             mod3.ptr, cgrid.ptr, nullptr, 2);                 \
    CUFFT(cufftExecC2R(planB2, cgrid.ptr, grid.ptr));                                                          \
    /* both gathers with gatherT<FINAL> (frec as scratch): faster than one kernel for both on sm_120 */      \
    if (useGather8) {                                                                                          \
        gatherField8<PC><<<blocksFor(8*(size_t) n, 128), 128, 0, q>>>(n, K1, K2, K3, theta.ptr, dtheta.ptr,      \
                                                                       gidx.ptr, grid.ptr, erec.ptr);            \
        gatherField8<PC><<<blocksFor(8*(size_t) n, 128), 128, 0, q>>>(n, K1, K2, K3, theta.ptr, dtheta.ptr,      \
                                                                       gidx.ptr, grid.ptr+gsize, erec2.ptr);     \
    }                                                                                                          \
    else {                                                                                                     \
    gatherT<PC, true><<<(n+63)/64, 64, gatherSmem, q>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr,         \
                                                         dtheta.ptr, d2theta.ptr, gidx.ptr, grid.ptr, erec.ptr,   \
                                                         frec.ptr);                                               \
    gatherT<PC, true><<<(n+63)/64, 64, gatherSmem, q>>>(n, P, posq.ptr, dipF2.ptr, K1, K2, K3, theta.ptr,        \
                                                         dtheta.ptr, d2theta.ptr, gidx.ptr, grid.ptr+gsize,       \
                                                         erec2.ptr, frec.ptr);                                    \
    }
    switch (P) {
        case 4: PGM_DUAL(4); break;
        case 5: PGM_DUAL(5); break;
        case 6: PGM_DUAL(6); break;
        default: PGM_DUAL(8); break;   // dualOrder()
    }
#undef PGM_DUAL
    join();
    // b and T x are combined from the field parts in cgInitK
}

/** Pairs, covalent dipoles, splines; polarizable: b = the field of the charges and permanent dipoles, the
 * initial guess x and T x (dual: the guess from the history, both fields together). */
void Engine::Impl::uploadPositions() {
    if (devPos) {
        packPositions4<<<blocksFor(n, 256), 256, 0, st>>>(n, devPos, qF.ptr, posq.ptr);
        return;
    }
    CU(cudaMemcpyAsync(pos3.ptr, regPtr, 3*(size_t) n*sizeof(float), cudaMemcpyHostToDevice, st));
    packPositions<<<blocksFor(n, 256), 256, 0, st>>>(n, pos3.ptr, qF.ptr, posq.ptr);
}

/** Registers the caller's position array (once per address and size); false if CUDA refuses. */
bool Engine::Impl::registerPositions(const float* p) {
    const size_t bytes = 3*(size_t) n*sizeof(float);
    if (p == regPtr && bytes == regBytes)
        return true;
    if (regPtr) {
        cudaHostUnregister(const_cast<float*>(regPtr));
        regPtr = nullptr;
        regBytes = 0;
    }
    if (std::getenv("PGM_CUDA_NO_REGISTER")
        || cudaHostRegister(const_cast<float*>(p), bytes, cudaHostRegisterDefault) != cudaSuccess) {
        cudaGetLastError();   // (clear it)
        return false;
    }
    regPtr = p;
    regBytes = bytes;
    pos3.resize(3*(size_t) n);
    g_allocations++;   // the graph's copy reads from this address
    return true;
}

void Engine::Impl::enqueuePre(bool dual) {
    const int m = 3*n;
    zeroSmall<<<1, 64, 0, st>>>(maxDisp.ptr, 2, scal.ptr, 16, acc.ptr, NACC, srOver.ptr, srOver.ptr ? 1 : 0);
    if (usePosF || devPos)
        uploadPositions();
    if (!externalList) {
        displacements<<<std::min(blocksFor(n, 256), 1024), 256, 0, st>>>(n, posq.ptr, posBuild.ptr, posPrev.ptr,
                                                                          maxDisp.ptr);
        if (useExport)
            copyWords<<<1, 32, 0, st>>>(maxDisp.ptr, dispH.ptr, 2, nullptr, nullptr, 0);
        else
            CU(cudaMemcpyAsync(dispH.ptr, maxDisp.ptr, 2*sizeof(unsigned), cudaMemcpyDeviceToHost, st));
    }
    const int warpsPerBlock = 4, threads = 32*warpsPerBlock;
    const int pblocks = (n+warpsPerBlock-1)/warpsPerBlock;
    const float rc2 = (float) (s.cutoff*s.cutoff), rp2 = (float) (pairRadius()*pairRadius()), beta = (float) s.beta;
    // the pairs together with the first field pass (dual, own list, stored kernels), or on their own
    const bool fused = dual && polarizable && !externalList && storeKernels;
    permDipolesKernel<<<blocksFor(n, 128), 128, 0, st>>>(n, posq.ptr, covStart.ptr, covPartnerSorted.ptr,
                                                         covStrengthSorted.ptr, perm.ptr);
#define PGM_SPLINES(PC)                                                                                     \
    splines<PC><<<blocksFor(3*n, 128), 128, 0, st>>>(n, s.order, posq.ptr, s.grid[0], s.grid[1], s.grid[2],    \
                                                     theta.ptr, dtheta.ptr, d2theta.ptr, gidx.ptr)
    switch (s.order) {
        case 4: PGM_SPLINES(4); break;
        case 5: PGM_SPLINES(5); break;
        case 6: PGM_SPLINES(6); break;
        case 8: PGM_SPLINES(8); break;
        default: PGM_SPLINES(0);
    }
#undef PGM_SPLINES
    if (fused) {   // the pair setup with the first pair pass, beside the PME of both dipole sets
        initialGuess<<<blocksFor(m, 256), 256, 0, st>>>(n, ctl.ptr, globalIdx.ptr, histPtrs(), alphaD.ptr, b.ptr,
                                                        x.ptr);   // the predictor: b not needed
        toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, perm.ptr, nullptr, dipF.ptr);
        toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, x.ptr, nullptr, dipF2.ptr);
        const bool pre = preFor(true);
        fork();
        pairAfterSpread([&] {
            pairSetupDual<<<pblocks, threads, 0, st>>>(n, cellAtomSorted.ptr, atomStart.ptr, candJ.ptr, posq.ptr,
                                                       rg.ptr, rc2, rp2, beta, dipF.ptr, dipF2.ptr, PJ.ptr, G12.ptr,
                                                       storeG03 ? G03.ptr : nullptr, pairEnd.ptr, fld.ptr, fld2.ptr,
                                                       pre ? (float) (rpre*rpre) : 0.0f, pre ? srCount.ptr : nullptr,
                                                       srJ.ptr, srG.ptr, srOver.ptr);
        });
        fieldDual(true);   // (its join)
        return;
    }
    if (externalList)
        pairSetup<true><<<pblocks, threads, 0, st>>>(n, nullptr, atomStart.ptr, candJ.ptr, candT.ptr, posq.ptr,
                                                     rg.ptr, rc2, rp2, beta, PJ.ptr, storeKernels ? G12.ptr : nullptr,
                                                     storeG03 ? G03.ptr : nullptr, pairEnd.ptr);
    else
        pairSetup<false><<<pblocks, threads, 0, st>>>(n, cellAtomSorted.ptr, atomStart.ptr, candJ.ptr, nullptr,
                                                      posq.ptr, rg.ptr, rc2, rp2, beta, PJ.ptr,
                                                      storeKernels ? G12.ptr : nullptr, storeG03 ? G03.ptr : nullptr,
                                                      pairEnd.ptr);
    if (!polarizable) {
        CU(cudaMemsetAsync(x.ptr, 0, m*sizeof(double), st));
        return;
    }
    if (dual) {
        initialGuess<<<blocksFor(m, 256), 256, 0, st>>>(n, ctl.ptr, globalIdx.ptr, histPtrs(), alphaD.ptr,
                                                        b.ptr, x.ptr);   // the predictor: b not needed
        fieldDual(false);
        return;
    }
    field(true, perm.ptr, b.ptr);
    initialGuess<<<blocksFor(m, 256), 256, 0, st>>>(n, ctl.ptr, globalIdx.ptr, histPtrs(), alphaD.ptr, b.ptr,
                                                    x.ptr);
    field(false, x.ptr, T.ptr);
}

void Engine::Impl::enqueueCgInit(CondHandle h, bool useHandle, bool dual) {
    const int m = 3*n, cgBlocks = std::max(1, std::min(blocksFor(m, CG_THREADS), 1024));
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    const bool pre = preFor(dual);
    curPre = pre;
    cgInitK<<<cgBlocks, CG_THREADS, 0, st>>>(m, b.ptr, T.ptr, dual ? fld.ptr : nullptr, erec.ptr, perm.ptr, fld2.ptr,
                                             erec2.ptr, cs, alphaD.ptr, x.ptr, r.ptr, z.ptr, p.ptr, scal.ptr, s.tol,
                                             s.maxIter, h, useHandle && !pre, dipF.ptr, pre, jacobiSelf());
    if (pre)
        precondK<true><<<std::max(1, std::min(blocksFor(n, CG_THREADS), 1024)), CG_THREADS, 0, st>>>(
                n, posq.ptr, srCount.ptr, srJ.ptr, srG.ptr, srOver.ptr, alphaD.ptr, r.ptr, nullptr, z.ptr, p.ptr,
                dipF.ptr, scal.ptr, s.tol, s.maxIter, h, useHandle);
}

void Engine::Impl::enqueueCgIteration(CondHandle h, bool useHandle) {
    const int m = 3*n, cgBlocks = std::max(1, std::min(blocksFor(m, CG_THREADS), 1024));
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    field(false, p.ptr, nullptr, true);   // T p combined in cgApply; p in dipF already (cgInitK, cgDirection)
    cgApply<<<cgBlocks, CG_THREADS, 0, st>>>(m, fld.ptr, erec.ptr, cs, alphaD.ptr, p.ptr, Ap.ptr, scal.ptr);
    cgUpdate<<<cgBlocks, CG_THREADS, 0, st>>>(m, alphaD.ptr, x.ptr, r.ptr, z.ptr, p.ptr, Ap.ptr, scal.ptr, curPre,
                                              jacobiSelf());
    if (curPre)
        precondK<false><<<std::max(1, std::min(blocksFor(n, CG_THREADS), 1024)), CG_THREADS, 0, st>>>(
                n, posq.ptr, srCount.ptr, srJ.ptr, srG.ptr, srOver.ptr, alphaD.ptr, r.ptr, Ap.ptr, z.ptr, nullptr,
                nullptr, scal.ptr, s.tol, s.maxIter, h, false);
    cgDirection<<<cgBlocks, CG_THREADS, 0, st>>>(m, z.ptr, p.ptr, scal.ptr, s.tol, s.maxIter, h, useHandle, dipF.ptr);
}

/** Peek, the history, energies, forces, the field at the total dipoles; the copies to the host. */
void Engine::Impl::enqueuePost(bool strain) {
    const int m = 3*n;
    const int warpsPerBlock = 4, threads = 32*warpsPerBlock;
    const int pblocks = (n+warpsPerBlock-1)/warpsPerBlock;
    if (polarizable) {
        if (s.peek != 0)
            peekStep<<<blocksFor(m, 256), 256, 0, st>>>(n, s.peek, alphaD.ptr, r.ptr, x.ptr);
        zeroHistorySlot<<<blocksFor(3*(size_t) nGlobal, 256), 256, 0, st>>>(3*(size_t) nGlobal, ctl.ptr, histPtrs());
        pushHistory<<<blocksFor(m, 256), 256, 0, st>>>(n, ctl.ptr, globalIdx.ptr, histPtrs(), x.ptr);
    }
    addVec<<<blocksFor(m, 256), 256, 0, st>>>(m, perm.ptr, x.ptr, dTot.ptr);
    toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, dTot.ptr, nullptr, dipF.ptr);
    const float rvdw2 = (float) (rvdw*rvdw), invKe = (float) (1.0/s.ke);
    fork();
#define PGM_FINAL(STRAIN, GV)                                                                                    \
    finalPass<STRAIN, GV><<<pblocks, threads, 0, st>>>(n, order(), atomStart.ptr, pairEnd.ptr, PJ.ptr, posq.ptr,  \
                                                       dipF.ptr, kernels(), rg.ptr, gvP.ptr, gvdwSlater,         \
                                                       rvdw2, invKe, fld.ptr, fpair.ptr, acc.ptr)
    pairAfterSpread([&] {
        if (strain && gvdwOn)
            PGM_FINAL(true, true);
        else if (strain)
            PGM_FINAL(true, false);
        else if (gvdwOn)
            PGM_FINAL(false, true);
        else
            PGM_FINAL(false, false);
    });
#undef PGM_FINAL
    pme(true, true, strain ? 2 : 1, acc.ptr+10);
    join();
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    combineField<<<blocksFor(m, 256), 256, 0, st>>>(m, fld.ptr, erec.ptr, dTot.ptr, cs, Etot.ptr);
    if (ncov > 0)
        covalentForces<<<blocksFor(ncov, 128), 128, 0, st>>>(ncov, posq.ptr, covAtom.ptr, covPartner.ptr,
                                                             covStrength.ptr, Etot.ptr, frc.ptr,
                                                             strain ? acc.ptr+20 : nullptr);
    const int rblocks = std::min(blocksFor(n, 256), 1024);
    double* fHost = nullptr;   // (finalReduce writing pinned host memory directly: 4 ms for 4096 waters)
    if (strain)
        finalReduce<true><<<rblocks, 256, 0, st>>>(n, s.ke, frc.ptr, fpair.ptr, frec.ptr, erec.ptr, dTot.ptr, x.ptr,
                                                   alphaD.ptr, fOut.ptr, acc.ptr, fHost);
    else
        finalReduce<false><<<rblocks, 256, 0, st>>>(n, s.ke, frc.ptr, fpair.ptr, frec.ptr, erec.ptr, dTot.ptr, x.ptr,
                                                    alphaD.ptr, fOut.ptr, acc.ptr, fHost);
    if (devForce)
        addForcesFixed<<<blocksFor(m, 256), 256, 0, st>>>(n, fOut.ptr, devForce, devStride, devScale);
    else if (useExport)   // coalesced 16-byte stores to pinned memory (a copy node costs more)
        copyToHost<<<std::min(blocksFor((size_t) m/2, 256), 1024), 256, 0, st>>>((const uint4*) fOut.ptr,
                                                                                  (uint4*) fH.ptr, (size_t) m/2,
                                                                                  fOut.ptr, fH.ptr, m);
    else
        CU(cudaMemcpyAsync(fH.ptr, fOut.ptr, m*sizeof(double), cudaMemcpyDeviceToHost, st));
    if (useExport)
        copyWords<<<1, 256, 0, st>>>((const unsigned*) acc.ptr, (unsigned*) accH.ptr, 2*NACC,
                                     (const unsigned*) scal.ptr, (unsigned*) scalH.ptr, 32);
    else {
        CU(cudaMemcpyAsync(accH.ptr, acc.ptr, NACC*sizeof(double), cudaMemcpyDeviceToHost, st));
        CU(cudaMemcpyAsync(scalH.ptr, scal.ptr, 16*sizeof(double), cudaMemcpyDeviceToHost, st));
    }
}

void Engine::Impl::releaseGraphs() {
    for (int v = 0; v < 4; v++)
        if (exec[v]) {
            cudaGraphExecDestroy(exec[v]);
            exec[v] = nullptr;
        }
}

/** captureGraph; on a failure the capture is ended (the stream usable again), false after a second one. */
bool Engine::Impl::tryCapture(bool strain, bool dual) {
#if PGM_CUDA_GRAPHS
    for (int attempt = 0; attempt < 2; attempt++) {
        try {
            captureGraph(strain, dual);
            return true;
        }
        catch (const std::exception& e) {
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(st, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
                cudaGraph_t g = nullptr;
                cudaStreamEndCapture(st, &g);
                if (g)
                    cudaGraphDestroy(g);
            }
            cudaGetLastError();
            fprintf(stderr, "pgm_cuda: graph capture failed (%s)%s\n", e.what(),
                    attempt == 0 ? ", retrying" : ", running without graphs");
        }
    }
#endif
    (void) strain;
    (void) dual;
    return false;
}

/** The whole evaluation as one graph: pre, CG init, while (CG iteration), post. */
void Engine::Impl::captureGraph(bool strain, bool dual) {
#if PGM_CUDA_GRAPHS
    const int v = (strain ? 1 : 0)+(dual ? 2 : 0);
    if (exec[v]) {
        cudaGraphExecDestroy(exec[v]);
        exec[v] = nullptr;
    }
    cudaGraph_t graph;
    CU(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle = 0;
    if (polarizable)   // reset to 0 at every launch; the CG kernels set it
        CU(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
    CU(cudaStreamBeginCaptureToGraph(st, graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    enqueuePre(dual);
    cudaGraph_t body = nullptr;
    if (polarizable) {
        const int K = curUnroll;   // iterations before the while node, unconditional
        enqueueCgInit(handle, K == 0, dual);
        for (int k = 0; k < K; k++)
            enqueueCgIteration(handle, k == K-1);   // the last one sets the condition of the while node
        // the while node after what was captured so far, the rest after it
        cudaStreamCaptureStatus status;
        const cudaGraphNode_t* deps = nullptr;
        size_t numDeps = 0;
        cudaGraph_t capturing;
        CU(cudaStreamGetCaptureInfo(st, &status, nullptr, &capturing, &deps, &numDeps));
        cudaGraphNodeParams cp = {};
        cp.type = cudaGraphNodeTypeConditional;
        cp.conditional.handle = handle;
        cp.conditional.type = cudaGraphCondTypeWhile;
        cp.conditional.size = 1;
        cudaGraphNode_t node;
        CU(cudaGraphAddNode(&node, capturing, deps, numDeps, &cp));
        body = cp.conditional.phGraph_out[0];
        CU(cudaStreamUpdateCaptureDependencies(st, &node, 1, cudaStreamSetCaptureDependencies));
    }
    enqueuePost(strain);
    CU(cudaStreamEndCapture(st, &graph));
    if (body) {
        CU(cudaStreamBeginCaptureToGraph(st, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
        enqueueCgIteration(handle, true);
        CU(cudaStreamEndCapture(st, &body));
    }
    CU(cudaGraphInstantiate(&exec[v], graph, 0));
    CU(cudaGraphDestroy(graph));
    execAllocations[v] = g_allocations;
    execKey[v] = graphKey();
    execUnroll[v] = curUnroll;
#else
    (void) strain;
    (void) dual;
#endif
}

Engine::Engine(const Settings& settings) : impl(new Impl) {
    impl->s = settings;
    if (settings.order < 3 || settings.order > MAX_ORDER)
        throw std::runtime_error("pgm_cuda: PME order must be 3..12");
    if (settings.device >= 0)
        CU(cudaSetDevice(settings.device));
    // non-blocking: no implicit order with the legacy stream, whose use by another thread would invalidate a
    // capture; the synchronous set-up copies wait for their completion (DevArray)
    CU(cudaStreamCreateWithFlags(&impl->st, cudaStreamNonBlocking));
    {   // the PME branch first: its blocks start before the pair pass fills the GPU (the pair pass takes the rest)
        int least = 0, greatest = 0;
        CU(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        const bool prio = std::getenv("PGM_CUDA_NO_PRIORITY") == nullptr;
        CU(cudaStreamCreateWithPriority(&impl->st2, cudaStreamNonBlocking, prio ? greatest : least));
    }
    CU(cudaEventCreateWithFlags(&impl->evFork, cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&impl->evJoin, cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&impl->evMid, cudaEventDisableTiming));
    const int K1 = settings.grid[0], K2 = settings.grid[1], K3 = settings.grid[2];
    CUFFT(cufftPlan3d(&impl->planF, K1, K2, K3, CUFFT_R2C));
    CUFFT(cufftPlan3d(&impl->planB, K1, K2, K3, CUFFT_C2R));
    int dims[3] = {K1, K2, K3};
    CUFFT(cufftPlanMany(&impl->planF2, 3, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, 2));
    CUFFT(cufftPlanMany(&impl->planB2, 3, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, 2));
    for (cufftHandle h : {impl->planF, impl->planB, impl->planF2, impl->planB2})
        CUFFT(cufftSetStream(h, impl->ps()));
    impl->gridFix.zero(2*(size_t) K1*K2*K3);   // two grids for fieldDual; cleared again by fixedToReal
    impl->grid.resize(2*(size_t) K1*K2*K3);
    impl->cgrid.resize(2*(size_t) K1*K2*(K3/2+1));
    std::vector<double> m[3] = {pgmcore::bsplineModuli(K1, settings.order), pgmcore::bsplineModuli(K2, settings.order),
                                pgmcore::bsplineModuli(K3, settings.order)};
    std::vector<float> mf[3];
    for (int d = 0; d < 3; d++)
        mf[d].assign(m[d].begin(), m[d].end());
    impl->mod1.upload(mf[0].data(), K1);
    impl->mod2.upload(mf[1].data(), K2);
    impl->mod3.upload(mf[2].data(), K3);
    impl->scal.resize(16);
    impl->acc.resize(NACC);
    impl->maxDisp.resize(2);
    impl->ctl.resize(1);
    impl->accH.resize(NACC);
    impl->scalH.resize(16);
    impl->dispH.resize(2);
    impl->boxH.resize(1);
    impl->ctlH.resize(1);
    impl->useGraphs = PGM_CUDA_GRAPHS && std::getenv("PGM_CUDA_NO_GRAPH") == nullptr;
}

Engine::~Engine() {
    if (impl->inFlight)
        cudaStreamSynchronize(impl->st);
    impl->releaseGraphs();
    if (impl->doneEvent)
        cudaEventDestroy(impl->doneEvent);
    if (impl->devEvent)
        cudaEventDestroy(impl->devEvent);
    if (impl->regPtr)
        cudaHostUnregister(const_cast<float*>(impl->regPtr));
    if (impl->planF)
        cufftDestroy(impl->planF);
    if (impl->planB)
        cufftDestroy(impl->planB);
    if (impl->planF2)
        cufftDestroy(impl->planF2);
    if (impl->planB2)
        cufftDestroy(impl->planB2);
    if (impl->evFork)
        cudaEventDestroy(impl->evFork);
    if (impl->evJoin)
        cudaEventDestroy(impl->evJoin);
    if (impl->evMid)
        cudaEventDestroy(impl->evMid);
    if (impl->st2)
        cudaStreamDestroy(impl->st2);
    if (impl->st)
        cudaStreamDestroy(impl->st);
    delete impl;
}

void Engine::setAtoms(int n, const double* q, const double* radius, const double* alpha, const int* globalIndex,
                      int numGlobal) {
    Impl& I = *impl;
    I.settle();
    if (I.s.device >= 0)
        CU(cudaSetDevice(I.s.device));
    CU(cudaStreamSynchronize(I.st));
    const bool resized = (n != I.n || numGlobal != I.nGlobal);
    if (resized) {
        I.numHist = 0;
        I.releasePositions();   // another atom count: another position array of the caller too
    }
    // the same atoms in another order: keep the list (renamed) instead of building it again
    std::vector<int> newOf;
    if (!resized && I.haveList && !I.externalList && globalIndex && (int) I.giHost.size() == n) {
        std::vector<int> localOf(numGlobal, -1);
        for (int i = 0; i < n; i++)
            if (globalIndex[i] >= 0 && globalIndex[i] < numGlobal)
                localOf[globalIndex[i]] = i;
        newOf.resize(n);
        for (int i = 0; i < n && !newOf.empty(); i++) {
            const int g = I.giHost[i];
            if (g < 0 || g >= numGlobal || localOf[g] < 0)
                newOf.clear();
            else
                newOf[i] = localOf[g];
        }
    }
    I.n = n;
    I.nGlobal = numGlobal;
    I.alpha.resize(n);
    I.q2 = I.Qtot = 0;
    std::vector<float> rgH(n);
    std::vector<int> gi(n);
    for (int i = 0; i < n; i++) {
        I.alpha[i] = (alpha[i] > pgmcore::MIN_POLARIZABILITY ? alpha[i] : 0.0);
        rgH[i] = (float) (2.0*radius[i]*radius[i]);
        gi[i] = (globalIndex ? globalIndex[i] : i);
        I.q2 += q[i]*q[i];
        I.Qtot += q[i];
    }
    I.polarizable = std::any_of(I.alpha.begin(), I.alpha.end(), [](double a) { return a > 0; });
    I.rg.upload(rgH.data(), n);
    I.alphaD.upload(I.alpha.data(), n);
    I.globalIdx.upload(gi.data(), n);
    I.giHost = gi;
    {
        std::vector<float> qh(q, q+n);
        I.qF.upload(qh.data(), n);
    }
    const int m = 3*n;
    I.posq.resize(n);
    I.posBuild.resize(n);
    I.posPrev.resize(n);
    I.posqH.resize(n);
    I.fH.resize(m);
    for (int i = 0; i < n; i++)
        I.posqH.ptr[i] = make_float4(0, 0, 0, (float) q[i]);
    if (resized) {   // (the same atoms in another order, as at every GROMACS pair search: nothing to clear)
        for (auto* a : {&I.perm, &I.b, &I.x, &I.r, &I.z, &I.p, &I.Ap, &I.T, &I.erec, &I.frec, &I.dTot, &I.Etot,
                        &I.fld, &I.fpair, &I.fOut, &I.erec2, &I.fld2})
            a->zero(m);
        I.dipF2.zero(n);
        for (auto& h : I.hist)
            h.zero(3*(size_t) numGlobal);
        I.lastS2 = 0;
        I.dipF.zero(n);
        I.atomStart.zero(n+1);
        I.pairEnd.zero(n);
    }
    I.frc.zero(m);   // then cleared by finalReduce
    const size_t np = (size_t) 3*n*I.s.order;
    I.theta.resize(np);
    I.dtheta.resize(np);
    I.d2theta.resize(np);
    I.gidx.resize((size_t) 3*n);
    I.mu.assign(m, 0.0);
    I.permH.assign(m, 0.0);
    const bool keep = !newOf.empty();
    if (keep) {
        I.remapList(newOf);
        I.numRemaps++;
    }
    else {
        I.haveList = false;
        I.prevValid = false;
    }
    I.externalList = false;
    I.gvdwOn = false;   // setGvdw again for the new order
    I.skipsMarked = false;
    I.cellAtomSorted.resize(n);
    if (I.rpre > 0) {
        I.srCount.resize(n);
        I.srJ.resize((size_t) n*PRE_CAP);
        I.srG.resize((size_t) n*PRE_CAP);
        I.srOver.resize(1);
    }
}

void Engine::Impl::remapList(const std::vector<int>& newOf) {
    // (persistent scratch: a reallocation would mean capturing the graph again)
    DevArray<int>& newOfD = rmNewOf;
    DevArray<int>& start2 = rmStart;
    DevArray<int>& cand2 = rmCand;
    newOfD.upload(newOf.data(), n);
    count.resize(n+1);
    remapCounts<<<blocksFor(n+1, 256), 256, 0, st>>>(n, newOfD.ptr, atomStart.ptr, count.ptr);
    start2.resize(n+1);
    size_t bytes = 0;
    CU(cub::DeviceScan::ExclusiveSum(nullptr, bytes, count.ptr, start2.ptr, n+1, st));
    cubTemp.resize(bytes);
    CU(cub::DeviceScan::ExclusiveSum(cubTemp.ptr, bytes, count.ptr, start2.ptr, n+1, st));
    cand2.resize(std::max(candJ.size, (size_t) 1));
    remapRows<<<(n+3)/4, 128, 0, st>>>(n, newOfD.ptr, atomStart.ptr, candJ.ptr, start2.ptr, cand2.ptr);
    // back into the same buffers (the graph keeps its pointers)
    CU(cudaMemcpyAsync(atomStart.ptr, start2.ptr, (n+1)*sizeof(int), cudaMemcpyDeviceToDevice, st));
    CU(cudaMemcpyAsync(candJ.ptr, cand2.ptr, ncand*sizeof(int), cudaMemcpyDeviceToDevice, st));
    DevArray<float4>& a2 = rmA;
    DevArray<float4>& b2 = rmB;
    a2.resize(n);
    b2.resize(n);
    remapAtoms<<<blocksFor(n, 256), 256, 0, st>>>(n, newOfD.ptr, posBuild.ptr, posPrev.ptr, a2.ptr, b2.ptr,
                                                  cellAtomSorted.ptr);
    CU(cudaMemcpyAsync(posBuild.ptr, a2.ptr, n*sizeof(float4), cudaMemcpyDeviceToDevice, st));
    CU(cudaMemcpyAsync(posPrev.ptr, b2.ptr, n*sizeof(float4), cudaMemcpyDeviceToDevice, st));
    CU(cudaStreamSynchronize(st));
}

void Engine::setGvdw(int repulsion, double rvdw, const double* sqrtA, const double* sqrtC6, const double* b,
                     int nskip, const int* skipI, const int* skipJ) {
    Impl& I = *impl;
    I.settle();
    CU(cudaStreamSynchronize(I.st));
    const int n = I.n;
    std::vector<float4> gvH(n);
    for (int i = 0; i < n; i++)
        gvH[i] = make_float4((float) sqrtA[i], (float) sqrtC6[i], (float) b[i], 0.0f);
    I.gvP.upload(gvH.data(), n);
    // both directions, sorted per atom
    std::vector<int> start(n+1, 0);
    for (int k = 0; k < nskip; k++) {
        start[skipI[k]+1]++;
        start[skipJ[k]+1]++;
    }
    for (int i = 0; i < n; i++)
        start[i+1] += start[i];
    std::vector<int> fill(start.begin(), start.end()-1), part(std::max(start[n], 1));
    for (int k = 0; k < nskip; k++) {
        part[fill[skipI[k]]++] = skipJ[k];
        part[fill[skipJ[k]]++] = skipI[k];
    }
    for (int i = 0; i < n; i++)
        std::sort(part.begin()+start[i], part.begin()+start[i+1]);
    I.skipStart.upload(start.data(), n+1);
    I.skipJ.upload(part.data(), part.size());
    I.gvdwOn = true;
    I.gvdwSlater = (repulsion == 1);
    I.rvdw = rvdw;
    if (I.pairRadius() > I.builtRadius)
        I.haveList = false;   // the list is for a smaller radius
    I.skipsMarked = false;
}

void Engine::setCovalent(int ncov, const int* atom, const int* partner, const double* strength) {
    Impl& I = *impl;
    I.settle();
    CU(cudaStreamSynchronize(I.st));
    I.ncov = ncov;
    std::vector<int> start(I.n+1, 0), partnerS(ncov);
    std::vector<double> strengthS(ncov);
    for (int m = 0; m < ncov; m++)
        start[atom[m]+1]++;
    for (int i = 0; i < I.n; i++)
        start[i+1] += start[i];
    std::vector<int> fill(start.begin(), start.end()-1);
    for (int m = 0; m < ncov; m++) {
        const int k = fill[atom[m]]++;
        partnerS[k] = partner[m];
        strengthS[k] = strength[m];
    }
    I.covStart.upload(start.data(), I.n+1);
    I.covPartnerSorted.upload(partnerS.data(), ncov);
    I.covStrengthSorted.upload(strengthS.data(), ncov);
    I.covAtom.upload(atom, ncov);
    I.covPartner.upload(partner, ncov);
    I.covStrength.upload(strength, ncov);
}

void Engine::setCandidates(int nseg, const int* segI, const int* segShift, const int* segBegin, const int* candJ) {
    Impl& I = *impl;
    I.settle();
    CU(cudaStreamSynchronize(I.st));
    const int n = I.n;
    // the full list: (i, j, t) and (j, i, -t), per atom; shifts packed 4 bits per component (|t| <= 7)
    std::vector<int> count(n+1, 0);
    for (int sg = 0; sg < nseg; sg++)
        for (int k = segBegin[sg]; k < segBegin[sg+1]; k++) {
            count[segI[sg]+1]++;
            count[candJ[k]+1]++;
        }
    for (int i = 0; i < n; i++)
        count[i+1] += count[i];
    I.ncand = count[n];
    std::vector<int> fill(count.begin(), count.end()-1), jj(I.ncand), tt(I.ncand);
    auto pack = [](int a, int b, int c) {
        if (std::abs(a) > 7 || std::abs(b) > 7 || std::abs(c) > 7)
            throw std::runtime_error("pgm_cuda: periodic shift beyond 7 box vectors");
        return (a+8)|((b+8)<<4)|((c+8)<<8);
    };
    for (int sg = 0; sg < nseg; sg++) {
        const int i = segI[sg], *t = segShift+3*sg;
        for (int k = segBegin[sg]; k < segBegin[sg+1]; k++) {
            const int j = candJ[k];
            int q = fill[i]++;
            jj[q] = j;
            tt[q] = pack(t[0], t[1], t[2]);
            q = fill[j]++;
            jj[q] = i;
            tt[q] = pack(-t[0], -t[1], -t[2]);
        }
    }
    I.atomStart.upload(count.data(), n+1);
    I.candJ.upload(jj.data(), I.ncand);
    I.skipsMarked = false;
    I.candT.upload(tt.data(), I.ncand);
    I.PJ.resize(I.ncand);
    if (I.storeKernels)
        I.G12.resize(I.ncand);
    if (I.storeKernels && I.storeG03)
        I.G03.resize(I.ncand);
    I.externalList = true;
}

Terms Engine::compute(const double* pos, const double box[3][3], bool newStep, double* f, double* strain) {
    return impl->evaluate(pos, nullptr, box, newStep, f, strain);
}

Terms Engine::computeF(const float* pos, const double box[3][3], bool newStep, double* strain) {
    return impl->evaluate(nullptr, pos, box, newStep, nullptr, strain);
}

const double* Engine::forces() const { return impl->fH.ptr; }

void Engine::releasePositions() {
    impl->settle();
    impl->releasePositions();
}

void Engine::launchDevice(const void* posq4, const double box[3][3], bool newStep, long long* forceFixed, int stride,
                          double fixedScale, void* stream) {
    Impl& I = *impl;
    I.settle();   // the bookkeeping of the evaluation before (deferred: its list check now)
    if (!I.devEvent)
        CU(cudaEventCreateWithFlags(&I.devEvent, cudaEventDisableTiming));
    if (!I.doneEvent)
        CU(cudaEventCreateWithFlags(&I.doneEvent, cudaEventDisableTiming));
    // the work of the caller so far (its positions, its cleared force buffer) before ours
    CU(cudaEventRecord(I.devEvent, (cudaStream_t) stream));
    CU(cudaStreamWaitEvent(I.st, I.devEvent, 0));
    I.devPos = (const float4*) posq4;
    I.devForce = forceFixed;
    I.devStride = stride;
    I.devScale = fixedScale;
    I.begin(nullptr, nullptr, box, newStep, false);
    CU(cudaEventRecord(I.doneEvent, I.st));
    I.inFlight = true;
    I.devPos = nullptr;
    I.devForce = nullptr;
}

void Engine::waitDevice(void* stream) {
    if (impl->inFlight)
        CU(cudaStreamWaitEvent((cudaStream_t) stream, impl->doneEvent, 0));
}

bool Engine::finishDevice(Terms& terms) {
    if (!impl->inFlight) {
        terms = impl->lastTerms;
        return true;
    }
    return impl->finish(nullptr, nullptr, terms);
}

Terms Engine::computeDevice(const void* posq4, const double box[3][3], bool newStep, long long* forceFixed,
                            int stride, double fixedScale, void* stream) {
    Terms t;
    do
        launchDevice(posq4, box, newStep, forceFixed, stride, fixedScale, stream);
    while (!finishDevice(t));
    return t;
}

Terms Engine::Impl::evaluate(const double* pos, const float* posF, const double box[3][3], bool newStep, double* f,
                             double* strain) {
    settle();
    Terms t;
    do
        begin(pos, posF, box, newStep, strain != nullptr);
    while (!finish(f, strain, t));
    return t;
}

void Engine::Impl::begin(const double* pos, const float* posF, const double box[3][3], bool newStep,
                         bool wantStrain) {
    Impl& I = *this;
    const int n = I.n;
    const Settings& s = I.s;
    if (s.device >= 0)
        CU(cudaSetDevice(s.device));   // the calling thread may be another than the constructor's
    const double tA = I.timing ? wallSeconds() : 0;
    // box, reciprocal box, volume
    const double (*H)[3] = box;
    const double det = H[0][0]*(H[1][1]*H[2][2]-H[1][2]*H[2][1])-H[0][1]*(H[1][0]*H[2][2]-H[1][2]*H[2][0])
                       +H[0][2]*(H[1][0]*H[2][1]-H[1][1]*H[2][0]);
    I.volume = std::fabs(det);
    double R[3][3];
    R[0][0] = (H[1][1]*H[2][2]-H[1][2]*H[2][1])/det;
    R[0][1] = (H[0][2]*H[2][1]-H[0][1]*H[2][2])/det;
    R[0][2] = (H[0][1]*H[1][2]-H[0][2]*H[1][1])/det;
    R[1][0] = (H[1][2]*H[2][0]-H[1][0]*H[2][2])/det;
    R[1][1] = (H[0][0]*H[2][2]-H[0][2]*H[2][0])/det;
    R[1][2] = (H[0][2]*H[1][0]-H[0][0]*H[1][2])/det;
    R[2][0] = (H[1][0]*H[2][1]-H[1][1]*H[2][0])/det;
    R[2][1] = (H[0][1]*H[2][0]-H[0][0]*H[2][1])/det;
    R[2][2] = (H[0][0]*H[1][1]-H[0][1]*H[1][0])/det;
    Box& bx = *I.boxH.ptr;
    for (int a = 0; a < 3; a++)
        for (int c = 0; c < 3; c++) {
            bx.b[3*a+c] = (float) H[a][c];
            bx.r[3*a+c] = (float) R[a][c];
            I.boxD[a][c] = H[a][c];
            I.recipD[a][c] = R[a][c];
        }
    bx.vol = (float) I.volume;
    float4* pq = I.posqH.ptr;
    // float positions in host memory registered with CUDA: the upload and the packing with the charges
    // are part of the evaluation (graph); else packed here into pinned memory
    I.usePosF = posF && I.registerPositions(posF);
    // (computeDevice: packed on the device)
    if (!I.devPos && posF && !I.usePosF)
        for (int i = 0; i < n; i++) {
            pq[i].x = posF[3*i];
            pq[i].y = posF[3*i+1];
            pq[i].z = posF[3*i+2];
        }
    else if (!I.devPos && !posF)
        for (int i = 0; i < n; i++) {
            pq[i].x = (float) pos[3*i];
            pq[i].y = (float) pos[3*i+1];
            pq[i].z = (float) pos[3*i+2];
        }
    CU(cudaMemcpyToSymbolAsync(c_box, I.boxH.ptr, sizeof(Box), 0, cudaMemcpyHostToDevice, I.st));
    if (!I.usePosF && !I.devPos)
        CU(cudaMemcpyAsync(I.posq.ptr, pq, n*sizeof(float4), cudaMemcpyHostToDevice, I.st));
    else if (I.listStaleBefore())
        I.uploadPositions();   // the list build needs them before the evaluation
    I.muValid = I.permValid = false;
    const double tB = I.timing ? wallSeconds() : 0;
    if (I.listStaleBefore())
        I.buildList();

    // the initial guess and the history slots (a repeated evaluation: those of the first)
    Control& c = *I.ctlH.ptr;
    if (!I.retry) {
        const int h = I.numHist;
        for (int k = 0; k < 4; k++)
            c.c[k] = 0;
        if (!newStep && h > 0)
            c.c[0] = 1;
        else if (s.predictor == 2 && h >= 4)
            c.c[0] = 4, c.c[1] = -6, c.c[2] = 4, c.c[3] = -1;
        else if (s.predictor >= 1 && h >= 3)
            c.c[0] = 3, c.c[1] = -3, c.c[2] = 1;
        else if (s.predictor >= 1 && h >= 1)
            c.c[0] = 1;
        c.head = I.head;
        I.pendingPush = I.polarizable && (newStep || h == 0);
        c.pushSlot = I.pendingPush ? (I.head+NHIST-1)%NHIST : -1;   // a slot the guess does not read: the newest
    }
    CU(cudaMemcpyAsync(I.ctl.ptr, I.ctlH.ptr, sizeof(Control), cudaMemcpyHostToDevice, I.st));
    // the permanent field and the guess's field together when the guess comes from the history
    static const bool noDual = std::getenv("PGM_CUDA_NO_DUAL") != nullptr;
    const bool dual = I.polarizable && c.c[0] != 0 && I.dualOrder() && !noDual;
    const double tC = I.timing ? wallSeconds() : 0;
    if (I.timing) {
        I.tPhase[0] += tB-tA;
        I.tPhase[1] += tC-tB;
        I.tStamp[0] = tC;
    }
    if (I.gvdwOn && !I.skipsMarked)
        I.markSkipList();
    const int v = (wantStrain ? 1 : 0)+(dual ? 2 : 0);
    I.curUnroll = I.polarizable ? I.unrollWanted() : 0;
    if (I.useGraphs && (!I.exec[v] || I.execAllocations[v] != g_allocations || I.execKey[v] != I.graphKey()
                        || I.execUnroll[v] != I.curUnroll))
        if (!I.tryCapture(wantStrain, dual))
            I.useGraphs = false;
    if (I.useGraphs) {
        const double tL = I.timing ? wallSeconds() : 0;
        CU(cudaGraphLaunch(I.exec[v], I.st));
        if (I.timing)
            I.tLaunch += wallSeconds()-tL;
        return;
    }
    int it = 0;
    double err = 0;
    I.enqueuePre(dual);
    if (I.polarizable) {
        I.enqueueCgInit(0, false, dual);
        CU(cudaMemcpyAsync(I.scalH.ptr, I.scal.ptr, 16*sizeof(double), cudaMemcpyDeviceToHost, I.st));
        CU(cudaStreamSynchronize(I.st));
        err = I.scalH.ptr[2];
        while (err > s.tol && it < s.maxIter) {
            I.enqueueCgIteration(0, false);
            CU(cudaMemcpyAsync(I.scalH.ptr, I.scal.ptr, 16*sizeof(double), cudaMemcpyDeviceToHost, I.st));
            CU(cudaStreamSynchronize(I.st));
            err = I.scalH.ptr[2];
            it++;
        }
    }
    I.enqueuePost(wantStrain);
    I.pendingIt = it;
    I.pendingErr = err;
}

/** Waits for the evaluation that begin() enqueued; false: its list was out of date (rebuilt now), repeat it. */
bool Engine::Impl::finish(double* f, double* strain, Terms& t, bool canRepeat) {
    const int m = 3*n;
    inFlight = false;
    CU(cudaStreamSynchronize(st));
    int it = pendingIt;
    double err = pendingErr;
    if (useGraphs) {
        it = (int) scalH.ptr[9];
        err = scalH.ptr[2];
    }
    if (!retry && !externalList) {   // the largest move of this step (not over a reordering)
        float s2;
        std::memcpy(&s2, dispH.ptr+1, sizeof(float));
        if (prevValid)
            lastS2 = s2;
        prevValid = true;
    }
    bool rebuilt = false;
    if (!retry && listStaleAfter()) {
        buildList();   // the prediction failed: rebuild and repeat
        if (canRepeat) {
            numRepeats++;
            retry = true;
            return false;
        }
        numLate++;   // too late to repeat: the next evaluation has the new list
        rebuilt = true;
    }
    retry = false;
    if (!externalList && !rebuilt) {
        float d2;
        std::memcpy(&d2, dispH.ptr, sizeof(float));
        lastD2 = d2;
    }
    const double tD = timing ? wallSeconds() : 0;
    if (!polarizable)
        it = 0, err = 0;
    if (pendingPush) {
        head = ctlH.ptr->pushSlot;
        numHist = std::min(numHist+1, 4);
    }
    lastIterations = it;
    lastResidual = err;
    if (polarizable) {
        // the iterations the tolerance needed (unrolled ones run regardless: counting those would keep them)
        const int need = (useGraphs && scalH.ptr[11] < 1e8) ? (int) scalH.ptr[11] : it;
        recentIt[recentPos] = need;
        recentPos = (recentPos+1)%16;
        numRecent = std::min(numRecent+1, 16);
    }
    const double* acc = accH.ptr;
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    t = Terms();
    t.direct = s.ke*acc[0];
    t.reciprocal = s.ke*acc[10];
    t.self = s.ke*(-(s.beta/pgmcore::SQRT_PI)*q2-0.5*cs*acc[29]);
    const double background = -pgmcore::PI*Qtot*Qtot/(2.0*volume*s.beta*s.beta);
    t.background = s.ke*background;
    t.polarization = s.ke*acc[30];
    t.gvdw = gvdwOn ? acc[40] : 0.0;
    if (f)
        for (int k = 0; k < m; k++)
            f[k] += fH.ptr[k];
    if (strain) {
        double S[9];
        for (int a = 0; a < 9; a++)
            S[a] = acc[1+a]+acc[11+a]+acc[20+a]+acc[31+a];
        for (int a = 0; a < 3; a++)
            S[4*a] -= background;
        for (int a = 0; a < 9; a++)
            strain[a] += s.ke*S[a];
    }
    CU(cudaGetLastError());
    if (timing) {
        const double tE = wallSeconds();
        tPhase[2] += tD-tStamp[0];
        tPhase[3] += tE-tD;
        if (++numCalls%1000 == 0)
            fprintf(stderr,
                    "pgm_cuda ms per call (%ld): inputs %.3f, list builds %.3f, GPU %.3f (graph launch call %.3f), "
                    "outputs %.3f; %ld builds, %ld repeated evaluations, %ld found late; CG unrolled %d\n",
                    numCalls, 1e3*tPhase[0]/numCalls, 1e3*tPhase[1]/numCalls, 1e3*tPhase[2]/numCalls,
                    1e3*tLaunch/numCalls, 1e3*tPhase[3]/numCalls, numBuilds, numRepeats, numLate, curUnroll);
    }
    lastTerms = t;
    return true;
}

int Engine::iterations() const {
    impl->settle();
    return impl->lastIterations;
}
double Engine::residual() const {
    impl->settle();
    return impl->lastResidual;
}
void Engine::inducedDipoles(double* mu) const {
    impl->settle();
    if (!impl->muValid)
        impl->x.download(impl->mu.data(), 3*(size_t) impl->n);
    impl->muValid = true;
    std::copy(impl->mu.begin(), impl->mu.end(), mu);
}
void Engine::permanentDipoles(double* p) const {
    impl->settle();
    if (!impl->permValid)
        impl->perm.download(impl->permH.data(), 3*(size_t) impl->n);
    impl->permValid = true;
    std::copy(impl->permH.begin(), impl->permH.end(), p);
}
void Engine::resetHistory() {
    impl->settle();
    impl->numHist = 0;
}
long Engine::lateListChecks() const { return impl->numLate; }
long Engine::listBuilds() const { return impl->numBuilds; }
int Engine::numCandidates() const { return impl->ncand; }

} // namespace pgmcuda

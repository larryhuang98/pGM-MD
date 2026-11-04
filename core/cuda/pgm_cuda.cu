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

__global__ void gatherPositions(int n, const int* order, const float4* posq, float4* out) {
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < n; k += gridDim.x*blockDim.x)
        out[k] = posq[order[k]];
}

/** max |r_i - r_i(build)|^2 (minimum image) as float bits. */
__global__ void maxDisplacement(int n, const float4* posq, const float4* ref, unsigned* out) {
    const Box bx = c_box;
    float m = 0;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const float4 a = posq[i], b = ref[i];
        float3 d = make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
        minImage(d, bx);
        m = fmaxf(m, d.x*d.x+d.y*d.y+d.z*d.z);
    }
    for (int o = 16; o > 0; o >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if (threadIdx.x%32 == 0)
        atomicMax(out, __float_as_uint(m));
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

__global__ void fixedToReal(size_t count, const ull* in, float* out) {
    for (size_t k = blockIdx.x*(size_t) blockDim.x+threadIdx.x; k < count; k += (size_t) gridDim.x*blockDim.x)
        out[k] = (float) ((double) (long long) in[k]/FIX);
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

struct HistPtrs {   // the four history slots (3 nGlobal, the global order)
    double* h[4];
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
                v += c.c[h]*hp.h[(c.head+h)%4][g];
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
__global__ void finalReduce(int n, double ke, const ull* frc, const double* fpair, const double* frec,
                            const double* erec, const double* d, const double* mu, const double* alpha, double* f,
                            double* acc) {
    double d2 = 0, pol = 0, S[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        for (int k = 0; k < 3; k++) {
            const int c = 3*i+k;
            f[c] = ke*((double) (long long) frc[c]/FIX+fpair[c]+frec[c]);
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
                        double tol, int maxIter, CondHandle h, bool useHandle) {
    double rz = 0, nrm = 0, err = 0;
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x) {
        const double a = alpha[k/3];
        const double bk = fldB ? fldB[k]+erecB[k]+cs*perm[k] : b[k];
        const double tk = fldB ? fldT[k]+erecT[k]+cs*x[k] : Tx[k];
        const double rr = (a > 0 ? bk-(x[k]/a-tk) : 0.0), zz = a*rr;
        r[k] = rr;
        z[k] = zz;
        p[k] = zz;
        rz += rr*zz;
        nrm += fabs(a*bk);
        err = fmax(err, fabs(zz));
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
                         double* scal) {
    const double step = scal[0]/fmax(scal[5], 1e-300);
    double num = 0, rz = 0, err = 0;
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x) {
        const double a = alpha[k/3];
        x[k] += step*p[k];
        const double rn = r[k]-step*Ap[k], zn = a*rn;
        num += zn*(rn-r[k]);
        rz += rn*zn;
        err = fmax(err, fabs(zn));
        r[k] = rn;
        z[k] = zn;
    }
    blockAdd(num, scal+6);
    blockAdd(rz, scal+7);
    blockMaxBits(err, (ull*) (scal+8));
}

/** p = z + beta p, beta = num / r.z (Polak-Ribiere); then (the last block) the scalars of the next
 * iteration and the loop condition. */
__global__ void cgDirection(int m, const double* z, double* p, double* scal, double tol, int maxIter, CondHandle h,
                            bool useHandle) {
    const double beta = scal[6]/fmax(scal[0], 1e-300);
    for (int k = blockIdx.x*blockDim.x+threadIdx.x; k < m; k += gridDim.x*blockDim.x)
        p[k] = z[k]+beta*p[k];
    if (lastBlock(scal)) {
        scal[3] = scal[0]/fmax(scal[5], 1e-300);
        scal[4] = scal[6]/fmax(scal[0], 1e-300);
        scal[0] = scal[7];
        scal[2] = __longlong_as_double((long long) *(ull*) (scal+8))/scal[1];
        scal[5] = scal[6] = scal[7] = 0;
        scal[9] += 1;
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
        if (n)
            CU(cudaMemcpy(ptr, h, n*sizeof(T), cudaMemcpyHostToDevice));
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
    // own pair list
    bool externalList = false, haveList = false;
    long numBuilds = 0;
    double buildBox[3][3] = {}, skinNow = 0;   // the box and the skin of the current list
    DevArray<int> cellKey, cellKeySorted, cellAtom, cellAtomSorted, cellStart, cellEnd, count, slabCount;
    DevArray<unsigned char> cubTemp;
    DevArray<unsigned> maxDisp;
    // predictor history (device, global order): slot (head + h) % 4 is h steps back
    DevArray<double> hist[4];
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
    std::array<long long, 6> execKey[4];
    std::array<long long, 6> graphKey() const {   // the host values the graph's kernels take as parameters
        return {n, ncov, nGlobal, polarizable ? 1 : 0, externalList ? 1 : 0, gvdwOn ? 1 : 0};
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

    void field(bool withQ, const double* dvec, double* out);   // out == nullptr: leave fld + erec
    void fieldDual();   // b = field(q, perm), T = field(x), together
    bool dualOrder() const { return s.order == 4 || s.order == 5 || s.order == 6 || s.order == 8; }
    void pme(bool withQ, bool final, int energyMode, double* acc10);
    bool needList();
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
    HistPtrs histPtrs() { return HistPtrs{{hist[0].ptr, hist[1].ptr, hist[2].ptr, hist[3].ptr}}; }
    void enqueuePre(bool dual);
    void enqueueCgInit(CondHandle h, bool useHandle, bool dual);
    void enqueueCgIteration(CondHandle h, bool useHandle);
    void enqueuePost(bool strain);
    void captureGraph(bool strain, bool dual);
    void releaseGraphs();
};

/** The list is stale when 2 max displacement + the box change exceeds the skin. */
bool Engine::Impl::needList() {
    if (externalList)
        return false;
    if (!haveList)
        return true;
    double dH = 0;
    for (int a = 0; a < 3; a++) {
        double r2 = 0;
        for (int c = 0; c < 3; c++)
            r2 += (boxD[a][c]-buildBox[a][c])*(boxD[a][c]-buildBox[a][c]);
        dH += std::sqrt(r2);
    }
    CU(cudaMemsetAsync(maxDisp.ptr, 0, sizeof(unsigned), st));
    maxDisplacement<<<std::min(blocksFor(n, 256), 1024), 256, 0, st>>>(n, posq.ptr, posBuild.ptr, maxDisp.ptr);
    CU(cudaMemcpyAsync(dispH.ptr, maxDisp.ptr, sizeof(unsigned), cudaMemcpyDeviceToHost, st));
    CU(cudaStreamSynchronize(st));
    float d2;
    std::memcpy(&d2, dispH.ptr, sizeof(float));
    return 2*std::sqrt((double) d2)+dH > skinNow;
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
    CU(cudaMemsetAsync(gridFix.ptr, 0, gsize*sizeof(ull), st));
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
        spread<PC, true><<<(n+3)/4, 128, 0, st>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr, dtheta.ptr,  \
                                                  gidx.ptr, gridFix.ptr);                                       \
    else                                                                                                        \
        spread<PC, false><<<(n+3)/4, 128, 0, st>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr, dtheta.ptr, \
                                                   gidx.ptr, gridFix.ptr)
    PGM_ORDER_DISPATCH(PGM_SPREAD)
    fixedToReal<<<blocksFor(gsize, 256), 256, 0, st>>>(gsize, gridFix.ptr, grid.ptr);
    CUFFT(cufftExecR2C(planF, grid.ptr, cgrid.ptr));
    const size_t csize = (size_t) K1*K2*(K3/2+1);
    const float beta = (float) s.beta;
    if (final && energyMode == 2)
        convolution<2><<<blocksFor(csize, 256), 256, 0, st>>>(K1, K2, K3, beta, mod1.ptr, mod2.ptr, mod3.ptr,
                                                               cgrid.ptr, accDev, 1);
    else if (final && energyMode == 1)
        convolution<1><<<blocksFor(csize, 256), 256, 0, st>>>(K1, K2, K3, beta, mod1.ptr, mod2.ptr, mod3.ptr,
                                                               cgrid.ptr, accDev, 1);
    else
        convolution<0><<<blocksFor(csize, 256), 256, 0, st>>>(K1, K2, K3, beta, mod1.ptr, mod2.ptr, mod3.ptr,
                                                               cgrid.ptr, nullptr, 1);
    CUFFT(cufftExecC2R(planB, cgrid.ptr, grid.ptr));
    // the usual orders: one thread per atom (gatherT, FINAL also for the field gathers, see there)
#define PGM_GATHER(PC)                                                                                          \
    if (PC > 0)                                                                                                 \
        gatherT<PC ? PC : 4, true><<<(n+63)/64, 64, gatherSmem, st>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr,   \
                                                            dtheta.ptr, d2theta.ptr, gidx.ptr, grid.ptr,       \
                                                            erec.ptr, frec.ptr);                               \
    else if (final)                                                                                             \
        gather<PC, true><<<(n+3)/4, 128, 0, st>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr, dtheta.ptr,  \
                                                  d2theta.ptr, gidx.ptr, grid.ptr, erec.ptr, frec.ptr);         \
    else                                                                                                        \
        gather<PC, false><<<(n+3)/4, 128, 0, st>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr, dtheta.ptr, \
                                                   d2theta.ptr, gidx.ptr, grid.ptr, erec.ptr, nullptr)
    PGM_ORDER_DISPATCH(PGM_GATHER)
#undef PGM_GATHER
#undef PGM_SPREAD
#undef PGM_ORDER_DISPATCH
}

/** out = T d (+ the charges' field if withQ): direct + reciprocal + self, for the dipoles dvec (double). */
void Engine::Impl::field(bool withQ, const double* dvec, double* out) {
    const int m = 3*n;
    toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, dvec, nullptr, dipF.ptr);
    const int warpsPerBlock = 4, threads = 32*warpsPerBlock;
    const int blocks = (n+warpsPerBlock-1)/warpsPerBlock;
    if (withQ)
        fieldPass<true><<<blocks, threads, 0, st>>>(n, order(), atomStart.ptr, pairEnd.ptr, PJ.ptr, posq.ptr,
                                                    dipF.ptr, kernels(), fld.ptr);
    else
        fieldPass<false><<<blocks, threads, 0, st>>>(n, order(), atomStart.ptr, pairEnd.ptr, PJ.ptr, posq.ptr,
                                                     dipF.ptr, kernels(), fld.ptr);
    pme(withQ, false, 0, nullptr);
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    if (out)
        combineField<<<blocksFor(m, 256), 256, 0, st>>>(m, fld.ptr, erec.ptr, dvec, cs, out);
}

void Engine::Impl::fieldDual() {
    const int m = 3*n, K1 = s.grid[0], K2 = s.grid[1], K3 = s.grid[2], P = s.order;
    const size_t gsize = (size_t) K1*K2*K3, csize = (size_t) K1*K2*(K3/2+1);
    toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, perm.ptr, nullptr, dipF.ptr);
    toFloat<<<blocksFor(n, 256), 256, 0, st>>>(m, x.ptr, nullptr, dipF2.ptr);
    const int warpsPerBlock = 4, threads = 32*warpsPerBlock, blocks = (n+warpsPerBlock-1)/warpsPerBlock;
    fieldPassDual<<<blocks, threads, 0, st>>>(n, order(), atomStart.ptr, pairEnd.ptr, PJ.ptr, posq.ptr, dipF.ptr,
                                              dipF2.ptr, kernels(), fld.ptr, fld2.ptr);
    CU(cudaMemsetAsync(gridFix.ptr, 0, 2*gsize*sizeof(ull), st));
#define PGM_DUAL(PC)                                                                                           \
    spreadDual<PC><<<(n+3)/4, 128, 0, st>>>(n, posq.ptr, dipF.ptr, dipF2.ptr, K1, K2, K3, theta.ptr, dtheta.ptr, \
                                            gidx.ptr, gridFix.ptr);                                            \
    fixedToReal<<<blocksFor(2*gsize, 256), 256, 0, st>>>(2*gsize, gridFix.ptr, grid.ptr);                      \
    CUFFT(cufftExecR2C(planF2, grid.ptr, cgrid.ptr));                                                          \
    convolution<0><<<blocksFor(2*csize, 256), 256, 0, st>>>(K1, K2, K3, (float) s.beta, mod1.ptr, mod2.ptr,     \
                                                             mod3.ptr, cgrid.ptr, nullptr, 2);                 \
    CUFFT(cufftExecC2R(planB2, cgrid.ptr, grid.ptr));                                                          \
    /* both gathers with gatherT<FINAL> (frec as scratch): faster than one kernel for both on sm_120 */      \
    gatherT<PC, true><<<(n+63)/64, 64, gatherSmem, st>>>(n, P, posq.ptr, dipF.ptr, K1, K2, K3, theta.ptr,         \
                                                         dtheta.ptr, d2theta.ptr, gidx.ptr, grid.ptr, erec.ptr,   \
                                                         frec.ptr);                                               \
    gatherT<PC, true><<<(n+63)/64, 64, gatherSmem, st>>>(n, P, posq.ptr, dipF2.ptr, K1, K2, K3, theta.ptr,        \
                                                         dtheta.ptr, d2theta.ptr, gidx.ptr, grid.ptr+gsize,       \
                                                         erec2.ptr, frec.ptr)
    switch (P) {
        case 4: PGM_DUAL(4); break;
        case 5: PGM_DUAL(5); break;
        case 6: PGM_DUAL(6); break;
        default: PGM_DUAL(8); break;   // dualOrder()
    }
#undef PGM_DUAL
    // b and T x are combined from the field parts in cgInitK
}

/** Pairs, covalent dipoles, splines; polarizable: b = the field of the charges and permanent dipoles, the
 * initial guess x and T x (dual: the guess from the history, both fields together). */
void Engine::Impl::enqueuePre(bool dual) {
    const int m = 3*n;
    const int warpsPerBlock = 4, threads = 32*warpsPerBlock;
    const int pblocks = (n+warpsPerBlock-1)/warpsPerBlock;
    const float rc2 = (float) (s.cutoff*s.cutoff), rp2 = (float) (pairRadius()*pairRadius()), beta = (float) s.beta;
    if (externalList)
        pairSetup<true><<<pblocks, threads, 0, st>>>(n, nullptr, atomStart.ptr, candJ.ptr, candT.ptr, posq.ptr,
                                                     rg.ptr, rc2, rp2, beta, PJ.ptr, storeKernels ? G12.ptr : nullptr, storeG03 ? G03.ptr : nullptr, pairEnd.ptr);
    else
        pairSetup<false><<<pblocks, threads, 0, st>>>(n, cellAtomSorted.ptr, atomStart.ptr, candJ.ptr, nullptr,
                                                      posq.ptr, rg.ptr, rc2, rp2, beta, PJ.ptr, storeKernels ? G12.ptr : nullptr, storeG03 ? G03.ptr : nullptr,
                                                      pairEnd.ptr);
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
    if (!polarizable) {
        CU(cudaMemsetAsync(x.ptr, 0, m*sizeof(double), st));
        return;
    }
    if (dual) {
        initialGuess<<<blocksFor(m, 256), 256, 0, st>>>(n, ctl.ptr, globalIdx.ptr, histPtrs(), alphaD.ptr, b.ptr,
                                                        x.ptr);   // the predictor: b not needed
        fieldDual();
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
    CU(cudaMemsetAsync(scal.ptr, 0, 16*sizeof(double), st));
    cgInitK<<<cgBlocks, CG_THREADS, 0, st>>>(m, b.ptr, T.ptr, dual ? fld.ptr : nullptr, erec.ptr, perm.ptr, fld2.ptr,
                                             erec2.ptr, cs, alphaD.ptr, x.ptr, r.ptr, z.ptr, p.ptr, scal.ptr, s.tol,
                                             s.maxIter, h, useHandle);
}

void Engine::Impl::enqueueCgIteration(CondHandle h, bool useHandle) {
    const int m = 3*n, cgBlocks = std::max(1, std::min(blocksFor(m, CG_THREADS), 1024));
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    field(false, p.ptr, nullptr);   // T p combined in cgApply
    cgApply<<<cgBlocks, CG_THREADS, 0, st>>>(m, fld.ptr, erec.ptr, cs, alphaD.ptr, p.ptr, Ap.ptr, scal.ptr);
    cgUpdate<<<cgBlocks, CG_THREADS, 0, st>>>(m, alphaD.ptr, x.ptr, r.ptr, z.ptr, p.ptr, Ap.ptr, scal.ptr);
    cgDirection<<<cgBlocks, CG_THREADS, 0, st>>>(m, z.ptr, p.ptr, scal.ptr, s.tol, s.maxIter, h, useHandle);
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
    CU(cudaMemsetAsync(frc.ptr, 0, m*sizeof(ull), st));
    CU(cudaMemsetAsync(acc.ptr, 0, NACC*sizeof(double), st));
    const float rvdw2 = (float) (rvdw*rvdw), invKe = (float) (1.0/s.ke);
#define PGM_FINAL(STRAIN, GV)                                                                                    \
    finalPass<STRAIN, GV><<<pblocks, threads, 0, st>>>(n, order(), atomStart.ptr, pairEnd.ptr, PJ.ptr, posq.ptr,  \
                                                       dipF.ptr, kernels(), rg.ptr, gvP.ptr, gvdwSlater,         \
                                                       rvdw2, invKe, fld.ptr, fpair.ptr, acc.ptr)
    if (strain && gvdwOn)
        PGM_FINAL(true, true);
    else if (strain)
        PGM_FINAL(true, false);
    else if (gvdwOn)
        PGM_FINAL(false, true);
    else
        PGM_FINAL(false, false);
#undef PGM_FINAL
    pme(true, true, strain ? 2 : 1, acc.ptr+10);
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    combineField<<<blocksFor(m, 256), 256, 0, st>>>(m, fld.ptr, erec.ptr, dTot.ptr, cs, Etot.ptr);
    if (ncov > 0)
        covalentForces<<<blocksFor(ncov, 128), 128, 0, st>>>(ncov, posq.ptr, covAtom.ptr, covPartner.ptr,
                                                             covStrength.ptr, Etot.ptr, frc.ptr,
                                                             strain ? acc.ptr+20 : nullptr);
    const int rblocks = std::min(blocksFor(n, 256), 1024);
    if (strain)
        finalReduce<true><<<rblocks, 256, 0, st>>>(n, s.ke, frc.ptr, fpair.ptr, frec.ptr, erec.ptr, dTot.ptr, x.ptr,
                                                   alphaD.ptr, fOut.ptr, acc.ptr);
    else
        finalReduce<false><<<rblocks, 256, 0, st>>>(n, s.ke, frc.ptr, fpair.ptr, frec.ptr, erec.ptr, dTot.ptr, x.ptr,
                                                    alphaD.ptr, fOut.ptr, acc.ptr);
    CU(cudaMemcpyAsync(fH.ptr, fOut.ptr, m*sizeof(double), cudaMemcpyDeviceToHost, st));
    CU(cudaMemcpyAsync(accH.ptr, acc.ptr, NACC*sizeof(double), cudaMemcpyDeviceToHost, st));
    CU(cudaMemcpyAsync(scalH.ptr, scal.ptr, 16*sizeof(double), cudaMemcpyDeviceToHost, st));
}

void Engine::Impl::releaseGraphs() {
    for (int v = 0; v < 4; v++)
        if (exec[v]) {
            cudaGraphExecDestroy(exec[v]);
            exec[v] = nullptr;
        }
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
        enqueueCgInit(handle, true, dual);
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
#else
    (void) strain;
    (void) dual;
#endif
}

Engine::Engine(const Settings& settings) : impl(new Impl) {
    impl->s = settings;
    if (settings.order < 3 || settings.order > MAX_ORDER)
        throw std::runtime_error("pgm_cuda: PME order must be 3..12");
    CU(cudaSetDevice(settings.device));
    CU(cudaStreamCreate(&impl->st));   // blocking: ordered after the synchronous set-up copies
    const int K1 = settings.grid[0], K2 = settings.grid[1], K3 = settings.grid[2];
    CUFFT(cufftPlan3d(&impl->planF, K1, K2, K3, CUFFT_R2C));
    CUFFT(cufftPlan3d(&impl->planB, K1, K2, K3, CUFFT_C2R));
    int dims[3] = {K1, K2, K3};
    CUFFT(cufftPlanMany(&impl->planF2, 3, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, 2));
    CUFFT(cufftPlanMany(&impl->planB2, 3, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, 2));
    for (cufftHandle h : {impl->planF, impl->planB, impl->planF2, impl->planB2})
        CUFFT(cufftSetStream(h, impl->st));
    impl->gridFix.resize(2*(size_t) K1*K2*K3);   // two grids for fieldDual
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
    impl->maxDisp.resize(1);
    impl->ctl.resize(1);
    impl->accH.resize(NACC);
    impl->scalH.resize(16);
    impl->dispH.resize(1);
    impl->boxH.resize(1);
    impl->ctlH.resize(1);
    impl->useGraphs = PGM_CUDA_GRAPHS && std::getenv("PGM_CUDA_NO_GRAPH") == nullptr;
}

Engine::~Engine() {
    impl->releaseGraphs();
    if (impl->planF)
        cufftDestroy(impl->planF);
    if (impl->planB)
        cufftDestroy(impl->planB);
    if (impl->planF2)
        cufftDestroy(impl->planF2);
    if (impl->planB2)
        cufftDestroy(impl->planB2);
    if (impl->st)
        cudaStreamDestroy(impl->st);
    delete impl;
}

void Engine::setAtoms(int n, const double* q, const double* radius, const double* alpha, const int* globalIndex,
                      int numGlobal) {
    Impl& I = *impl;
    CU(cudaSetDevice(I.s.device));
    CU(cudaStreamSynchronize(I.st));
    const bool resized = (n != I.n || numGlobal != I.nGlobal);
    if (resized)
        I.numHist = 0;
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
    const int m = 3*n;
    I.posq.resize(n);
    I.posBuild.resize(n);
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
        I.dipF.zero(n);
        I.atomStart.zero(n+1);
        I.pairEnd.zero(n);
    }
    I.frc.resize(m);
    const size_t np = (size_t) 3*n*I.s.order;
    I.theta.resize(np);
    I.dtheta.resize(np);
    I.d2theta.resize(np);
    I.gidx.resize((size_t) 3*n);
    I.mu.assign(m, 0.0);
    I.permH.assign(m, 0.0);
    I.haveList = false;
    I.externalList = false;
    I.gvdwOn = false;   // setGvdw again for the new order
    I.cellAtomSorted.resize(n);
}

void Engine::setGvdw(int repulsion, double rvdw, const double* sqrtA, const double* sqrtC6, const double* b,
                     int nskip, const int* skipI, const int* skipJ) {
    Impl& I = *impl;
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
    const bool radiusChanged = !I.gvdwOn || I.rvdw != rvdw;
    I.gvdwOn = true;
    I.gvdwSlater = (repulsion == 1);
    I.rvdw = rvdw;
    if (radiusChanged)
        I.haveList = false;   // the list radius may change
    I.skipsMarked = false;
}

void Engine::setCovalent(int ncov, const int* atom, const int* partner, const double* strength) {
    Impl& I = *impl;
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
    Impl& I = *impl;
    const int n = I.n, m = 3*n;
    const Settings& s = I.s;
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
    for (int i = 0; i < n; i++) {
        pq[i].x = (float) pos[3*i];
        pq[i].y = (float) pos[3*i+1];
        pq[i].z = (float) pos[3*i+2];
    }
    CU(cudaMemcpyToSymbolAsync(c_box, I.boxH.ptr, sizeof(Box), 0, cudaMemcpyHostToDevice, I.st));
    CU(cudaMemcpyAsync(I.posq.ptr, pq, n*sizeof(float4), cudaMemcpyHostToDevice, I.st));
    I.muValid = I.permValid = false;
    const double tB = I.timing ? wallSeconds() : 0;
    if (I.needList())
        I.buildList();
    if (I.gvdwOn && !I.skipsMarked)
        I.markSkipList();

    // the initial guess and the history slots
    Control& c = *I.ctlH.ptr;
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
    const bool push = I.polarizable && (newStep || h == 0);
    c.pushSlot = push ? (I.head+3)%4 : -1;   // the oldest slot becomes the newest
    CU(cudaMemcpyAsync(I.ctl.ptr, I.ctlH.ptr, sizeof(Control), cudaMemcpyHostToDevice, I.st));

    const bool wantStrain = (strain != nullptr);
    // the permanent field and the guess's field together when the guess comes from the history
    static const bool noDual = std::getenv("PGM_CUDA_NO_DUAL") != nullptr;
    const bool dual = I.polarizable && c.c[0] != 0 && I.dualOrder() && !noDual;
    const double tC = I.timing ? wallSeconds() : 0;
    int it = 0;
    double err = 0;
    if (I.useGraphs) {
        const int v = (wantStrain ? 1 : 0)+(dual ? 2 : 0);
        if (!I.exec[v] || I.execAllocations[v] != g_allocations || I.execKey[v] != I.graphKey())
            I.captureGraph(wantStrain, dual);
        CU(cudaGraphLaunch(I.exec[v], I.st));
        CU(cudaStreamSynchronize(I.st));
        it = (int) I.scalH.ptr[9];
        err = I.scalH.ptr[2];
    }
    else {
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
        CU(cudaStreamSynchronize(I.st));
    }
    const double tD = I.timing ? wallSeconds() : 0;
    if (!I.polarizable)
        it = 0, err = 0;
    if (push) {
        I.head = c.pushSlot;
        I.numHist = std::min(I.numHist+1, 4);
    }
    I.lastIterations = it;
    I.lastResidual = err;

    const double* accH = I.accH.ptr;
    const double cs = 4.0*s.beta*s.beta*s.beta/(3.0*pgmcore::SQRT_PI);
    Terms t;
    t.direct = s.ke*accH[0];
    t.reciprocal = s.ke*accH[10];
    t.self = s.ke*(-(s.beta/pgmcore::SQRT_PI)*I.q2-0.5*cs*accH[29]);
    const double background = -pgmcore::PI*I.Qtot*I.Qtot/(2.0*I.volume*s.beta*s.beta);
    t.background = s.ke*background;
    t.polarization = s.ke*accH[30];
    t.gvdw = I.gvdwOn ? accH[40] : 0.0;
    if (f)
        for (int k = 0; k < m; k++)
            f[k] += I.fH.ptr[k];
    if (strain) {
        double S[9];
        for (int a = 0; a < 9; a++)
            S[a] = accH[1+a]+accH[11+a]+accH[20+a]+accH[31+a];
        for (int a = 0; a < 3; a++)
            S[4*a] -= background;
        for (int a = 0; a < 9; a++)
            strain[a] += s.ke*S[a];
    }
    CU(cudaGetLastError());
    if (I.timing) {
        const double tE = wallSeconds();
        I.tPhase[0] += tB-tA;
        I.tPhase[1] += tC-tB;
        I.tPhase[2] += tD-tC;
        I.tPhase[3] += tE-tD;
        if (++I.numCalls%1000 == 0)
            fprintf(stderr, "pgm_cuda ms per call (%ld): inputs %.3f, list check/build %.3f, GPU %.3f, outputs %.3f\n",
                    I.numCalls, 1e3*I.tPhase[0]/I.numCalls, 1e3*I.tPhase[1]/I.numCalls, 1e3*I.tPhase[2]/I.numCalls,
                    1e3*I.tPhase[3]/I.numCalls);
    }
    return t;
}

int Engine::iterations() const { return impl->lastIterations; }
double Engine::residual() const { return impl->lastResidual; }
void Engine::inducedDipoles(double* mu) const {
    if (!impl->muValid)
        impl->x.download(impl->mu.data(), 3*(size_t) impl->n);
    impl->muValid = true;
    std::copy(impl->mu.begin(), impl->mu.end(), mu);
}
void Engine::permanentDipoles(double* p) const {
    if (!impl->permValid)
        impl->perm.download(impl->permH.data(), 3*(size_t) impl->n);
    impl->permValid = true;
    std::copy(impl->permH.begin(), impl->permH.end(), p);
}
void Engine::resetHistory() { impl->numHist = 0; }
long Engine::listBuilds() const { return impl->numBuilds; }
int Engine::numCandidates() const { return impl->ncand; }

} // namespace pgmcuda

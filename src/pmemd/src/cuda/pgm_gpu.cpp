#include <string.h>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include "copyright_PGM.i"

//---------------------------------------------------------------------------------------------
// PGM CUDA GPU Interface
//
// Implements extern "C" functions called from Fortran (pGM_gpu_interface.F90).
// Uses the existing pmemd gpuContext - no separate PGM context.
//
// Workflow:
//   1. Upload PGM prmtop data (once): charges, local dipoles, covalent bonds, r_gauss, pol
//   2. Local-to-global dipole conversion (every step, uses coords already on GPU)
//   3. Permanent field: direct + self + recip
//   4. SCF for induced dipoles
//   5. Forces
//---------------------------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "gpuContext.h"
#include "gpu.h"

// Access the existing global gpu pointer from gpu.cpp
namespace gppImpl {
  extern gpuContext gpu;
}
using namespace gppImpl;

// Named constants
// Coulomb constant: MUST match constants.F90 value exactly (NOT 332.0522173)
static const double PGM_COULOMB_CONST = 332.05382;

// Batched CG iteration batch sizes (tuned for convergence vs launch overhead)
static const int PGM_CG_BATCH_COLD    = 8;  // cold start (no history)
static const int PGM_CG_BATCH_WARM    = 6;  // warm start (1-2 history points)
static const int PGM_CG_BATCH_EXTRAP  = 3;  // extrapolated (3+ history points)

extern "C" void kPGMExtrapolateBase(gpuContext gpu, double* dst, const double* base,
                                    double* h0, double* h1, double* h2, double* h3, int K, int cubic);
extern "C" void kPGMVecSub(gpuContext gpu, double* dst, const double* x, const double* b);
extern "C" void kPGMExtrapolate8(gpuContext gpu, double* dst, const double* base, double** H);
extern "C" void kCalculatePMEDirectFieldCacheWriteH_PGM(gpuContext gpu);
extern "C" void kPGMXLUpdate(gpuContext gpu, double* dst, const double* mu, double** H, int K, double kappa, double alpha, const double* c);
// Extended-Lagrangian 0-SCF (PGM_GPU_PRED=xl): auxiliary dipoles n propagated with Niklasson's dissipative
// integrator; each step: fused sweep gives r0 = b - A n, mu = n + w*r0, forces from mu, then
// n_{k+1} = 2n_k - n_{k-1} + kappa*(mu - n_k) + alpha*sum_j c_j n_{k-j}.   env PGM_XL_W = w (default 0.727)
static double pgm_xl_w = 0.727;
#include <vector>
// ---- CLAUDE 2026-10-08: mode 8 'cpuls' = exact port of the CPU predictor (pGM_induced.F90 extrap_ind_dipole,
//      dipole_scf_init=3, dipole_scf_init_order=3, dipole_scf_init_step=2).  Records live on the device so they can be
//      remapped across neighbor-list rebuilds with kPGMRemapDipHistory; the tiny least-squares math runs on the host.
static const int CPULS_STEP = 2, CPULS_ORDER = 3;
static GpuBuffer<double>* cpuls_R[4][CPULS_STEP] = {{NULL,NULL},{NULL,NULL},{NULL,NULL},{NULL,NULL}};  // record1..4
static GpuBuffer<double>* cpuls_O[2] = {NULL, NULL};   // oldrec(:,:,1:2) = stage-1 / stage-2 predictions
static GpuBuffer<double>* cpuls_tmp = NULL;
static int cpuls_cnt[4] = {0,0,0,0};                   // reccnt(1:4)
static int cpuls_slot[4] = {-1,-1,-1,-1};              // last written slot per record (ring of CPULS_STEP)
static void cpuls_alloc(int stride) {
  if (cpuls_tmp) return;
  for (int k = 0; k < 4; k++) for (int j = 0; j < CPULS_STEP; j++) cpuls_R[k][j] = new GpuBuffer<double>(3 * stride);
  cpuls_O[0] = new GpuBuffer<double>(3 * stride); cpuls_O[1] = new GpuBuffer<double>(3 * stride);
  cpuls_tmp = new GpuBuffer<double>(3 * stride);
}
static void cpuls_push(int k, const std::vector<double>& v) {        // update_record()
  cpuls_slot[k] = (cpuls_slot[k] + 1) % CPULS_STEP;
  cudaMemcpy(cpuls_R[k][cpuls_slot[k]]->_pDevData, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice);
  cpuls_cnt[k]++;
}
static void cpuls_fetch(GpuBuffer<double>* b, std::vector<double>& v) {
  cudaMemcpy(v.data(), b->_pDevData, v.size() * sizeof(double), cudaMemcpyDeviceToHost);
}
static double cpuls_dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0; for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i]; return s;
}
// least_square_extrap(): c = argmin || sum_j c_j record_j - target ||^2, solved with init_step CG iterations as on the CPU
static void cpuls_ls(const std::vector<double> R[CPULS_STEP], const std::vector<double>& target, double c[CPULS_STEP]) {
  const int n = CPULS_STEP; double M[CPULS_STEP][CPULS_STEP], b[CPULS_STEP];
  for (int i = 0; i < n; i++) { b[i] = cpuls_dot(R[i], target); for (int j = i; j < n; j++) { M[i][j] = cpuls_dot(R[i], R[j]); M[j][i] = M[i][j]; } }
  double r[CPULS_STEP], p[CPULS_STEP], Mp[CPULS_STEP];
  for (int i = 0; i < n; i++) { c[i] = 0.0; r[i] = b[i]; p[i] = r[i]; }
  for (int it = 0; it < n; it++) {
    for (int j = 0; j < n; j++) { Mp[j] = 0.0; for (int k = 0; k < n; k++) Mp[j] += M[j][k] * p[k]; }
    double pMp = 0.0, rr = 0.0; for (int j = 0; j < n; j++) { pMp += Mp[j] * p[j]; rr += r[j] * r[j]; }
    double alpha = (fabs(pMp) > 1.0e-50) ? rr / pMp : 0.0;
    for (int j = 0; j < n; j++) { c[j] += alpha * p[j]; r[j] -= alpha * Mp[j]; }
    double rr2 = 0.0; for (int j = 0; j < n; j++) rr2 += r[j] * r[j];
    double beta = (rr > 1.0e-50) ? rr2 / rr : 0.0;
    for (int j = 0; j < n; j++) p[j] = r[j] + beta * p[j];
  }
}
// extrap_ind_dipole(): called before the SCF solve; sets x0 in pPGMCGx
static void cpuls_predict(gpuContext gpu, int stride) {
  cpuls_alloc(stride);
  const int n3 = 3 * gpu->sim.atoms;
  std::vector<double> b(n3), pol(gpu->sim.atoms), B(n3), x0(n3), R[CPULS_STEP];
  cudaMemcpy(b.data(), gpu->sim.pPGMCGb, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(pol.data(), gpu->sim.pPGMPolarizability, gpu->sim.atoms * sizeof(double), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n3; i++) B[i] = pol[i / 3] * b[i];          // ind_dip_new = pol * (alpha*E_perm), as on the CPU
  for (int j = 0; j < CPULS_STEP; j++) R[j].resize(n3);
  double c[CPULS_STEP] = {0.0, 0.0};
  bool have_c = false;
  if (cpuls_cnt[0] >= CPULS_STEP) { for (int j = 0; j < CPULS_STEP; j++) cpuls_fetch(cpuls_R[0][j], R[j]); cpuls_ls(R, B, c); have_c = true; }
  else { x0 = B; cudaMemcpy(gpu->sim.pPGMCGx, x0.data(), n3 * sizeof(double), cudaMemcpyHostToDevice); }
  cpuls_push(0, B);
  // stage 1: total induced dipoles (record2)
  if (cpuls_cnt[1] >= CPULS_STEP && have_c) {
    for (int j = 0; j < CPULS_STEP; j++) cpuls_fetch(cpuls_R[1][j], R[j]);
    for (int i = 0; i < n3; i++) { double s = 0.0; for (int j = 0; j < CPULS_STEP; j++) s += c[j] * R[j][i]; x0[i] = s; }
    cudaMemcpy(cpuls_O[0]->_pDevData, x0.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);
  } else {
    if (cpuls_cnt[0] == 1 || CPULS_ORDER == 1) { x0 = B; cudaMemcpy(gpu->sim.pPGMCGx, x0.data(), n3 * sizeof(double), cudaMemcpyHostToDevice); }
    return;   // otherwise keep previous converged dipoles already in pPGMCGx (CPU behaviour)
  }
  // stage 2: first-order differences (record3)
  if (CPULS_ORDER >= 2 && cpuls_cnt[2] >= CPULS_STEP) {
    for (int j = 0; j < CPULS_STEP; j++) cpuls_fetch(cpuls_R[2][j], R[j]);
    for (int i = 0; i < n3; i++) { double s = 0.0; for (int j = 0; j < CPULS_STEP; j++) s += c[j] * R[j][i]; x0[i] += s; }
    cudaMemcpy(cpuls_O[1]->_pDevData, x0.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);
  }
  // stage 3: second-order differences (record4)
  if (CPULS_ORDER == 3 && cpuls_cnt[3] >= CPULS_STEP) {
    for (int j = 0; j < CPULS_STEP; j++) cpuls_fetch(cpuls_R[3][j], R[j]);
    for (int i = 0; i < n3; i++) { double s = 0.0; for (int j = 0; j < CPULS_STEP; j++) s += c[j] * R[j][i]; x0[i] += s; }
  }
  cudaMemcpy(gpu->sim.pPGMCGx, x0.data(), n3 * sizeof(double), cudaMemcpyHostToDevice);
}
// post-solve record updates (pGM_INDUCED_eval after the solver)
static void cpuls_update(gpuContext gpu, int stride) {
  const int n3 = 3 * gpu->sim.atoms;
  std::vector<double> mu(n3), o(n3), d(n3);
  cudaMemcpy(mu.data(), gpu->sim.pPGMCGx, n3 * sizeof(double), cudaMemcpyDeviceToHost);
  cpuls_push(1, mu);
  if (CPULS_ORDER >= 2 && cpuls_cnt[1] > CPULS_STEP) {
    cpuls_fetch(cpuls_O[0], o); for (int i = 0; i < n3; i++) d[i] = mu[i] - o[i]; cpuls_push(2, d);
  }
  if (CPULS_ORDER == 3 && cpuls_cnt[2] > CPULS_STEP) {
    cpuls_fetch(cpuls_O[1], o); for (int i = 0; i < n3; i++) d[i] = mu[i] - o[i]; cpuls_push(3, d);
  }
}
// ---- device implementation (mode 'cpulsd'): identical algorithm and slot bookkeeping, math in kCPULS_PGM.cu
extern "C" void cpuls_dev_scale_pol(double* B, const double* b, const double* pol, int n3);
extern "C" void cpuls_dev_ls(const double* R0, const double* R1, const double* T, int n3, double* part, double* c);
extern "C" void cpuls_dev_comb(double* dst, const double* R0, const double* R1, const double* c, int n3, int accumulate);
extern "C" void cpuls_dev_sub(double* d, const double* a, const double* b, int n3);
extern "C" int  cpuls_dev_scratch_size();
extern "C" double cpuls_dev_power(const double* r, const double* x, double* xprev, const double* pol, int n3, double* part, int skip);
// CLAUDE: env PGM_POWER=N -> every N steps print mean per-step dH = -sum r.dP/alpha (kcal/mol) and its ns^-1 rate at 1 fs
static int pgm_power_every = -1; static bool pgm_power_skip = true; static GpuBuffer<double>* pgm_power_xprev = NULL;
static GpuBuffer<double>* pgm_power_part = NULL; static double pgm_power_sum = 0.0, pgm_power_sum_all = 0.0; static long pgm_power_n = 0, pgm_power_n_all = 0;
static int cpuls_device = 0;
static GpuBuffer<double>* cpuls_part = NULL;
static int cpuls_advance(int k) { cpuls_slot[k] = (cpuls_slot[k] + 1) % CPULS_STEP; cpuls_cnt[k]++; return cpuls_slot[k]; }
static void cpulsd_predict(gpuContext gpu, int stride) {
  cpuls_alloc(stride);
  if (!cpuls_part) cpuls_part = new GpuBuffer<double>(cpuls_dev_scratch_size());
  const int n3 = 3 * gpu->sim.atoms; const size_t bytes = n3 * sizeof(double);
  double* c = cpuls_part->_pDevData + cpuls_dev_scratch_size() - 2;
  double* x = gpu->sim.pPGMCGx; double* B = cpuls_tmp->_pDevData;
  cpuls_dev_scale_pol(B, gpu->sim.pPGMCGb, gpu->sim.pPGMPolarizability, n3);
  bool have_c = false;
  if (cpuls_cnt[0] >= CPULS_STEP) { cpuls_dev_ls(cpuls_R[0][0]->_pDevData, cpuls_R[0][1]->_pDevData, B, n3, cpuls_part->_pDevData, c); have_c = true; }
  else cudaMemcpyAsync(x, B, bytes, cudaMemcpyDeviceToDevice, 0);
  { int s = cpuls_advance(0); cudaMemcpyAsync(cpuls_R[0][s]->_pDevData, B, bytes, cudaMemcpyDeviceToDevice, 0); }
  if (!(cpuls_cnt[1] >= CPULS_STEP && have_c)) {
    if (cpuls_cnt[0] == 1 || CPULS_ORDER == 1) cudaMemcpyAsync(x, B, bytes, cudaMemcpyDeviceToDevice, 0);
    return;
  }
  cpuls_dev_comb(x, cpuls_R[1][0]->_pDevData, cpuls_R[1][1]->_pDevData, c, n3, 0);
  cudaMemcpyAsync(cpuls_O[0]->_pDevData, x, bytes, cudaMemcpyDeviceToDevice, 0);
  if (CPULS_ORDER >= 2 && cpuls_cnt[2] >= CPULS_STEP) {
    cpuls_dev_comb(x, cpuls_R[2][0]->_pDevData, cpuls_R[2][1]->_pDevData, c, n3, 1);
    cudaMemcpyAsync(cpuls_O[1]->_pDevData, x, bytes, cudaMemcpyDeviceToDevice, 0);
  }
  if (CPULS_ORDER == 3 && cpuls_cnt[3] >= CPULS_STEP)
    cpuls_dev_comb(x, cpuls_R[3][0]->_pDevData, cpuls_R[3][1]->_pDevData, c, n3, 1);
}
static void cpulsd_update(gpuContext gpu, int stride) {
  const int n3 = 3 * gpu->sim.atoms; const size_t bytes = n3 * sizeof(double);
  double* x = gpu->sim.pPGMCGx;
  { int s = cpuls_advance(1); cudaMemcpyAsync(cpuls_R[1][s]->_pDevData, x, bytes, cudaMemcpyDeviceToDevice, 0); }
  if (CPULS_ORDER >= 2 && cpuls_cnt[1] > CPULS_STEP) { int s = cpuls_advance(2); cpuls_dev_sub(cpuls_R[2][s]->_pDevData, x, cpuls_O[0]->_pDevData, n3); }
  if (CPULS_ORDER == 3 && cpuls_cnt[2] > CPULS_STEP) { int s = cpuls_advance(3); cpuls_dev_sub(cpuls_R[3][s]->_pDevData, x, cpuls_O[1]->_pDevData, n3); }
}
// MC-barostat trial save/restore of the predictor state (both cpuls modes)
static GpuBuffer<double>* cpuls_saved[10] = {NULL};
static int cpuls_saved_cnt[4], cpuls_saved_slot[4]; static bool cpuls_saved_valid = false;
static GpuBuffer<double>* cpuls_buf(int i) { return i < 8 ? cpuls_R[i / 2][i % 2] : cpuls_O[i - 8]; }
static void cpuls_save_state(gpuContext gpu) {
  if (!cpuls_tmp) { cpuls_saved_valid = false; return; }
  size_t bytes = 3 * gpu->sim.stride * sizeof(double);
  for (int i = 0; i < 10; i++) {
    if (!cpuls_saved[i]) cpuls_saved[i] = new GpuBuffer<double>(3 * gpu->sim.stride);
    cudaMemcpy(cpuls_saved[i]->_pDevData, cpuls_buf(i)->_pDevData, bytes, cudaMemcpyDeviceToDevice);
  }
  for (int k = 0; k < 4; k++) { cpuls_saved_cnt[k] = cpuls_cnt[k]; cpuls_saved_slot[k] = cpuls_slot[k]; }
  cpuls_saved_valid = true;
}
static void cpuls_restore_state(gpuContext gpu, const unsigned int* saved_lookup) {
  if (!cpuls_saved_valid) return;
  for (int i = 0; i < 10; i++)   // saved data are in the ordering of the save; remap into the current ordering
    kPGMRemapDipHistory(gpu, cpuls_buf(i)->_pDevData, cpuls_saved[i]->_pDevData, saved_lookup);
  for (int k = 0; k < 4; k++) { cpuls_cnt[k] = cpuls_saved_cnt[k]; cpuls_slot[k] = cpuls_saved_slot[k]; }
}
// remap all records (and the warm-start x) to the new image ordering after a neighbor-list rebuild
static void cpuls_remap(gpuContext gpu, int stride) {
  unsigned int* old_lookup = gpu->pbPGMOldImageLookup->_pDevData;
  size_t bytes = 3 * stride * sizeof(double);
  for (int k = 0; k < 4; k++) for (int j = 0; j < CPULS_STEP; j++) {
    cudaMemcpy(cpuls_tmp->_pDevData, cpuls_R[k][j]->_pDevData, bytes, cudaMemcpyDeviceToDevice);
    kPGMRemapDipHistory(gpu, cpuls_R[k][j]->_pDevData, cpuls_tmp->_pDevData, old_lookup);
  }
  for (int k = 0; k < 2; k++) {
    cudaMemcpy(cpuls_tmp->_pDevData, cpuls_O[k]->_pDevData, bytes, cudaMemcpyDeviceToDevice);
    kPGMRemapDipHistory(gpu, cpuls_O[k]->_pDevData, cpuls_tmp->_pDevData, old_lookup);
  }
  cudaMemcpy(cpuls_tmp->_pDevData, gpu->sim.pPGMCGx, bytes, cudaMemcpyDeviceToDevice);
  kPGMRemapDipHistory(gpu, gpu->sim.pPGMCGx, cpuls_tmp->_pDevData, old_lookup);
}

static const double pgm_xl_kappa = 1.82, pgm_xl_alpha = 0.018;
static const double pgm_xl_c[6] = {-6.0, 14.0, -8.0, -3.0, 4.0, -1.0};   // K=5 Niklasson coefficients
extern "C" void kCalculatePMEDirectFieldCacheReadH_PGM(gpuContext gpu);
// Predictor mode (env PGM_GPU_PRED): 0 = quadratic on mu (default, legacy)
//   1 = b + quadratic on Delta ; 2 = b + cubic on Delta ; 3 = b + LS-quadratic(4) on Delta
//   where Delta = mu - b is the many-body correction beyond alpha*E_perm.
extern "C" void kPGMExtrapolate5(gpuContext gpu, double* dst, const double* base,
                                 double* h0, double* h1, double* h2, double* h3, double* h4);
extern "C" void kPGMAddScaledDip(gpuContext gpu, const double* v, double scale);
extern "C" void kPGMAxpy(gpuContext gpu, double* x, const double* r, double w);
//   4 = cubic on mu (K=4) ; 5 = quartic on mu (K=5)   [no dependence on the current field:
//   these allow the fused permanent-field + initial-residual sweep, env PGM_FUSED=1]
static int pgm_overlap = 0;                        // env PGM_OVERLAP: run reciprocal chain concurrently with the direct kernel
static cudaStream_t pgm_s2 = NULL;
static cudaEvent_t pgm_ev_fork = NULL, pgm_ev_join = NULL;
static void pgm_overlap_init() {
  if (pgm_s2) return;
  cudaStreamCreateWithFlags(&pgm_s2, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(&pgm_ev_fork, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&pgm_ev_join, cudaEventDisableTiming);
}
static int pgm_cache16 = 0;                        // env PGM_CACHE16: half2 B-factor cache for CG matvecs (SPFP)
static bool pgm_in_cg_loop = false;
static int pgm_pred_mode = -1;
static int pgm_fused = 0;                          // env PGM_FUSED: fuse perm-field and r0 sweeps
static double pgm_peek_w = 0.0;                    // env PGM_PEEK: Richardson step weight after CG
static double pgm_bnorm_frozen = -1.0;             // exact mean|b| captured on a non-fused step
static int pgm_cg_batch_env = 0;                   // env PGM_CG_BATCH overrides all batch sizes if > 0
static GpuBuffer<double>* pgm_b_prev = NULL;      // previous step's b (image order)
static int pgm_b_prev_valid = 0;
static int pgm_pred_hist_max() {
  if (pgm_pred_mode == 7) return 6;
  if (pgm_pred_mode == 4) return 4;
  if (pgm_pred_mode == 5) return 5;
  if (pgm_pred_mode == 6) return 8;
  return (pgm_pred_mode >= 2) ? 4 : 3;
}
// Solver options from the &pol_gauss namelist (pgm_gpu_set_options_); environment variables
// PGM_GPU_PRED, PGM_FUSED, PGM_OVERLAP, PGM_CG_BATCH and PGM_POWER override them (developer use).
static int pgm_opt_pred = 1, pgm_opt_check = 3, pgm_opt_overlap = 1, pgm_opt_power = 0;
static double pgm_power_dt = 0.001;   // ps, set from the MD time step
static void pgm_pred_init() {
  if (pgm_pred_mode >= 0) return;
  const char* e = getenv("PGM_GPU_PRED");
  // defaults from mdin: 0 quadratic, 1 cubic + combined field, 2 MOE (device)
  pgm_pred_mode = (pgm_opt_pred == 1) ? 4 : 0;
  if (pgm_opt_pred == 2) { pgm_pred_mode = 8; cpuls_device = 1; }
  pgm_fused = (pgm_opt_pred == 1) ? 1 : 0;
  pgm_overlap = pgm_opt_overlap ? 1 : 0;
  pgm_cg_batch_env = pgm_opt_check;
  if (e) {
    pgm_pred_mode = 0; cpuls_device = 0;
    if (!strcmp(e, "delta3")) pgm_pred_mode = 1;
    else if (!strcmp(e, "delta4")) pgm_pred_mode = 2;
    else if (!strcmp(e, "delta4ls")) pgm_pred_mode = 3;
    else if (!strcmp(e, "mu4")) pgm_pred_mode = 4;
    else if (!strcmp(e, "mu5")) pgm_pred_mode = 5;
    else if (!strcmp(e, "delta8")) pgm_pred_mode = 6;   // b + LS quartic over 8 Delta points
    else if (!strcmp(e, "xl")) pgm_pred_mode = 7;        // extended-Lagrangian 0-SCF (2 sweeps/step, no CG)
    else if (!strcmp(e, "cpuls")) pgm_pred_mode = 8;     // CLAUDE: exact CPU field-anchored LS predictor (order 3, step 2), host math
    else if (!strcmp(e, "cpulsd")) { pgm_pred_mode = 8; cpuls_device = 1; }  // CLAUDE: same, device math (production path)
  }
  { const char* w = getenv("PGM_XL_W"); if (w) pgm_xl_w = atof(w); if (pgm_pred_mode == 7) fprintf(stderr, "PGM XL: w = %.3f kappa %.2f alpha %.3f\n", pgm_xl_w, pgm_xl_kappa, pgm_xl_alpha);
  }
  const char* f = getenv("PGM_FUSED");
  if (f || e) pgm_fused = ((f && atoi(f) > 0 && (pgm_pred_mode == 4 || pgm_pred_mode == 5)) || pgm_pred_mode == 7) ? 1 : 0;
  { const char* ov = getenv("PGM_OVERLAP"); if (ov) pgm_overlap = (atoi(ov) > 0) ? 1 : 0; }
  const char* c16 = getenv("PGM_CACHE16");
#ifndef use_DPFP
  pgm_cache16 = (c16 && atoi(c16) > 0) ? 1 : 0;
#endif
  const char* pk = getenv("PGM_PEEK");
  if (pk) pgm_peek_w = atof(pk);
  const char* bsz = getenv("PGM_CG_BATCH");
  if (bsz) pgm_cg_batch_env = atoi(bsz);
  if (e || f || bsz || getenv("PGM_OVERLAP"))
    fprintf(stderr, "PGM GPU solver (environment override): predictor mode = %d (%s), check = %d, fused = %d, overlap = %d\n",
            pgm_pred_mode, e ? e : "mdin", pgm_cg_batch_env, pgm_fused, pgm_overlap);
}

// Helper: check CUDA graph API return code
#define CUDA_GRAPH_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "PGM CUDA Graph error: %s at %s:%d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
    RTERROR(err, #call); \
  } \
} while(0)

//---------------------------------------------------------------------------------------------
// pgm_gpu_init_: initialize PGM data structures within existing gpuContext
//
// Called from Fortran: pGM_GPU_init(natom)
// NOTE: gpu and gpu->sim.atoms already set up by standard pmemd gpu_init_.
//       This just allocates PGM-specific buffers.
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_init_(int* natom)
{
  int n = *natom;

  gpu->ipgm = 1;
  gpu->sim.ipgm = 1;
  gpu->sim.invCoulombConst = (PMEFloat)(1.0 / PGM_COULOMB_CONST);

  int stride = gpu->sim.stride;

  // Per-atom PGM data
  gpu->pbPGMPermChg       = new GpuBuffer<double>(stride);
  gpu->pbPGMRGauss        = new GpuBuffer<double>(stride);
  gpu->pbPGMPolarizability = new GpuBuffer<double>(stride);
  gpu->pbPGMPrecondInv     = new GpuBuffer<double>(stride);

  // Original-order copies for GPU NL remap (uploaded once, scatter on each NL rebuild)
  gpu->pbPGMOrigChg    = new GpuBuffer<double>(stride);
  gpu->pbPGMOrigRGauss = new GpuBuffer<double>(stride);
  gpu->pbPGMOrigPol    = new GpuBuffer<double>(stride);
  gpu->sim.pPGMOrigChg    = gpu->pbPGMOrigChg->_pDevData;
  gpu->sim.pPGMOrigRGauss = gpu->pbPGMOrigRGauss->_pDevData;
  gpu->sim.pPGMOrigPol    = gpu->pbPGMOrigPol->_pDevData;

  // Permanent dipoles (global frame, computed on GPU)
  gpu->pbPermDipX  = new GpuBuffer<double>(stride);
  gpu->pbPermDipY  = new GpuBuffer<double>(stride);
  gpu->pbPermDipZ  = new GpuBuffer<double>(stride);
  gpu->pbPDipXYSP  = new GpuBuffer<PMEFloat2>(stride);
  gpu->pbPDipZSP   = new GpuBuffer<PMEFloat>(stride);

  // Fractional dipoles (for reciprocal grid fill)
  gpu->pbPGMFracDipX = new GpuBuffer<PMEFloat>(stride);
  gpu->pbPGMFracDipY = new GpuBuffer<PMEFloat>(stride);
  gpu->pbPGMFracDipZ = new GpuBuffer<PMEFloat>(stride);

  // Induced dipoles
  gpu->pbIndDipX  = new GpuBuffer<double>(stride);
  gpu->pbIndDipY  = new GpuBuffer<double>(stride);
  gpu->pbIndDipZ  = new GpuBuffer<double>(stride);
  gpu->pbIDipXYSP = new GpuBuffer<PMEFloat2>(stride);
  gpu->pbIDipZSP  = new GpuBuffer<PMEFloat>(stride);

  // Field accumulators
  gpu->pbPGMFieldXAccumulator = new GpuBuffer<PMEAccumulator>(stride);
  gpu->pbPGMFieldYAccumulator = new GpuBuffer<PMEAccumulator>(stride);
  gpu->pbPGMFieldZAccumulator = new GpuBuffer<PMEAccumulator>(stride);

  // Potential accumulator (phi_000, for energy)
  gpu->pbPGMPotentialAccumulator = new GpuBuffer<PMEAccumulator>(stride);

  // Wire device pointers into cSim
  gpu->sim.pPGMPermChg       = gpu->pbPGMPermChg->_pDevData;
  gpu->sim.pPGMRGauss        = gpu->pbPGMRGauss->_pDevData;
  gpu->sim.pPGMPolarizability = gpu->pbPGMPolarizability->_pDevData;
  gpu->sim.pPGMPrecondInv     = gpu->pbPGMPrecondInv->_pDevData;

  gpu->sim.pPermDipX  = gpu->pbPermDipX->_pDevData;
  gpu->sim.pPermDipY  = gpu->pbPermDipY->_pDevData;
  gpu->sim.pPermDipZ  = gpu->pbPermDipZ->_pDevData;
  gpu->sim.pPDipXYSP  = gpu->pbPDipXYSP->_pDevData;
  gpu->sim.pPDipZSP   = gpu->pbPDipZSP->_pDevData;

  gpu->sim.pPGMFracDipX = gpu->pbPGMFracDipX->_pDevData;
  gpu->sim.pPGMFracDipY = gpu->pbPGMFracDipY->_pDevData;
  gpu->sim.pPGMFracDipZ = gpu->pbPGMFracDipZ->_pDevData;

  gpu->sim.pIndDipX  = gpu->pbIndDipX->_pDevData;
  gpu->sim.pIndDipY  = gpu->pbIndDipY->_pDevData;
  gpu->sim.pIndDipZ  = gpu->pbIndDipZ->_pDevData;
  gpu->sim.pIDipXYSP = gpu->pbIDipXYSP->_pDevData;
  gpu->sim.pIDipZSP  = gpu->pbIDipZSP->_pDevData;

  gpu->sim.pPGMFieldXAccumulator = gpu->pbPGMFieldXAccumulator->_pDevData;
  gpu->sim.pPGMFieldYAccumulator = gpu->pbPGMFieldYAccumulator->_pDevData;
  gpu->sim.pPGMFieldZAccumulator = gpu->pbPGMFieldZAccumulator->_pDevData;
  gpu->sim.pPGMPotentialAccumulator = gpu->pbPGMPotentialAccumulator->_pDevData;

  // Total dipoles (perm + induced)
  gpu->pbTotDipX     = new GpuBuffer<double>(stride);
  gpu->pbTotDipY     = new GpuBuffer<double>(stride);
  gpu->pbTotDipZ     = new GpuBuffer<double>(stride);
  gpu->pbTDipXYSP    = new GpuBuffer<PMEFloat2>(stride);
  gpu->pbTDipZSP     = new GpuBuffer<PMEFloat>(stride);
  gpu->pbPGMFracTotDipX = new GpuBuffer<PMEFloat>(stride);
  gpu->pbPGMFracTotDipY = new GpuBuffer<PMEFloat>(stride);
  gpu->pbPGMFracTotDipZ = new GpuBuffer<PMEFloat>(stride);

  gpu->sim.pTotDipX         = gpu->pbTotDipX->_pDevData;
  gpu->sim.pTotDipY         = gpu->pbTotDipY->_pDevData;
  gpu->sim.pTotDipZ         = gpu->pbTotDipZ->_pDevData;
  gpu->sim.pTDipXYSP        = gpu->pbTDipXYSP->_pDevData;
  gpu->sim.pTDipZSP         = gpu->pbTDipZSP->_pDevData;
  gpu->sim.pPGMFracTotDipX  = gpu->pbPGMFracTotDipX->_pDevData;
  gpu->sim.pPGMFracTotDipY  = gpu->pbPGMFracTotDipY->_pDevData;
  gpu->sim.pPGMFracTotDipZ  = gpu->pbPGMFracTotDipZ->_pDevData;

  // CG solver vectors (interleaved xyz, 3*stride)
  int cg_size = 3 * stride;
  gpu->pbPGMCGx       = new GpuBuffer<double>(cg_size);
  gpu->pbPGMCGr       = new GpuBuffer<double>(cg_size);
  gpu->pbPGMCGp       = new GpuBuffer<double>(cg_size);
  gpu->pbPGMCGz       = new GpuBuffer<double>(cg_size);
  gpu->pbPGMCGb       = new GpuBuffer<double>(cg_size);
  gpu->pbPGMCGScalars = new GpuBuffer<double>(16);

  gpu->sim.pPGMCGx       = gpu->pbPGMCGx->_pDevData;
  gpu->sim.pPGMCGr       = gpu->pbPGMCGr->_pDevData;
  gpu->sim.pPGMCGp       = gpu->pbPGMCGp->_pDevData;
  gpu->sim.pPGMCGz       = gpu->pbPGMCGz->_pDevData;
  gpu->sim.pPGMCGb       = gpu->pbPGMCGb->_pDevData;
  gpu->sim.pPGMCGScalars = gpu->pbPGMCGScalars->_pDevData;

  // PCG inner solver vectors
  gpu->pbPGMPCGzpc = new GpuBuffer<double>(cg_size);
  gpu->pbPGMPCGru  = new GpuBuffer<double>(cg_size);
  gpu->pbPGMPCGpu  = new GpuBuffer<double>(cg_size);
  gpu->sim.pPGMPCGzpc = gpu->pbPGMPCGzpc->_pDevData;
  gpu->sim.pPGMPCGru  = gpu->pbPGMPCGru->_pDevData;
  gpu->sim.pPGMPCGpu  = gpu->pbPGMPCGpu->_pDevData;

  // Saved charges/dipoles for SCF (device-side save/restore)
  gpu->pbPGMSavedPermChg    = new GpuBuffer<double>(stride);
  gpu->pbPGMSavedAtomChgSP  = new GpuBuffer<PMEFloat>(stride);
  gpu->pbPGMSavedPermDipX   = new GpuBuffer<double>(stride);
  gpu->pbPGMSavedPermDipY   = new GpuBuffer<double>(stride);
  gpu->pbPGMSavedPermDipZ   = new GpuBuffer<double>(stride);

  gpu->sim.pPGMSavedPermChg    = gpu->pbPGMSavedPermChg->_pDevData;
  gpu->sim.pPGMSavedAtomChgSP  = gpu->pbPGMSavedAtomChgSP->_pDevData;
  gpu->sim.pPGMSavedPermDipX   = gpu->pbPGMSavedPermDipX->_pDevData;
  gpu->sim.pPGMSavedPermDipY   = gpu->pbPGMSavedPermDipY->_pDevData;
  gpu->sim.pPGMSavedPermDipZ   = gpu->pbPGMSavedPermDipZ->_pDevData;

  // Partial reduction buffer for CG (256 blocks max)
  int cg_nblocks = (n + 127) / 128;
  if (cg_nblocks > 256) cg_nblocks = 256;
  gpu->pbPGMCGPartials = new GpuBuffer<double>(cg_nblocks * 4);  // 4 scalars per block
  gpu->sim.pPGMCGPartials = gpu->pbPGMCGPartials->_pDevData;
  gpu->sim.pgmCGnblocks   = cg_nblocks;

  // T2 tensor cache (allocated lazily after first NL build)
  gpu->pbPGMT2Cache       = NULL;
  gpu->pbPGMBFactorOffset = NULL;
  gpu->pgmBFactorMaxPairs   = 0;
  gpu->pgmBFactorMaxPairsH = 0;
  gpu->pbPGMT2CacheH = NULL;
  gpu->sim.pPGMBD_h2 = NULL;
  gpu->pgmBFactorMaxEntries = 0;
  gpu->sim.pPGMBD_nB1        = NULL;
  gpu->sim.pPGMBD_nB2        = NULL;
  gpu->sim.pPGMBD_dx         = NULL;
  gpu->sim.pPGMBD_dy         = NULL;
  gpu->sim.pPGMBD_dz         = NULL;
  gpu->sim.pPGMBFactorOffset = NULL;
  gpu->sim.pgmBFactorTotalPairs = 0;

  // Short-range preconditioner CSR cache (allocated lazily at first NL rebuild)
  gpu->pbPGMShortPairOffset = NULL;
  gpu->pbPGMShortPairAtom   = NULL;
  gpu->pbPGMShortPairCache  = NULL;
  gpu->pbPGMShortPairCount  = NULL;
  gpu->pgmShortPairMaxPairs = 0;
  gpu->pgmShortPairMaxAtoms = 0;
  gpu->sim.pPGMShortPairOffset = NULL;
  gpu->sim.pPGMShortPairAtom   = NULL;
  gpu->sim.pPGMShortPairNB1    = NULL;
  gpu->sim.pPGMShortPairNB2    = NULL;
  gpu->sim.pPGMShortPairDX     = NULL;
  gpu->sim.pPGMShortPairDY     = NULL;
  gpu->sim.pPGMShortPairDZ     = NULL;
  gpu->sim.pPGMShortPairCount  = NULL;
  gpu->sim.pgmShortPairTotalPairs = 0;

  // Dipole history for multi-step LS extrapolation (initial guess predictor)
  // 6 history points → LS quadratic with noise amplification 3.6 (vs 5.0 for 3 points)
  for (int i = 0; i < 8; i++) {
    gpu->pbPGMDipHistory[i] = new GpuBuffer<double>(cg_size);
    gpu->pgm_dip_hist_dev[i] = gpu->pbPGMDipHistory[i]->_pDevData;
  }
  gpu->pgm_dip_history_count = 0;

  // Old image lookup for remapping dipole history across NL rebuilds
  gpu->pbPGMOldImageLookup = new GpuBuffer<unsigned int>(stride);

  // PCG solver defaults (overridden by pgm_gpu_set_pcg_params_)
  gpu->pgm_scf_solv_opt    = 1;    // standard CG by default
  gpu->pgm_scf_local_niter = 3;
  gpu->pgm_scf_local_cut   = 3.0;
  gpu->pgm_scf_sor_coeff   = 0.65;
  gpu->sim.scfLocalCut2    = (PMEFloat)(3.0 * 3.0);
  gpu->sim.pgmIGVDW        = 0;
  gpu->sim.pgmGVDWRepForm  = 0;
  gpu->sim.pgmBRepScale    = (PMEFloat)1.0;

  // CUDA graph for CG iteration (eliminates kernel launch overhead)
  gpu->pgmStream = 0;
  cudaStreamCreate(&gpu->pgmCaptureStream);
  gpu->pgmCGGraph = NULL;
  gpu->pgmCGGraphExec = NULL;
  gpu->pgmPermGraph = NULL;
  gpu->pgmPermGraphExec = NULL;
  gpu->pgmForceGraph = NULL;
  gpu->pgmForceGraphExec = NULL;
  gpu->pgmGraphsValid = false;

  // Bond CSR arrays allocated later in pgm_gpu_upload_bonds_
  gpu->pbPGMCovalentPtr = NULL;
  gpu->pbPGMCovalentAtm = NULL;
  gpu->pbPGMCovalentDip = NULL;

}


//---------------------------------------------------------------------------------------------
// pgm_gpu_set_params_: store Ewald and SCF parameters
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_set_params_(double* ew_coeff, double* tol, int* max_iter,
                                     int* solver, int* precond)
{
  gpu->pgm_scf_tol     = *tol;
  gpu->pgm_scf_maxiter = *max_iter;
  gpu->pgm_scf_solv_opt    = *solver;
  gpu->pgm_scf_local_niter = *precond;
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_set_options_: GPU solver and van der Waals options from the &pol_gauss namelist
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_set_options_(int* pred, int* check, int* overlap, int* power,
                                     int* igvdw, int* rep_form, double* b_rep_scale)
{
  pgm_opt_pred    = *pred;
  pgm_opt_check   = *check;
  pgm_opt_overlap = *overlap;
  pgm_opt_power   = *power;
  gpu->sim.pgmIGVDW       = *igvdw;
  gpu->sim.pgmGVDWRepForm = *rep_form;
  gpu->sim.pgmBRepScale   = (PMEFloat)(*b_rep_scale);
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_set_pcg_params_: store PCG-specific parameters from Fortran
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_set_pcg_params_(int* solv_opt, int* local_niter,
                                          double* local_cut, double* sor_coeff)
{
  gpu->pgm_scf_solv_opt    = *solv_opt;
  gpu->pgm_scf_local_niter = *local_niter;
  gpu->pgm_scf_local_cut   = *local_cut;
  gpu->pgm_scf_sor_coeff   = *sor_coeff;
  gpu->sim.scfLocalCut2    = (PMEFloat)((*local_cut) * (*local_cut));
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_upload_multipoles_: upload permanent charges, r_gauss, polarizability
//
// Called once from pGM_gpu_setup after reading prmtop.
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_upload_multipoles_(double* charge, double* rGauss, double* pol)
{
  int atoms = gpu->sim.atoms;

  // Copy into host-side buffer, then upload to device
  memcpy(gpu->pbPGMPermChg->_pSysData, charge, atoms * sizeof(double));
  gpu->pbPGMPermChg->Upload();

  memcpy(gpu->pbPGMRGauss->_pSysData, rGauss, atoms * sizeof(double));
  gpu->pbPGMRGauss->Upload();

  memcpy(gpu->pbPGMPolarizability->_pSysData, pol, atoms * sizeof(double));
  gpu->pbPGMPolarizability->Upload();
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_upload_bonds_: upload covalent bond CSR topology for local-to-global conversion
//
// Called once from pGM_gpu_setup.
// bondPtr: CSR row pointers [natom+1], 1-based Fortran indexing
// bondAtom: bonded atom indices [numBonds], 1-based Fortran indexing
// bondDip: bond dipole magnitudes [numBonds]
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_upload_bonds_(int* bondPtr, int* bondAtom,
                                       double* bondDip, int* numBonds)
{
  int atoms = gpu->sim.atoms;
  int nbonds = *numBonds;

  gpu->pgmNumCovalent = nbonds;
  gpu->sim.pgmNumCovalent = nbonds;

  // Allocate CSR buffers
  gpu->pbPGMCovalentPtr = new GpuBuffer<int>(atoms + 1);
  gpu->pbPGMCovalentAtm = new GpuBuffer<int>(nbonds);
  gpu->pbPGMCovalentDip = new GpuBuffer<double>(nbonds);

  // CSR row pointers are already 0-based from Fortran (covalent_ptr(1) = 0)
  memcpy(gpu->pbPGMCovalentPtr->_pSysData, bondPtr, (atoms + 1) * sizeof(int));
  gpu->pbPGMCovalentPtr->Upload();

  // Convert Fortran 1-based atom indices to C 0-based
  for (int j = 0; j < nbonds; j++) {
    gpu->pbPGMCovalentAtm->_pSysData[j] = bondAtom[j] - 1;
  }
  gpu->pbPGMCovalentAtm->Upload();

  memcpy(gpu->pbPGMCovalentDip->_pSysData, bondDip, nbonds * sizeof(double));
  gpu->pbPGMCovalentDip->Upload();

  // Wire into cSim
  gpu->sim.pPGMCovalentPtr = gpu->pbPGMCovalentPtr->_pDevData;
  gpu->sim.pPGMCovalentAtm = gpu->pbPGMCovalentAtm->_pDevData;
  gpu->sim.pPGMCovalentDip = gpu->pbPGMCovalentDip->_pDevData;
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_upload_prmtop_: store PGM prmtop data in host memory for NL-rebuild re-upload
//
// Called ONCE from Fortran init. Stores original-atom-order copies of charges, r_gauss,
// polarizability so that pgm_gpu_pme_ene_/force_ can re-upload with NL remap when needed
// without any Fortran involvement.
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_upload_prmtop_(double* perm_chg, double* r_gauss,
    double* pol, double* scf_tol, int* scf_maxiter, int* natom)
{
  int atoms = *natom;

  // Allocate host copies
  gpu->pgm_host_perm_chg = new double[atoms];
  gpu->pgm_host_r_gauss  = new double[atoms];
  gpu->pgm_host_pol      = new double[atoms];

  memcpy(gpu->pgm_host_perm_chg, perm_chg, atoms * sizeof(double));
  memcpy(gpu->pgm_host_r_gauss,  r_gauss,  atoms * sizeof(double));
  memcpy(gpu->pgm_host_pol,      pol,      atoms * sizeof(double));

  // Upload original-order data to GPU (for GPU-side NL remap scatter)
  memcpy(gpu->pbPGMOrigChg->_pSysData, perm_chg, atoms * sizeof(double));
  gpu->pbPGMOrigChg->Upload();
  memcpy(gpu->pbPGMOrigRGauss->_pSysData, r_gauss, atoms * sizeof(double));
  gpu->pbPGMOrigRGauss->Upload();
  memcpy(gpu->pbPGMOrigPol->_pSysData, pol, atoms * sizeof(double));
  gpu->pbPGMOrigPol->Upload();

  gpu->pgm_scf_tol     = *scf_tol;
  gpu->pgm_scf_maxiter = *scf_maxiter;
  gpu->pgm_diprms      = 0.0;
  gpu->pgm_dipiter     = 0.0;
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_get_scf_info_: retrieve SCF convergence info after pgm_gpu_pme_ene_
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_get_scf_info_(double* diprms, double* dipiter)
{
  *diprms  = gpu->pgm_diprms;
  *dipiter = gpu->pgm_dipiter;
}


// Forward declarations for SCF kernel dispatchers
extern "C" void kPGMInitSCF(gpuContext gpu);
extern "C" void kPGMCopyDipToBuffers(gpuContext gpu, double* src_dev);
extern "C" void kPGMComputeAp(gpuContext gpu, double* src_dev, double* dst_dev);
extern "C" void kPGMInitResidual(gpuContext gpu);
extern "C" void kPGMDotPZ(gpuContext gpu);
extern "C" void kPGMCGUpdate(gpuContext gpu);
extern "C" void kPGMCGUpdateP(gpuContext gpu);
extern "C" void kPGMStoreInducedDipoles(gpuContext gpu);
extern "C" void kPGMComputeResidual(gpuContext gpu);
extern "C" void kPGMFusedCopyDipClearSelf(gpuContext gpu, double* src_dev);
extern "C" void kPGMFusedComputeApDotPZ(gpuContext gpu, double* src_dev, double* dst_dev);
extern void SetkSCF_PGMSim(gpuContext gpu);

// PCG forward declarations
extern "C" void kPGMPrecondInit(gpuContext gpu);
extern "C" void kPGMPrecondCopyDipClear(gpuContext gpu, double* src_dev);
extern "C" void kPGMPrecondFusedApDotPZ(gpuContext gpu);
extern "C" void kPGMPrecondUpdate(gpuContext gpu);
extern "C" void kPGMPrecondUpdateP(gpuContext gpu);
extern "C" void kPGMPrecondPeek(gpuContext gpu);
extern "C" void kPGMPCGInitResidual(gpuContext gpu);
extern "C" void kPGMPCGSetPFromZpc(gpuContext gpu);
extern "C" void kPGMPCGUpdate(gpuContext gpu);
extern "C" void kPGMPCGDotRZpc(gpuContext gpu);
extern "C" void kPGMPCGUpdateP(gpuContext gpu);
extern "C" void kCalculatePMEDirectFieldShort_PGM(gpuContext gpu);
extern "C" void kPGMBuildShortRangeCSR(gpuContext gpu);
extern "C" void kPGMShortRangeSpMV(gpuContext gpu, double* src_dev);
extern void SetkPrecondCache_PGMSim(gpuContext gpu);
// Helper: run the FULL matvec pipeline (self + direct + reciprocal with FFT)
//
// DPFP: uses BD cache (CacheRead) — compute-bound, cache saves expensive erfc/exp.
// SPFP: recomputes from positions — bandwidth-bound, fasterfc is cheap, avoiding
//       240+ MB cache reads per iteration is a 3-4x win at large system sizes.
// Reciprocal chain (clear grid, fill, reduce, FFT, scalar sum, IFFT, gradient sum) on stream st.
static void pgm_gpu_run_recip(gpuContext gpu, cudaStream_t st, bool force_mode = false)
{
  cudaStream_t saved = gpu->pgmStream; gpu->pgmStream = st;
  cufftSetStream(gpu->forwardPlan, st); cufftSetStream(gpu->backwardPlan, st);
  kPMEClearChargeGridBuffer_s(gpu, st);
  kPMEFillChargeGridBuffer_PGM(gpu);
  kPMEReduceChargeGridBuffer_s(gpu, st);
#ifdef use_DPFP
  cufftExecD2Z(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#else
  cufftExecR2C(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#endif
  double vol = (double)gpu->sim.uc_volume;
  kPMEScalarSumRC_s(gpu, vol, st);
#ifdef use_DPFP
  cufftExecZ2D(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#else
  cufftExecC2R(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#endif
  if (force_mode) kPMEGradSumForce_PGM(gpu); else kPMEGradSum_PGM(gpu);
  gpu->pgmStream = saved;
  cufftSetStream(gpu->forwardPlan, saved); cufftSetStream(gpu->backwardPlan, saved);
}
// Overlapped matvec body: after ClearAndSelf on stream s, fork the reciprocal chain to s2 and run the
// direct kernel on s concurrently; both accumulate atomically into the field accumulators; join.
static void pgm_gpu_run_direct_recip_overlapped(gpuContext gpu, bool cache_read)
{
  cudaStream_t s = gpu->pgmStream;
  pgm_overlap_init();
  cudaEventRecord(pgm_ev_fork, s);
  cudaStreamWaitEvent(pgm_s2, pgm_ev_fork, 0);
  pgm_gpu_run_recip(gpu, pgm_s2);
  cudaEventRecord(pgm_ev_join, pgm_s2);
#ifdef use_DPFP
  if (cache_read) kCalculatePMEDirectFieldCacheRead_PGM(gpu); else kCalculatePMEDirectFieldCacheWrite_PGM(gpu);
#else
  if (cache_read && pgm_cache16) kCalculatePMEDirectFieldCacheReadH_PGM(gpu);
  else if (!cache_read && pgm_cache16) kCalculatePMEDirectFieldCacheWriteH_PGM(gpu);
  else kCalculatePMEDirectField_PGM(gpu);
#endif
  cudaStreamWaitEvent(s, pgm_ev_join, 0);
}
static void pgm_gpu_run_matvec(gpuContext gpu)
{
  cudaStream_t s = gpu->pgmStream;

  kPGMClearAndSelfField(gpu);
  if (pgm_overlap && s != 0) { pgm_gpu_run_direct_recip_overlapped(gpu, pgm_in_cg_loop); return; }

#ifdef use_DPFP
  kCalculatePMEDirectFieldCacheRead_PGM(gpu);
#else
  if (pgm_cache16 && pgm_in_cg_loop) kCalculatePMEDirectFieldCacheReadH_PGM(gpu);
  else                               kCalculatePMEDirectField_PGM(gpu);
#endif

  kPMEClearChargeGridBuffer_s(gpu, s);
  kPMEFillChargeGridBuffer_PGM(gpu);
  kPMEReduceChargeGridBuffer_s(gpu, s);
#ifdef use_DPFP
  cufftExecD2Z(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#else
  cufftExecR2C(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#endif
  double vol = (double)gpu->sim.uc_volume;
  kPMEScalarSumRC_s(gpu, vol, s);
#ifdef use_DPFP
  cufftExecZ2D(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#else
  cufftExecC2R(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#endif
  kPMEGradSum_PGM(gpu);
}

// Helper: direct-space only matvec (self + direct, NO reciprocal space).
//
// Used for CG iterations during SCF.  The key insight: during SCF, the CG
// direction vector p is a small correction.  The reciprocal-space contribution
// to T·p is smooth and small (long-wavelength modes).  By using only the
// direct-space operator for CG iterations, we eliminate 7 of 9 operations
// per matvec (ClearGrid, FillGrid, ReduceGrid, FFT, ScalarSum, IFFT, GradSum).
//
// The RHS b already includes the FULL permanent field (with reciprocal), so
// the long-range effect enters through b.  The direct-only operator captures
// the dominant short-range dipole-dipole interaction for CG convergence.
//
// This saves 3-4 FFT round-trips per step, which is ~10% of total step time
// for large systems.
static void pgm_gpu_run_matvec_direct_only(gpuContext gpu)
{
  kPGMClearAndSelfField(gpu);

#ifdef use_DPFP
  kCalculatePMEDirectFieldCacheRead_PGM(gpu);
#else
  kCalculatePMEDirectField_PGM(gpu);
#endif
  // NO reciprocal space — skip ClearGrid, FillGrid, ReduceGrid, FFT,
  // ScalarSum, IFFT, GradSum.  Saves ~150-300 µs per CG iteration on 32K.
}

// Helper: matvec without ClearAndSelf — used with kPGMFusedCopyDipClearSelf
// which combines dipole copy + accumulator clear + self-field in a single kernel.
// Saves one kernel launch and one memory pass per CG iteration.
static void pgm_gpu_run_matvec_no_clear(gpuContext gpu)
{
  cudaStream_t s = gpu->pgmStream;

#ifdef use_DPFP
  kCalculatePMEDirectFieldCacheRead_PGM(gpu);
#else
  kCalculatePMEDirectField_PGM(gpu);
#endif

  kPMEClearChargeGridBuffer_s(gpu, s);
  kPMEFillChargeGridBuffer_PGM(gpu);
  kPMEReduceChargeGridBuffer_s(gpu, s);
#ifdef use_DPFP
  cufftExecD2Z(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#else
  cufftExecR2C(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#endif
  double vol = (double)gpu->sim.uc_volume;
  kPMEScalarSumRC_s(gpu, vol, s);
#ifdef use_DPFP
  cufftExecZ2D(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#else
  cufftExecC2R(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#endif
  kPMEGradSum_PGM(gpu);
}

// Helper: permanent field (self + direct[CacheWrite] + reciprocal)
// Like matvec but uses CacheWrite (DPFP) to populate BD cache during perm field.
static void pgm_gpu_run_perm_field(gpuContext gpu)
{
  cudaStream_t s = gpu->pgmStream;

  kPGMClearAndSelfField(gpu);
  if (pgm_overlap && s != 0) { pgm_gpu_run_direct_recip_overlapped(gpu, false); return; }

#ifdef use_DPFP
  kCalculatePMEDirectFieldCacheWrite_PGM(gpu);
#else
  if (pgm_cache16) kCalculatePMEDirectFieldCacheWriteH_PGM(gpu);
  else             kCalculatePMEDirectField_PGM(gpu);
#endif

  kPMEClearChargeGridBuffer_s(gpu, s);
  kPMEFillChargeGridBuffer_PGM(gpu);
  kPMEReduceChargeGridBuffer_s(gpu, s);
#ifdef use_DPFP
  cufftExecD2Z(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#else
  cufftExecR2C(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#endif
  double vol = (double)gpu->sim.uc_volume;
  kPMEScalarSumRC_s(gpu, vol, s);
#ifdef use_DPFP
  cufftExecZ2D(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#else
  cufftExecC2R(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#endif
  kPMEGradSum_PGM(gpu);
}

// Helper: force pipeline (self + direct_force + recip_force + chain_rule + energy + scale)
// Everything except kExecuteBondWorkUnits (which varies by need_energy).
static void pgm_gpu_run_force_field(gpuContext gpu, double coulomb_const)
{
  cudaStream_t s = gpu->pgmStream;

  kPGMClearAndSelfField(gpu);
  if (pgm_overlap && s != 0) {
    pgm_overlap_init();
    cudaEventRecord(pgm_ev_fork, s);
    cudaStreamWaitEvent(pgm_s2, pgm_ev_fork, 0);
    pgm_gpu_run_recip(gpu, pgm_s2, true);
    cudaEventRecord(pgm_ev_join, pgm_s2);
    kCalculatePMEDirectForce_PGM(gpu);
    cudaStreamWaitEvent(s, pgm_ev_join, 0);
    kPGMChainRuleForce(gpu);
    kPGMComputeEnergy(gpu, coulomb_const);
    kPGMScaleForces(gpu, coulomb_const);
    return;
  }
  kCalculatePMEDirectForce_PGM(gpu);

  kPMEClearChargeGridBuffer_s(gpu, s);
  kPMEFillChargeGridBuffer_PGM(gpu);
  kPMEReduceChargeGridBuffer_s(gpu, s);
#ifdef use_DPFP
  cufftExecD2Z(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#else
  cufftExecR2C(gpu->forwardPlan, gpu->sim.pXYZ_q, gpu->sim.pXYZ_qt);
#endif
  double vol = (double)gpu->sim.uc_volume;
  kPMEScalarSumRC_s(gpu, vol, s);
#ifdef use_DPFP
  cufftExecZ2D(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#else
  cufftExecC2R(gpu->backwardPlan, gpu->sim.pXYZ_qt, gpu->sim.pXYZ_q);
#endif
  kPMEGradSumForce_PGM(gpu);
  kPGMChainRuleForce(gpu);
  kPGMComputeEnergy(gpu, coulomb_const);
  kPGMScaleForces(gpu, coulomb_const);
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_run_short_matvec: short-range direct-space field only (no self, no reciprocal)
//
// Used by the local preconditioner inner CG.  Dipoles must already be in
// SP dipole buffers and field accumulators must already be cleared.
//---------------------------------------------------------------------------------------------
static void pgm_gpu_run_short_matvec(gpuContext gpu)
{
  kCalculatePMEDirectFieldShort_PGM(gpu);
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_local_precondition: GPU implementation of local_precondition()
//
// Solves (1 + alpha*T_short)*zpc = r using niter inner CG iterations
// with zero initial guess, followed by an SOR peek step.
// Input:  outer residual r in pPGMCGr
// Output: preconditioned residual in pPGMPCGzpc
//---------------------------------------------------------------------------------------------
static void pgm_gpu_local_precondition(gpuContext gpu)
{
  int niter = gpu->pgm_scf_local_niter;

  // Initialize: xu=0, ru=r, pu=r, rdotr=dot(r,r)
  kPGMPrecondInit(gpu);

  // Inner CG iterations
  for (int i = 0; i < niter; i++) {
    // Phase 2: SpMV from pre-computed short-range CSR cache
    // Replaces kPGMPrecondCopyDipClear + pgm_gpu_run_short_matvec
    kPGMShortRangeSpMV(gpu, gpu->sim.pPGMPCGpu);

    // zu = pu + alpha*field; pdotz = dot(pu, zu); alpha_cg = rdotr/pdotz
    kPGMPrecondFusedApDotPZ(gpu);

    // xu += alpha_cg*pu; ru -= alpha_cg*zu; rdotr_new; beta
    kPGMPrecondUpdate(gpu);

    // pu = ru + beta*pu (skip on last iteration)
    if (i < niter - 1) {
      kPGMPrecondUpdateP(gpu);
    }
  }

  // Peek step: xu += ru * alpha_i
  kPGMPrecondPeek(gpu);
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_save_dipole_state_ / pgm_gpu_restore_dipole_state_
//
// Save and restore the induced-dipole SCF state around MC barostat trials.
// The PGM pipeline updates dipole history (pPGMCGx and pgm_dip_hist_dev[0..2])
// every time it runs.  When a barostat trial is rejected, the history would
// contain dipoles converged at the trial geometry, contaminating the
// extrapolated initial guess for the next real step.
//
// Usage (from Fortran, inside #ifdef PGM / #ifdef CUDA guards):
//   call pgm_gpu_save_dipole_state()     -- before mcbar_trial pme_force call
//   call pgm_gpu_restore_dipole_state()  -- on rejection, before gpu_upload_frc
//---------------------------------------------------------------------------------------------

// Static save buffers (allocated on first use)
static GpuBuffer<double>* pgm_saved_CGx = NULL;
static GpuBuffer<double>* pgm_saved_hist[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
static double* pgm_saved_hist_ptrs[8];
static int     pgm_saved_history_count = 0;
static GpuBuffer<unsigned int>* pgm_saved_lookup = NULL;

static bool pgm_pred_done = false;
static void pgm_run_predictor(gpuContext gpu, bool needColdStart, int stride)
{
  pgm_pred_init();
  if (pgm_pred_mode > 0 && !pgm_b_prev) pgm_b_prev = new GpuBuffer<double>(3 * stride);
  if (pgm_pred_mode == 8) {
    if (pgm_fused) { fprintf(stderr, "FATAL: PGM_GPU_PRED=cpuls requires PGM_FUSED=0\n"); exit(EXIT_FAILURE); }
    if (cpuls_device) cpulsd_predict(gpu, stride); else cpuls_predict(gpu, stride);
    return;
  }
  if (needColdStart) {
    // Genuine cold start (first step, no history)
    cudaMemcpyAsync(gpu->sim.pPGMCGx, gpu->sim.pPGMCGb,
               3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
    pgm_b_prev_valid = 0;
  } else if (pgm_pred_mode == 7) {
    // ---- XL: x0 = auxiliary dipoles n_k (H[0]); on the first warm step seed the history with the converged x ----
    if (gpu->pgm_dip_history_count == 0) {
      for (int h = 0; h < 6; h++)
        cudaMemcpyAsync(gpu->pgm_dip_hist_dev[h], gpu->sim.pPGMCGx, 3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
      gpu->pgm_dip_history_count = 6;
    }
    cudaMemcpyAsync(gpu->sim.pPGMCGx, gpu->pgm_dip_hist_dev[0], 3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
  } else if (pgm_pred_mode == 4 || pgm_pred_mode == 5) {
    // ---- polynomial extrapolation of mu with K=4 (cubic) or K=5 (quartic); no field dependence ----
    int hmax = pgm_pred_hist_max();
    double* oldest = gpu->pgm_dip_hist_dev[hmax - 1];
    for (int h = hmax - 1; h > 0; h--) gpu->pgm_dip_hist_dev[h] = gpu->pgm_dip_hist_dev[h - 1];
    gpu->pgm_dip_hist_dev[0] = oldest;
    cudaMemcpyAsync(gpu->pgm_dip_hist_dev[0], gpu->sim.pPGMCGx,
               3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
    if (gpu->pgm_dip_history_count < hmax) gpu->pgm_dip_history_count++;
    int K = gpu->pgm_dip_history_count;
    double** H = gpu->pgm_dip_hist_dev;
    if (K >= 5 && hmax == 5)      kPGMExtrapolate5(gpu, gpu->sim.pPGMCGx, NULL, H[0], H[1], H[2], H[3], H[4]);
    else if (K >= 4)              kPGMExtrapolateBase(gpu, gpu->sim.pPGMCGx, NULL, H[0], H[1], H[2], H[3], 4, 1);
    else if (K >= 3)              kPGMExtrapolateBase(gpu, gpu->sim.pPGMCGx, NULL, H[0], H[1], H[2], NULL, 3, 0);
    else if (K >= 2)              kPGMExtrapolateBase(gpu, gpu->sim.pPGMCGx, NULL, H[0], H[1], NULL, NULL, 2, 0);
  } else if (pgm_pred_mode == 0) {
    // ---- legacy: quadratic extrapolation of mu ----
    double* oldest = gpu->pgm_dip_hist_dev[2];
    gpu->pgm_dip_hist_dev[2] = gpu->pgm_dip_hist_dev[1];
    gpu->pgm_dip_hist_dev[1] = gpu->pgm_dip_hist_dev[0];
    gpu->pgm_dip_hist_dev[0] = oldest;
    cudaMemcpyAsync(gpu->pgm_dip_hist_dev[0], gpu->sim.pPGMCGx,
               3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
    if (gpu->pgm_dip_history_count < 3) gpu->pgm_dip_history_count++;
    int K = gpu->pgm_dip_history_count;
    if (K >= 3) {
      kPGMExtrapolateLS(gpu, gpu->sim.pPGMCGx,
                        gpu->pgm_dip_hist_dev[0], gpu->pgm_dip_hist_dev[1],
                        gpu->pgm_dip_hist_dev[2], NULL, NULL, NULL, 3);
    } else if (K >= 2) {
      kPGMExtrapolateLS(gpu, gpu->sim.pPGMCGx,
                        gpu->pgm_dip_hist_dev[0], gpu->pgm_dip_hist_dev[1],
                        NULL, NULL, NULL, NULL, 2);
    }
  } else {
    // ---- field-anchored predictor: history holds Delta = mu - b; x0 = b_n + extrap(Delta) ----
    int hmax = pgm_pred_hist_max();
    if (pgm_b_prev_valid) {
      double* oldest = gpu->pgm_dip_hist_dev[hmax - 1];
      for (int h = hmax - 1; h > 0; h--) gpu->pgm_dip_hist_dev[h] = gpu->pgm_dip_hist_dev[h - 1];
      gpu->pgm_dip_hist_dev[0] = oldest;
      kPGMVecSub(gpu, gpu->pgm_dip_hist_dev[0], gpu->sim.pPGMCGx, pgm_b_prev->_pDevData);
      if (gpu->pgm_dip_history_count < hmax) gpu->pgm_dip_history_count++;
    }
    int K = gpu->pgm_dip_history_count;
    double** H = gpu->pgm_dip_hist_dev;
    if (K >= 8 && pgm_pred_mode == 6) {
      kPGMExtrapolate8(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGb, H);
    } else if (K >= 4 && pgm_pred_mode >= 2) {
      kPGMExtrapolateBase(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGb, H[0], H[1], H[2], H[3], 4,
                          (pgm_pred_mode == 2 || pgm_pred_mode == 6) ? 1 : 0);        // cubic until 8 points exist
    } else if (K >= 3) {
      kPGMExtrapolateBase(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGb, H[0], H[1], H[2], NULL, 3, 0);
    } else if (K >= 2) {
      kPGMExtrapolateBase(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGb, H[0], H[1], NULL, NULL, 2, 0);
    } else if (K >= 1) {
      kPGMExtrapolateBase(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGb, H[0], NULL, NULL, NULL, 1, 0);
    }
    // K == 0 (first step after a cold start): keep x = previous converged x (warm start)
  }
  if (pgm_pred_mode > 0) {
    cudaMemcpyAsync(pgm_b_prev->_pDevData, gpu->sim.pPGMCGb,
               3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
    pgm_b_prev_valid = 1;
  }
}

extern "C" void pgm_gpu_save_dipole_state_()
{
  int size = 3 * gpu->sim.stride;

  // Lazy allocation
  if (!pgm_saved_CGx) {
    pgm_saved_CGx = new GpuBuffer<double>(size);
    pgm_saved_lookup = new GpuBuffer<unsigned int>(gpu->sim.atoms);
    for (int i = 0; i < 8; i++)
      pgm_saved_hist[i] = new GpuBuffer<double>(size);
  }

  cudaMemcpy(pgm_saved_lookup->_pDevData, gpu->sim.pImageAtomLookup,
             gpu->sim.atoms * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

  // Save current converged dipoles (CGx)
  cudaMemcpy(pgm_saved_CGx->_pDevData, gpu->sim.pPGMCGx,
             size * sizeof(double), cudaMemcpyDeviceToDevice);

  // Save history buffer data and pointer order (FIFO rotation swaps pointers)
  for (int i = 0; i < 8; i++) {
    pgm_saved_hist_ptrs[i] = gpu->pgm_dip_hist_dev[i];
    cudaMemcpy(pgm_saved_hist[i]->_pDevData, gpu->pgm_dip_hist_dev[i],
               size * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  pgm_saved_history_count = gpu->pgm_dip_history_count;
  if (pgm_pred_mode == 8) cpuls_save_state(gpu);
}

extern "C" void pgm_gpu_restore_dipole_state_()
{
  if (!pgm_saved_CGx) return;  // save was never called

  int size = 3 * gpu->sim.stride;

  // Restore converged dipoles
  cudaMemcpy(gpu->sim.pPGMCGx, pgm_saved_CGx->_pDevData,
             size * sizeof(double), cudaMemcpyDeviceToDevice);

  kPGMRemapDipHistory(gpu, gpu->sim.pPGMCGx, pgm_saved_CGx->_pDevData,
                     pgm_saved_lookup->_pDevData);

  // Restore history pointer order and data
  for (int i = 0; i < 8; i++) {
    gpu->pgm_dip_hist_dev[i] = pgm_saved_hist_ptrs[i];
    cudaMemcpy(gpu->pgm_dip_hist_dev[i], pgm_saved_hist[i]->_pDevData,
               size * sizeof(double), cudaMemcpyDeviceToDevice);
    kPGMRemapDipHistory(gpu, gpu->pgm_dip_hist_dev[i], pgm_saved_hist[i]->_pDevData,
                       pgm_saved_lookup->_pDevData);
  }

  gpu->pgm_dip_history_count = pgm_saved_history_count;
  if (pgm_pred_mode == 8) cpuls_restore_state(gpu, pgm_saved_lookup->_pDevData);
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_cleanup_: free PGM-specific GPU buffers
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_cleanup_()
{
  delete gpu->pbPGMPermChg;
  delete gpu->pbPGMRGauss;
  delete gpu->pbPGMPolarizability;
  delete gpu->pbPGMPrecondInv;

  delete gpu->pbPGMCovalentPtr;
  delete gpu->pbPGMCovalentAtm;
  delete gpu->pbPGMCovalentDip;

  delete gpu->pbPermDipX;
  delete gpu->pbPermDipY;
  delete gpu->pbPermDipZ;
  delete gpu->pbPDipXYSP;
  delete gpu->pbPDipZSP;

  delete gpu->pbPGMFracDipX;
  delete gpu->pbPGMFracDipY;
  delete gpu->pbPGMFracDipZ;

  delete gpu->pbIndDipX;
  delete gpu->pbIndDipY;
  delete gpu->pbIndDipZ;
  delete gpu->pbIDipXYSP;
  delete gpu->pbIDipZSP;

  delete gpu->pbPGMFieldXAccumulator;
  delete gpu->pbPGMFieldYAccumulator;
  delete gpu->pbPGMFieldZAccumulator;
  delete gpu->pbPGMPotentialAccumulator;

  delete gpu->pbTotDipX;
  delete gpu->pbTotDipY;
  delete gpu->pbTotDipZ;
  delete gpu->pbTDipXYSP;
  delete gpu->pbTDipZSP;
  delete gpu->pbPGMFracTotDipX;
  delete gpu->pbPGMFracTotDipY;
  delete gpu->pbPGMFracTotDipZ;

  // CG solver
  delete gpu->pbPGMCGx;
  delete gpu->pbPGMCGr;
  delete gpu->pbPGMCGp;
  delete gpu->pbPGMCGz;
  delete gpu->pbPGMCGb;
  delete gpu->pbPGMCGScalars;
  delete gpu->pbPGMSavedPermChg;
  delete gpu->pbPGMSavedAtomChgSP;
  delete gpu->pbPGMSavedPermDipX;
  delete gpu->pbPGMSavedPermDipY;
  delete gpu->pbPGMSavedPermDipZ;
  delete gpu->pbPGMCGPartials;

  // T2 tensor cache
  delete gpu->pbPGMT2Cache;
  delete gpu->pbPGMBFactorOffset;

  // Short-range preconditioner CSR cache
  delete gpu->pbPGMShortPairOffset;
  delete gpu->pbPGMShortPairAtom;
  delete gpu->pbPGMShortPairCache;
  delete gpu->pbPGMShortPairCount;

  // Dipole history + old lookup
  for (int i = 0; i < 8; i++) delete gpu->pbPGMDipHistory[i];
  delete gpu->pbPGMOldImageLookup;

  // MC barostat dipole save buffers
  delete pgm_saved_CGx;  pgm_saved_CGx = NULL;
  delete pgm_saved_lookup; pgm_saved_lookup = NULL;
  for (int i = 0; i < 8; i++) { delete pgm_saved_hist[i]; pgm_saved_hist[i] = NULL; }

  // CUDA graph cleanup
  if (gpu->pgmCGGraphExec) cudaGraphExecDestroy(gpu->pgmCGGraphExec);
  if (gpu->pgmCGGraph) cudaGraphDestroy(gpu->pgmCGGraph);
  if (gpu->pgmPermGraphExec) cudaGraphExecDestroy(gpu->pgmPermGraphExec);
  if (gpu->pgmPermGraph) cudaGraphDestroy(gpu->pgmPermGraph);
  if (gpu->pgmForceGraphExec) cudaGraphExecDestroy(gpu->pgmForceGraphExec);
  if (gpu->pgmForceGraph) cudaGraphDestroy(gpu->pgmForceGraph);
  cudaStreamDestroy(gpu->pgmCaptureStream);

  gpu->ipgm = 0;
  gpu->sim.ipgm = 0;

}


//=====================================================================================
//
// pgm_gpu_pme_ene_ / pgm_gpu_pme_force_
//
// Restructured PGM GPU entry points matching the standard gpu_pme_ene_/gpu_pme_force_
// pattern.  ONE C++ function call from Fortran does everything:
//   NL rebuild → clear forces → grid weights → PGM data re-upload → dipoles →
//   perm field → SCF → force → energy → scale → bonded → energy download
//
// Standard gpu_download_frc_ (called from runmd.F90) handles force download.
//
//=====================================================================================

// PGM pipeline profiling infrastructure (compile with -DPGM_DEBUG to enable)
#ifdef PGM_DEBUG
static struct PGMProfile {
  bool initialized;
  int step_count;
  cudaEvent_t e[20];  // timing events
  double t_nl, t_clear, t_gridwt, t_remap;
  double t_dipconv;
  double t_perm_self, t_perm_direct, t_perm_recip;
  double t_scf_init, t_scf_matvec, t_scf_other;
  double t_force_direct, t_force_recip, t_force_chain;
  double t_force_other;
  double t_bonded;
  double t_total;
  int    total_cg_iters;
} pgm_prof = {false, 0};

static void pgm_profile_init() {
  if (!pgm_prof.initialized) {
    for (int i = 0; i < 20; i++) cudaEventCreate(&pgm_prof.e[i]);
    pgm_prof.initialized = true;
    pgm_prof.t_nl = pgm_prof.t_clear = pgm_prof.t_gridwt = pgm_prof.t_remap = 0;
    pgm_prof.t_dipconv = 0;
    pgm_prof.t_perm_self = pgm_prof.t_perm_direct = pgm_prof.t_perm_recip = 0;
    pgm_prof.t_scf_init = pgm_prof.t_scf_matvec = pgm_prof.t_scf_other = 0;
    pgm_prof.t_force_direct = pgm_prof.t_force_recip = pgm_prof.t_force_chain = 0;
    pgm_prof.t_force_other = pgm_prof.t_bonded = pgm_prof.t_total = 0;
    pgm_prof.total_cg_iters = 0;
  }
}

static float pgm_elapsed(int i, int j) {
  float ms = 0;
  cudaEventElapsedTime(&ms, pgm_prof.e[i], pgm_prof.e[j]);
  return ms;
}
#define PGM_EVENT_RECORD(idx) cudaEventRecord(pgm_prof.e[idx])
#else
#define PGM_EVENT_RECORD(idx)
#endif // PGM_DEBUG

// Internal helper: full PGM pipeline (shared by ene and force variants)
static void pgm_gpu_pme_pipeline(bool need_energy)
{
  int atoms = gpu->sim.atoms;
  int stride = gpu->sim.stride;
  double coulomb_const = PGM_COULOMB_CONST;

#ifdef PGM_DEBUG
  pgm_profile_init();
#endif

  PGM_EVENT_RECORD(0);  // pipeline start

  // Invalidate CUDA graphs if box volume changed (e.g., MC barostat trial).
  // CUDA graphs freeze kernel arguments by value at capture time, so a volume
  // change requires recapture.
  static double pgm_last_vol = 0.0;
  bool pgm_vol_changed = (gpu->sim.uc_volume != pgm_last_vol);
  if (pgm_vol_changed) {
    gpu->pgmGraphsValid = false;
    pgm_last_vol = gpu->sim.uc_volume;
  }

  static bool pgm_cSim_initialized = false;

  // Save current image lookup before NL build (for history remap if NL rebuilds)
  if (pgm_cSim_initialized) {
    cudaMemcpy(gpu->pbPGMOldImageLookup->_pDevData, gpu->sim.pImageAtomLookup,
               gpu->sim.atoms * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  }

  // ===== 1. Build neighbor list (only when needed) =====
  //
  // Detect fresh NL rebuilds using clear-and-restore on bNewNeighborList.
  // We must NOT permanently clear bNewNeighborList — the download/upload
  // functions (gpu_download_vel_, gpu_upload_frc_, gpu_download_frc_, etc.)
  // check it to re-download the image atom index from the GPU.  If they
  // miss the notification, forces get uploaded to wrong image slots on MC
  // barostat rejection (journal #23).
  //
  // But we DO need to detect each fresh NL rebuild so the PGM NL remap runs.
  // The previous approach (pgm_nl_processed static flag) failed when multiple
  // NL rebuilds occurred between MC trials (mcbarint > skinnb update interval):
  // bNewNeighborList was never cleared to reset the flag, so subsequent remaps
  // were skipped and PGM data stayed in stale image ordering (journal #24).
  //
  // Solution: temporarily clear bNewNeighborList before the NL build so we
  // can detect a fresh rebuild, then OR the result with the saved value to
  // preserve the notification for download/upload consumers.
  bool had_pending_nl = gpu->bNewNeighborList;
  gpu->bNewNeighborList = false;
  gpu_build_neighbor_list_();
  bool pgm_fused_active = false;
  bool needColdStart = gpu->bNewNeighborList;  // true iff NL rebuilt THIS call
  if (needColdStart) pgm_power_skip = true;
  gpu->bNewNeighborList = gpu->bNewNeighborList || had_pending_nl;
  PGM_EVENT_RECORD(1);

  // ===== 2. Clear forces + energy buffer =====
  kClearForces(gpu, gpu->sim.NLNonbondEnergyWarps);
  PGM_EVENT_RECORD(2);

  // ===== 3. Compute grid weights from current coordinates =====
  kPMEGetGridWeights(gpu);
  PGM_EVENT_RECORD(3);

  // ===== 4. NL-conditional PGM data remap (GPU-side) =====
  if (needColdStart) {

    // Invalidate all CUDA graphs — kernel parameters (cSim pointers) may change
    gpu->pgmGraphsValid = false;

    // GPU scatter: original→image order for charges, r_gauss, pol, preconditioner.
    // PGM kernel cSim copies are already up to date — gpuCopyConstants() (called
    // inside gpu_build_neighbor_list_) now includes all PGM SetSim calls.
    kPGMNLRemap(gpu);

#ifdef use_DPFP
    // DPFP: compute BD cache offsets and allocate buffer for CacheRead/CacheWrite
    {
      unsigned int nEntries;
      cudaMemcpy(&nEntries, gpu->sim.pNLEntries, sizeof(unsigned int),
                 cudaMemcpyDeviceToHost);

      // Prefix-sum kernel uses 256 threads × 256 chunks = 65536 max entries
      if (nEntries > 65536) {
        fprintf(stderr, "PGM ERROR: NL entries (%u) exceeds prefix-sum limit (65536).\n"
                "  Increase system cutoff or reduce box size.\n", nEntries);
        exit(1);
      }

      if ((int)(nEntries + 1) > gpu->pgmBFactorMaxEntries) {
        int allocEntries = (nEntries + 1) * 3;
        delete gpu->pbPGMBFactorOffset;
        gpu->pbPGMBFactorOffset = new GpuBuffer<int>(allocEntries);
        gpu->pgmBFactorMaxEntries = allocEntries;
        gpu->sim.pPGMBFactorOffset = gpu->pbPGMBFactorOffset->_pDevData;
      }

      SetkNLRemap_PGMSim(gpu);
      kPGMComputeT2Offsets(gpu);

      int totalPairs;
      cudaMemcpy(&totalPairs,
                 &gpu->pbPGMBFactorOffset->_pDevData[nEntries],
                 sizeof(int), cudaMemcpyDeviceToHost);

      if (totalPairs > gpu->pgmBFactorMaxPairs) {
        int allocSize = totalPairs * 3 / 2;
        delete gpu->pbPGMT2Cache;
        gpu->pbPGMT2Cache = new GpuBuffer<PMEFloat>(5 * allocSize);
        gpu->sim.pPGMBD_nB1 = gpu->pbPGMT2Cache->_pDevData + 0 * allocSize;
        gpu->sim.pPGMBD_nB2 = gpu->pbPGMT2Cache->_pDevData + 1 * allocSize;
        gpu->sim.pPGMBD_dx  = gpu->pbPGMT2Cache->_pDevData + 2 * allocSize;
        gpu->sim.pPGMBD_dy  = gpu->pbPGMT2Cache->_pDevData + 3 * allocSize;
        gpu->sim.pPGMBD_dz  = gpu->pbPGMT2Cache->_pDevData + 4 * allocSize;
        gpu->pgmBFactorMaxPairs = allocSize;
        gpu->sim.pgmBFactorTotalPairs = allocSize;
      }

      SetkCalculatePMENonbondEnergy_PGMSim(gpu);
    }
#endif  // use_DPFP (SPFP skips BD cache — recomputes from positions)
#ifndef use_DPFP
    pgm_pred_init();
    if (pgm_cache16) {
      unsigned int nEntries;
      cudaMemcpy(&nEntries, gpu->sim.pNLEntries, sizeof(unsigned int), cudaMemcpyDeviceToHost);
      if (nEntries > 65536) {
        fprintf(stderr, "PGM CACHE16: NL entries (%u) exceed prefix-sum limit; disabling cache16.\n", nEntries);
        pgm_cache16 = 0;
      } else {
        if ((int)(nEntries + 1) > gpu->pgmBFactorMaxEntries) {
          int allocEntries = (nEntries + 1) * 3;
          delete gpu->pbPGMBFactorOffset;
          gpu->pbPGMBFactorOffset = new GpuBuffer<int>(allocEntries);
          gpu->pgmBFactorMaxEntries = allocEntries;
          gpu->sim.pPGMBFactorOffset = gpu->pbPGMBFactorOffset->_pDevData;
        }
        SetkNLRemap_PGMSim(gpu);
        kPGMComputeT2Offsets(gpu);
        int totalPairs;
        cudaMemcpy(&totalPairs, &gpu->pbPGMBFactorOffset->_pDevData[nEntries], sizeof(int), cudaMemcpyDeviceToHost);
        if (totalPairs > gpu->pgmBFactorMaxPairsH) {
          int allocSize = totalPairs * 3 / 2;
          delete gpu->pbPGMT2CacheH;
          gpu->pbPGMT2CacheH = new GpuBuffer<unsigned int>(allocSize);
          gpu->sim.pPGMBD_h2 = gpu->pbPGMT2CacheH->_pDevData;
          gpu->pgmBFactorMaxPairsH = allocSize;
          gpu->sim.pgmBFactorTotalPairs = allocSize;
        }
        SetkCalculatePMENonbondEnergy_PGMSim(gpu);
      }
    }
#endif

    // Build short-range preconditioner CSR (Phase 2) if PCG is active
    if (gpu->pgm_scf_solv_opt >= 2) {
      // Ensure offset and count buffers are allocated
      if ((int)gpu->sim.atoms + 1 > gpu->pgmShortPairMaxAtoms) {
        int allocAtoms = (gpu->sim.atoms + 1) * 2;
        delete gpu->pbPGMShortPairOffset;
        delete gpu->pbPGMShortPairCount;
        gpu->pbPGMShortPairOffset = new GpuBuffer<int>(allocAtoms);
        gpu->pbPGMShortPairCount  = new GpuBuffer<int>(allocAtoms);
        gpu->pgmShortPairMaxAtoms = allocAtoms;
        gpu->sim.pPGMShortPairOffset = gpu->pbPGMShortPairOffset->_pDevData;
        gpu->sim.pPGMShortPairCount  = gpu->pbPGMShortPairCount->_pDevData;
      }

      // Update scfLocalCut2 before build
      gpu->sim.scfLocalCut2 = (PMEFloat)(gpu->pgm_scf_local_cut * gpu->pgm_scf_local_cut);

      SetkPrecondCache_PGMSim(gpu);  // push offset/count buffer pointers to cSim before build
      kPGMBuildShortRangeCSR(gpu);
      SetkPrecondCache_PGMSim(gpu);  // refresh after possible reallocation inside CSR build
    }

    pgm_cSim_initialized = true;
  }
  PGM_EVENT_RECORD(4);  // after NL remap

  // PGM kernel cSim copies are now updated by gpuCopyConstants(), which is
  // called from both gpu_pressure_scale_ (MC barostat volume changes) and
  // gpu_build_neighbor_list_ (NL rebuilds).  No separate volume-change
  // SetSim block is needed here.

  // ===== 5. Dipole conversion (every step) =====
  kLocalToGlobal_PGM(gpu);
  pgm_pred_init();
  if (pgm_fused) {
    // Fused permanent-field + initial-residual sweep: form x0 now (mu-only predictor) and add it
    // to the permanent dipoles, so the perm sweep returns gradphi(q,mu_perm) + T x0 in one pass.
    if (needColdStart && gpu->pgm_dip_history_count > 0) {
      unsigned int* old_lookup = gpu->pbPGMOldImageLookup->_pDevData;
      for (int h = 0; h < gpu->pgm_dip_history_count; h++) {
        cudaMemcpyAsync(gpu->sim.pPGMCGz, gpu->pgm_dip_hist_dev[h],
                   3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
        kPGMRemapDipHistory(gpu, gpu->pgm_dip_hist_dev[h], gpu->sim.pPGMCGz, old_lookup);
      }
      cudaMemcpyAsync(gpu->sim.pPGMCGz, gpu->sim.pPGMCGx,
                 3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
      kPGMRemapDipHistory(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGz, old_lookup);
      needColdStart = false;
    }
    if (!needColdStart && pgm_bnorm_frozen > 0.0) {
      pgm_run_predictor(gpu, needColdStart, stride);
      pgm_pred_done = true;
      kPGMAddScaledDip(gpu, gpu->sim.pPGMCGx, 1.0);
      pgm_fused_active = true;
    }
  }
  PGM_EVENT_RECORD(5);

  // ===== 6. Permanent field (self + direct + recip) =====
  // Captured as CUDA graph for near-zero launch overhead on replay steps.
  if (!gpu->pgmGraphsValid) {
    // Destroy old perm graph if exists
    if (gpu->pgmPermGraphExec) {
      cudaGraphExecDestroy(gpu->pgmPermGraphExec);
      gpu->pgmPermGraphExec = NULL;
    }
    if (gpu->pgmPermGraph) {
      cudaGraphDestroy(gpu->pgmPermGraph);
      gpu->pgmPermGraph = NULL;
    }

    // Capture perm field sequence
    gpu->pgmStream = gpu->pgmCaptureStream;
    cufftSetStream(gpu->forwardPlan, gpu->pgmCaptureStream);
    cufftSetStream(gpu->backwardPlan, gpu->pgmCaptureStream);

    CUDA_GRAPH_CHECK(cudaStreamBeginCapture(gpu->pgmCaptureStream, cudaStreamCaptureModeRelaxed));
    pgm_gpu_run_perm_field(gpu);
    CUDA_GRAPH_CHECK(cudaStreamEndCapture(gpu->pgmCaptureStream, &gpu->pgmPermGraph));
    CUDA_GRAPH_CHECK(cudaGraphInstantiate(&gpu->pgmPermGraphExec, gpu->pgmPermGraph, nullptr, nullptr, 0));

    gpu->pgmStream = 0;
    cufftSetStream(gpu->forwardPlan, 0);
    cufftSetStream(gpu->backwardPlan, 0);
  }
  CUDA_GRAPH_CHECK(cudaGraphLaunch(gpu->pgmPermGraphExec, 0));
  PGM_EVENT_RECORD(6);
  PGM_EVENT_RECORD(7);
  PGM_EVENT_RECORD(8);

  // ===== 7. SCF CG solver (GPU-resident) =====
  double tol = gpu->pgm_scf_tol;
  int maxiter = gpu->pgm_scf_maxiter;

  // 7a. Initialize: compute b = -alpha * E_perm
  kPGMInitSCF(gpu);
  if (pgm_fused_active) {
    kPGMAddScaledDip(gpu, gpu->sim.pPGMCGx, -1.0);          // restore mu_perm in the dipole buffers
    cudaMemcpyAsync(&gpu->sim.pPGMCGScalars[3], &pgm_bnorm_frozen, sizeof(double), cudaMemcpyHostToDevice, 0);
  } else if (pgm_fused && pgm_bnorm_frozen < 0.0) {
    cudaMemcpy(&pgm_bnorm_frozen, &gpu->sim.pPGMCGScalars[3], sizeof(double), cudaMemcpyDeviceToHost);
  }

  // Initial guess: multi-step extrapolation > warm start > cold start.
  // On NL rebuild, remap history to new atom ordering (preserves extrapolation).
  if (needColdStart && gpu->pgm_dip_history_count > 0) {
    unsigned int* old_lookup = gpu->pbPGMOldImageLookup->_pDevData;
    for (int h = 0; h < gpu->pgm_dip_history_count; h++) {
      cudaMemcpyAsync(gpu->sim.pPGMCGz, gpu->pgm_dip_hist_dev[h],
                 3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
      kPGMRemapDipHistory(gpu, gpu->pgm_dip_hist_dev[h],
                          gpu->sim.pPGMCGz, old_lookup);
    }
    cudaMemcpyAsync(gpu->sim.pPGMCGz, gpu->sim.pPGMCGx,
               3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
    kPGMRemapDipHistory(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGz, old_lookup);
    if (pgm_b_prev && pgm_b_prev_valid) {
      cudaMemcpyAsync(gpu->sim.pPGMCGz, pgm_b_prev->_pDevData,
                 3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
      kPGMRemapDipHistory(gpu, pgm_b_prev->_pDevData, gpu->sim.pPGMCGz, old_lookup);
    }
    needColdStart = false;  // history is now valid in new ordering
  }

  if (pgm_pred_mode == 8 && needColdStart && cpuls_cnt[0] > 0) { cpuls_remap(gpu, stride); needColdStart = false; }
  if (!pgm_pred_done) pgm_run_predictor(gpu, needColdStart, stride);
  pgm_pred_done = false;

  // 7b. Save charges/dipoles & zero charges (fused kernel: 7 ops → 1 launch)
  kPGMSaveAndZeroCharges(gpu);
  PGM_EVENT_RECORD(9);  // SCF start

  // Select solver based on scf_solv_opt:
  //   1 = standard CG (diagonal preconditioner)
  //   2 = preconditioned CG with local preconditioner
  //   3 = preconditioned CG with local preconditioner + peek step (default)
  // Options 2 and 3 use the same code path; peek is always applied in the
  // preconditioner (matches CPU behavior where local_precondition always peeks).

  double r_norm = 0.0;
  int iter = 0;
  double scalars[16];

  if (gpu->pgm_scf_solv_opt >= 2) {
  // ---------- PCG with local preconditioner (scf_solv_opt = 2 or 3) ----------

  // Update scfLocalCut2 in cSim for short-range kernel
  gpu->sim.scfLocalCut2 = (PMEFloat)(gpu->pgm_scf_local_cut * gpu->pgm_scf_local_cut);

  // 7c. Initial residual: r = b - A*x (FULL matvec for accurate residual)
  kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGx);
  pgm_gpu_run_matvec(gpu);
  kPGMComputeAp(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGz);
  kPGMPCGInitResidual(gpu);

  // Apply preconditioner: zpc = precond(r)
  pgm_gpu_local_precondition(gpu);

  // p = zpc; rdotz = dot(r, zpc)
  kPGMPCGSetPFromZpc(gpu);

  // 7d. PCG iteration loop
  while (true) {
    // A*p → z
    kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGp);
    pgm_gpu_run_matvec(gpu);
    kPGMComputeAp(gpu, gpu->sim.pPGMCGp, gpu->sim.pPGMCGz);
    kPGMDotPZ(gpu);
    iter++;

    // x += alpha*p; r -= alpha*z; rmax
    kPGMPCGUpdate(gpu);

    // Read convergence metric
    cudaMemcpy(&scalars[4], &gpu->sim.pPGMCGScalars[4], sizeof(double),
               cudaMemcpyDeviceToHost);
    r_norm = scalars[4];

    if (r_norm <= tol || iter >= maxiter) break;

    // Apply preconditioner: zpc = precond(r)
    pgm_gpu_local_precondition(gpu);

    // rdotz_new = dot(r, zpc); beta = rdotz_new / rdotz
    kPGMPCGDotRZpc(gpu);

    // p = zpc + beta * p
    kPGMPCGUpdateP(gpu);
  }
  // CLAUDE 2026-10-08: final peek after PCG convergence, as in CPU pGM_pkcg (ind_dip += scf_sor_coefficient * rv).
  // Weight from env PGM_PCG_PEEK: a number, or "cpu" = use scf_sor_coefficient from &pol_gauss. Absent/0 = original GPU behaviour.
  { static double pcg_peek = -1.0;
    if (pcg_peek < 0.0) { const char* e = getenv("PGM_PCG_PEEK"); pcg_peek = 0.0;
      if (e) pcg_peek = (!strcmp(e, "cpu")) ? gpu->pgm_scf_sor_coeff : atof(e);
      fprintf(stderr, "PGM PCG final peek weight = %.4f (scf_solv_opt=%d)\n", pcg_peek, gpu->pgm_scf_solv_opt); }
    if (pcg_peek > 0.0 && gpu->pgm_scf_solv_opt == 3) kPGMAxpy(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGr, pcg_peek);
  }

  } else {
  // ---------- Standard CG (scf_solv_opt = 1) ----------

#ifdef PGM_STANDARD_CG
  // ---------- Standard CG: every iteration uses full matvec (with FFT) ----------

  // 7c. Initial residual: r = b - A*x (FULL matvec for accurate starting residual)
  kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGx);
  pgm_gpu_run_matvec(gpu);
  kPGMComputeAp(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGz);
  kPGMInitResidual(gpu);

  // 7d. CG iteration loop
  while (true) {
    kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGp);
    pgm_gpu_run_matvec(gpu);
    kPGMComputeAp(gpu, gpu->sim.pPGMCGp, gpu->sim.pPGMCGz);
    kPGMDotPZ(gpu);
    iter++;
    kPGMCGUpdate(gpu);

    // Read convergence metric
    cudaMemcpy(&scalars[4], &gpu->sim.pPGMCGScalars[4], sizeof(double),
               cudaMemcpyDeviceToHost);
    r_norm = scalars[4];

    if (r_norm <= tol || iter >= maxiter) break;
    kPGMCGUpdateP(gpu);
  }

#else
  pgm_in_cg_loop = true;   // CG-iteration matvecs may use the inexact half2 cache
  { static int dbg2 = -1; if (dbg2 < 0) { const char* e = getenv("PGM_ITERDBG"); dbg2 = (e && atoi(e) > 0) ? 1 : 0; }
    if (dbg2) { double s4; kPGMComputeResidual(gpu); cudaMemcpy(&s4, &gpu->sim.pPGMCGScalars[4], sizeof(double), cudaMemcpyDeviceToHost);
      fprintf(stderr, "PGM START r0 %.3e  hist %d  cold %d  fusedactive %d\n", s4, gpu->pgm_dip_history_count, needColdStart ? 1 : 0, pgm_fused_active ? 1 : 0); } }
  {
    static int chk = -1; if (chk < 0) { const char* e = getenv("PGM_CACHECHECK"); chk = (e && atoi(e) > 0) ? 1 : 0; }
    if (chk && pgm_cache16) {
      int n3 = 3 * gpu->sim.atoms;
      // exact A*p0 -> pPGMPCGzpc ; cached A*p0 -> pPGMCGz
      pgm_in_cg_loop = false;
      kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGp); pgm_gpu_run_matvec(gpu); kPGMComputeAp(gpu, gpu->sim.pPGMCGp, gpu->sim.pPGMPCGzpc);
      pgm_in_cg_loop = true;
      kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGp); pgm_gpu_run_matvec(gpu); kPGMComputeAp(gpu, gpu->sim.pPGMCGp, gpu->sim.pPGMCGz);
      std::vector<double> ze(n3), zc(n3), pv(n3);
      cudaMemcpy(ze.data(), gpu->sim.pPGMPCGzpc, n3*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(zc.data(), gpu->sim.pPGMCGz, n3*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(pv.data(), gpu->sim.pPGMCGp, n3*sizeof(double), cudaMemcpyDeviceToHost);
      double md = 0, mz = 0, mp = 0, sd = 0;
      for (int i = 0; i < n3; i++) { double d = fabs(ze[i]-zc[i]); if (d > md) md = d; if (fabs(ze[i]) > mz) mz = fabs(ze[i]); if (fabs(pv[i]) > mp) mp = fabs(pv[i]); sd += d*d; }
      fprintf(stderr, "PGM CACHECHECK: max|Ap_exact - Ap_cache| = %.3e  max|Ap| = %.3e  rel = %.3e  rms/max = %.3e  max|p| = %.3e\n", md, mz, md/mz, sqrt(sd/n3)/mz, mp);
    }
  }
  // ---------- Batched CG with CUDA Graph: capture iteration body, replay N times ----------
  //
  // CUDA Graph captures the CG iteration body (matvec + scalar ops)
  // as a single graph object, then replays it with near-zero launch overhead.

  // With quadratic extrapolation (hist>=3), converges in ~3 iters instead of ~6.
  int batch_size = needColdStart ? PGM_CG_BATCH_COLD :
                   (gpu->pgm_dip_history_count >= 3) ? PGM_CG_BATCH_EXTRAP :
                   PGM_CG_BATCH_WARM;
  if (pgm_cg_batch_env > 0 && !needColdStart) batch_size = pgm_cg_batch_env;

  // 7c. Initial residual: r = b - A*x (FULL matvec for accurate starting residual)
  if (pgm_fused_active) {
    // b now holds b - alpha*T*x0 (fused sweep), so r0 = b_fused - x0: no matvec needed.
    cudaMemcpyAsync(gpu->sim.pPGMCGz, gpu->sim.pPGMCGx, 3 * stride * sizeof(double),
                    cudaMemcpyDeviceToDevice, 0);
    kPGMInitResidual(gpu);
  } else {
    static int r0c = -1; if (r0c < 0) { const char* e = getenv("PGM_R0CACHE"); r0c = (e && atoi(e) > 0) ? 1 : 0; }
    if (r0c) pgm_in_cg_loop = true;
    kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGx);
    pgm_gpu_run_matvec(gpu);
    kPGMComputeAp(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGz);
    kPGMInitResidual(gpu);
    pgm_in_cg_loop = false;
  }
  { static int dbg3 = -1; if (dbg3 < 0) { const char* e = getenv("PGM_ITERDBG"); dbg3 = (e && atoi(e) > 0) ? 1 : 0; }
    if (dbg3) { double s4, x6[6], b6[6], h6[4][6]; kPGMComputeResidual(gpu);
      cudaMemcpy(&s4, &gpu->sim.pPGMCGScalars[4], sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(x6, gpu->sim.pPGMCGx, 6*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(b6, gpu->sim.pPGMCGb, 6*sizeof(double), cudaMemcpyDeviceToHost);
      for (int h = 0; h < 4 && h < gpu->pgm_dip_history_count; h++) cudaMemcpy(h6[h], gpu->pgm_dip_hist_dev[h], 6*sizeof(double), cudaMemcpyDeviceToHost);
      fprintf(stderr, "PGM R0 %.3e  x0[0..2] %.6f %.6f %.6f  b[0..2] %.6f %.6f %.6f  H0 %.6f H1 %.6f H2 %.6f H3 %.6f  hist %d fused %d\n",
              s4, x6[0],x6[1],x6[2], b6[0],b6[1],b6[2], h6[0][0], h6[1][0], h6[2][0], h6[3][0], gpu->pgm_dip_history_count, pgm_fused_active?1:0); } }

  // Capture CG iteration graph if not yet valid (first step after NL rebuild)
  if (!gpu->pgmGraphsValid) {
    // Destroy old graph if exists
    if (gpu->pgmCGGraphExec) {
      cudaGraphExecDestroy(gpu->pgmCGGraphExec);
      gpu->pgmCGGraphExec = NULL;
    }
    if (gpu->pgmCGGraph) {
      cudaGraphDestroy(gpu->pgmCGGraph);
      gpu->pgmCGGraph = NULL;
    }

    // Set all PGM kernels to go to capture stream
    gpu->pgmStream = gpu->pgmCaptureStream;
    cufftSetStream(gpu->forwardPlan, gpu->pgmCaptureStream);
    cufftSetStream(gpu->backwardPlan, gpu->pgmCaptureStream);

    // Begin capture
    CUDA_GRAPH_CHECK(cudaStreamBeginCapture(gpu->pgmCaptureStream, cudaStreamCaptureModeRelaxed));

    // One CG iteration body (including UpdateP for uniform graph)
    kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGp);
    pgm_gpu_run_matvec(gpu);
    kPGMComputeAp(gpu, gpu->sim.pPGMCGp, gpu->sim.pPGMCGz);
    kPGMDotPZ(gpu);
    kPGMCGUpdate(gpu);
    kPGMCGUpdateP(gpu);

    // End capture and instantiate
    CUDA_GRAPH_CHECK(cudaStreamEndCapture(gpu->pgmCaptureStream, &gpu->pgmCGGraph));
    CUDA_GRAPH_CHECK(cudaGraphInstantiate(&gpu->pgmCGGraphExec, gpu->pgmCGGraph, nullptr, nullptr, 0));

    // Reset to default stream
    gpu->pgmStream = 0;
    cufftSetStream(gpu->forwardPlan, 0);
    cufftSetStream(gpu->backwardPlan, 0);
  }

  // 7d. CG iteration loop (graph replay: launch batch_size graph instances, check once)
  if (pgm_pred_mode == 7 && pgm_fused_active) {
    // XL 0-SCF: Richardson correction(s) on the exact residual from the fused sweep; no CG.
    //   x1 = x0 + w r0 ; optionally r1 = r0 - w A r0 (one more sweep), x2 = x1 + w r1  (env PGM_XL_STEPS)
    static int xl_steps = -1; if (xl_steps < 0) { const char* e = getenv("PGM_XL_STEPS"); xl_steps = (e && atoi(e) > 1) ? atoi(e) : 1; }
    kPGMAxpy(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGr, pgm_xl_w);
    for (int sstep = 1; sstep < xl_steps; sstep++) {
      kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGr);
      pgm_gpu_run_matvec(gpu);
      kPGMComputeAp(gpu, gpu->sim.pPGMCGr, gpu->sim.pPGMCGz);          // z = A r
      kPGMAxpy(gpu, gpu->sim.pPGMCGr, gpu->sim.pPGMCGz, -pgm_xl_w);     // r <- r - w A r
      kPGMAxpy(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGr, pgm_xl_w);     // x <- x + w r
    }
    { static int xd = -1; if (xd < 0) { const char* e = getenv("PGM_STATS"); xd = (e && atoi(e) > 0) ? atoi(e) : 0; }
      static long xs = 0; xs++;
      if (xd > 0 && xs % xd == 0) { double s4; kPGMComputeResidual(gpu); cudaMemcpy(&s4, &gpu->sim.pPGMCGScalars[4], sizeof(double), cudaMemcpyDeviceToHost);
        fprintf(stderr, "PGM XL: step %ld residual(mu) %.3e\n", xs, s4); } }
    maxiter = 0;
  }
  while (iter < maxiter) {
    int n_this = batch_size;
    if (iter + n_this > maxiter) n_this = maxiter - iter;

    // Replay CG iteration graph n_this times (stream-ordered on default stream)
    for (int i = 0; i < n_this; i++) {
      CUDA_GRAPH_CHECK(cudaGraphLaunch(gpu->pgmCGGraphExec, 0));
      iter++;
    }

    // ONE sync per batch: check convergence
    cudaMemcpy(&scalars[4], &gpu->sim.pPGMCGScalars[4], sizeof(double),
               cudaMemcpyDeviceToHost);
    r_norm = scalars[4];
    { static int dbg = -1; if (dbg < 0) { const char* e = getenv("PGM_ITERDBG"); dbg = (e && atoi(e) > 0) ? 1 : 0; }
      if (dbg) fprintf(stderr, "PGM ITER %d rnorm %.3e\n", iter, r_norm);
      if (dbg && pgm_cache16 && iter == 1) {
        // compare the graph's z (= A_cache * p_old) against an exact recomputation on the same p_old:
        // p_old is no longer available (UpdateP ran), so instead recompute both on the CURRENT p.
        int n3 = 3 * gpu->sim.atoms; std::vector<double> ze(n3), zc(n3);
        pgm_in_cg_loop = false; kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGp); pgm_gpu_run_matvec(gpu); kPGMComputeAp(gpu, gpu->sim.pPGMCGp, gpu->sim.pPGMPCGzpc);
        pgm_in_cg_loop = true;  kPGMCopyDipToBuffers(gpu, gpu->sim.pPGMCGp); pgm_gpu_run_matvec(gpu); kPGMComputeAp(gpu, gpu->sim.pPGMCGp, gpu->sim.pPGMCGz);
        cudaMemcpy(ze.data(), gpu->sim.pPGMPCGzpc, n3*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(zc.data(), gpu->sim.pPGMCGz, n3*sizeof(double), cudaMemcpyDeviceToHost);
        double md=0,mz=0; for (int i=0;i<n3;i++){ double d=fabs(ze[i]-zc[i]); if(d>md)md=d; if(fabs(ze[i])>mz)mz=fabs(ze[i]); }
        unsigned int hv[4]; cudaMemcpy(hv, gpu->sim.pPGMBD_h2, 4*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        fprintf(stderr, "PGM INLOOP step-check: rel|Ac p - Ae p| = %.3e  cache ptr %p  first words %08x %08x %08x %08x  totalPairs %d\n", md/mz, (void*)gpu->sim.pPGMBD_h2, hv[0],hv[1],hv[2],hv[3], gpu->sim.pgmBFactorTotalPairs);
      } }

    if (r_norm <= tol) break;
  }
  if (pgm_peek_w > 0.0) kPGMAxpy(gpu, gpu->sim.pPGMCGx, gpu->sim.pPGMCGr, pgm_peek_w);
  pgm_in_cg_loop = false;

#endif  // PGM_STANDARD_CG

  } // end solver selection (scf_solv_opt)

  // Every CG/PCG branch reads the recurrence norm before leaving its loop.
  // Do not accept an unconverged cap-limited state as an induced-dipole solution.
  // The experimental XL zero-SCF branch deliberately sets maxiter=0.
  if (maxiter > 0 && iter >= maxiter && !(r_norm <= tol)) {
    fprintf(stderr,
            "FATAL PGM GPU SCF: iteration cap exhausted without convergence "
            "(solver=%d, iterations=%d, cap=%d, checked recurrence norm=%.17e, "
            "tolerance=%.17e).\n",
            gpu->pgm_scf_solv_opt, iter, maxiter, r_norm, tol);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }

  if (pgm_power_every < 0) { const char* e = getenv("PGM_POWER"); pgm_power_every = e ? ((atoi(e) > 0) ? atoi(e) : 0) : pgm_opt_power;
#ifndef use_DPFP
    if (pgm_power_every > 0) fprintf(stderr, "PGM POWER: note: in SPFP the stored residual contains rounding contributions; use DPFP for the estimate\n");
#endif
  }
  if (pgm_power_every > 0) {
    int n3 = 3 * gpu->sim.atoms;
    if (!pgm_power_xprev) { pgm_power_xprev = new GpuBuffer<double>(3 * stride); pgm_power_part = new GpuBuffer<double>(256); pgm_power_skip = true; }
    double dh = cpuls_dev_power(gpu->sim.pPGMCGr, gpu->sim.pPGMCGx, pgm_power_xprev->_pDevData, gpu->sim.pPGMPolarizability, n3, pgm_power_part->_pDevData, pgm_power_skip ? 1 : 0);
    if (!pgm_power_skip) { pgm_power_sum += dh; pgm_power_n++; pgm_power_sum_all += dh; pgm_power_n_all++; }
    pgm_power_skip = false;
    if (pgm_power_n >= pgm_power_every) {
      // dH per step in e^2/A (Gaussian units of the solver) -> kcal/mol, rate per ns at the actual time step
      const double cc = 332.05382, per_ns = 1.0e3 / pgm_power_dt;
      fprintf(stderr, "PGM POWER: induction power (eq 12) block %.4f kcal/mol/ns  cumulative %.4f kcal/mol/ns  (steps %ld / %ld)\n",
              cc * per_ns * pgm_power_sum / pgm_power_n, cc * per_ns * pgm_power_sum_all / pgm_power_n_all, pgm_power_n, pgm_power_n_all);
      pgm_power_sum = 0.0; pgm_power_n = 0;
    }
  }
  if (pgm_pred_mode == 8) { if (cpuls_device) cpulsd_update(gpu, stride); else cpuls_update(gpu, stride); }
  // 7e. Store induced dipoles
  kPGMStoreInducedDipoles(gpu);
  if (pgm_pred_mode == 7 && pgm_fused_active) {
    // n_{k+1} -> oldest slot, then rotate so H[0] = n_{k+1}
    double** H = gpu->pgm_dip_hist_dev;
    kPGMXLUpdate(gpu, gpu->sim.pPGMCGz /*tmp*/, gpu->sim.pPGMCGx, H, 5, pgm_xl_kappa, pgm_xl_alpha, pgm_xl_c);
    double* oldest = H[5];
    for (int h = 5; h > 0; h--) H[h] = H[h - 1];
    H[0] = oldest;
    cudaMemcpyAsync(H[0], gpu->sim.pPGMCGz, 3 * stride * sizeof(double), cudaMemcpyDeviceToDevice, 0);
  }

  // 7f. Restore charges/dipoles + copy total dipoles to active (fused: 10 ops → 1 launch)
  kPGMRestoreAndCopyTotalDip(gpu);

  // Store SCF convergence info in gpu context
  gpu->pgm_diprms  = r_norm;
  gpu->pgm_dipiter = (double)iter;
  {
    static long pgm_stat_steps = 0, pgm_stat_iters = 0; static int pgm_stat_on = -1;
    if (pgm_stat_on < 0) { const char* s = getenv("PGM_STATS"); pgm_stat_on = (s && atoi(s) > 0) ? atoi(s) : 0; }
    if (pgm_stat_on > 0) {
      pgm_stat_steps++; pgm_stat_iters += iter;
      if (pgm_stat_steps % pgm_stat_on == 0)
        fprintf(stderr, "PGM STATS: steps %ld  mean CG iters %.3f  last iters %d  last rnorm %.2e\n",
                pgm_stat_steps, (double)pgm_stat_iters / pgm_stat_steps, iter, r_norm);
    }
  }

#ifdef PGM_DEBUG
  // Iteration count log
  static int pgm_step = 0;
  static int pgm_total_iters = 0;
  pgm_step++;
  pgm_total_iters += iter;
  if (pgm_step % 1000 == 0) {
    fprintf(stderr, "  PGM SCF: step %d, avg %.1f iters, last iters=%d rnorm=%.2e\n",
            pgm_step, (double)pgm_total_iters / pgm_step, iter, r_norm);
  }
  pgm_prof.total_cg_iters += iter;
  fprintf(stderr, "  step %d: CG iters=%d cold=%d hist=%d rnorm=%.2e\n",
          pgm_prof.step_count, iter, needColdStart ? 1 : 0,
          gpu->pgm_dip_history_count, r_norm);
#endif
  PGM_EVENT_RECORD(10);  // SCF end

  // ===== 8. Force step =====
  if (!gpu->pgmGraphsValid) {
      if (gpu->pgmForceGraphExec) {
        cudaGraphExecDestroy(gpu->pgmForceGraphExec);
        gpu->pgmForceGraphExec = NULL;
      }
      if (gpu->pgmForceGraph) {
        cudaGraphDestroy(gpu->pgmForceGraph);
        gpu->pgmForceGraph = NULL;
      }
      gpu->pgmStream = gpu->pgmCaptureStream;
      cufftSetStream(gpu->forwardPlan, gpu->pgmCaptureStream);
      cufftSetStream(gpu->backwardPlan, gpu->pgmCaptureStream);
      CUDA_GRAPH_CHECK(cudaStreamBeginCapture(gpu->pgmCaptureStream, cudaStreamCaptureModeRelaxed));
      pgm_gpu_run_force_field(gpu, coulomb_const);
      CUDA_GRAPH_CHECK(cudaStreamEndCapture(gpu->pgmCaptureStream, &gpu->pgmForceGraph));
      CUDA_GRAPH_CHECK(cudaGraphInstantiate(&gpu->pgmForceGraphExec, gpu->pgmForceGraph, nullptr, nullptr, 0));
      gpu->pgmStream = 0;
      cufftSetStream(gpu->forwardPlan, 0);
      cufftSetStream(gpu->backwardPlan, 0);
      gpu->pgmGraphsValid = true;
    }
  CUDA_GRAPH_CHECK(cudaGraphLaunch(gpu->pgmForceGraphExec, 0));
  PGM_EVENT_RECORD(11);
  PGM_EVENT_RECORD(12);
  PGM_EVENT_RECORD(13);  // after force field

  // ===== 9. Bonded forces + energies =====
  // Called AFTER kPGMScaleForces so bonded forces (already in kcal/mol/A) are NOT scaled.
  if (need_energy) {
    kExecuteBondWorkUnits(gpu, 1);  // with energy
  } else {
    kExecuteBondWorkUnits(gpu, 0);  // forces only
  }

#ifdef PGM_DEBUG
  cudaEventRecord(pgm_prof.e[14]);  // pipeline end

  // ===== Profile accumulation =====
  pgm_prof.step_count++;
  cudaEventSynchronize(pgm_prof.e[14]);
  pgm_prof.t_nl       += pgm_elapsed(0, 1);
  pgm_prof.t_clear    += pgm_elapsed(1, 2);
  pgm_prof.t_gridwt   += pgm_elapsed(2, 3);
  pgm_prof.t_remap    += pgm_elapsed(3, 4);
  pgm_prof.t_dipconv  += pgm_elapsed(4, 5);
  pgm_prof.t_perm_self   += pgm_elapsed(5, 8);
  pgm_prof.t_scf_init    += pgm_elapsed(8, 9);
  pgm_prof.t_scf_matvec  += pgm_elapsed(9, 10);
  pgm_prof.t_force_direct += pgm_elapsed(10, 13);
  pgm_prof.t_bonded       += pgm_elapsed(13, 14);
  pgm_prof.t_total        += pgm_elapsed(0, 14);

  if (pgm_prof.step_count % 50 == 0) {
    int n = pgm_prof.step_count;
    double avg_iters = (double)pgm_prof.total_cg_iters / n;
    fprintf(stderr, "\n===== PGM PROFILE (%d steps, avg %.1f CG iters) =====\n", n, avg_iters);
    fprintf(stderr, "  NL build:      %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_nl/n, 100*pgm_prof.t_nl/pgm_prof.t_total);
    fprintf(stderr, "  Clear forces:  %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_clear/n, 100*pgm_prof.t_clear/pgm_prof.t_total);
    fprintf(stderr, "  Grid weights:  %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_gridwt/n, 100*pgm_prof.t_gridwt/pgm_prof.t_total);
    fprintf(stderr, "  NL remap:      %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_remap/n, 100*pgm_prof.t_remap/pgm_prof.t_total);
    fprintf(stderr, "  Dip convert:   %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_dipconv/n, 100*pgm_prof.t_dipconv/pgm_prof.t_total);
    fprintf(stderr, "  Perm field:    %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_perm_self/n, 100*pgm_prof.t_perm_self/pgm_prof.t_total);
    fprintf(stderr, "  SCF init:      %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_scf_init/n, 100*pgm_prof.t_scf_init/pgm_prof.t_total);
    fprintf(stderr, "  SCF CG:        %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_scf_matvec/n, 100*pgm_prof.t_scf_matvec/pgm_prof.t_total);
    fprintf(stderr, "  Force field:   %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_force_direct/n, 100*pgm_prof.t_force_direct/pgm_prof.t_total);
    fprintf(stderr, "  Bonded:        %8.2f ms/step (%5.1f%%)\n", pgm_prof.t_bonded/n, 100*pgm_prof.t_bonded/pgm_prof.t_total);
    fprintf(stderr, "  TOTAL:         %8.2f ms/step\n", pgm_prof.t_total/n);
    fprintf(stderr, "================================================\n\n");
  }
#endif // PGM_DEBUG
}

//---------------------------------------------------------------------------------------------
// pgm_gpu_pme_ene_: compute PGM forces + energies (matches standard gpu_pme_ene_ signature)
//
// Called from pme_force.F90 when ipgm > 0 and need_pot_enes = .true.
// Does everything in ONE call: NL, bonded, PGM (perm field, SCF, forces, energy).
// Energy buffer downloaded, pEnergy struct filled (same as standard gpu_pme_ene_).
// Forces stay on GPU — standard gpu_download_frc_ (runmd.F90:680) handles download.
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_pme_ene_(double* ewaldcof, double* vol,
    pme_pot_ene_rec* pEnergy, double enmr[3], double virial[3],
    double ekcmt[3], int* nstep, double* dt,
    double atm_crd[][3], double atm_frc[][3])
{
  pgm_power_dt = *dt;
  PRINTMETHOD("pgm_gpu_pme_ene");

  // Run full PGM pipeline with energy
  pgm_gpu_pme_pipeline(true);

  // Download energy buffer (sync point)
  gpu->pbEnergyBuffer->Download();

  // Decode energy buffer (same as standard gpu_pme_ene_)
  PMEDouble energy[EXTENDED_ENERGY_TERMS];
  for (int i = 0; i < EXTENDED_ENERGY_TERMS; i++) {
    energy[i] = 0.0;
  }
  pEnergy->total = 0.0;
  for (int i = 0; i < VIRIAL_OFFSET; i++) {
    unsigned long long int val = gpu->pbEnergyBuffer->_pSysData[i];
    if (val >= 0x8000000000000000ull) {
      energy[i] = -(PMEDouble)(val ^ 0xffffffffffffffffull) / ENERGYSCALE;
    } else {
      energy[i] = (PMEDouble)val / ENERGYSCALE;
    }
    pEnergy->total += energy[i];
  }
  // Add long-range VDW correction (set by vdw_correction() in Fortran before this call)
  pEnergy->total += pEnergy->vdw_recip;

  // Fill pEnergy struct
  // PGM electrostatic: slot 0 (pEELT) — pre-scaled by coulomb_const in kPGMComputeEnergy
  // VDW: slot 1 (pEVDW) — already in kcal/mol
  // Bonded: slots 3-8 from kExecuteBondWorkUnits
  pEnergy->vdw_dir    = energy[1];
  // vdw_recip is set by vdw_correction() in pme_force.F90 BEFORE this call.
  // Do NOT zero it — it contains the long-range VDW correction.
  pEnergy->vdw_tot    = energy[1] + pEnergy->vdw_recip;
  pEnergy->elec_dir   = energy[0];   // PGM total electrostatic (via phi formula)
  pEnergy->elec_recip = 0.0;         // included in phi
  pEnergy->elec_self  = 0.0;         // included in phi
  pEnergy->elec_nb_adjust = 0.0;
  pEnergy->elec_tot   = energy[0];
  pEnergy->hbond      = 0.0;
  pEnergy->bond       = energy[3];
  pEnergy->angle      = energy[4];
  pEnergy->dihedral   = energy[5];
  pEnergy->elec_14    = energy[6];
  pEnergy->vdw_14     = energy[7];
  pEnergy->restraint  = energy[8] + energy[14] + energy[15] + energy[16];
  pEnergy->angle_ub   = energy[11];
  pEnergy->imp        = energy[12];
  pEnergy->cmap       = energy[13];
  enmr[0]             = energy[14];
  enmr[1]             = energy[15];
  enmr[2]             = energy[16];
  pEnergy->efield     = energy[17];

}

//---------------------------------------------------------------------------------------------
// pgm_gpu_pme_force_: compute PGM forces without energy (matches standard gpu_pme_force_)
//
// Called from pme_force.F90 when ipgm > 0 and need_pot_enes = .false.
// Fully async (no sync needed).
//---------------------------------------------------------------------------------------------
extern "C" void pgm_gpu_pme_force_(double* ewaldcof, double* vol,
    double virial[3], double ekcmt[3], int* nstep, double* dt,
    double atm_crd[][3], double atm_frc[][3])
{
  pgm_power_dt = *dt;
  PRINTMETHOD("pgm_gpu_pme_force");

  // Run full PGM pipeline without energy
  pgm_gpu_pme_pipeline(false);
}

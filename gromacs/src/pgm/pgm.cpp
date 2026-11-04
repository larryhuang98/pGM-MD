/*
 * pGM electrostatics for GROMACS, phase G1: scalar CPU implementation (pair kernels, splines, CG and
 * forces in double precision; the PME grid and FFT in GROMACS's `real`).
 *
 * The model and the formulas are those of docs/spec.md of pgm-native and of its reference engine
 * core/pgm_core.cpp (checked against pGM-JAX); this file ports that engine to the GROMACS data
 * structures: direct-space pairs from the nbnxm cluster pair lists, the PME FFT from gmx_parallel_3dfft,
 * parameters from the PgmAtom / PgmCovalentDipole interaction lists.
 */
#include "gmxpre.h"

#include "pgm.h"

#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "gromacs/domdec/domdec_internal.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/fft/fft.h"
#include "gromacs/fft/parallel_3dfft.h"
#include "gromacs/math/gmxcomplex.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/nbnxm/pairlistset.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/simd/simd.h"
#include "gromacs/simd/simd_math.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/gmxomp.h"
#include "gromacs/utility/alignedallocator.h"
#include "gromacs/utility/listoflists.h"

#ifdef __linux__
#    include <pthread.h>
#    include <sched.h>
#    include <unistd.h>
#endif

#include "gvdw_kernel.h" // pgm-native core/, copied next to this file: the GVDW pair
#include "pgm_math.h"    // and the shared numerics (erf kernels, B-spline moduli, minimum image)
#if GMX_GPU_CUDA
#    include "pgm_cuda.h" // pgm-native core/cuda: the pGM engine on the GPU (GMX_PGM_GPU=1)
#endif

namespace gmx
{

namespace
{

/*! \brief One worker thread that runs one job at a time (the GPU part of a step while the calling
 * thread does the CPU short-range work).  Not pinned: the mdrun threads are, and the worker mostly
 * waits on the GPU.  Both hand-overs spin for a while before they block (a futex wake-up of a thread
 * that went to sleep costs up to 0.1 ms, a tenth of a step): the worker waits for the next job, the
 * caller for the result. */
class PgmWorker
{
public:
    ~PgmWorker()
    {
        if (thread_.joinable())
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                quit_ = true;
            }
            wake_.notify_all();
            thread_.join();
        }
    }
    void run(std::function<void()> job)
    {
        if (!thread_.joinable())
        {
            thread_ = std::thread([this] { loop(); });
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_   = std::move(job);
            error_ = nullptr;
            busy_.store(true, std::memory_order_release);
            pending_.store(true, std::memory_order_release);
        }
        wake_.notify_all();
    }
    //! wait for the job; rethrows its exception
    void wait()
    {
        spinWhile([this] { return busy_.load(std::memory_order_acquire); });
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return !busy_.load(std::memory_order_acquire); });
        if (error_)
        {
            std::rethrow_exception(error_);
        }
    }

private:
    //! spin while cond() holds, at most c_spinSeconds
    template<class Cond>
    static void spinWhile(Cond cond)
    {
        const auto t0 = std::chrono::steady_clock::now();
        int        k  = 0;
        while (cond())
        {
            if (++k % 1024 == 0
                && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > c_spinSeconds)
            {
                return;
            }
        }
    }
    static constexpr double c_spinSeconds = 0.005;
    void                    loop()
    {
#ifdef __linux__
        cpu_set_t set;
        CPU_ZERO(&set);
        const long numCpus = sysconf(_SC_NPROCESSORS_CONF);
        for (long c = 0; c < numCpus && c < CPU_SETSIZE; c++)
        {
            CPU_SET(c, &set);
        }
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
        while (true)
        {
            spinWhile([this] { return !pending_.load(std::memory_order_acquire) && !quit_.load(); });
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return quit_.load() || pending_.load(std::memory_order_acquire); });
            if (quit_.load())
            {
                return;
            }
            std::function<void()> job = std::move(job_);
            job_                      = nullptr;
            pending_.store(false, std::memory_order_release);
            lock.unlock();
            std::exception_ptr error;
            try
            {
                job();
            }
            catch (...)
            {
                error = std::current_exception();
            }
            lock.lock();
            error_ = error;
            busy_.store(false, std::memory_order_release);
            lock.unlock();
            done_.notify_all();
        }
    }
    std::thread             thread_;
    std::mutex              mutex_;
    std::condition_variable wake_, done_;
    std::function<void()>   job_;
    std::atomic<bool>       busy_{ false }, pending_{ false }, quit_{ false };
    std::exception_ptr      error_;
};

} // namespace

namespace
{

// the numerics shared with pgm_core and the OpenMM plugin (pgm-native core/pgm_math.h)
using pgmcore::bsplineModuli;
using pgmcore::ewaldCoefficient;
constexpr double c_sqrtPi = pgmcore::SQRT_PI;
//! polarizabilities at or below this [nm^3] are not polarizable (pmemd-pgm)
constexpr double c_minPolarizability = pgmcore::MIN_POLARIZABILITY;
//! largest PME order (readir: pme-order <= 12)
constexpr int c_maxPmeOrder = 12;

inline double dot3(const DVec& a, const DVec& b)
{
    return a[XX] * b[XX] + a[YY] * b[YY] + a[ZZ] * b[ZZ];
}

using PairBuffer = std::vector<double, AlignedAllocator<double>>;

#if GMX_SIMD && GMX_SIMD_HAVE_DOUBLE
constexpr int c_pairSimdWidth = GMX_SIMD_DOUBLE_WIDTH;
#else
constexpr int c_pairSimdWidth = 1;
#endif

/*! \brief The Ewald pair kernels G_n = Bc_n(beta) - Bc_n(a), n = 0..3, of erfc(x r)/r (the form of
 * pgmcore::ewaldPairKernels for a r >= 1) for n pairs, n a multiple of c_pairSimdWidth, arrays aligned.
 * In double-precision SIMD where available (erfc and exp to the last bits). */
void erfcPairKernels(const double* r, const double* a, int n, double beta, double* G0, double* G1, double* G2, double* G3)
{
#if GMX_SIMD && GMX_SIMD_HAVE_DOUBLE
    const SimdDouble vBeta(beta), cBeta(2.0 * beta / c_sqrtPi), b2(2.0 * beta * beta), b4(4.0 * beta * beta * beta * beta);
    const SimdDouble c2sqpi(2.0 / c_sqrtPi), two(2.0), three(3.0), five(5.0);
    for (int k = 0; k < n; k += c_pairSimdWidth)
    {
        const SimdDouble vr    = load<SimdDouble>(r + k);
        const SimdDouble va    = load<SimdDouble>(a + k);
        const SimdDouble invR  = inv(vr);
        const SimdDouble invR2 = invR * invR;
        const SimdDouble xb    = vBeta * vr;
        const SimdDouble cb    = cBeta * exp(-(xb * xb));
        const SimdDouble Bb0   = erfc(xb) * invR;
        const SimdDouble Bb1   = (Bb0 + cb) * invR2;
        const SimdDouble Bb2   = fma(three, Bb1, b2 * cb) * invR2;
        const SimdDouble Bb3   = fma(five, Bb2, b4 * cb) * invR2;
        const SimdDouble xa    = va * vr;
        const SimdDouble a2    = two * va * va;
        const SimdDouble ca    = c2sqpi * va * exp(-(xa * xa));
        const SimdDouble Ba0   = erfc(xa) * invR;
        const SimdDouble Ba1   = (Ba0 + ca) * invR2;
        const SimdDouble Ba2   = fma(three, Ba1, a2 * ca) * invR2;
        const SimdDouble Ba3   = fma(five, Ba2, a2 * a2 * ca) * invR2;
        store(G0 + k, Bb0 - Ba0);
        store(G1 + k, Bb1 - Ba1);
        store(G2 + k, Bb2 - Ba2);
        store(G3 + k, Bb3 - Ba3);
    }
#else
    for (int k = 0; k < n; k++)
    {
        double Gk[4], Bb[4];
        pgmcore::erfcKernels(beta, r[k], 4, Gk);
        pgmcore::erfcKernels(a[k], r[k], 4, Bb);
        G0[k] = Gk[0] - Bb[0];
        G1[k] = Gk[1] - Bb[1];
        G2[k] = Gk[2] - Bb[2];
        G3[k] = Gk[3] - Bb[3];
    }
#endif
}

} // namespace

class PgmElectrostatics::Impl
{
public:
    Impl(const t_inputrec& ir, FILE* fplog);
    ~Impl();

    void calculate(ArrayRef<const RVec>          x,
                   const matrix                  box,
                   int                           numHomeAtoms,
                   ArrayRef<const real>          charges,
                   const nonbonded_verlet_t&     nbv,
                   const InteractionDefinitions& idef,
                   const ListOfLists<int>&       exclusions,
                   ArrayRef<const RVec>          shiftVectors,
                   ArrayRef<const int>           globalIndex,
                   const gmx_domdec_t*           dd,
                   ForceWithVirial*              forceWithVirial,
                   gmx_enerdata_t*               enerd,
                   bool                          doPairSearch);

    /* Domain decomposition (G4).  Local atoms 0..nHome_-1 are this rank's home atoms, nHome_..n_-1
     * the halo (the DD zones), whose coordinates and charges GROMACS has communicated.  The pairs of the
     * local and non-local nbnxm lists are each on exactly one rank; per-atom data the pairs need on
     * the halo (radii, GVDW parameters, dipoles) are broadcast from the owners, partial sums on the halo
     * (fields, permanent dipoles, forces) are reduced back to them.  The CG vectors and the PME live on
     * the home atoms; the PME grid is summed over the ranks and every rank does the (small) FFT; scalar
     * sums and the predictor history (global order) are reduced over the ranks.  dd_ = nullptr: one
     * rank, all of this is the identity. */
    const gmx_domdec_t* dd_     = nullptr;
    int                 nHome_  = 0;
    int                 nGlobal_ = 0;
    bool                isMain_ = true; // energies and virial terms computed redundantly on every rank count here only
    //! the owners' values on the halo atoms (as dd_move_x, no shift)
    void haloBroadcast(std::vector<DVec>& v) const;
    //! add the halo atoms' values to their owners' (as dd_move_f); the halo entries are then undefined
    void haloReduce(std::vector<DVec>& v) const;
    void sendRecv(int dimIndex, bool forward, std::vector<DVec>& send, std::vector<DVec>& recv) const;
    //! sum over the PP ranks (in place)
    void sumAll(double* data, int count) const;
    double maxAll(double v) const;
    //! global index of local atom i
    int globalOf(int i) const { return globalIndex_.empty() ? i : globalIndex_[i]; }

    // settings
    double rc_, beta_, ke_, tol_, peek_;

    // GVDW (Gaussian-density van der Waals): per local atom sqrt(A), sqrt(C6), b; pair overrides
    bool                 gvdw_ = false;
    int                  gvdwRep_ = pgmcore::GVDWSlater;
    double               rvdw_ = 0;
    DispersionCorrectionType dispCorr_ = DispersionCorrectionType::No;
    std::vector<double>  gvdwA_, gvdwC_, gvdwB_;
    std::vector<char>    gvdwActive_;
    struct GvdwPair
    {
        int    i, j;
        double A, C6, B;
    };
    std::vector<GvdwPair>        gvdwPairs_;   // [ pgm_gvdw_pairs ]: replace the rule, any distance
    std::unordered_set<int64_t> gvdwPairSet_; // their keys gi*N+gj (global indices, gi < gj; of all ranks)
    const ListOfLists<int>*      excls_ = nullptr;
    // per call: GVDW forces (per thread, reduced into [0]), energies and strain
    std::vector<std::vector<DVec>> gvdwForce_;
    double                         gvdwEnergy_ = 0, gvdwPairEnergy_ = 0, gvdwTail_ = 0;
    double                         gvdwS_[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    void                           gvdwOverridesAndTail();
    //! whether the pair takes part in the combination-rule GVDW: both atoms GVDW atoms, not excluded, not overridden
    bool gvdwRulePair(int ai, int aj) const
    {
        if (!gvdw_ || !gvdwActive_[ai] || !gvdwActive_[aj])
        {
            return false;
        }
        for (const int e : (*excls_)[ai])
        {
            if (e == aj)
            {
                return false;
            }
        }
        const int64_t gi = globalOf(ai), gj = globalOf(aj);
        return gvdwPairSet_.empty() || gvdwPairSet_.count(std::min(gi, gj) * nGlobal_ + std::max(gi, gj)) == 0;
    }
    //! one combination-rule GVDW pair of the pair loop (x = r_i - r_j + shift)
    void gvdwPairInLoop(int ai, int aj, const DVec& x, double r2, std::vector<DVec>& F, double* sum) const
    {
        const double r   = std::sqrt(r2);
        const double bij = 1.0 / std::sqrt(2.0 * (radius_[ai] * radius_[ai] + radius_[aj] * radius_[aj]));
        double       dEdR;
        sum[0] += pgmcore::gvdwPair(
                r, bij, gvdwA_[ai] * gvdwA_[aj], gvdwC_[ai] * gvdwC_[aj], 0.5 * (gvdwB_[ai] + gvdwB_[aj]), gvdwRep_, dEdR);
        for (int k = 0; k < DIM; k++)
        {
            const double fk = dEdR * x[k]; // force on i; dU/dx_k = -fk
            F[ai][k] += fk;
            F[aj][k] -= fk;
            for (int b = 0; b < DIM; b++)
            {
                sum[1 + 3 * k + b] -= fk * x[b];
            }
        }
    }
    int    grid_[DIM], order_, maxIter_, predictor_;
    bool   polarizable_ = true;

    // topology
    int                 n_ = -1;
    std::vector<double> q_, radius_, alpha_;
    std::vector<int>    covAtom_, covPartner_;
    std::vector<double> covStrength_;

    // geometry
    std::vector<DVec> pos_;
    double            box_[DIM][DIM], recip_[DIM][DIM], volume_ = 0;

    /*! \brief The pairs of one thread's share of the pair list (cache-line aligned: the vector headers of
     * different threads do not share cache lines).
     *
     * The candidates are taken from the nbnxm cluster lists at the pair-search steps: every atom pair
     * of the listed cluster pairs, each once, in segments of one i atom and one periodic shift.  At
     * every step the pairs within the cut-off are compacted into structure-of-arrays segments of one i
     * atom (r_i - r_j + shift and the kernels G_0..G_3), so the pair passes accumulate atom i in registers.
     */
    struct alignas(64) ThreadPairs
    {
        std::vector<int>  candSegI, candSegShift, candSegBegin, candJ; // candSegBegin: one past the end too
        std::vector<char> candGvdw; // the candidate takes part in the combination-rule GVDW
        std::vector<int>    segI, segBegin, j; // within the cut-off at this step; segBegin: + end
        std::vector<double> x[DIM];
        PairBuffer       G[4], r, a; // kernels; |r_ij| and the Gaussian pair exponent (padded)
        size_t              numPairs() const { return j.size(); }
    };
    std::vector<ThreadPairs>        tp_;
    std::vector<std::array<int, 2>> ciItems_; // (pair list, i-cluster entry)
    bool                            haveCandidates_ = false;
    //! the candidate pairs from the current nbnxm lists (at pair-search steps)
    void searchCandidates(const nonbonded_verlet_t& nbv);
    //! GMX_PGM_CHECK_PAIRS: compare the number of pairs within rc (all ranks) with all pairs, each step (O(N^2))
    bool checkPairs_ = (std::getenv("GMX_PGM_CHECK_PAIRS") != nullptr);
    long numPairErrors_ = 0;
    void checkPairCount();

    // OpenMP: per-thread scratch for the scatter loops
    int                              nt_ = 1;
    std::vector<std::vector<DVec>>   threadVec_;
    /* PME grids in double with the z dimension padded by order - 1 (kzPad_ = nz + order - 1): an atom
     * with z base point b covers the padded points b .. b + order - 1 contiguously (grid point
     * (t - order + 1) mod nz), so the inner spreading and gathering loops run over contiguous memory with
     * the z splines in that order (zTheta_ ...).  Spreading is owner-computes over tiles of (x, y) grid
     * columns: each tile adds the parts of its atoms that fall into its columns, no two tiles write the
     * same point, no per-thread grids, no colouring. */
    int                 kzPad_ = 0;
    std::vector<double> zTheta_, zDTheta_, zD2Theta_; // per atom, the z splines of padded points b + m
    int                 numTiles_[2] = { 1, 1 };
    std::vector<int>    columnTile_[2]; // tile of each x (y) grid index
    std::vector<int>    tileStart_, tileAtoms_;
    std::vector<double> spreadGrid_; // nx*ny*kzPad_
    void                assignTiles();
    size_t              padIndex(int g1, int g2) const
    {
        return (static_cast<size_t>(g1) * grid_[YY] + g2) * kzPad_;
    }
    //! sum the per-thread vectors 0..nt_-1 into out
    void reduceThreadVec(std::vector<DVec>& out);

    // GPU (G3): GMX_PGM_GPU=1 in a CUDA build, one rank: the pGM electrostatics and the combination-rule
    // GVDW on the GPU (pgm_cuda, with its own pair list); GVDW overrides and the tail stay here
    bool useGpu_ = (std::getenv("GMX_PGM_GPU") != nullptr);
#if GMX_GPU_CUDA
    std::unique_ptr<pgmcuda::Engine> gpu_;
    std::vector<int>                 gpuAtoms_; // the global indices the engine has
    std::vector<double>              gpuF_, gpuS_;
    pgmcuda::Terms                   gpuTerms_;
#endif
    PgmWorker gpuWorker_;
    bool      gpuPending_ = false, launched_ = false, lastTopologyChanged_ = false, wantVirial_ = true;
    double    tWait_ = 0, tPrep_ = 0, tFinish_ = 0, tLaunchEnd_ = 0, tBetween_ = 0;
    double    tSubmit_ = 0, tJobEnd_ = 0, tWakeWorker_ = 0, tWakeMain_ = 0; // worker hand-over latencies
    void      prepare(ArrayRef<const RVec>          xIn,
                      const matrix                  box,
                      int                           numHomeAtoms,
                      ArrayRef<const real>          charges,
                      const InteractionDefinitions& idef,
                      ArrayRef<const int>           globalIndex,
                      const gmx_domdec_t*           dd,
                      bool                          doPairSearch);
    void      startGpu(bool topologyChanged);
    void      finishGpu(ForceWithVirial* forceWithVirial, gmx_enerdata_t* enerd);
    bool                             gpuMuStale_ = false;
    void                             fetchGpuDipoles()
    {
#if GMX_GPU_CUDA
        if (gpuMuStale_ && gpu_)
        {
            std::vector<double> mu(3 * n_);
            gpu_->inducedDipoles(mu.data());
            mu_.resize(n_);
            for (int i = 0; i < n_; i++)
            {
                mu_[i] = DVec(mu[3 * i], mu[3 * i + 1], mu[3 * i + 2]);
            }
        }
#endif
        gpuMuStale_ = false;
    }
    void addGvdwOutputs(ArrayRef<RVec> f, ForceWithVirial* forceWithVirial, gmx_enerdata_t* enerd);

    // dipoles and predictor history
    std::vector<DVec>              perm_, mu_;
    std::deque<std::vector<DVec>>  history_;
    std::vector<DVec>              lastPos_;     // global order
    std::vector<int>               globalIndex_; // global index of the local atoms (empty: identity)
    bool                           havePos_        = false;
    int                            lastIterations_ = 0;
    double                         lastResidual_   = 0;
    long                           numCalls_ = 0, sumIterations_ = 0;

    // wall-clock seconds per phase (GMX_PGM_TIMING): pair setup, splines + influence, pair passes of the
    // field, PME, final forces
    double tPairs_ = 0, tSplines_ = 0, tFieldPairs_ = 0, tPme_ = 0, tFinal_ = 0, tCand_ = 0;
    // within PME: spreading (with the grid reduction), the two FFTs, the convolution, the gathers
    double tSpread_ = 0, tFft_ = 0, tConv_ = 0, tGather_ = 0;
    double numCandidates_ = 0, numPairs_ = 0, numSearches_ = 0;
    double tStart_ = -1; // the first call (GPU timing)
    static double now()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // PME
    gmx_parallel_3dfft_t fft_        = nullptr;
    int                  fftThreads_ = 0;
    void                 initFft(int numThreads);
    void                 fftExecute(gmx_fft_direction dir);
    real*                realGrid_    = nullptr;
    t_complex*           complexGrid_ = nullptr;
    ivec                 rNData_, rOffset_, rSize_, cOrder_, cNData_, cOffset_, cSize_;
    std::vector<double>  moduli_[DIM];
    std::vector<double>  influence_; // per complex grid element
    double               influenceBox_[DIM][DIM] = {}; // the box influence_ was computed for
    std::vector<int>     gridIndex_[DIM]; // per atom and spline point j: grid index (base - j) mod K
    std::vector<double>  theta_[DIM], dtheta_[DIM], d2theta_[DIM];
    std::vector<double>  phi_; // reciprocal potential on the padded grid (nx*ny*kzPad_, see padIndex)

    void setTopology(ArrayRef<const real> charges, const InteractionDefinitions& idef, bool topologyChanged);
    void setBox(const matrix box);
    void buildPairs();
    void minImage(DVec& v) const;
    void permanentDipoles();
    void splines();
    void computeInfluence();
    std::vector<DVec> fractionalDipoles(const std::vector<DVec>& d) const;
    double pmePotential(const std::vector<double>* q, const std::vector<DVec>& d, double* strain);
    void   pmeField(std::vector<DVec>& E);
    //! the field at the home atoms of the charges q (nullptr: none) and the home atoms' dipoles dHome
    std::vector<DVec> field(const std::vector<double>* q, const std::vector<DVec>& dHome);
    void              solve();
    //! the home atoms' values of all ranks in the global order (summed over the ranks), and back
    std::vector<DVec> toGlobal(const std::vector<DVec>& v) const
    {
        std::vector<DVec> g(nGlobal_, DVec{ 0, 0, 0 });
        for (int i = 0; i < nHome_; i++)
        {
            g[globalOf(i)] = v[i];
        }
        sumAll(reinterpret_cast<double*>(g.data()), 3 * nGlobal_);
        return g;
    }
    std::vector<DVec> toLocal(const std::vector<DVec>& g) const
    {
        std::vector<DVec> v(n_, DVec{ 0, 0, 0 });
        for (int i = 0; i < nHome_; i++)
        {
            v[i] = g[globalOf(i)];
        }
        return v;
    }

    size_t realIndex(int g1, int g2, int g3) const
    {
        return (static_cast<size_t>(g1) * rSize_[YY] + g2) * rSize_[ZZ] + g3;
    }
};

PgmElectrostatics::Impl::Impl(const t_inputrec& ir, FILE* fplog)
{
    GMX_RELEASE_ASSERT(usingPgm(ir.coulombtype), "PgmElectrostatics needs coulombtype = pGM");
    rc_        = ir.rcoulomb;
    beta_      = ewaldCoefficient(ir.rcoulomb, ir.ewald_rtol);
    if (ir.epsilon_r == 0)
    {
        gmx_fatal(FARGS, "coulombtype = pGM needs a finite epsilon-r");
    }
    if (ir.opts.ngener > 1)
    {
        gmx_fatal(FARGS, "coulombtype = pGM does not support energy groups yet (energygrps)");
    }
    ke_        = c_one4PiEps0 / ir.epsilon_r;
    tol_       = ir.pgm_dipole_tol;
    maxIter_   = ir.pgm_max_iter;
    peek_      = ir.pgm_peek;
    predictor_ = ir.pgm_predictor;
    order_     = ir.pme_order;
    rvdw_      = ir.rvdw;
    gvdwRep_   = (ir.pgm_gvdw_rep_form == 0 ? pgmcore::GVDWGaussian : pgmcore::GVDWSlater);
    dispCorr_  = ir.eDispCorr;
    grid_[XX]  = ir.nkx;
    grid_[YY]  = ir.nky;
    grid_[ZZ]  = ir.nkz;
    for (int d = 0; d < DIM; d++)
    {
        moduli_[d] = bsplineModuli(grid_[d], order_);
    }

    if (fplog)
    {
        fprintf(fplog,
                "\npGM electrostatics (CPU; double-precision kernels and PME spreading, FFT in real):\n"
                "  rcoulomb %g nm, Ewald coefficient %.10g 1/nm, PME grid %d x %d x %d, order %d\n"
                "  Coulomb constant %.10g, dipole tolerance %g, max. iterations %d, predictor %d, "
                "peek %g\n  GVDW (if the topology has [ pgm_gvdw ]): rvdw %g nm, %s repulsion, dispersion "
                "correction %s\n\n",
                rc_,
                beta_,
                grid_[XX],
                grid_[YY],
                grid_[ZZ],
                order_,
                ke_,
                tol_,
                maxIter_,
                predictor_,
                peek_,
                rvdw_,
                gvdwRep_ == pgmcore::GVDWSlater ? "Slater" : "Gaussian",
                enumValueToString(dispCorr_));
    }
}

PgmElectrostatics::Impl::~Impl()
{
    if (numCalls_ > 0 && std::getenv("GMX_PGM_QUIET") == nullptr)
    {
        fprintf(stderr,
                "pGM: %ld force calls, %.2f conjugate-gradient iterations per call\n",
                numCalls_,
                static_cast<double>(sumIterations_) / numCalls_);
        if (std::getenv("GMX_PGM_TIMING"))
        {
            fprintf(stderr,
                    "pGM ms per call: pair setup %.2f, splines+influence %.2f, field pair passes %.2f, "
                    "PME %.2f, final forces (excl. PME) %.2f\n",
                    1e3 * tPairs_ / numCalls_, 1e3 * tSplines_ / numCalls_, 1e3 * tFieldPairs_ / numCalls_,
                    1e3 * tPme_ / numCalls_, 1e3 * tFinal_ / numCalls_);
            fprintf(stderr,
                    "pGM PME ms per call: spread %.2f, FFTs %.2f, convolution %.2f, gathers %.2f\n",
                    1e3 * tSpread_ / numCalls_, 1e3 * tFft_ / numCalls_, 1e3 * tConv_ / numCalls_,
                    1e3 * tGather_ / numCalls_);
            fprintf(stderr,
                    "pGM pairs: %.0f cluster-list atom pairs per pair search (%.0f searches), %.0f within the "
                    "cut-off per call; %d threads\n",
                    numCandidates_ / std::max(1.0, numSearches_), numSearches_, numPairs_ / numCalls_, nt_);
        }
    }
    if (checkPairs_)
    {
        fprintf(stderr, "pGM pair check: %ld of %ld calls with missing or extra pairs\n", numPairErrors_, numCalls_);
    }
    if (fft_)
    {
        gmx_parallel_3dfft_destroy(fft_); // frees fft_
    }
}

void PgmElectrostatics::Impl::initFft(int numThreads)
{
    if (fft_)
    {
        gmx_parallel_3dfft_destroy(fft_); // frees fft_ and the grids
        fft_ = nullptr;
    }
    MPI_Comm comm[2] = { MPI_COMM_NULL, MPI_COMM_NULL };
    ivec     ndata   = { grid_[XX], grid_[YY], grid_[ZZ] };
    gmx_parallel_3dfft_init(&fft_, ndata, &realGrid_, &complexGrid_, comm, TRUE, numThreads);
    gmx_parallel_3dfft_real_limits(fft_, rNData_, rOffset_, rSize_);
    gmx_parallel_3dfft_complex_limits(fft_, cOrder_, cNData_, cOffset_, cSize_);
    // the complex grid is stored y major, x minor (as in pme_solve: (iy size_z + iz) size_x + ix); the
    // limits are indexed by dimension; complex_order reports {0, 1, 2} regardless
    cOrder_[0] = YY;
    cOrder_[1] = ZZ;
    cOrder_[2] = XX;
    for (int d = 0; d < DIM; d++)
    {
        GMX_RELEASE_ASSERT(rNData_[d] == grid_[d] && rOffset_[d] == 0 && cOffset_[d] == 0,
                           "pGM: the PME FFT must be local");
    }
    GMX_RELEASE_ASSERT(cNData_[XX] == grid_[XX] && cNData_[YY] == grid_[YY]
                               && cNData_[ZZ] == grid_[ZZ] / 2 + 1,
                       "pGM: unexpected complex FFT layout");

    fftThreads_ = numThreads;
    influence_.clear();
}

void PgmElectrostatics::Impl::fftExecute(gmx_fft_direction dir)
{
    if (fftThreads_ == 1)
    {
        gmx_parallel_3dfft_execute(fft_, dir, 0, nullptr);
        return;
    }
    // as in pme.cpp: every thread of the team calls execute with its index
#pragma omp parallel num_threads(fftThreads_)
    {
        gmx_parallel_3dfft_execute(fft_, dir, gmx_omp_get_thread_num(), nullptr);
    }
}

void PgmElectrostatics::Impl::setTopology(ArrayRef<const real>          charges,
                                          const InteractionDefinitions& idef,
                                          bool                          topologyChanged)
{
    // n_ (local atoms, home + halo) and nHome_ are set; the per-atom interactions are on the owner
    const int n = n_;
    q_.assign(charges.begin(), charges.begin() + n);
    radius_.assign(n, -1.0);
    alpha_.assign(n, 0.0);
    const auto& ilAtom = idef.il[InteractionFunction::PgmAtom];
    for (int k = 0; k < ilAtom.size(); k += 2)
    {
        const t_iparams& ip = idef.iparams[ilAtom.iatoms[k]];
        const int        a  = ilAtom.iatoms[k + 1];
        radius_[a]          = ip.pgmAtom.radius;
        alpha_[a]           = (ip.pgmAtom.alpha > c_minPolarizability ? ip.pgmAtom.alpha : 0.0);
    }
    for (int a = 0; a < nHome_; a++)
    {
        if (radius_[a] <= 0)
        {
            gmx_fatal(FARGS,
                      "pGM: atom %d has no [ pgm_atoms ] entry with a positive radius; every atom needs one",
                      globalOf(a) + 1);
        }
    }
    covAtom_.clear();
    covPartner_.clear();
    covStrength_.clear();
    const auto& ilDip = idef.il[InteractionFunction::PgmCovalentDipole];
    for (int k = 0; k < ilDip.size(); k += 3)
    {
        covStrength_.push_back(idef.iparams[ilDip.iatoms[k]].pgmDipole.strength);
        covAtom_.push_back(ilDip.iatoms[k + 1]);
        covPartner_.push_back(ilDip.iatoms[k + 2]);
    }
    double numPolarizable = std::count_if(alpha_.begin(), alpha_.begin() + nHome_, [](double a) { return a > 0; });
    sumAll(&numPolarizable, 1);
    polarizable_ = numPolarizable > 0;

    // GVDW
    gvdwA_.assign(n, 0.0);
    gvdwC_.assign(n, 0.0);
    gvdwB_.assign(n, 0.0);
    gvdwActive_.assign(n, 0);
    const auto& ilGvdw = idef.il[InteractionFunction::PgmGvdwAtom];
    for (int k = 0; k < ilGvdw.size(); k += 2)
    {
        const t_iparams& ip = idef.iparams[ilGvdw.iatoms[k]];
        const int        a  = ilGvdw.iatoms[k + 1];
        gvdwA_[a]           = ip.pgmGvdwAtom.sqrtA;
        gvdwC_[a]           = ip.pgmGvdwAtom.sqrtC6;
        gvdwB_[a]           = ip.pgmGvdwAtom.b;
    }
    gvdwPairs_.clear();
    const auto& ilGvdwPair = idef.il[InteractionFunction::PgmGvdwPair];
    for (int k = 0; k < ilGvdwPair.size(); k += 3)
    {
        const t_iparams& ip = idef.iparams[ilGvdwPair.iatoms[k]];
        const int        i = std::min(ilGvdwPair.iatoms[k + 1], ilGvdwPair.iatoms[k + 2]);
        const int        j = std::max(ilGvdwPair.iatoms[k + 1], ilGvdwPair.iatoms[k + 2]);
        gvdwPairs_.push_back({ i, j, ip.pgmGvdwPair.A, ip.pgmGvdwPair.C6, ip.pgmGvdwPair.B });
    }
    double haveGvdw = (!ilGvdw.empty() || !ilGvdwPair.empty()) ? 1 : 0;
    sumAll(&haveGvdw, 1);
    gvdw_ = haveGvdw > 0;

    // the halo atoms' radii and GVDW parameters from their owners
    if (dd_)
    {
        std::vector<DVec> v(n), w(n);
        for (int a = 0; a < nHome_; a++)
        {
            v[a] = DVec(radius_[a], gvdwA_[a], gvdwC_[a]);
            w[a] = DVec(gvdwB_[a], 0, 0);
        }
        haloBroadcast(v);
        haloBroadcast(w);
        for (int a = nHome_; a < n; a++)
        {
            radius_[a] = v[a][XX];
            gvdwA_[a]  = v[a][YY];
            gvdwC_[a]  = v[a][ZZ];
            gvdwB_[a]  = w[a][XX];
        }
    }
    for (int a = 0; a < n; a++)
    {
        gvdwActive_[a] = (gvdwA_[a] != 0 || gvdwC_[a] != 0);
    }
    // the overridden pairs of all ranks, by global index (a pair's override and its place in the
    // pair lists can be on different ranks); they change only with the local topology
    if (topologyChanged)
    {
        std::vector<double> keys; // gi*N + gj, exact in double for N < 9.4e7
        for (const GvdwPair& gp : gvdwPairs_)
        {
            const int64_t gi = globalOf(gp.i), gj = globalOf(gp.j);
            keys.push_back(static_cast<double>(std::min(gi, gj) * nGlobal_ + std::max(gi, gj)));
        }
        if (dd_)
        {
            const int        numRanks = dd_->mpiComm().size(), rank = dd_->mpiComm().rank();
            std::vector<double> count(numRanks, 0.0);
            count[rank] = keys.size();
            sumAll(count.data(), numRanks);
            int offset = 0, total = 0;
            for (int r = 0; r < numRanks; r++)
            {
                offset += (r < rank ? static_cast<int>(count[r]) : 0);
                total += static_cast<int>(count[r]);
            }
            std::vector<double> all(std::max(total, 1), 0.0);
            std::copy(keys.begin(), keys.end(), all.begin() + offset);
            sumAll(all.data(), total);
            all.resize(total);
            keys = all;
        }
        gvdwPairSet_.clear();
        for (const double k : keys)
        {
            gvdwPairSet_.insert(static_cast<int64_t>(k));
        }
    }
    if (static_cast<int>(mu_.size()) != n)
    {
        perm_.assign(n, DVec{ 0, 0, 0 });
        mu_.assign(n, DVec{ 0, 0, 0 });
    }
}

void PgmElectrostatics::Impl::setBox(const matrix box)
{
    // H has the box vectors as rows (GROMACS: lower triangular); recip = H^-1
    double H[DIM][DIM];
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            H[i][j] = box_[i][j] = box[i][j];
        }
    }
    const double det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1])
                       - H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0])
                       + H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);
    volume_      = std::fabs(det);
    recip_[0][0] = (H[1][1] * H[2][2] - H[1][2] * H[2][1]) / det;
    recip_[0][1] = (H[0][2] * H[2][1] - H[0][1] * H[2][2]) / det;
    recip_[0][2] = (H[0][1] * H[1][2] - H[0][2] * H[1][1]) / det;
    recip_[1][0] = (H[1][2] * H[2][0] - H[1][0] * H[2][2]) / det;
    recip_[1][1] = (H[0][0] * H[2][2] - H[0][2] * H[2][0]) / det;
    recip_[1][2] = (H[0][2] * H[1][0] - H[0][0] * H[1][2]) / det;
    recip_[2][0] = (H[1][0] * H[2][1] - H[1][1] * H[2][0]) / det;
    recip_[2][1] = (H[0][1] * H[2][0] - H[0][0] * H[2][1]) / det;
    recip_[2][2] = (H[0][0] * H[1][1] - H[0][1] * H[1][0]) / det;
}

void PgmElectrostatics::Impl::minImage(DVec& v) const
{
    pgmcore::minImage(v, box_);
}

static_assert(sizeof(DVec) == 3 * sizeof(double), "DVec arrays are sent as doubles");

void PgmElectrostatics::Impl::sumAll(double* data, int count) const
{
    if (dd_ && count > 0)
    {
        dd_->mpiComm().sumReduce(count, data);
    }
}

double PgmElectrostatics::Impl::maxAll(double v) const
{
#if GMX_MPI
    if (dd_)
    {
        double out = v;
        MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, dd_->mpiComm().comm());
        return out;
    }
#endif
    return v;
}

void PgmElectrostatics::Impl::sendRecv(int dimIndex, bool forward, std::vector<DVec>& send, std::vector<DVec>& recv) const
{
#if GMX_MPI
    // the neighbours as ddSendrecv: forward sends to neighbor[d][0] and receives from neighbor[d][1]
    const int  sendRank = dd_->neighbor[dimIndex][forward ? 0 : 1];
    const int  recvRank = dd_->neighbor[dimIndex][forward ? 1 : 0];
    const int  tag      = 7461; // pGM's own messages, apart from GROMACS's
    MPI_Comm   comm     = dd_->mpiComm().comm();
    MPI_Status status;
    const int  ns = send.size() * sizeof(DVec), nr = recv.size() * sizeof(DVec);
    if (ns > 0 && nr > 0)
    {
        MPI_Sendrecv(send.data(), ns, MPI_BYTE, sendRank, tag, recv.data(), nr, MPI_BYTE, recvRank, tag, comm, &status);
    }
    else if (ns > 0)
    {
        MPI_Send(send.data(), ns, MPI_BYTE, sendRank, tag, comm);
    }
    else if (nr > 0)
    {
        MPI_Recv(recv.data(), nr, MPI_BYTE, recvRank, tag, comm, &status);
    }
#else
    GMX_UNUSED_VALUE(dimIndex);
    GMX_UNUSED_VALUE(forward);
    GMX_UNUSED_VALUE(send);
    GMX_UNUSED_VALUE(recv);
#endif
}

void PgmElectrostatics::Impl::haloBroadcast(std::vector<DVec>& v) const
{
    if (!dd_)
    {
        return;
    }
    // dd_move_x without the coordinate shift: pulse by pulse along the DD dimensions, the zones received
    // so far are sent on
    const gmx_domdec_comm_t& comm   = *dd_->comm;
    int                      nzone  = 1;
    int                      natTot = comm.atomRanges.numHomeAtoms();
    std::vector<DVec>        send, recv;
    for (int dimIndex = 0; dimIndex < dd_->ndim; dimIndex++)
    {
        const gmx_domdec_comm_dim_t& cd = comm.cd[dimIndex];
        for (const gmx_domdec_ind_t& ind : cd.ind)
        {
            send.resize(ind.nsend[nzone + 1]);
            int n = 0;
            for (const int j : ind.index)
            {
                send[n++] = v[j];
            }
            recv.resize(ind.nrecv[nzone + 1]);
            sendRecv(dimIndex, false, send, recv);
            if (cd.receiveInPlace)
            {
                std::copy(recv.begin(), recv.end(), v.begin() + natTot);
            }
            else
            {
                int j = 0;
                for (int zone = 0; zone < nzone; zone++)
                {
                    for (int i = ind.cell2at0[zone]; i < ind.cell2at1[zone]; i++)
                    {
                        v[i] = recv[j++];
                    }
                }
            }
            natTot += ind.nrecv[nzone + 1];
        }
        nzone += nzone;
    }
}

void PgmElectrostatics::Impl::haloReduce(std::vector<DVec>& v) const
{
    if (!dd_)
    {
        return;
    }
    // dd_move_f without shift forces: the pulses in reverse, the halo values sent back and added
    const gmx_domdec_comm_t& comm   = *dd_->comm;
    int                      nzone  = dd_->zones.numZones() / 2;
    int                      natTot = comm.atomRanges.end(DDAtomRanges::Type::Zones);
    std::vector<DVec>        send, recv;
    for (int dimIndex = dd_->ndim - 1; dimIndex >= 0; dimIndex--)
    {
        const gmx_domdec_comm_dim_t& cd = comm.cd[dimIndex];
        for (int pulse = cd.numPulses() - 1; pulse >= 0; pulse--)
        {
            const gmx_domdec_ind_t& ind = cd.ind[pulse];
            natTot -= ind.nrecv[nzone + 1];
            send.resize(ind.nrecv[nzone + 1]);
            if (cd.receiveInPlace)
            {
                std::copy(v.begin() + natTot, v.begin() + natTot + send.size(), send.begin());
            }
            else
            {
                int j = 0;
                for (int zone = 0; zone < nzone; zone++)
                {
                    for (int i = ind.cell2at0[zone]; i < ind.cell2at1[zone]; i++)
                    {
                        send[j++] = v[i];
                    }
                }
            }
            recv.resize(ind.nsend[nzone + 1]);
            sendRecv(dimIndex, true, send, recv);
            int n = 0;
            for (const int j : ind.index)
            {
                v[j] += recv[n++];
            }
        }
        nzone /= 2;
    }
}

void PgmElectrostatics::Impl::searchCandidates(const nonbonded_verlet_t& nbv)
{
    // the local lists, and with DD the non-local ones (pairs with halo atoms): one list vector
    std::vector<const NbnxnPairlistCpu*> lists;
    for (const InteractionLocality loc : { InteractionLocality::Local, InteractionLocality::NonLocal })
    {
        if (loc == InteractionLocality::NonLocal && !dd_)
        {
            continue;
        }
        for (const NbnxnPairlistCpu& nbl : nbv.pairlistSets().pairlistSet(loc).cpuLists())
        {
            lists.push_back(&nbl);
        }
    }
    if (lists.empty())
    {
        gmx_fatal(FARGS, "pGM electrostatics need the CPU pair lists (mdrun -nb cpu)");
    }
    // every atom pair of the cluster pairs is a candidate, as for the nbnxm LJ kernels: filtering at
    // the atomic list radius would drop the extra buffer the cluster bounding boxes give (water512
    // NVE, mixed: energy drift 1.0 instead of -0.07 kJ/mol over 4 ps)
    // With dynamic pruning, ci / cj are the inner list, pruned every few steps; the candidates must
    // last until the next pair search: the outer (unpruned) list ciOuter / cjOuter then.
    ArrayRef<const int> atomOrder = nbv.getLocalAtomOrder();
    auto useOuter = [](const NbnxnPairlistCpu& nbl) { return !nbl.ciOuter.empty(); };
    auto ciList   = [&](const NbnxnPairlistCpu& nbl) -> const FastVector<nbnxn_ci_t>& {
        return useOuter(nbl) ? nbl.ciOuter : nbl.ci;
    };
    // work items: the i-cluster entries of all lists, dealt out dynamically
    ciItems_.clear();
    for (int l = 0; l < static_cast<int>(lists.size()); l++)
    {
        for (int c = 0; c < static_cast<int>(ciList(*lists[l]).size()); c++)
        {
            ciItems_.push_back({ l, c });
            const nbnxn_ci_t& ci = ciList(*lists[l])[c];
            numCandidates_ += static_cast<double>(ci.cj_ind_end - ci.cj_ind_start) * lists[l]->na_ci * lists[l]->na_cj;
        }
    }
    numSearches_++;
    if (checkPairs_)
    {
        fprintf(stderr, "pGM pair search at call %ld: %s lists, %zu i-entries\n", numCalls_,
                useOuter(*lists[0]) ? "outer" : "plain", ciItems_.size());
    }
    const int numItems = ciItems_.size();
    tp_.resize(nt_);
#pragma omp parallel num_threads(nt_)
    {
        ThreadPairs& T = tp_[gmx_omp_get_thread_num()];
        T.candSegI.clear();
        T.candSegShift.clear();
        T.candSegBegin.clear();
        T.candJ.clear();
        T.candGvdw.clear();
#pragma omp for schedule(dynamic, 8)
        for (int item = 0; item < numItems; item++)
        {
            const NbnxnPairlistCpu& nbl     = *lists[ciItems_[item][0]];
            const bool              outer   = useOuter(nbl);
            const nbnxn_ci_t&       ciEntry = ciList(nbl)[ciItems_[item][1]];
            const int               naci = nbl.na_ci, nacj = nbl.na_cj;
            const int               ish     = ciEntry.shift & NBNXN_CI_SHIFT;
            const bool              central = (ish == c_centralShiftIndex);
            const int i0 = ciEntry.ci * naci;
            for (int ii = 0; ii < naci; ii++)
            {
                const int ai = atomOrder[i0 + ii];
                if (ai < 0)
                {
                    continue;
                }
                const int start = T.candJ.size();
                for (int cjIndex = ciEntry.cj_ind_start; cjIndex < ciEntry.cj_ind_end; cjIndex++)
                {
                    const int j0 = (outer ? nbl.cjOuter[cjIndex].cj : nbl.cj.cj(cjIndex)) * nacj;
                    // nbnxm counts the pairs of overlapping cluster pairs (the diagonal) once: j > i;
                    // the exclusion masks are ignored: pGM electrostatics have no exclusions
                    const bool overlap = central && i0 < j0 + nacj && j0 < i0 + naci;
                    for (int jj = 0; jj < nacj; jj++)
                    {
                        const int aj = atomOrder[j0 + jj];
                        if (aj < 0 || (overlap && j0 + jj <= i0 + ii) || aj == ai)
                        {
                            continue;
                        }
                        T.candJ.push_back(aj);
                        T.candGvdw.push_back(gvdwRulePair(ai, aj) ? 1 : 0);
                    }
                }
                if (static_cast<int>(T.candJ.size()) > start)
                {
                    T.candSegI.push_back(ai);
                    T.candSegShift.push_back(ish);
                    T.candSegBegin.push_back(start);
                }
            }
        }
        T.candSegBegin.push_back(T.candJ.size());
    }
    haveCandidates_ = true;
}

void PgmElectrostatics::Impl::checkPairCount()
{
    // all pairs of the global positions (the same on every rank) against the pairs found on all ranks
    const double            rc2 = rc_ * rc_;
    const std::vector<DVec> X   = toGlobal(pos_);
    long                    all = 0;
#pragma omp parallel for num_threads(nt_) schedule(dynamic, 16) reduction(+ : all)
    for (int i = 0; i < nGlobal_; i++)
    {
        for (int j = i + 1; j < nGlobal_; j++)
        {
            DVec x = X[i] - X[j];
            minImage(x);
            all += (dot3(x, x) < rc2);
        }
    }
    double numFound = 0;
    for (const auto& T : tp_)
    {
        numFound += T.numPairs();
    }
    sumAll(&numFound, 1);
    const long found = static_cast<long>(numFound);
    if (found != all)
    {
        numPairErrors_++;
        if (!dd_ && numPairErrors_ <= 2)
        {
            // which pairs: those within rc missing from the lists
            std::unordered_set<int64_t> foundSet;
            for (const auto& T : tp_)
            {
                for (int sg = 0; sg < static_cast<int>(T.segI.size()); sg++)
                {
                    for (int k = T.segBegin[sg]; k < T.segBegin[sg + 1]; k++)
                    {
                        const int64_t a = globalOf(T.segI[sg]), b = globalOf(T.j[k]);
                        foundSet.insert(std::min(a, b) * nGlobal_ + std::max(a, b));
                    }
                }
            }
            long extra = 0;
            for (const int64_t key : foundSet)
            {
                DVec x = X[key / nGlobal_] - X[key % nGlobal_];
                minImage(x);
                extra += (dot3(x, x) >= rc2);
            }
            fprintf(stderr, "  %zu distinct pairs in the lists, %ld of them beyond rc by the minimum image\n",
                    foundSet.size(), extra);
            int shown = 0;
            for (int i = 0; i < nGlobal_ && shown < 10; i++)
            {
                for (int j = i + 1; j < nGlobal_ && shown < 10; j++)
                {
                    DVec x = X[i] - X[j];
                    minImage(x);
                    if (dot3(x, x) < rc2 && foundSet.count(static_cast<int64_t>(i) * nGlobal_ + j) == 0)
                    {
                        fprintf(stderr, "  missing pair %d %d: r %.6f nm, x_i %.4f %.4f %.4f, x_j %.4f %.4f %.4f\n", i, j,
                                std::sqrt(dot3(x, x)), X[i][XX], X[i][YY], X[i][ZZ], X[j][XX], X[j][YY], X[j][ZZ]);
                        shown++;
                    }
                }
            }
        }
        if (isMain_)
        {
            fprintf(stderr, "pGM pair check, call %ld: %ld pairs within rc in the lists, %ld in all pairs\n",
                    numCalls_, found, all);
        }
    }
}

void PgmElectrostatics::Impl::buildPairs()
{
    const double rc2   = rc_ * rc_;
    const double rvdw2 = rvdw_ * rvdw_;
    double       gvdwSum[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    gvdwForce_.resize(nt_);
    for (auto& v : gvdwForce_)
    {
        v.assign(gvdw_ ? n_ : 0, DVec{ 0, 0, 0 });
    }
    const int numLists = tp_.size();
#pragma omp parallel num_threads(nt_)
    {
        std::vector<DVec>& Fg      = gvdwForce_[gmx_omp_get_thread_num()];
        double             sum[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // GVDW energy, strain (9)
        double             Gk[4];
#pragma omp for schedule(static)
        for (int l = 0; l < numLists; l++)
        {
            ThreadPairs& T = tp_[l];
            T.segI.clear();
            T.segBegin.clear();
            T.j.clear();
            for (auto& v : T.x)
            {
                v.clear();
            }
            // the pairs within the cut-off (and the GVDW of those within rvdw)
            const int numSeg = T.candSegI.size();
            for (int sg = 0; sg < numSeg; sg++)
            {
                const int  ai = T.candSegI[sg];
                const IVec t  = shiftIndexToXYZ(T.candSegShift[sg]);
                DVec       xi = pos_[ai];
                for (int d = 0; d < DIM; d++)
                {
                    xi[d] += t[XX] * box_[XX][d] + t[YY] * box_[YY][d] + t[ZZ] * box_[ZZ][d];
                }
                const int start = T.j.size();
                for (int k = T.candSegBegin[sg]; k < T.candSegBegin[sg + 1]; k++)
                {
                    const int    aj = T.candJ[k];
                    const DVec   dx = xi - pos_[aj];
                    const double r2 = dot3(dx, dx);
                    if (T.candGvdw[k] && r2 < rvdw2)
                    {
                        gvdwPairInLoop(ai, aj, dx, r2, Fg, sum);
                    }
                    if (r2 < rc2 && !useGpu_) // on the GPU the pGM pairs are done there
                    {
                        T.j.push_back(aj);
                        for (int d = 0; d < DIM; d++)
                        {
                            T.x[d].push_back(dx[d]);
                        }
                    }
                }
                if (static_cast<int>(T.j.size()) > start)
                {
                    T.segI.push_back(ai);
                    T.segBegin.push_back(start);
                }
            }
            T.segBegin.push_back(T.j.size());
            // the kernels G_n = B_n[erf(a r)/r] - B_n[erf(beta r)/r] (pgmcore::ewaldPairKernels): all
            // pairs in the erfc form in SIMD, then the few with a r < 1 (bonded neighbours) again from the series
            const int np    = T.j.size();
            const int npPad = ((np + c_pairSimdWidth - 1) / c_pairSimdWidth) * c_pairSimdWidth;
            for (auto* v : { &T.G[0], &T.G[1], &T.G[2], &T.G[3], &T.r, &T.a })
            {
                v->resize(npPad);
            }
            for (int sg = 0; sg < static_cast<int>(T.segI.size()); sg++)
            {
                const double rgi = 2.0 * radius_[T.segI[sg]] * radius_[T.segI[sg]];
                for (int k = T.segBegin[sg]; k < T.segBegin[sg + 1]; k++)
                {
                    const int aj = T.j[k];
                    T.r[k] = std::sqrt(T.x[XX][k] * T.x[XX][k] + T.x[YY][k] * T.x[YY][k] + T.x[ZZ][k] * T.x[ZZ][k]);
                    T.a[k] = 1.0 / std::sqrt(rgi + 2.0 * radius_[aj] * radius_[aj]);
                }
            }
            for (int k = np; k < npPad; k++)
            {
                T.r[k] = 1.0;
                T.a[k] = 1.0;
            }
            erfcPairKernels(T.r.data(), T.a.data(), npPad, beta_, T.G[0].data(), T.G[1].data(), T.G[2].data(), T.G[3].data());
            for (int k = 0; k < np; k++)
            {
                if (T.a[k] * T.r[k] < 1.0)
                {
                    pgmcore::ewaldPairKernels(T.a[k], beta_, T.r[k], 4, Gk);
                    for (int n = 0; n < 4; n++)
                    {
                        T.G[n][k] = Gk[n];
                    }
                }
            }
        }
        for (int k = 0; k < 10; k++)
        {
#pragma omp atomic
            gvdwSum[k] += sum[k];
        }
    }
    gvdwEnergy_ = gvdwSum[0];
    for (int k = 0; k < 9; k++)
    {
        gvdwS_[k] = gvdwSum[1 + k];
    }
    // reduce the GVDW forces into thread 0's buffer
    if (gvdw_)
    {
#pragma omp parallel for num_threads(nt_) schedule(static)
        for (int i = 0; i < n_; i++)
        {
            for (int t = 1; t < nt_; t++)
            {
                gvdwForce_[0][i] += gvdwForce_[t][i];
            }
        }
    }
}

void PgmElectrostatics::Impl::gvdwOverridesAndTail()
{
    gvdwPairEnergy_ = 0;
    gvdwTail_       = 0;
    if (!gvdw_)
    {
        return;
    }
    std::vector<DVec>& F = gvdwForce_[0];
    for (const GvdwPair& p : gvdwPairs_)
    {
        DVec x = pos_[p.i] - pos_[p.j];
        minImage(x);
        const double r   = std::sqrt(dot3(x, x));
        const double bij = 1.0 / std::sqrt(2.0 * (radius_[p.i] * radius_[p.i] + radius_[p.j] * radius_[p.j]));
        double       dEdR;
        gvdwPairEnergy_ += pgmcore::gvdwPair(r, bij, p.A, p.C6, p.B, gvdwRep_, dEdR);
        for (int k = 0; k < DIM; k++)
        {
            const double fk = dEdR * x[k];
            F[p.i][k] += fk;
            F[p.j][k] -= fk;
            for (int b = 0; b < DIM; b++)
            {
                gvdwS_[3 * k + b] -= fk * x[b];
            }
        }
    }
    // dispersion tail (Amber vdwmeth = 1 for the C6 term; GROMACS's DispCorr adds the LJ part)
    if (dispCorr_ != DispersionCorrectionType::No)
    {
        double sumC = 0;
        for (int i = 0; i < nHome_; i++)
        {
            sumC += gvdwC_[i];
        }
        sumAll(&sumC, 1);
        // the same on every rank: counted on the main rank
        gvdwTail_ = isMain_ ? pgmcore::gvdwTail(sumC, volume_, rvdw_) : 0.0;
        if (dispCorr_ == DispersionCorrectionType::EnerPres || dispCorr_ == DispersionCorrectionType::AllEnerPres)
        {
            for (int a = 0; a < DIM; a++)
            {
                gvdwS_[4 * a] -= gvdwTail_; // E ~ 1/V: dE/d eps_aa = -E
            }
        }
    }
}

void PgmElectrostatics::Impl::addGvdwOutputs(ArrayRef<RVec> f, ForceWithVirial* forceWithVirial, gmx_enerdata_t* enerd)
{
    if (!gvdw_)
    {
        return;
    }
    const std::vector<DVec>& Fg = gvdwForce_[0];
    for (int i = 0; i < nHome_; i++)
    {
        for (int k = 0; k < DIM; k++)
        {
            f[i][k] += Fg[i][k];
        }
    }
    matrix virG;
    for (int a = 0; a < DIM; a++)
    {
        for (int b = 0; b < DIM; b++)
        {
            virG[a][b] = 0.5 * gvdwS_[3 * b + a];
        }
    }
    forceWithVirial->addVirialContribution(virG);
    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR][0] += gvdwEnergy_;
    enerd->term[InteractionFunction::PgmGvdwPair] += gvdwPairEnergy_;
    if (dispCorr_ != DispersionCorrectionType::No)
    {
        enerd->term[InteractionFunction::DispersionCorrection] += gvdwTail_;
        if (dispCorr_ == DispersionCorrectionType::EnerPres || dispCorr_ == DispersionCorrectionType::AllEnerPres)
        {
            enerd->term[InteractionFunction::PressureDispersionCorrection] += c_presfac * gvdwTail_ / volume_;
        }
    }
}

void PgmElectrostatics::Impl::startGpu(bool topologyChanged)
{
#if GMX_GPU_CUDA
    if (dd_)
    {
        gmx_fatal(FARGS, "pGM on the GPU (GMX_PGM_GPU) needs a single rank");
    }
    if (!gpu_)
    {
        pgmcuda::Settings gs;
        gs.cutoff    = rc_;
        gs.beta      = beta_;
        for (int d = 0; d < DIM; d++)
        {
            gs.grid[d] = grid_[d];
        }
        gs.order     = order_;
        gs.tol       = tol_;
        gs.maxIter   = maxIter_;
        gs.peek      = peek_;
        gs.predictor = predictor_;
        gs.ke        = ke_;
        gs.device    = 0;
        if (const char* sk = std::getenv("GMX_PGM_GPU_SKIN"))
        {
            gs.skin = std::atof(sk);
        }
        gpu_            = std::make_unique<pgmcuda::Engine>(gs);
        topologyChanged = true;
        gpuAtoms_.clear();
    }
    const double tc0 = now();
    // the engine keeps its own pair list (GPU cell list, rebuilt by displacement): only the atoms
    if (topologyChanged)
    {
        std::vector<int> gi(n_);
        for (int i = 0; i < n_; i++)
        {
            gi[i] = globalOf(i);
        }
        if (gi != gpuAtoms_)
        {
            gpu_->setAtoms(n_, q_.data(), radius_.data(), alpha_.data(), gi.data(), nGlobal_);
            gpu_->setCovalent(covAtom_.size(), covAtom_.data(), covPartner_.data(), covStrength_.data());
            if (gvdw_)
            {
                // the combination rule on the GPU, without the excluded and the overridden pairs (those
                // and the tail stay here, gvdwOverridesAndTail)
                std::vector<int> si, sj;
                for (int i = 0; i < n_; i++)
                {
                    for (const int e : (*excls_)[i])
                    {
                        if (e > i)
                        {
                            si.push_back(i);
                            sj.push_back(e);
                        }
                    }
                }
                for (const GvdwPair& p : gvdwPairs_)
                {
                    si.push_back(p.i);
                    sj.push_back(p.j);
                }
                gpu_->setGvdw(gvdwRep_ == pgmcore::GVDWSlater ? 1 : 0, rvdw_, gvdwA_.data(), gvdwC_.data(),
                              gvdwB_.data(), si.size(), si.data(), sj.data());
            }
            gpuAtoms_ = gi;
        }
    }
    tCand_ += now() - tc0;
    gpuF_.assign(3 * n_, 0.0);
    gpuS_.assign(9, 0.0);
    static_assert(sizeof(DVec) == 3 * sizeof(double), "pos_ is passed as 3n doubles");
    tSubmit_ = now();
    auto job = [this]()
    {
        const double tg0 = now();
        tWakeWorker_ += tg0 - tSubmit_;
        gpuTerms_ = gpu_->compute(reinterpret_cast<const double*>(pos_.data()), box_, true, gpuF_.data(), wantVirial_ ? gpuS_.data() : nullptr);
        tJobEnd_ = now();
        tPme_ += tJobEnd_ - tg0; // the whole GPU part (GMX_PGM_TIMING: "PME")
    };
    if (std::getenv("GMX_PGM_GPU_SYNC"))
    {
        job();
    }
    else
    {
        gpuWorker_.run(job);
    }
    gpuPending_ = true;
#else
    GMX_UNUSED_VALUE(topologyChanged);
    gmx_fatal(FARGS, "GMX_PGM_GPU: this GROMACS build has no CUDA");
#endif
}

void PgmElectrostatics::Impl::finishGpu(ForceWithVirial* forceWithVirial, gmx_enerdata_t* enerd)
{
#if GMX_GPU_CUDA
    GMX_RELEASE_ASSERT(gpuPending_, "finishGpu without startGpu");
    const double tw0 = now();
    try
    {
        gpuWorker_.wait();
    }
    catch (const std::exception& e)
    {
        gmx_fatal(FARGS, "pGM on the GPU: %s", e.what());
    }
    tWait_ += now() - tw0;
    tWakeMain_ += std::max(0.0, now() - std::max(tJobEnd_, tw0));
    if (tLaunchEnd_ > 0)
    {
        tBetween_ += tw0 - tLaunchEnd_; // CPU work between the launch and here
    }
    gpuPending_                  = false;
    const pgmcuda::Terms&     t  = gpuTerms_;
    const std::vector<double>& F = gpuF_;
    const std::vector<double>& S = gpuS_;
    lastIterations_ = gpu_->iterations();
    lastResidual_   = gpu_->residual();
    gpuMuStale_ = true; // fetched by inducedDipoles() when asked for
    ArrayRef<RVec> f = forceWithVirial->force_;
    for (int i = 0; i < n_; i++)
    {
        for (int k = 0; k < DIM; k++)
        {
            f[i][k] += F[3 * i + k];
        }
    }
    matrix vir;
    for (int a = 0; a < DIM; a++)
    {
        for (int b = 0; b < DIM; b++)
        {
            vir[a][b] = 0.5 * S[3 * b + a]; // Xi = S^T / 2
        }
    }
    if (wantVirial_)
    {
        forceWithVirial->addVirialContribution(vir);
    }
    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR][0] += t.direct;
    enerd->term[InteractionFunction::CoulombReciprocalSpace] += t.reciprocal + t.self + t.background;
    enerd->term[InteractionFunction::PgmAtom] += t.polarization;
    gvdwEnergy_ = t.gvdw; // its forces and virial are in F and S
    addGvdwOutputs(f, forceWithVirial, enerd);
    tFinish_ += now() - tw0;
    if (std::getenv("GMX_PGM_TIMING") && (numCalls_ + 1) % 500 == 0)
    {
        const double c = 1e3 / (numCalls_ + 1);
        fprintf(stderr,
                "pGM GPU ms per call (%ld calls): launch (setup + atoms) %.3f, CPU work meanwhile %.3f, "
                "GPU engine %.3f, waiting + outputs %.3f (waiting %.3f; hand-over to the worker %.3f, back "
                "%.3f), step %.3f; %ld list builds, %d candidates\n",
                numCalls_ + 1, c * tPrep_, c * tBetween_, c * tPme_, c * tFinish_, c * tWait_, c * tWakeWorker_,
                c * tWakeMain_,
                c * (now() - tStart_), gpu_->listBuilds(), gpu_->numCandidates());
    }
#else
    GMX_UNUSED_VALUE(forceWithVirial);
    GMX_UNUSED_VALUE(enerd);
    gmx_fatal(FARGS, "GMX_PGM_GPU: this GROMACS build has no CUDA");
#endif
}

void PgmElectrostatics::Impl::reduceThreadVec(std::vector<DVec>& out)
{
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int i = 0; i < n_; i++)
    {
        for (int t = 0; t < nt_; t++)
        {
            out[i] += threadVec_[t][i];
        }
    }
}

void PgmElectrostatics::Impl::permanentDipoles()
{
    perm_.assign(n_, DVec{ 0, 0, 0 });
    for (size_t m = 0; m < covAtom_.size(); m++)
    {
        const int i = covAtom_[m], j = covPartner_[m];
        DVec      v = pos_[j] - pos_[i];
        minImage(v);
        const double L = std::sqrt(dot3(v, v));
        for (int k = 0; k < DIM; k++)
        {
            perm_[i][k] += covStrength_[m] * v[k] / L;
        }
    }
    // a covalent dipole of a home atom may sit on another rank, where the atom is in the halo
    haloReduce(perm_);
    std::fill(perm_.begin() + nHome_, perm_.end(), DVec{ 0, 0, 0 });
}

void PgmElectrostatics::Impl::assignTiles()
{
    // tiles at least order points wide: an atom's order x (y) points fall into at most two tiles
    const int p = order_;
    for (int a = 0; a < 2; a++)
    {
        numTiles_[a]  = std::max(1, grid_[a] / p);
        const int w   = grid_[a] / numTiles_[a];
        columnTile_[a].resize(grid_[a]);
        for (int g = 0; g < grid_[a]; g++)
        {
            columnTile_[a][g] = std::min(g / w, numTiles_[a] - 1);
        }
    }
    const int numTiles = numTiles_[XX] * numTiles_[YY];
    // the (at most four) tiles of each atom, as a list per tile
    std::vector<int> count(numTiles + 1, 0);
    auto             tilesOf = [&](int i, int* out) {
        int tx[2], ty[2], nx = 0, ny = 0;
        for (int j = 0; j < p; j++)
        {
            const int t = columnTile_[XX][gridIndex_[XX][i * p + j]];
            if (nx == 0 || (tx[0] != t && (nx == 1 || tx[1] != t)))
            {
                tx[nx++] = t;
            }
            const int u = columnTile_[YY][gridIndex_[YY][i * p + j]];
            if (ny == 0 || (ty[0] != u && (ny == 1 || ty[1] != u)))
            {
                ty[ny++] = u;
            }
        }
        int m = 0;
        for (int a = 0; a < nx; a++)
        {
            for (int b = 0; b < ny; b++)
            {
                out[m++] = tx[a] * numTiles_[YY] + ty[b];
            }
        }
        return m;
    };
    int tl[4];
    for (int i = 0; i < nHome_; i++)
    {
        const int m = tilesOf(i, tl);
        for (int k = 0; k < m; k++)
        {
            count[tl[k] + 1]++;
        }
    }
    for (int t = 0; t < numTiles; t++)
    {
        count[t + 1] += count[t];
    }
    tileStart_ = count;
    tileAtoms_.resize(count[numTiles]);
    for (int i = 0; i < nHome_; i++)
    {
        const int m = tilesOf(i, tl);
        for (int k = 0; k < m; k++)
        {
            tileAtoms_[count[tl[k]]++] = i;
        }
    }
}

void PgmElectrostatics::Impl::splines()
{
    const int p = order_;
    for (int a = 0; a < DIM; a++)
    {
        gridIndex_[a].resize(nHome_ * p);
        theta_[a].resize(nHome_ * p);
        dtheta_[a].resize(nHome_ * p);
        d2theta_[a].resize(nHome_ * p);
    }
    kzPad_ = grid_[ZZ] + p - 1;
    zTheta_.resize(nHome_ * p);
    zDTheta_.resize(nHome_ * p);
    zD2Theta_.resize(nHome_ * p);
    // the PME acts on the home atoms only (their grid sums are added over the ranks)
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int i = 0; i < nHome_; i++)
    {
        // M[k][j] = M_k(fr + j), j = 0..k-1 (cardinal B-spline of order k); grid point base - j
        double M[c_maxPmeOrder + 1][c_maxPmeOrder + 2] = {};
        auto   get = [&M](int k, int j) { return (j >= 0 && j < k) ? M[k][j] : 0.0; };
        for (int a = 0; a < DIM; a++)
        {
            double t = pos_[i][XX] * recip_[XX][a] + pos_[i][YY] * recip_[YY][a] + pos_[i][ZZ] * recip_[ZZ][a];
            t -= std::floor(t);
            const double w    = grid_[a] * t;
            int          base = static_cast<int>(std::floor(w));
            const double fr   = w - base;
            base              = base % grid_[a];
            M[1][0]           = 1.0;
            for (int k = 2; k <= p; k++)
            {
                for (int j = 0; j < k; j++)
                {
                    M[k][j] = ((fr + j) * get(k - 1, j) + (k - fr - j) * get(k - 1, j - 1)) / (k - 1);
                }
            }
            for (int j = 0; j < p; j++)
            {
                theta_[a][i * p + j]   = M[p][j];
                dtheta_[a][i * p + j]  = get(p - 1, j) - get(p - 1, j - 1);
                d2theta_[a][i * p + j] = get(p - 2, j) - 2 * get(p - 2, j - 1) + get(p - 2, j - 2);
            }
            for (int j = 0; j < p; j++)
            {
                gridIndex_[a][i * p + j] = (base - j + grid_[a]) % grid_[a];
            }
            if (a == ZZ)
            {
                // padded z points base + m hold grid point base - (p - 1 - m): spline index j = p - 1 - m
                for (int m = 0; m < p; m++)
                {
                    zTheta_[i * p + m]   = theta_[ZZ][i * p + p - 1 - m];
                    zDTheta_[i * p + m]  = dtheta_[ZZ][i * p + p - 1 - m];
                    zD2Theta_[i * p + m] = d2theta_[ZZ][i * p + p - 1 - m];
                }
            }
        }
    }
    assignTiles();
}

void PgmElectrostatics::Impl::computeInfluence()
{
    const int c0 = cOrder_[0], c1 = cOrder_[1], c2 = cOrder_[2];
    bool same = (influence_.size() == static_cast<size_t>(cSize_[c0]) * cSize_[c1] * cSize_[c2]);
    for (int a = 0; a < DIM; a++)
    {
        for (int b = 0; b < DIM; b++)
        {
            same = same && influenceBox_[a][b] == box_[a][b];
            influenceBox_[a][b] = box_[a][b];
        }
    }
    if (same)
    {
        return;
    }
    influence_.assign(static_cast<size_t>(cSize_[c0]) * cSize_[c1] * cSize_[c2], 0.0);
    const double fac = M_PI * M_PI / (beta_ * beta_);
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int i0 = 0; i0 < cNData_[c0]; i0++)
    {
        int k[DIM];
        for (int i1 = 0; i1 < cNData_[c1]; i1++)
        {
            for (int i2 = 0; i2 < cNData_[c2]; i2++)
            {
                k[c0] = i0;
                k[c1] = i1;
                k[c2] = i2;
                const int mm[DIM] = { k[XX] <= grid_[XX] / 2 ? k[XX] : k[XX] - grid_[XX],
                                      k[YY] <= grid_[YY] / 2 ? k[YY] : k[YY] - grid_[YY],
                                      k[ZZ] };
                if (mm[XX] == 0 && mm[YY] == 0 && mm[ZZ] == 0)
                {
                    continue;
                }
                DVec ms;
                for (int d = 0; d < DIM; d++)
                {
                    ms[d] = mm[XX] * recip_[d][XX] + mm[YY] * recip_[d][YY] + mm[ZZ] * recip_[d][ZZ];
                }
                const double m2 = dot3(ms, ms);
                influence_[(static_cast<size_t>(i0) * cSize_[c1] + i1) * cSize_[c2] + i2] =
                        std::exp(-fac * m2) / (M_PI * volume_ * m2)
                        / (moduli_[XX][k[XX]] * moduli_[YY][k[YY]] * moduli_[ZZ][k[ZZ]]);
            }
        }
    }
}

std::vector<DVec> PgmElectrostatics::Impl::fractionalDipoles(const std::vector<DVec>& d) const
{
    std::vector<DVec> e(nHome_);
    for (int i = 0; i < nHome_; i++)
    {
        for (int a = 0; a < DIM; a++)
        {
            e[i][a] = grid_[a] * (d[i][XX] * recip_[XX][a] + d[i][YY] * recip_[YY][a] + d[i][ZZ] * recip_[ZZ][a]);
        }
    }
    return e;
}

double PgmElectrostatics::Impl::pmePotential(const std::vector<double>* q, const std::vector<DVec>& d, double* strain)
{
    const int K1 = grid_[XX], K2 = grid_[YY], K3 = grid_[ZZ], p = order_;
    const double tp0 = now();
    std::fill(realGrid_, realGrid_ + static_cast<size_t>(rSize_[XX]) * rSize_[YY] * rSize_[ZZ], static_cast<real>(0));
    // spread in double into the padded grid, owner-computes over the tiles (assignTiles), then fold
    // the padding into the FFT grid
    const size_t      gridSize = static_cast<size_t>(K1) * K2 * kzPad_;
    std::vector<DVec> e        = fractionalDipoles(d);
    spreadGrid_.resize(gridSize);
    phi_.resize(gridSize);
#pragma omp parallel num_threads(nt_)
    {
#pragma omp for schedule(dynamic, 1)
        for (int tile = 0; tile < numTiles_[XX] * numTiles_[YY]; tile++)
        {
            const int tx = tile / numTiles_[YY], ty = tile % numTiles_[YY];
            // zero the tile's columns
            for (int g1 = 0; g1 < K1; g1++)
            {
                if (columnTile_[XX][g1] != tx)
                {
                    continue;
                }
                for (int g2 = 0; g2 < K2; g2++)
                {
                    if (columnTile_[YY][g2] == ty)
                    {
                        std::fill_n(spreadGrid_.data() + padIndex(g1, g2), kzPad_, 0.0);
                    }
                }
            }
            for (int ia = tileStart_[tile]; ia < tileStart_[tile + 1]; ia++)
            {
                const int     i  = tileAtoms_[ia];
                const double  qi = (q ? (*q)[i] : 0.0);
                const int     zb = gridIndex_[ZZ][i * p];
                const double* th = zTheta_.data() + i * p;
                const double* dt = zDTheta_.data() + i * p;
                for (int j1 = 0; j1 < p; j1++)
                {
                    const int g1 = gridIndex_[XX][i * p + j1];
                    if (columnTile_[XX][g1] != tx)
                    {
                        continue;
                    }
                    const double t1 = theta_[XX][i * p + j1], u1 = dtheta_[XX][i * p + j1];
                    for (int j2 = 0; j2 < p; j2++)
                    {
                        const int g2 = gridIndex_[YY][i * p + j2];
                        if (columnTile_[YY][g2] != ty)
                        {
                            continue;
                        }
                        const double t2  = theta_[YY][i * p + j2], u2 = dtheta_[YY][i * p + j2];
                        const double a0  = qi * t1 * t2 + e[i][XX] * u1 * t2 + e[i][YY] * t1 * u2;
                        const double a3  = e[i][ZZ] * t1 * t2;
                        double*      row = spreadGrid_.data() + padIndex(g1, g2) + zb;
                        for (int m = 0; m < p; m++)
                        {
                            row[m] += a0 * th[m] + a3 * dt[m];
                        }
                    }
                }
            }
        }
    }
    // the grid of all ranks' home atoms
    sumAll(spreadGrid_.data(), gridSize);
#pragma omp parallel num_threads(nt_)
    {
        // fold: grid point z is padded point z + p - 1, and z + p - 1 - nz for the top p - 1 points
#pragma omp for schedule(static)
        for (int g1 = 0; g1 < K1; g1++)
        {
            for (int g2 = 0; g2 < K2; g2++)
            {
                const double* row = spreadGrid_.data() + padIndex(g1, g2);
                for (int g3 = 0; g3 < K3; g3++)
                {
                    realGrid_[realIndex(g1, g2, g3)] =
                            row[g3 + p - 1] + (g3 + p - 1 >= K3 ? row[g3 + p - 1 - K3] : 0.0);
                }
            }
        }
    }
    const double tp1 = now();
    fftExecute(GMX_FFT_REAL_TO_COMPLEX);
    const double tp2 = now();

    const int c0 = cOrder_[0], c1 = cOrder_[1], c2 = cOrder_[2];
    double    energy = 0.0;
#pragma omp parallel num_threads(nt_) reduction(+ : energy)
    {
        double strainLocal[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        int    k[DIM];
#pragma omp for schedule(static)
        for (int i0 = 0; i0 < cNData_[c0]; i0++)
        {
            for (int i1 = 0; i1 < cNData_[c1]; i1++)
            {
                for (int i2 = 0; i2 < cNData_[c2]; i2++)
                {
                    const size_t idx = (static_cast<size_t>(i0) * cSize_[c1] + i1) * cSize_[c2] + i2;
                    k[c0]            = i0;
                    k[c1]            = i1;
                    k[c2]            = i2;
                    const double g   = influence_[idx];
                    const double w   = (k[ZZ] == 0 || (K3 % 2 == 0 && k[ZZ] == K3 / 2) ? 1.0 : 2.0);
                    t_complex&   F   = complexGrid_[idx];
                    const double F2  = static_cast<double>(F.re) * F.re + static_cast<double>(F.im) * F.im;
                    const double ek  = 0.5 * w * g * F2;
                    energy += ek;
                    if (strain && g != 0)
                    {
                        const int mm[DIM] = { k[XX] <= K1 / 2 ? k[XX] : k[XX] - K1,
                                              k[YY] <= K2 / 2 ? k[YY] : k[YY] - K2,
                                              k[ZZ] };
                        DVec      ms;
                        for (int dd = 0; dd < DIM; dd++)
                        {
                            ms[dd] = mm[XX] * recip_[dd][XX] + mm[YY] * recip_[dd][YY] + mm[ZZ] * recip_[dd][ZZ];
                        }
                        const double m2 = dot3(ms, ms);
                        const double cm = 2.0 * (M_PI * M_PI / (beta_ * beta_) + 1.0 / m2);
                        for (int a = 0; a < DIM; a++)
                        {
                            for (int b = 0; b < DIM; b++)
                            {
                                strainLocal[3 * a + b] += ek * ((a == b ? -1.0 : 0.0) + cm * ms[a] * ms[b]);
                            }
                        }
                    }
                    F.re *= g;
                    F.im *= g;
                }
            }
        }
        if (strain)
        {
            for (int a = 0; a < 9; a++)
            {
#pragma omp atomic
                strain[a] += strainLocal[a];
            }
        }
    }
    const double tp3 = now();
    fftExecute(GMX_FFT_COMPLEX_TO_REAL);
    tSpread_ += tp1 - tp0;
    tFft_ += (tp2 - tp1) + (now() - tp3);
    tConv_ += tp3 - tp2;
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int g1 = 0; g1 < K1; g1++)
    {
        for (int g2 = 0; g2 < K2; g2++)
        {
            double* row = phi_.data() + padIndex(g1, g2);
            for (int t = 0; t < kzPad_; t++)
            {
                row[t] = realGrid_[realIndex(g1, g2, (t - (p - 1) + K3) % K3)];
            }
        }
    }
    return energy;
}

void PgmElectrostatics::Impl::pmeField(std::vector<DVec>& E)
{
    const double tg0 = now();
    const int K1 = grid_[XX], K2 = grid_[YY], K3 = grid_[ZZ], p = order_;
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int i = 0; i < nHome_; i++)
    {
        double f[3] = { 0, 0, 0 };
        for (int j1 = 0; j1 < p; j1++)
        {
            const int    g1 = gridIndex_[XX][i * p + j1];
            const double t1 = theta_[XX][i * p + j1], u1 = dtheta_[XX][i * p + j1];
            for (int j2 = 0; j2 < p; j2++)
            {
                const int     g2  = gridIndex_[YY][i * p + j2];
                const double  t2  = theta_[YY][i * p + j2], u2 = dtheta_[YY][i * p + j2];
                const double* row = phi_.data() + padIndex(g1, g2) + gridIndex_[ZZ][i * p];
                const double* th  = zTheta_.data() + i * p;
                const double* dt  = zDTheta_.data() + i * p;
                double        s0 = 0, s3 = 0; // sum theta3 phi, sum dtheta3 phi
                for (int m = 0; m < p; m++)
                {
                    s0 += th[m] * row[m];
                    s3 += dt[m] * row[m];
                }
                f[0] += u1 * t2 * s0;
                f[1] += t1 * u2 * s0;
                f[2] += t1 * t2 * s3;
            }
        }
        for (int k = 0; k < DIM; k++)
        {
            E[i][k] -= K1 * recip_[k][XX] * f[0] + K2 * recip_[k][YY] * f[1] + K3 * recip_[k][ZZ] * f[2];
        }
    }
    tGather_ += now() - tg0;
}

std::vector<DVec> PgmElectrostatics::Impl::field(const std::vector<double>* q, const std::vector<DVec>& dHome)
{
    // dHome: the dipoles of the home atoms; the pairs need the halo's too
    const double      t0 = now();
    std::vector<DVec> d  = dHome;
    d.resize(n_, DVec{ 0, 0, 0 });
    haloBroadcast(d);
    std::vector<DVec> E(n_, DVec{ 0, 0, 0 });
    const int         numLists = tp_.size();
    for (auto& v : threadVec_)
    {
        v.assign(n_, DVec{ 0, 0, 0 });
    }
#pragma omp parallel num_threads(nt_)
    {
        std::vector<DVec>& Et = threadVec_[gmx_omp_get_thread_num()];
#pragma omp for schedule(static)
        for (int l = 0; l < numLists; l++)
        {
            const ThreadPairs& T = tp_[l];
            for (int sg = 0; sg < static_cast<int>(T.segI.size()); sg++)
            {
                const int    i  = T.segI[sg];
                const double qi = (q ? (*q)[i] : 0.0);
                const DVec   di = d[i];
                DVec         Ei{ 0, 0, 0 };
                for (int k = T.segBegin[sg]; k < T.segBegin[sg + 1]; k++)
                {
                    const int    j   = T.j[k];
                    const DVec   x(T.x[XX][k], T.x[YY][k], T.x[ZZ][k]);
                    const double G1 = T.G[1][k], G2 = T.G[2][k];
                    const double qj  = (q ? (*q)[j] : 0.0);
                    const DVec&  dj  = d[j];
                    const double djx = dot3(dj, x), dix = dot3(di, x);
                    for (int c = 0; c < DIM; c++)
                    {
                        Ei[c] += qj * x[c] * G1 - dj[c] * G1 + djx * x[c] * G2;
                        Et[j][c] += -qi * x[c] * G1 - di[c] * G1 + dix * x[c] * G2;
                    }
                }
                Et[i] += Ei;
            }
        }
    }
    reduceThreadVec(E);
    haloReduce(E); // the halo's field contributions to their owners
    std::fill(E.begin() + nHome_, E.end(), DVec{ 0, 0, 0 });
    double t1 = now();
    tFieldPairs_ += t1 - t0;
    pmePotential(q, d, nullptr);
    pmeField(E);
    tPme_ += now() - t1;
    const double cs = 4.0 * beta_ * beta_ * beta_ / (3.0 * c_sqrtPi);
    for (int i = 0; i < nHome_; i++)
    {
        for (int k = 0; k < DIM; k++)
        {
            E[i][k] += cs * d[i][k];
        }
    }
    return E;
}

void PgmElectrostatics::Impl::solve()
{
    // the CG runs on the home atoms; sums and maxima over all ranks
    const int                  nh = nHome_;
    std::vector<DVec>          b  = field(&q_, perm_);
    const std::vector<double>& al = alpha_;
    double                     norm = 0;
    for (int i = 0; i < nh; i++)
    {
        for (int k = 0; k < DIM; k++)
        {
            norm += std::fabs(al[i] * b[i][k]);
        }
    }
    sumAll(&norm, 1);
    norm = norm / (3.0 * nGlobal_) + 1e-300;

    // initial guess (history in the global atom order, the same on every rank)
    std::vector<DVec>       x(n_, DVec{ 0, 0, 0 });
    const std::vector<DVec> posGlobal = toGlobal(pos_);
    const bool              samePos   = havePos_ && lastPos_ == posGlobal;
    const int               h         = history_.size();
    if (samePos && h > 0)
    {
        x = toLocal(history_[0]);
    }
    else if (predictor_ == 2 && h >= 4)
    {
        std::vector<DVec> g(nGlobal_);
        for (int i = 0; i < nGlobal_; i++)
        {
            g[i] = 4.0 * history_[0][i] - 6.0 * history_[1][i] + 4.0 * history_[2][i] - history_[3][i];
        }
        x = toLocal(g);
    }
    else if (predictor_ >= 1 && h >= 3)
    {
        std::vector<DVec> g(nGlobal_);
        for (int i = 0; i < nGlobal_; i++)
        {
            g[i] = 3.0 * history_[0][i] - 3.0 * history_[1][i] + history_[2][i];
        }
        x = toLocal(g);
    }
    else if (predictor_ >= 1 && h >= 1)
    {
        x = toLocal(history_[0]);
    }
    else
    {
        for (int i = 0; i < nh; i++)
        {
            x[i] = al[i] * b[i];
        }
    }

    auto applyA = [&](const std::vector<DVec>& v) { // alpha^-1 v - T v (rows of polarizable atoms)
        std::vector<DVec> Tv = field(nullptr, v);
        std::vector<DVec> out(n_, DVec{ 0, 0, 0 });
        for (int i = 0; i < nh; i++)
        {
            for (int k = 0; k < DIM; k++)
            {
                out[i][k] = (al[i] > 0 ? v[i][k] / al[i] - Tv[i][k] : 0.0);
            }
        }
        return out;
    };
    auto vdot = [&](const std::vector<DVec>& a, const std::vector<DVec>& c) {
        double sum = 0;
        for (int i = 0; i < nh; i++)
        {
            sum += dot3(a[i], c[i]);
        }
        sumAll(&sum, 1);
        return sum;
    };
    auto errOf = [&](const std::vector<DVec>& r) {
        double m = 0;
        for (int i = 0; i < nh; i++)
        {
            for (int k = 0; k < DIM; k++)
            {
                m = std::max(m, std::fabs(al[i] * r[i][k]));
            }
        }
        return maxAll(m) / norm;
    };
    // r = b - A x, Jacobi-preconditioned CG (Polak-Ribiere)
    std::vector<DVec> Ax = applyA(x), r(n_, DVec{ 0, 0, 0 }), z(n_, DVec{ 0, 0, 0 }), pdir;
    for (int i = 0; i < nh; i++)
    {
        for (int k = 0; k < DIM; k++)
        {
            r[i][k] = (al[i] > 0 ? b[i][k] - Ax[i][k] : 0.0);
            z[i][k] = al[i] * r[i][k];
        }
    }
    pdir       = z;
    double rz  = vdot(r, z);
    double err = errOf(r);
    int    it  = 0;
    while (err > tol_ && it < maxIter_)
    {
        std::vector<DVec> Ap   = applyA(pdir);
        const double      alph = rz / std::max(vdot(pdir, Ap), 1e-300);
        std::vector<DVec> rnew(n_, DVec{ 0, 0, 0 }), znew(n_, DVec{ 0, 0, 0 });
        double            beta = 0;
        for (int i = 0; i < nh; i++)
        {
            for (int k = 0; k < DIM; k++)
            {
                x[i][k] += alph * pdir[i][k];
                rnew[i][k] = r[i][k] - alph * Ap[i][k];
                znew[i][k] = al[i] * rnew[i][k];
                beta += znew[i][k] * (rnew[i][k] - r[i][k]);
            }
        }
        sumAll(&beta, 1);
        beta /= std::max(rz, 1e-300);
        for (int i = 0; i < nh; i++)
        {
            pdir[i] = znew[i] + beta * pdir[i];
        }
        r   = rnew;
        z   = znew;
        rz  = vdot(r, z);
        err = errOf(r);
        it++;
    }
    if (peek_ != 0)
    {
        for (int i = 0; i < nh; i++)
        {
            x[i] += (peek_ * al[i]) * r[i];
        }
    }
    mu_             = x;
    lastIterations_ = it;
    lastResidual_   = err;
    if (!samePos)
    {
        history_.push_front(toGlobal(mu_));
        while (history_.size() > 4)
        {
            history_.pop_back();
        }
    }
    lastPos_ = posGlobal;
    havePos_ = true;
}

void PgmElectrostatics::Impl::prepare(ArrayRef<const RVec>          xIn,
                                      const matrix                  box,
                                      int                           numHomeAtoms,
                                      ArrayRef<const real>          charges,
                                      const InteractionDefinitions& idef,
                                      ArrayRef<const int>           globalIndex,
                                      const gmx_domdec_t*           dd,
                                      bool                          doPairSearch)
{
    dd_    = dd;
    if (dd_ && dd_->unitCellInfo.haveScrewPBC)
    {
        gmx_fatal(FARGS, "coulombtype = pGM does not support screw PBC");
    }
    if (dd_ && dd_->haloExchange)
    {
        gmx_fatal(FARGS, "coulombtype = pGM needs the classic DD halo communication (unset GMX_FILLERS_IN_LOCAL_STATE)");
    }
    isMain_ = (dd_ == nullptr || DDMAIN(dd_));
    nHome_  = numHomeAtoms;
    n_      = dd_ ? dd_->comm->atomRanges.end(DDAtomRanges::Type::Zones) : numHomeAtoms;
    if (nGlobal_ == 0)
    {
        double ng = nHome_;
        sumAll(&ng, 1);
        nGlobal_ = static_cast<int>(ng);
    }
    const bool topologyChanged = doPairSearch || numCalls_ == 0;
    lastTopologyChanged_       = topologyChanged;
    // the local atom order changes at repartitioning (also on a single rank), only at pair-search
    // steps: the parameters are read from the local topology then, the predictor history is kept in
    // the global order
    nt_ = std::max(1, gmx_omp_nthreads_get(ModuleMultiThread::Default));
    threadVec_.resize(nt_);
    if (fftThreads_ != nt_)
    {
        initFft(nt_);
    }
    globalIndex_.assign(globalIndex.begin(), globalIndex.begin() + (globalIndex.empty() ? 0 : n_));
    if (topologyChanged || static_cast<int>(q_.size()) != n_)
    {
        setTopology(charges, idef, topologyChanged);
    }
    pos_.resize(n_);
    for (int i = 0; i < n_; i++)
    {
        pos_[i] = DVec(xIn[i][XX], xIn[i][YY], xIn[i][ZZ]);
    }
    setBox(box);
}

void PgmElectrostatics::Impl::calculate(ArrayRef<const RVec>          xIn,
                                        const matrix                  box,
                                        int                           numHomeAtoms,
                                        ArrayRef<const real>          charges,
                                        const nonbonded_verlet_t&     nbv,
                                        const InteractionDefinitions& idef,
                                        const ListOfLists<int>&       exclusions,
                                        ArrayRef<const RVec>          shiftVectors,
                                        ArrayRef<const int>           globalIndex,
                                        const gmx_domdec_t*           dd,
                                        ForceWithVirial*              forceWithVirial,
                                        gmx_enerdata_t*               enerd,
                                        bool                          doPairSearch)
{
    excls_ = &exclusions;
    if (!launched_)
    {
        prepare(xIn, box, numHomeAtoms, charges, idef, globalIndex, dd, doPairSearch);
        if (useGpu_)
        {
            startGpu(lastTopologyChanged_);
        }
    }
    launched_                  = false;
    const bool topologyChanged = lastTopologyChanged_;
    if (!useGpu_)
    {
        permanentDipoles();
    }
    double t0 = now();
    // the nbnxm candidates: for the CPU kernels, GVDW and the pair check (the GPU engine has its own list)
    const bool needCandidates = !useGpu_ || checkPairs_;
    if (needCandidates && (topologyChanged || !haveCandidates_ || static_cast<int>(tp_.size()) != nt_))
    {
        searchCandidates(nbv);
    }
    GMX_UNUSED_VALUE(shiftVectors);
    if (!useGpu_)
    {
        buildPairs();
    }
    else if (gvdw_) // the rule part comes from the GPU (finishGpu), the overrides and the tail from here
    {
        gvdwForce_.resize(1);
        gvdwForce_[0].assign(n_, DVec{ 0, 0, 0 });
        gvdwEnergy_ = 0;
        std::fill(std::begin(gvdwS_), std::end(gvdwS_), 0.0);
    }
    if (checkPairs_ && !useGpu_)
    {
        checkPairCount();
    }
    gvdwOverridesAndTail();
    if (gvdw_)
    {
        haloReduce(gvdwForce_[0]); // GVDW forces on halo atoms to their owners
    }
    const double tPairs1 = now();
    tPairs_ += tPairs1 - t0;
    for (const auto& T : tp_)
    {
        numPairs_ += T.numPairs();
    }
    if (numCalls_ == 0 && std::getenv("GMX_PGM_TIMING"))
    {
        fprintf(stderr, "pGM pairs per thread:");
        for (const auto& T : tp_)
        {
            fprintf(stderr, " %zu", T.numPairs());
        }
        fprintf(stderr, " (%d i-cluster items)\n", static_cast<int>(ciItems_.size()));
    }
    if (useGpu_)
    {
        if (tStart_ < 0)
        {
            tStart_ = t0;
        }
        finishGpu(forceWithVirial, enerd);
        numCalls_++;
        sumIterations_ += lastIterations_;
        return;
    }
    splines();
    computeInfluence();
    tSplines_ += now() - tPairs1;

    if (polarizable_)
    {
        solve();
    }
    else
    {
        mu_.assign(n_, DVec{ 0, 0, 0 });
        lastIterations_ = 0;
    }
    numCalls_++;
    sumIterations_ += lastIterations_;
    const double tFinal0 = now();

    // total dipoles: home atoms, then the halo's from their owners
    std::vector<DVec> d(n_, DVec{ 0, 0, 0 });
    for (int i = 0; i < nHome_; i++)
    {
        d[i] = perm_[i] + mu_[i];
    }
    haloBroadcast(d);
    const std::vector<double>& q = q_;
    double                     direct = 0, reciprocal = 0, self = 0, background = 0, polarization = 0;
    std::vector<DVec>          E(n_, DVec{ 0, 0, 0 }); // total field at fixed d [e/nm^2]
    std::vector<DVec>          F(n_, DVec{ 0, 0, 0 }); // forces at fixed d [e^2/nm^2]
    double                     S[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // strain derivative [e^2/nm]
    // pairs: per-thread forces in threadVec_, fields in threadE, strain and energy
    std::vector<std::vector<DVec>>      threadE(nt_, std::vector<DVec>(n_, DVec{ 0, 0, 0 }));
    std::array<double, 10>              threadSum{}; // S (9), direct energy: summed over threads
    const int                           numLists = tp_.size();
    for (auto& v : threadVec_)
    {
        v.assign(n_, DVec{ 0, 0, 0 });
    }
#pragma omp parallel num_threads(nt_)
    {
        const int          t   = gmx_omp_get_thread_num();
        std::vector<DVec>& Ft  = threadVec_[t];
        std::vector<DVec>& Et  = threadE[t];
        std::array<double, 10> sum{}; // S (9), direct energy: local, stored at the end
#pragma omp for schedule(static)
        for (int l = 0; l < numLists; l++)
        {
            const ThreadPairs& T = tp_[l];
            for (int sg = 0; sg < static_cast<int>(T.segI.size()); sg++)
            {
                const int    i  = T.segI[sg];
                const double qi = q[i];
                const DVec   di = d[i];
                DVec         Fi{ 0, 0, 0 }, Ei{ 0, 0, 0 };
                for (int k = T.segBegin[sg]; k < T.segBegin[sg + 1]; k++)
                {
                    const int    j  = T.j[k];
                    const DVec   x(T.x[XX][k], T.x[YY][k], T.x[ZZ][k]);
                    const double G[4] = { T.G[0][k], T.G[1][k], T.G[2][k], T.G[3][k] };
                    const double qj   = q[j];
                    const DVec&  dj   = d[j];
                    const double dix = dot3(di, x), djx = dot3(dj, x), didj = dot3(di, dj);
                    sum[9] += qi * qj * G[0] + (qi * djx - qj * dix) * G[1] + didj * G[1] - dix * djx * G[2];
                    const double sc =
                            -qi * qj * G[1] - (qi * djx - qj * dix) * G[2] - didj * G[2] + dix * djx * G[3];
                    for (int c = 0; c < DIM; c++)
                    {
                        const double dedx = sc * x[c] + (qi * dj[c] - qj * di[c]) * G[1]
                                            - (djx * di[c] + dix * dj[c]) * G[2];
                        for (int b = 0; b < DIM; b++)
                        {
                            sum[3 * c + b] += dedx * x[b];
                        }
                        Fi[c] -= dedx;
                        Ft[j][c] += dedx;
                        Ei[c] += qj * x[c] * G[1] - dj[c] * G[1] + djx * x[c] * G[2];
                        Et[j][c] += -qi * x[c] * G[1] - di[c] * G[1] + dix * x[c] * G[2];
                    }
                }
                Ft[i] += Fi;
                Et[i] += Ei;
            }
        }
        for (int k = 0; k < 10; k++)
        {
#pragma omp atomic
            threadSum[k] += sum[k];
        }
    }
    reduceThreadVec(F);
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int i = 0; i < n_; i++)
    {
        for (int t = 0; t < nt_; t++)
        {
            E[i] += threadE[t][i];
        }
    }
    haloReduce(E); // the halo's direct fields to their owners (F: below, with the covalent forces)
    std::fill(E.begin() + nHome_, E.end(), DVec{ 0, 0, 0 });
    for (int k = 0; k < 9; k++)
    {
        S[k] += threadSum[k];
    }
    direct += threadSum[9];

    // reciprocal space: the home atoms; the k-space energy and strain are the same on every rank
    const double tPme0 = now();
    double       Sk[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    reciprocal         = pmePotential(&q, d, Sk);
    if (isMain_)
    {
        for (int k = 0; k < 9; k++)
        {
            S[k] += Sk[k];
        }
    }
    else
    {
        reciprocal = 0;
    }
    std::vector<DVec> Erec(n_, DVec{ 0, 0, 0 });
    pmeField(Erec);
    const double tPme1 = now();
    for (int i = 0; i < nHome_; i++)
    {
        for (int a = 0; a < DIM; a++)
        {
            E[i][a] += Erec[i][a];
            for (int b = 0; b < DIM; b++)
            {
                S[3 * a + b] += Erec[i][a] * d[i][b]; // fractional dipoles change with the box
            }
        }
    }
    // reciprocal forces on the home atoms: -[q grad phi + (grad grad phi) . d] (added after the halo
    // reduction of F below)
    std::vector<DVec> Frec(nHome_, DVec{ 0, 0, 0 });
    {
        const int         p = order_;
        std::vector<DVec> e = fractionalDipoles(d);
#pragma omp parallel for num_threads(nt_) schedule(static)
        for (int i = 0; i < nHome_; i++)
        {
            double f1[3] = { 0, 0, 0 }, f2[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
            for (int j1 = 0; j1 < p; j1++)
            {
                const int    g1 = gridIndex_[XX][i * p + j1];
                const double t1 = theta_[XX][i * p + j1], u1 = dtheta_[XX][i * p + j1],
                             v1 = d2theta_[XX][i * p + j1];
                for (int j2 = 0; j2 < p; j2++)
                {
                    const int     g2 = gridIndex_[YY][i * p + j2];
                    const double  t2 = theta_[YY][i * p + j2], u2 = dtheta_[YY][i * p + j2],
                                 v2 = d2theta_[YY][i * p + j2];
                    const double* row = phi_.data() + padIndex(g1, g2) + gridIndex_[ZZ][i * p];
                    const double* th  = zTheta_.data() + i * p;
                    const double* dt  = zDTheta_.data() + i * p;
                    const double* d2t = zD2Theta_.data() + i * p;
                    double        s0 = 0, s1 = 0, s2 = 0; // sums of theta3, dtheta3, d2theta3 times phi
                    for (int m = 0; m < p; m++)
                    {
                        s0 += th[m] * row[m];
                        s1 += dt[m] * row[m];
                        s2 += d2t[m] * row[m];
                    }
                    f1[0] += u1 * t2 * s0;
                    f1[1] += t1 * u2 * s0;
                    f1[2] += t1 * t2 * s1;
                    f2[0][0] += v1 * t2 * s0;
                    f2[1][1] += t1 * v2 * s0;
                    f2[2][2] += t1 * t2 * s2;
                    f2[0][1] += u1 * u2 * s0;
                    f2[0][2] += u1 * t2 * s1;
                    f2[1][2] += t1 * u2 * s1;
                }
            }
            f2[1][0] = f2[0][1];
            f2[2][0] = f2[0][2];
            f2[2][1] = f2[1][2];
            for (int a = 0; a < DIM; a++)
            {
                const double c = q[i] * f1[a] + e[i][0] * f2[a][0] + e[i][1] * f2[a][1] + e[i][2] * f2[a][2];
                for (int k = 0; k < DIM; k++)
                {
                    Frec[i][k] -= grid_[a] * recip_[k][a] * c;
                }
            }
        }
    }
    const double cs = 4.0 * beta_ * beta_ * beta_ / (3.0 * c_sqrtPi);
    double       sums[3] = { 0, 0, 0 }; // sum q^2, sum |d|^2 (home: self energy), sum q (all ranks)
    for (int i = 0; i < nHome_; i++)
    {
        sums[0] += q[i] * q[i];
        sums[1] += dot3(d[i], d[i]);
        sums[2] += q[i];
        for (int k = 0; k < DIM; k++)
        {
            E[i][k] += cs * d[i][k];
        }
    }
    self = -(beta_ / c_sqrtPi) * sums[0] - 0.5 * cs * sums[1];
    sumAll(&sums[2], 1);
    if (isMain_)
    {
        background = -M_PI * sums[2] * sums[2] / (2.0 * volume_ * beta_ * beta_);
        for (int a = 0; a < DIM; a++)
        {
            S[4 * a] -= background; // E_bg ~ 1/V
        }
    }
    for (int i = 0; i < nHome_; i++)
    {
        if (alpha_[i] > 0)
        {
            polarization += dot3(mu_[i], mu_[i]) / (2.0 * alpha_[i]);
        }
    }
    // covalent dipoles: dE/dr_partner = c (g - (g.u) u)/|v|, g = dE/dd_atom = -E_atom; the dipole's atom
    // may be in the halo: its total field from the owner
    haloBroadcast(E);
    for (size_t m = 0; m < covAtom_.size(); m++)
    {
        const int i = covAtom_[m], j = covPartner_[m];
        DVec      v = pos_[j] - pos_[i];
        minImage(v);
        const double L  = std::sqrt(dot3(v, v));
        const DVec   u  = (1.0 / L) * v;
        const DVec   g  = -1.0 * E[i];
        const double gu = dot3(g, u);
        for (int k = 0; k < DIM; k++)
        {
            const double t = covStrength_[m] * (g[k] - gu * u[k]) / L;
            F[j][k] -= t;
            F[i][k] += t;
            for (int b = 0; b < DIM; b++)
            {
                S[3 * k + b] += t * v[b];
            }
        }
    }
    haloReduce(F); // pair and covalent forces on the halo to their owners
    for (int i = 0; i < nHome_; i++)
    {
        F[i] += Frec[i];
    }

    tFinal_ += (now() - tFinal0) - (tPme1 - tPme0);
    tPme_ += tPme1 - tPme0;

    // outputs
    ArrayRef<RVec> f = forceWithVirial->force_;
    for (int i = 0; i < nHome_; i++)
    {
        for (int k = 0; k < DIM; k++)
        {
            f[i][k] += ke_ * F[i][k];
        }
    }
    matrix vir;
    for (int a = 0; a < DIM; a++)
    {
        for (int b = 0; b < DIM; b++)
        {
            vir[a][b] = 0.5 * ke_ * S[3 * b + a]; // Xi = S^T / 2
        }
    }
    forceWithVirial->addVirialContribution(vir);
    enerd->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR][0] += ke_ * direct;
    enerd->term[InteractionFunction::CoulombReciprocalSpace] += ke_ * (reciprocal + self + background);
    enerd->term[InteractionFunction::PgmAtom] += ke_ * polarization;
    addGvdwOutputs(f, forceWithVirial, enerd);

    static const char* dump = std::getenv("GMX_PGM_DUMP");
    if (dump && numCalls_ == 1 && dd_)
    {
        fprintf(stderr, "pGM: GMX_PGM_DUMP needs one rank; no dump written\n");
    }
    if (dump && numCalls_ == 1 && !dd_)
    {
        // everything needed to recompute this call with core/pgm_core (core/tests/run_dump.cpp)
        FILE* fp = std::fopen(dump, "w");
        fprintf(fp, "# pGM dump: n ncov beta rc K1 K2 K3 order ke\n%d %d %.17g %.17g %d %d %d %d %.17g\n",
                n_, static_cast<int>(covAtom_.size()), beta_, rc_, grid_[XX], grid_[YY], grid_[ZZ], order_, ke_);
        fprintf(fp,
                "# terms [kJ/mol]: direct reciprocal self background polarization; iterations residual\n"
                "%.17g %.17g %.17g %.17g %.17g %d %.6g\n",
                ke_ * direct, ke_ * reciprocal, ke_ * self, ke_ * background, ke_ * polarization,
                lastIterations_, lastResidual_);
        fprintf(fp, "# box rows [nm]\n");
        for (int a = 0; a < DIM; a++)
        {
            fprintf(fp, "%.17g %.17g %.17g\n", box_[a][XX], box_[a][YY], box_[a][ZZ]);
        }
        fprintf(fp, "# virial (Xi = S^T/2) [kJ/mol]\n");
        for (int a = 0; a < DIM; a++)
        {
            fprintf(fp, "%.17g %.17g %.17g\n", vir[a][XX], vir[a][YY], vir[a][ZZ]);
        }
        fprintf(fp,
                "# atom (global order): x(3) [nm] q radius alpha, pGM force (3) [kJ/mol/nm], induced dipole "
                "(3) [e nm], permanent dipole (3)\n");
        std::vector<int> localOf(n_);
        for (int i = 0; i < n_; i++)
        {
            localOf[globalIndex_.empty() ? i : globalIndex_[i]] = i;
        }
        for (int g = 0; g < n_; g++)
        {
            const int i = localOf[g];
            fprintf(fp,
                    "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g "
                    "%.17g\n",
                    pos_[i][XX], pos_[i][YY], pos_[i][ZZ], q_[i], radius_[i], alpha_[i],
                    ke_ * F[i][XX], ke_ * F[i][YY], ke_ * F[i][ZZ],
                    mu_[i][XX], mu_[i][YY], mu_[i][ZZ],
                    perm_[i][XX], perm_[i][YY], perm_[i][ZZ]);
        }
        fprintf(fp, "# covalent dipoles: atom partner strength [e nm] (0-based, global)\n");
        for (size_t m = 0; m < covAtom_.size(); m++)
        {
            const int a = covAtom_[m], b = covPartner_[m];
            fprintf(fp, "%d %d %.17g\n", globalIndex_.empty() ? a : globalIndex_[a],
                    globalIndex_.empty() ? b : globalIndex_[b], covStrength_[m]);
        }
        if (gvdw_)
        {
            fprintf(fp,
                    "# GVDW: energy (pairs) energy (overrides) energy (tail) [kJ/mol]; virial (3x3); per atom "
                    "(global order) force [kJ/mol/nm]\n%.17g %.17g %.17g\n",
                    gvdwEnergy_, gvdwPairEnergy_, gvdwTail_);
            for (int a = 0; a < DIM; a++)
            {
                fprintf(fp, "%.17g %.17g %.17g\n", 0.5 * gvdwS_[a], 0.5 * gvdwS_[3 + a], 0.5 * gvdwS_[6 + a]);
            }
            for (int g = 0; g < n_; g++)
            {
                const int i = localOf[g];
                fprintf(fp, "%.17g %.17g %.17g\n", gvdwForce_[0][i][XX], gvdwForce_[0][i][YY], gvdwForce_[0][i][ZZ]);
            }
        }
        std::fclose(fp);
    }
}

// ------------------------------------------------------------------------- public interface

PgmElectrostatics::PgmElectrostatics(const t_inputrec& ir, FILE* fplog) :
    impl_(std::make_unique<Impl>(ir, fplog))
{
}

PgmElectrostatics::~PgmElectrostatics() = default;

void PgmElectrostatics::calculate(ArrayRef<const RVec>          x,
                                  const matrix                  box,
                                  int                           numHomeAtoms,
                                  ArrayRef<const real>          charges,
                                  const nonbonded_verlet_t&     nbv,
                                  const InteractionDefinitions& idef,
                                  const ListOfLists<int>&       exclusions,
                                  ArrayRef<const RVec>          shiftVectors,
                                  ArrayRef<const int>           globalIndex,
                                  const gmx_domdec_t*           dd,
                                  ForceWithVirial*              forceWithVirial,
                                  gmx_enerdata_t*               enerd,
                                  const StepWorkload&           stepWork)
{
    if (!impl_->launched_)
    {
        impl_->wantVirial_ = stepWork.computeVirial;
    }
    impl_->calculate(x, box, numHomeAtoms, charges, nbv, idef, exclusions, shiftVectors, globalIndex, dd,
                     forceWithVirial, enerd, stepWork.doNeighborSearch);
}

void PgmElectrostatics::launch(ArrayRef<const RVec>          x,
                               const matrix                  box,
                               int                           numHomeAtoms,
                               ArrayRef<const real>          charges,
                               const InteractionDefinitions& idef,
                               const ListOfLists<int>&       exclusions,
                               ArrayRef<const int>           globalIndex,
                               const gmx_domdec_t*           dd,
                               const StepWorkload&           stepWork)
{
    if (!impl_->useGpu_)
    {
        return;
    }
    const double t0     = Impl::now();
    impl_->wantVirial_ = stepWork.computeVirial; // the GPU strain derivative only when needed
    impl_->excls_      = &exclusions;
    impl_->prepare(x, box, numHomeAtoms, charges, idef, globalIndex, dd, stepWork.doNeighborSearch);
    impl_->startGpu(impl_->lastTopologyChanged_);
    impl_->launched_ = true;
    impl_->tLaunchEnd_ = Impl::now();
    impl_->tPrep_ += impl_->tLaunchEnd_ - t0;
}

ArrayRef<const DVec> PgmElectrostatics::inducedDipoles() const
{
    impl_->fetchGpuDipoles();
    return impl_->mu_;
}

ArrayRef<const DVec> PgmElectrostatics::permanentDipoles() const
{
    return impl_->perm_;
}

int PgmElectrostatics::iterations() const
{
    return impl_->lastIterations_;
}

} // namespace gmx

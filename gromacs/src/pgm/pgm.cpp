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
#include <chrono>
#include <deque>
#include <vector>

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
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/gmxomp.h"

namespace gmx
{

namespace
{

constexpr double c_sqrtPi = 1.7724538509055160273;
//! polarizabilities at or below this [nm^3] are not polarizable (pmemd-pgm; core/pgm_constants.h)
constexpr double c_minPolarizability = 1e-9;
//! largest PME order (readir: pme-order <= 12)
constexpr int c_maxPmeOrder = 12;

inline double dot3(const DVec& a, const DVec& b)
{
    return a[XX] * b[XX] + a[YY] * b[YY] + a[ZZ] * b[ZZ];
}

/*! \brief Radial kernels B_0..B_{nmax-1} of erf(a r)/r, B_n = -(1/r d/dr) B_{n-1}:
 * a series for (a r)^2 < 1, the upward recursion elsewhere. */
void erfKernels(double a, double r, int nmax, double* B)
{
    const double x = a * r, s = x * x;
    const double c0 = 2.0 * a / c_sqrtPi, a2 = 2.0 * a * a;
    if (s < 1.0)
    {
        // B_n = c0 (2a^2)^n sum_m (-s)^m / (m! (2m+2n+1)), m = 0..14
        double t[15];
        t[0] = 1.0;
        for (int m = 1; m < 15; m++)
        {
            t[m] = t[m - 1] * (-s) / m;
        }
        double pw = 1.0;
        for (int k = 0; k < nmax; k++)
        {
            double sum = 0.0;
            for (int m = 14; m >= 0; m--)
            {
                sum += t[m] / (2 * m + 2 * k + 1);
            }
            B[k] = c0 * pw * sum;
            pw *= a2;
        }
    }
    else
    {
        const double c = c0 * std::exp(-s), ir2 = 1.0 / (r * r);
        B[0]           = std::erf(x) / r;
        double pw      = 1.0;
        for (int k = 1; k < nmax; k++)
        {
            B[k] = ((2 * k - 1) * B[k - 1] - pw * c) * ir2;
            pw *= a2;
        }
    }
}

//! Ewald coefficient beta with erfc(beta rc) = rtol, in double (calc_ewaldcoeff_q in real)
double ewaldCoefficient(double rc, double rtol)
{
    double beta = 5, low = 0, high;
    int    n = 0;
    do
    {
        n++;
        beta *= 2;
    } while (std::erfc(beta * rc) > rtol);
    n += 60;
    high = beta;
    for (int i = 0; i < n; i++)
    {
        beta = 0.5 * (low + high);
        if (std::erfc(beta * rc) > rtol)
        {
            low = beta;
        }
        else
        {
            high = beta;
        }
    }
    return beta;
}

//! |sum_k M_p(k+1) exp(2 pi i m k / K)|^2, zeros replaced by the mean of the neighbours
std::vector<double> bsplineModuli(int K, int p)
{
    std::vector<double> M(p + 1, 0.0), tmp(p + 1, 0.0);
    M[0] = 1.0;
    for (int k = 2; k <= p; k++)
    {
        for (int j = 0; j < k; j++)
        {
            const double cur = (j < k - 1 ? M[j] : 0.0), prev = (j > 0 ? M[j - 1] : 0.0);
            tmp[j]           = (j * cur + (k - j) * prev) / (k - 1);
        }
        for (int j = 0; j < k; j++)
        {
            M[j] = tmp[j];
        }
    }
    std::vector<double> data(K, 0.0);
    for (int k = 0; k < p - 1 && k < K; k++)
    {
        data[k] = M[k + 1];
    }
    std::vector<double> mod(K);
    for (int m = 0; m < K; m++)
    {
        double sc = 0, ss = 0;
        for (int k = 0; k < K; k++)
        {
            const double arg = 2 * M_PI * m * k / K;
            sc += data[k] * std::cos(arg);
            ss += data[k] * std::sin(arg);
        }
        mod[m] = sc * sc + ss * ss;
    }
    for (int m = 0; m < K; m++)
    {
        if (mod[m] < 1e-7)
        {
            mod[m] = 0.5 * (mod[(m - 1 + K) % K] + mod[(m + 1) % K]);
        }
    }
    return mod;
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
                   ArrayRef<const RVec>          shiftVectors,
                   ArrayRef<const int>           globalIndex,
                   ForceWithVirial*              forceWithVirial,
                   gmx_enerdata_t*               enerd);

    // settings
    double rc_, beta_, ke_, tol_, peek_;
    int    grid_[DIM], order_, maxIter_, predictor_;
    bool   polarizable_ = true;

    // topology
    int                 n_ = -1;
    std::vector<double> q_, radius_, alpha_;
    std::vector<int>    covAtom_, covPartner_;
    std::vector<double> covStrength_;

    // geometry
    struct Pair
    {
        int    i, j;
        DVec   x; // r_i - r_j (+ shift)
        double G[4];
    };
    std::vector<DVec> pos_;
    double            box_[DIM][DIM], recip_[DIM][DIM], volume_ = 0;
    std::vector<std::vector<Pair>> pairs_;   // per thread
    std::vector<std::array<int, 2>> ciItems_; // (pair list, i-cluster entry)

    // OpenMP: per-thread scratch for the scatter loops
    int                              nt_ = 1;
    std::vector<std::vector<DVec>>   threadVec_;
    std::vector<std::vector<double>> threadGrid_;
    //! sum the per-thread vectors 0..nt_-1 into out
    void reduceThreadVec(std::vector<DVec>& out);

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
    double tPairs_ = 0, tSplines_ = 0, tFieldPairs_ = 0, tPme_ = 0, tFinal_ = 0;
    double numCandidates_ = 0, numPairs_ = 0;
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
    std::vector<int>     gridIndex_[DIM];
    std::vector<double>  theta_[DIM], dtheta_[DIM], d2theta_[DIM];
    std::vector<double>  phi_; // reciprocal potential on the grid (real layout, nx*ny*nz)

    void setTopology(int n, ArrayRef<const real> charges, const InteractionDefinitions& idef);
    void setBox(const matrix box);
    void buildPairs(ArrayRef<const RVec> x, const nonbonded_verlet_t& nbv, ArrayRef<const RVec> shiftVectors);
    void minImage(DVec& v) const;
    void permanentDipoles();
    void splines();
    void computeInfluence();
    std::vector<DVec> fractionalDipoles(const std::vector<DVec>& d) const;
    double pmePotential(const std::vector<double>* q, const std::vector<DVec>& d, double* strain);
    void   pmeField(std::vector<DVec>& E) const;
    std::vector<DVec> field(const std::vector<double>* q, const std::vector<DVec>& d);
    void              solve();
    //! local -> global order (history), and back
    std::vector<DVec> toGlobal(const std::vector<DVec>& v) const
    {
        if (globalIndex_.empty())
        {
            return v;
        }
        std::vector<DVec> g(n_);
        for (int i = 0; i < n_; i++)
        {
            g[globalIndex_[i]] = v[i];
        }
        return g;
    }
    std::vector<DVec> toLocal(const std::vector<DVec>& g) const
    {
        if (globalIndex_.empty())
        {
            return g;
        }
        std::vector<DVec> v(n_);
        for (int i = 0; i < n_; i++)
        {
            v[i] = g[globalIndex_[i]];
        }
        return v;
    }

    size_t realIndex(int g1, int g2, int g3) const
    {
        return (static_cast<size_t>(g1) * rSize_[YY] + g2) * rSize_[ZZ] + g3;
    }
    size_t phiIndex(int g1, int g2, int g3) const
    {
        return (static_cast<size_t>(g1) * grid_[YY] + g2) * grid_[ZZ] + g3;
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
                "\npGM electrostatics (G1, CPU; double-precision kernels, PME grid in real):\n"
                "  rcoulomb %g nm, Ewald coefficient %.10g 1/nm, PME grid %d x %d x %d, order %d\n"
                "  Coulomb constant %.10g, dipole tolerance %g, max. iterations %d, predictor %d, "
                "peek %g\n\n",
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
                peek_);
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
            fprintf(stderr, "pGM pairs per call: %.0f cluster-list atom pairs, %.0f within the cut-off; %d threads\n",
                    numCandidates_ / numCalls_, numPairs_ / numCalls_, nt_);
        }
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

void PgmElectrostatics::Impl::setTopology(int n, ArrayRef<const real> charges, const InteractionDefinitions& idef)
{
    n_ = n;
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
    for (int a = 0; a < n; a++)
    {
        if (radius_[a] <= 0)
        {
            gmx_fatal(FARGS,
                      "pGM: atom %d has no [ pgm_atoms ] entry with a positive radius; every atom needs one",
                      a + 1);
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
    polarizable_ = std::any_of(alpha_.begin(), alpha_.end(), [](double a) { return a > 0; });
    if (static_cast<int>(mu_.size()) != n)
    {
        perm_.assign(n, DVec{ 0, 0, 0 });
        mu_.assign(n, DVec{ 0, 0, 0 });
        history_.clear();
        havePos_ = false;
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
    for (int c = ZZ; c >= XX; c--)
    {
        const double k = std::round(v[c] / box_[c][c]);
        for (int j = 0; j < DIM; j++)
        {
            v[j] -= k * box_[c][j];
        }
    }
}

void PgmElectrostatics::Impl::buildPairs(ArrayRef<const RVec>      x,
                                         const nonbonded_verlet_t& nbv,
                                         ArrayRef<const RVec>      shiftVectors)
{
    const double rc2   = rc_ * rc_;
    const auto&  lists = nbv.pairlistSets().pairlistSet(InteractionLocality::Local).cpuLists();
    if (lists.empty())
    {
        gmx_fatal(FARGS, "pGM electrostatics need the CPU pair lists (mdrun -nb cpu)");
    }
    ArrayRef<const int> atomOrder = nbv.getLocalAtomOrder();
    // work items: the i-cluster entries of all lists, dealt out dynamically; one pair vector per thread
    ciItems_.clear();
    for (int l = 0; l < lists.ssize(); l++)
    {
        for (int c = 0; c < static_cast<int>(lists[l].ci.size()); c++)
        {
            ciItems_.push_back({ l, c });
        }
    }
    const int numItems = ciItems_.size();
    for (int item = 0; item < numItems; item++)
    {
        const NbnxnPairlistCpu& nbl = lists[ciItems_[item][0]];
        const nbnxn_ci_t&       ci  = nbl.ci[ciItems_[item][1]];
        numCandidates_ += static_cast<double>(ci.cj_ind_end - ci.cj_ind_start) * nbl.na_ci * nbl.na_cj;
    }
    pairs_.resize(nt_);
    for (auto& v : pairs_)
    {
        v.clear();
    }
#pragma omp parallel num_threads(nt_)
    {
        // fill a local vector (no false sharing of the vector headers in pairs_), keep its capacity
        std::vector<Pair> pairs;
        pairs.swap(pairs_[gmx_omp_get_thread_num()]);
        double Bb[4];
#pragma omp for schedule(dynamic, 8)
        for (int item = 0; item < numItems; item++)
        {
            const NbnxnPairlistCpu& nbl     = lists[ciItems_[item][0]];
            const nbnxn_ci_t&       ciEntry = nbl.ci[ciItems_[item][1]];
            const int               naci = nbl.na_ci, nacj = nbl.na_cj;
            const int  ish     = ciEntry.shift & NBNXN_CI_SHIFT;
            const bool central = (ish == c_centralShiftIndex);
            const DVec sv(shiftVectors[ish][XX], shiftVectors[ish][YY], shiftVectors[ish][ZZ]);
            const int  i0 = ciEntry.ci * naci;
            for (int cjIndex = ciEntry.cj_ind_start; cjIndex < ciEntry.cj_ind_end; cjIndex++)
            {
                const int j0 = nbl.cj.cj(cjIndex) * nacj;
                // nbnxm counts the pairs of overlapping cluster pairs (the diagonal) once: j > i;
                // the exclusion masks are ignored: pGM electrostatics have no exclusions
                const bool overlap = central && i0 < j0 + nacj && j0 < i0 + naci;
                for (int ii = 0; ii < naci; ii++)
                {
                    const int ai = atomOrder[i0 + ii];
                    if (ai < 0)
                    {
                        continue;
                    }
                    for (int jj = 0; jj < nacj; jj++)
                    {
                        const int aj = atomOrder[j0 + jj];
                        if (aj < 0 || (overlap && j0 + jj <= i0 + ii) || aj == ai)
                        {
                            continue;
                        }
                        Pair p;
                        p.i = ai;
                        p.j = aj;
                        for (int d = 0; d < DIM; d++)
                        {
                            p.x[d] = (static_cast<double>(x[ai][d]) + sv[d]) - static_cast<double>(x[aj][d]);
                        }
                        const double r2 = dot3(p.x, p.x);
                        if (r2 >= rc2)
                        {
                            continue;
                        }
                        const double r = std::sqrt(r2);
                        const double a = 1.0
                                         / std::sqrt(2.0
                                                     * (radius_[ai] * radius_[ai] + radius_[aj] * radius_[aj]));
                        erfKernels(a, r, 4, p.G);
                        erfKernels(beta_, r, 4, Bb);
                        for (int k = 0; k < 4; k++)
                        {
                            p.G[k] -= Bb[k];
                        }
                        pairs.push_back(p);
                    }
                }
            }
        }
        pairs.swap(pairs_[gmx_omp_get_thread_num()]);
    }
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
}

void PgmElectrostatics::Impl::splines()
{
    const int p = order_;
    for (int a = 0; a < DIM; a++)
    {
        gridIndex_[a].resize(n_);
        theta_[a].resize(n_ * p);
        dtheta_[a].resize(n_ * p);
        d2theta_[a].resize(n_ * p);
    }
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int i = 0; i < n_; i++)
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
            gridIndex_[a][i] = base;
        }
    }
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
    std::vector<DVec> e(n_);
    for (int i = 0; i < n_; i++)
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
    std::fill(realGrid_, realGrid_ + static_cast<size_t>(rSize_[XX]) * rSize_[YY] * rSize_[ZZ], static_cast<real>(0));
    phi_.resize(static_cast<size_t>(K1) * K2 * K3);
    // spread in double on per-thread grids, sum them into the FFT grid
    const size_t      gridSize = static_cast<size_t>(K1) * K2 * K3;
    std::vector<DVec> e        = fractionalDipoles(d);
#pragma omp parallel num_threads(nt_)
    {
#pragma omp for schedule(static)
        for (int t = 0; t < nt_; t++) // all of them, also of threads that do not take part
        {
            threadGrid_[t].assign(gridSize, 0.0);
        }
        std::vector<double>& grid = threadGrid_[gmx_omp_get_thread_num()];
#pragma omp for schedule(static)
        for (int i = 0; i < n_; i++)
        {
            const double qi = (q ? (*q)[i] : 0.0);
            for (int j1 = 0; j1 < p; j1++)
            {
                const int    g1 = ((gridIndex_[XX][i] - j1) % K1 + K1) % K1;
                const double t1 = theta_[XX][i * p + j1], u1 = dtheta_[XX][i * p + j1];
                for (int j2 = 0; j2 < p; j2++)
                {
                    const int    g2 = ((gridIndex_[YY][i] - j2) % K2 + K2) % K2;
                    const double t2 = theta_[YY][i * p + j2], u2 = dtheta_[YY][i * p + j2];
                    const double a0 = qi * t1 * t2 + e[i][XX] * u1 * t2 + e[i][YY] * t1 * u2;
                    const double a3 = e[i][ZZ] * t1 * t2;
                    for (int j3 = 0; j3 < p; j3++)
                    {
                        const int g3 = ((gridIndex_[ZZ][i] - j3) % K3 + K3) % K3;
                        grid[phiIndex(g1, g2, g3)] += a0 * theta_[ZZ][i * p + j3] + a3 * dtheta_[ZZ][i * p + j3];
                    }
                }
            }
        }
#pragma omp for schedule(static)
        for (int g1 = 0; g1 < K1; g1++)
        {
            for (int g2 = 0; g2 < K2; g2++)
            {
                for (int g3 = 0; g3 < K3; g3++)
                {
                    double sum = 0;
                    for (int t = 0; t < nt_; t++)
                    {
                        sum += threadGrid_[t][phiIndex(g1, g2, g3)];
                    }
                    realGrid_[realIndex(g1, g2, g3)] = sum;
                }
            }
        }
    }
    fftExecute(GMX_FFT_REAL_TO_COMPLEX);

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
    fftExecute(GMX_FFT_COMPLEX_TO_REAL);
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int g1 = 0; g1 < K1; g1++)
    {
        for (int g2 = 0; g2 < K2; g2++)
        {
            for (int g3 = 0; g3 < K3; g3++)
            {
                phi_[phiIndex(g1, g2, g3)] = realGrid_[realIndex(g1, g2, g3)];
            }
        }
    }
    return energy;
}

void PgmElectrostatics::Impl::pmeField(std::vector<DVec>& E) const
{
    const int K1 = grid_[XX], K2 = grid_[YY], K3 = grid_[ZZ], p = order_;
#pragma omp parallel for num_threads(nt_) schedule(static)
    for (int i = 0; i < n_; i++)
    {
        double f[3] = { 0, 0, 0 };
        for (int j1 = 0; j1 < p; j1++)
        {
            const int    g1 = ((gridIndex_[XX][i] - j1) % K1 + K1) % K1;
            const double t1 = theta_[XX][i * p + j1], u1 = dtheta_[XX][i * p + j1];
            for (int j2 = 0; j2 < p; j2++)
            {
                const int    g2 = ((gridIndex_[YY][i] - j2) % K2 + K2) % K2;
                const double t2 = theta_[YY][i * p + j2], u2 = dtheta_[YY][i * p + j2];
                for (int j3 = 0; j3 < p; j3++)
                {
                    const int    g3 = ((gridIndex_[ZZ][i] - j3) % K3 + K3) % K3;
                    const double t3 = theta_[ZZ][i * p + j3], u3 = dtheta_[ZZ][i * p + j3];
                    const double ph = phi_[phiIndex(g1, g2, g3)];
                    f[0] += u1 * t2 * t3 * ph;
                    f[1] += t1 * u2 * t3 * ph;
                    f[2] += t1 * t2 * u3 * ph;
                }
            }
        }
        for (int k = 0; k < DIM; k++)
        {
            E[i][k] -= K1 * recip_[k][XX] * f[0] + K2 * recip_[k][YY] * f[1] + K3 * recip_[k][ZZ] * f[2];
        }
    }
}

std::vector<DVec> PgmElectrostatics::Impl::field(const std::vector<double>* q, const std::vector<DVec>& d)
{
    const double      t0 = now();
    std::vector<DVec> E(n_, DVec{ 0, 0, 0 });
    const int         numLists = pairs_.size();
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
            for (const Pair& p : pairs_[l])
            {
                const DVec&  x  = p.x;
                const double qi = (q ? (*q)[p.i] : 0.0), qj = (q ? (*q)[p.j] : 0.0);
                const DVec&  di = d[p.i];
                const DVec&  dj = d[p.j];
                const double djx = dot3(dj, x), dix = dot3(di, x);
                for (int k = 0; k < DIM; k++)
                {
                    Et[p.i][k] += qj * x[k] * p.G[1] - dj[k] * p.G[1] + djx * x[k] * p.G[2];
                    Et[p.j][k] += -qi * x[k] * p.G[1] - di[k] * p.G[1] + dix * x[k] * p.G[2];
                }
            }
        }
    }
    reduceThreadVec(E);
    double t1 = now();
    tFieldPairs_ += t1 - t0;
    pmePotential(q, d, nullptr);
    pmeField(E);
    tPme_ += now() - t1;
    const double cs = 4.0 * beta_ * beta_ * beta_ / (3.0 * c_sqrtPi);
    for (int i = 0; i < n_; i++)
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
    std::vector<DVec>          b  = field(&q_, perm_);
    const std::vector<double>& al = alpha_;
    double                     norm = 0;
    for (int i = 0; i < n_; i++)
    {
        for (int k = 0; k < DIM; k++)
        {
            norm += std::fabs(al[i] * b[i][k]);
        }
    }
    norm = norm / (3.0 * n_) + 1e-300;

    // initial guess (history in the global atom order)
    std::vector<DVec> x(n_, DVec{ 0, 0, 0 });
    const std::vector<DVec> posGlobal = toGlobal(pos_);
    const bool              samePos   = havePos_ && lastPos_ == posGlobal;
    const int               h         = history_.size();
    if (samePos && h > 0)
    {
        x = toLocal(history_[0]);
    }
    else if (predictor_ == 2 && h >= 4)
    {
        std::vector<DVec> g(n_);
        for (int i = 0; i < n_; i++)
        {
            g[i] = 4.0 * history_[0][i] - 6.0 * history_[1][i] + 4.0 * history_[2][i] - history_[3][i];
        }
        x = toLocal(g);
    }
    else if (predictor_ >= 1 && h >= 3)
    {
        std::vector<DVec> g(n_);
        for (int i = 0; i < n_; i++)
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
        for (int i = 0; i < n_; i++)
        {
            x[i] = al[i] * b[i];
        }
    }

    auto applyA = [&](const std::vector<DVec>& v) { // alpha^-1 v - T v (rows of polarizable atoms)
        std::vector<DVec> Tv = field(nullptr, v);
        std::vector<DVec> out(n_);
        for (int i = 0; i < n_; i++)
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
        for (int i = 0; i < n_; i++)
        {
            sum += dot3(a[i], c[i]);
        }
        return sum;
    };
    auto errOf = [&](const std::vector<DVec>& r) {
        double m = 0;
        for (int i = 0; i < n_; i++)
        {
            for (int k = 0; k < DIM; k++)
            {
                m = std::max(m, std::fabs(al[i] * r[i][k]));
            }
        }
        return m / norm;
    };
    // r = b - A x, Jacobi-preconditioned CG (Polak-Ribiere)
    std::vector<DVec> Ax = applyA(x), r(n_), z(n_), pdir(n_);
    for (int i = 0; i < n_; i++)
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
        std::vector<DVec> rnew(n_), znew(n_);
        for (int i = 0; i < n_; i++)
        {
            for (int k = 0; k < DIM; k++)
            {
                x[i][k] += alph * pdir[i][k];
                rnew[i][k] = r[i][k] - alph * Ap[i][k];
                znew[i][k] = al[i] * rnew[i][k];
            }
        }
        double beta = 0;
        for (int i = 0; i < n_; i++)
        {
            for (int k = 0; k < DIM; k++)
            {
                beta += znew[i][k] * (rnew[i][k] - r[i][k]);
            }
        }
        beta /= std::max(rz, 1e-300);
        for (int i = 0; i < n_; i++)
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
        for (int i = 0; i < n_; i++)
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

void PgmElectrostatics::Impl::calculate(ArrayRef<const RVec>          xIn,
                                        const matrix                  box,
                                        int                           numHomeAtoms,
                                        ArrayRef<const real>          charges,
                                        const nonbonded_verlet_t&     nbv,
                                        const InteractionDefinitions& idef,
                                        ArrayRef<const RVec>          shiftVectors,
                                        ArrayRef<const int>           globalIndex,
                                        ForceWithVirial*              forceWithVirial,
                                        gmx_enerdata_t*               enerd)
{
    // the local atom order changes at repartitioning (also on a single rank): parameters are read
    // from the local topology at every call, the predictor history is kept in the global order
    nt_ = std::max(1, gmx_omp_nthreads_get(ModuleMultiThread::Default));
    threadVec_.resize(nt_);
    threadGrid_.resize(nt_);
    if (fftThreads_ != nt_)
    {
        initFft(nt_);
    }
    setTopology(numHomeAtoms, charges, idef);
    globalIndex_.assign(globalIndex.begin(), globalIndex.begin() + (globalIndex.empty() ? 0 : n_));
    pos_.resize(n_);
    for (int i = 0; i < n_; i++)
    {
        pos_[i] = DVec(xIn[i][XX], xIn[i][YY], xIn[i][ZZ]);
    }
    setBox(box);
    permanentDipoles();
    double t0 = now();
    buildPairs(xIn, nbv, shiftVectors);
    const double tPairs1 = now();
    tPairs_ += tPairs1 - t0;
    for (const auto& v : pairs_)
    {
        numPairs_ += v.size();
    }
    if (numCalls_ == 0 && std::getenv("GMX_PGM_TIMING"))
    {
        fprintf(stderr, "pGM pairs per thread:");
        for (const auto& v : pairs_)
        {
            fprintf(stderr, " %zu", v.size());
        }
        fprintf(stderr, " (%d i-cluster items)\n", static_cast<int>(ciItems_.size()));
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

    std::vector<DVec> d(n_);
    for (int i = 0; i < n_; i++)
    {
        d[i] = perm_[i] + mu_[i];
    }
    const std::vector<double>& q = q_;
    double                     direct = 0, reciprocal = 0, self = 0, background = 0, polarization = 0;
    std::vector<DVec>          E(n_, DVec{ 0, 0, 0 }); // total field at fixed d [e/nm^2]
    std::vector<DVec>          F(n_, DVec{ 0, 0, 0 }); // forces at fixed d [e^2/nm^2]
    double                     S[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // strain derivative [e^2/nm]
    // pairs: per-thread forces in threadVec_, fields in threadE, strain and energy
    std::vector<std::vector<DVec>>      threadE(nt_, std::vector<DVec>(n_, DVec{ 0, 0, 0 }));
    std::array<double, 10>              threadSum{}; // S (9), direct energy: summed over threads
    const int                           numLists = pairs_.size();
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
            for (const Pair& p : pairs_[l])
            {
                const DVec&   x  = p.x;
                const double  qi = q[p.i], qj = q[p.j];
                const DVec&   di = d[p.i];
                const DVec&   dj = d[p.j];
                const double  dix = dot3(di, x), djx = dot3(dj, x), didj = dot3(di, dj);
                const double* G = p.G;
                sum[9] += qi * qj * G[0] + (qi * djx - qj * dix) * G[1] + didj * G[1] - dix * djx * G[2];
                const double sc =
                        -qi * qj * G[1] - (qi * djx - qj * dix) * G[2] - didj * G[2] + dix * djx * G[3];
                for (int k = 0; k < DIM; k++)
                {
                    const double dedx = sc * x[k] + (qi * dj[k] - qj * di[k]) * G[1]
                                        - (djx * di[k] + dix * dj[k]) * G[2];
                    for (int b = 0; b < DIM; b++)
                    {
                        sum[3 * k + b] += dedx * x[b];
                    }
                    Ft[p.i][k] -= dedx;
                    Ft[p.j][k] += dedx;
                    Et[p.i][k] += qj * x[k] * G[1] - dj[k] * G[1] + djx * x[k] * G[2];
                    Et[p.j][k] += -qi * x[k] * G[1] - di[k] * G[1] + dix * x[k] * G[2];
                }
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
    for (int k = 0; k < 9; k++)
    {
        S[k] += threadSum[k];
    }
    direct += threadSum[9];

    const double tPme0 = now();
    reciprocal         = pmePotential(&q, d, S);
    std::vector<DVec> Erec(n_, DVec{ 0, 0, 0 });
    pmeField(Erec);
    const double tPme1 = now();
    for (int i = 0; i < n_; i++)
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
    // reciprocal forces: -[q grad phi + (grad grad phi) . d]
    {
        const int         K1 = grid_[XX], K2 = grid_[YY], K3 = grid_[ZZ], p = order_;
        std::vector<DVec> e = fractionalDipoles(d);
#pragma omp parallel for num_threads(nt_) schedule(static)
        for (int i = 0; i < n_; i++)
        {
            double f1[3] = { 0, 0, 0 }, f2[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
            for (int j1 = 0; j1 < p; j1++)
            {
                const int    g1 = ((gridIndex_[XX][i] - j1) % K1 + K1) % K1;
                const double t1 = theta_[XX][i * p + j1], u1 = dtheta_[XX][i * p + j1],
                             v1 = d2theta_[XX][i * p + j1];
                for (int j2 = 0; j2 < p; j2++)
                {
                    const int    g2 = ((gridIndex_[YY][i] - j2) % K2 + K2) % K2;
                    const double t2 = theta_[YY][i * p + j2], u2 = dtheta_[YY][i * p + j2],
                                 v2 = d2theta_[YY][i * p + j2];
                    for (int j3 = 0; j3 < p; j3++)
                    {
                        const int    g3 = ((gridIndex_[ZZ][i] - j3) % K3 + K3) % K3;
                        const double t3 = theta_[ZZ][i * p + j3], u3 = dtheta_[ZZ][i * p + j3],
                                     v3 = d2theta_[ZZ][i * p + j3];
                        const double ph = phi_[phiIndex(g1, g2, g3)];
                        f1[0] += u1 * t2 * t3 * ph;
                        f1[1] += t1 * u2 * t3 * ph;
                        f1[2] += t1 * t2 * u3 * ph;
                        f2[0][0] += v1 * t2 * t3 * ph;
                        f2[1][1] += t1 * v2 * t3 * ph;
                        f2[2][2] += t1 * t2 * v3 * ph;
                        f2[0][1] += u1 * u2 * t3 * ph;
                        f2[0][2] += u1 * t2 * u3 * ph;
                        f2[1][2] += t1 * u2 * u3 * ph;
                    }
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
                    F[i][k] -= grid_[a] * recip_[k][a] * c;
                }
            }
        }
    }
    const double cs = 4.0 * beta_ * beta_ * beta_ / (3.0 * c_sqrtPi);
    double       q2 = 0, d2 = 0, Q = 0;
    for (int i = 0; i < n_; i++)
    {
        q2 += q[i] * q[i];
        d2 += dot3(d[i], d[i]);
        Q += q[i];
        for (int k = 0; k < DIM; k++)
        {
            E[i][k] += cs * d[i][k];
        }
    }
    self       = -(beta_ / c_sqrtPi) * q2 - 0.5 * cs * d2;
    background = -M_PI * Q * Q / (2.0 * volume_ * beta_ * beta_);
    for (int a = 0; a < DIM; a++)
    {
        S[4 * a] -= background; // E_bg ~ 1/V
    }
    for (int i = 0; i < n_; i++)
    {
        if (alpha_[i] > 0)
        {
            polarization += dot3(mu_[i], mu_[i]) / (2.0 * alpha_[i]);
        }
    }
    // covalent dipoles: dE/dr_partner = c (g - (g.u) u)/|v|, g = dE/dd_atom = -E_atom
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

    tFinal_ += (now() - tFinal0) - (tPme1 - tPme0);
    tPme_ += tPme1 - tPme0;

    // outputs
    ArrayRef<RVec> f = forceWithVirial->force_;
    for (int i = 0; i < n_; i++)
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

    static const char* dump = std::getenv("GMX_PGM_DUMP");
    if (dump && numCalls_ == 1)
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
                                  ArrayRef<const RVec>          shiftVectors,
                                  ArrayRef<const int>           globalIndex,
                                  ForceWithVirial*              forceWithVirial,
                                  gmx_enerdata_t*               enerd,
                                  const StepWorkload& /*stepWork*/)
{
    impl_->calculate(x, box, numHomeAtoms, charges, nbv, idef, shiftVectors, globalIndex, forceWithVirial, enerd);
}

ArrayRef<const DVec> PgmElectrostatics::inducedDipoles() const
{
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

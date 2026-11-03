#ifndef PGM_CORE_H_
#define PGM_CORE_H_

/* pgm_core: a standalone double-precision implementation of pGM electrostatics.
 *
 * Gaussian charges, covalent permanent dipoles and Gaussian induced dipoles; smooth PME for
 * periodic systems or plain all-pairs sums without a box.  It is the reference implementation of
 * the OpenMM plugin (Reference platform) and of the GROMACS code, written from docs/spec.md and
 * checked against pGM-JAX.  No dependencies beyond the C++ standard library and pocketfft.
 *
 * Units: nm, e, e nm, nm^3, kJ/mol (the Coulomb constant is a parameter).
 */

#include <array>
#include <deque>
#include <vector>

namespace pgmcore {

typedef std::array<double, 3> V3;

struct Settings {
    bool periodic = false;      // PME (true) or no cutoff / no box (false)
    double cutoff = 0.9;        // direct-space cutoff [nm] (periodic only)
    double beta = 0.0;          // Ewald coefficient [1/nm]
    int grid[3] = {0, 0, 0};    // PME grid
    int order = 6;              // B-spline order
    bool polarizable = true;    // induced dipoles on / off
    double tol = 1e-5;          // max|alpha r| / mean|alpha E_perm|
    int maxIter = 100;
    double peek = 0.65;         // final mu += peek alpha r
    int predictor = 2;          // 0 none, 1 mu3, 2 mu4
    double ke = 138.935456;     // Coulomb constant [kJ mol^-1 nm e^-2]
};

struct Topology {
    std::vector<double> q, radius, alpha;   // per atom [e], [nm], [nm^3]
    std::vector<int> covAtom, covPartner;   // covalent dipoles: p_atom += c unit(r_partner - r_atom)
    std::vector<double> covStrength;        // [e nm]
};

struct EnergyTerms {
    double direct = 0, reciprocal = 0, self = 0, background = 0, polarization = 0;
    double total() const { return direct+reciprocal+self+background+polarization; }
};

class Engine {
public:
    Engine(const Topology& top, const Settings& settings);
    /**
     * Energy [kJ/mol] and, if forces != nullptr, forces [kJ/mol/nm] (added to *forces).
     * box: the three box vectors (OpenMM's reduced form: a = (ax,0,0), b = (bx,by,0), c = (cx,cy,cz)),
     * ignored for non-periodic systems.
     */
    double compute(const std::vector<V3>& pos, const V3 box[3], std::vector<V3>* forces);
    /**
     * Key of the predictor history (an MD step count; -1: none).  The history holds one solution per key:
     * a later evaluation with the same key (a barostat trial, an energy query) starts from that solution
     * and leaves the history alone; without a key, evaluations at unchanged positions do the same.
     */
    void setHistoryKey(long long key) { historyKey = key; }
    const std::vector<V3>& permanentDipoles() const { return perm; }
    const std::vector<V3>& inducedDipoles() const { return mu; }
    int iterations() const { return lastIterations; }
    double residual() const { return lastResidual; }
    const EnergyTerms& terms() const { return lastTerms; }
    /**
     * Strain derivative S_ab = dE/d eps_ab [kJ/mol] of the last compute() with forces (atomic: every
     * position and the box deformed as r -> (1 + eps) r, induced dipoles fixed (E is variational in
     * them), covalent dipoles following their bonds).  Row-major 3x3.  GROMACS's virial is
     * Xi = S^T / 2 (Xi = -1/2 sum r (x) F for pair forces); pressure P = 2/(3V) (E_kin - Xi).
     */
    const std::array<double, 9>& strainDerivative() const { return lastStrain; }
    void setTopology(const Topology& top);   // new parameters, same atoms (predictor history kept)
    void resetHistory() { history.clear(); havePos = false; }
    /** Field [e/nm^2] of charges q and dipoles d at every atom (direct + reciprocal + self), for tests. */
    std::vector<V3> field(const std::vector<double>& q, const std::vector<V3>& d);
    /** Prepare geometry (pairs, splines) for positions and box; compute() calls it. */
    void setGeometry(const std::vector<V3>& pos, const V3 box[3]);

private:
    struct Pair { int i, j; V3 x; double G[4]; };
    Topology top;
    Settings s;
    int n;
    std::vector<V3> perm, mu;
    std::deque<std::vector<V3> > history;
    std::vector<V3> lastPos;
    bool havePos = false;
    long long historyKey = -1, lastKey = -1;
    int lastIterations = 0;
    double lastResidual = 0;
    EnergyTerms lastTerms;
    std::array<double, 9> lastStrain{};
    // geometry
    std::vector<V3> pos;
    V3 box[3], recip[3];   // recip[k][a] = (H^-1)_{k a}: fractional coordinate a = sum_k r_k recip[k][a]
    double volume = 0;
    std::vector<Pair> pairs;
    // PME
    std::vector<int> gridIndex[3];                    // per atom, per dimension: base index floor(w)
    std::vector<double> theta[3], dtheta[3], d2theta[3];   // per atom * order
    std::vector<double> influence;                    // half spectrum, nx*ny*(nz/2+1)
    void perm_dipoles();
    void build_pairs();
    void setup_pme();
    void compute_influence();
    // reciprocal-space potential on the grid of charges q and dipoles d; returns the energy [e^2/nm]
    // with strain != nullptr, adds the strain derivative of the energy at fixed fractional positions and
    // fixed fractional dipoles: (1/2) sum_m G |F|^2 [-delta + 2 (pi^2/beta^2 + 1/m^2) m m] [e^2/nm]
    double pme_potential(const std::vector<double>* q, const std::vector<V3>& d, std::vector<double>& phi, double* strain = nullptr);
    void pme_field(const std::vector<double>& phi, std::vector<V3>& E) const;   // adds -grad phi
    std::vector<V3> frac_dipoles(const std::vector<V3>& d) const;
    void solve();
};

/** Radial kernels B_0..B_{nmax-1} of erf(a r)/r: series for (a r)^2 < 1, closed form elsewhere. */
void erfKernels(double a, double r, int nmax, double* B);

} // namespace pgmcore

#endif

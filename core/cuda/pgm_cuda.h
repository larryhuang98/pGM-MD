#ifndef PGM_CUDA_H_
#define PGM_CUDA_H_

/* pgm_cuda: the pGM electrostatics on an NVIDIA GPU, for engines that do their own pair search.
 *
 * The caller provides the atoms and the covalent dipoles; the pairs come from the library's own list
 * (cell list on the GPU, minimum image, radius cutoff + skin, rebuilt when an atom has moved by more than
 * half the skin) or from the caller's candidates (setCandidates).  The library does the direct-space
 * pair kernels, the smooth PME of charges and dipoles
 * (cuFFT), the Jacobi-preconditioned CG of the induced dipoles, energies, forces and the strain
 * derivative, as core/pgm_core.cpp (docs/spec.md) does on the CPU.
 *
 * Precision (as OpenMM's mixed precision): pair kernels, splines and the PME grid in float; fields,
 * forces and the PME grid accumulated in 64-bit fixed point; the CG vectors, dipoles, energies and the
 * strain derivative in double.  The predictor history (global atom order) is kept on the GPU.
 *
 * Units: nm, e, e nm, nm^3; energies, forces and the strain derivative in kJ/mol(/nm) with the given
 * Coulomb constant.  Not thread safe: one thread at a time per engine (any thread).
 */

#include <array>

namespace pgmcuda {

struct Settings {
    double cutoff = 0.9;     // direct-space cutoff [nm]
    double beta = 3.5;       // Ewald coefficient [1/nm]
    int grid[3] = {36, 36, 36};
    int order = 6;           // B-spline order, 3..12
    double tol = 1e-5;       // max|alpha r| / mean|alpha b|
    int maxIter = 100;
    double peek = 0.65;
    int predictor = 2;       // 0 none, 1 mu3, 2 mu4 (history in the global order)
    double ke = 138.935458;  // Coulomb constant [kJ mol^-1 nm e^-2]
    int device = 0;          // < 0: the CUDA context current on the calling thread (e.g. OpenMM's), never
                             // switched
    double skin = 0.15;      // own pair list: radius cutoff + skin (less in small boxes), rebuilt when atoms
                             // moved > skin / 2 [nm]
};

struct Terms {
    double direct = 0, reciprocal = 0, self = 0, background = 0, polarization = 0;
    double gvdw = 0;   // the combination-rule GVDW (setGvdw)
    double total() const { return direct + reciprocal + self + background + polarization + gvdw; }
};

class Engine {
public:
    explicit Engine(const Settings& settings);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /** The atoms in the caller's (local) order; globalIndex[i] (nullptr: i) keys the predictor history. */
    void setAtoms(int n, const double* q, const double* radius, const double* alpha, const int* globalIndex,
                  int numGlobal);
    /**
     * GVDW (core/gvdw_kernel.h) by the combination rule for all pairs within rvdw except the skipped ones
     * (exclusions, pairs with their own parameters: those and the dispersion tail are the caller's):
     * per atom sqrt(A), sqrt(C6), b (local order; zeros: none), repulsion 0 Gaussian, 1 Slater; skip pairs
     * (skipI[k], skipJ[k]) once each, local indices.  Its energy is Terms::gvdw, its forces and strain are
     * added to those of compute().  After every setAtoms.
     */
    void setGvdw(int repulsion, double rvdw, const double* sqrtA, const double* sqrtC6, const double* b, int nskip,
                 const int* skipI, const int* skipJ);
    /** Covalent dipoles: p_atom += strength unit(r_partner - r_atom), local indices. */
    void setCovalent(int ncov, const int* atom, const int* partner, const double* strength);
    /**
     * Optional (instead of the own pair list, until the next setAtoms). Pair candidates: segment s holds the pairs (segI[s], candJ[k]), k in [segBegin[s], segBegin[s+1]),
     * with x = r_i + t.box - r_j, t = (segShift[3s], segShift[3s+1], segShift[3s+2]) integer box
     * vectors.  Every pair at most once; those beyond the cut-off are skipped at every step.
     */
    void setCandidates(int nseg, const int* segI, const int* segShift, const int* segBegin, const int* candJ);
    /**
     * Energy terms; forces added to f (n*3, may be nullptr), strain derivative S_ab = dE/d eps_ab added
     * to strain (9, may be nullptr; GROMACS virial = S^T/2).  box: rows a, b, c (reduced triclinic).
     * newStep: this evaluation starts a new MD step (its solution enters the predictor history).
     */
    Terms compute(const double* pos, const double box[3][3], bool newStep, double* f, double* strain);
    /** compute() with float positions (3n), the forces left in forces() (3n, kJ/mol/nm, valid until the
     * next call): no copies for callers in single precision. */
    Terms computeF(const float* pos, const double box[3][3], bool newStep, double* strain);
    const double* forces() const;
    /**
     * compute() on device data of a caller on the same GPU context (OpenMM): positions xyz from the
     * device array posq4 (float4, w ignored, local order), forces added atomically to the device fixed-point
     * buffer forceFixed (x of atom i at i, y at i + stride, z at i + 2 stride, scale fixedScale); stream:
     * the caller's stream (cudaStream_t), whose work so far comes first.  Returns after the evaluation.
     */
    Terms computeDevice(const void* posq4, const double box[3][3], bool newStep, long long* forceFixed, int stride,
                        double fixedScale, void* stream);

    int iterations() const;
    double residual() const;
    /** Induced / permanent dipoles of the last compute (n*3, e nm). */
    void inducedDipoles(double* mu) const;
    void permanentDipoles(double* p) const;
    void resetHistory();
    /** Builds of the own pair list so far; candidates (full list, both directions) of the current one. */
    long listBuilds() const;
    int numCandidates() const;

private:
    struct Impl;
    Impl* impl;
};

} // namespace pgmcuda

#endif

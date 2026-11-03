#ifndef REFERENCE_GVDW_KERNELS_H_
#define REFERENCE_GVDW_KERNELS_H_

#include "PGMKernels.h"
#include "openmm/Vec3.h"
#include <set>
#include <vector>

namespace PGMPlugin {

/**
 * Reference implementation of GVDWForce in double precision (core/gvdw_kernel.h): all pairs of the atoms
 * that carry GVDW (a or c nonzero), minimum image in the reduced triclinic box; exceptions one by one.
 * O(N_GVDW^2): for validation and small systems.
 */
class ReferenceCalcGVDWForceKernel : public CalcGVDWForceKernel {
public:
    ReferenceCalcGVDWForceKernel(std::string name, const OpenMM::Platform& platform) : CalcGVDWForceKernel(name, platform) {
    }
    void initialize(const OpenMM::System& system, const GVDWForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const GVDWForce& force);

private:
    void setParameters(const GVDWForce& force);
    int method, rep;
    double cutoff, sumC;
    bool dispersionCorrection, exceptionsPeriodic;
    std::vector<double> R2, sqrtA, sqrtC6, b;
    std::vector<int> active;                       // atoms with a or c nonzero
    std::vector<std::set<int> > activeExclusions;  // exception pairs, in the indices of `active`
    struct Exception {
        int i, j;
        double A, C6, B;
    };
    std::vector<Exception> exceptions;             // the ones with A or C6 nonzero
    std::vector<std::pair<int, int> > exceptionPairs;   // all of them (fixed after initialize)
    std::vector<int> evaluated;                    // the force's indices of the ones with parameters
};

} // namespace PGMPlugin

#endif

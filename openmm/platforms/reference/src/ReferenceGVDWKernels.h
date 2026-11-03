#ifndef REFERENCE_GVDW_KERNELS_H_
#define REFERENCE_GVDW_KERNELS_H_

#include "PGMKernels.h"
#include <set>
#include <vector>

namespace PGMPlugin {

/**
 * Reference implementation of GVDWForce: double precision, all pairs of the atoms that carry GVDW
 * (a or c nonzero), minimum image in OpenMM's reduced triclinic box.
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
    double cutoff;
    bool dispersionCorrection;
    double sumC;
    std::vector<double> R2, sqrtA, sqrtC6, b;
    std::vector<int> active;                  // atoms with a or c nonzero
    std::vector<std::set<int> > exclusions;
};

} // namespace PGMPlugin

#endif

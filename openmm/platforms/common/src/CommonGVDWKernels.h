#ifndef COMMON_GVDW_KERNELS_H_
#define COMMON_GVDW_KERNELS_H_

#include "PGMKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"

namespace PGMPlugin {

/**
 * GVDWForce on the Common platforms (CUDA, OpenCL, HIP): combination-rule pairs in OpenMM's nonbonded
 * tile kernel (NonbondedUtilities; every exception pair excluded there), exceptions with parameters as
 * a BondedUtilities interaction, the dispersion correction on the host.  Device code:
 * kernels/gvdwPair.cc (the pair), gvdwNonbonded.cc, gvdwExceptions.cc.
 */
class CommonCalcGVDWForceKernel : public CalcGVDWForceKernel {
public:
    CommonCalcGVDWForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcGVDWForceKernel(name, platform), cc(cc), system(system) {
    }
    void initialize(const OpenMM::System& system, const GVDWForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const GVDWForce& force);

private:
    class ForceInfo;
    void uploadParameters(const GVDWForce& force);
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    ForceInfo* info = nullptr;
    OpenMM::ComputeArray params;            // per atom (R^2, sqrt(A), sqrt(C6), b), original atom order
    OpenMM::ComputeArray exceptionParams;   // per exception with parameters (A, C6, B, b_ij)
    std::vector<int> exceptionIndex;        // the force's index of each of them
    std::vector<std::pair<int, int> > exceptionPairs;   // all exceptions (the excluded pairs)
    bool dispersionCorrection = false;
    double cutoff = 0, sumC = 0;
};

} // namespace PGMPlugin

#endif

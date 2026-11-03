#ifndef COMMON_GVDW_KERNELS_H_
#define COMMON_GVDW_KERNELS_H_

#include "PGMKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"

namespace PGMPlugin {

/**
 * GVDWForce on the Common platforms (CUDA, OpenCL, HIP): a pair interaction in OpenMM's
 * nonbonded tile kernel (NonbondedUtilities, with exclusions), the dispersion correction on the host.
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
    OpenMM::ComputeArray params;   // per atom (R^2, sqrt(A), sqrt(C6), b), in the context's atom order
    ForceInfo* info = nullptr;
    bool dispersionCorrection = false;
    double cutoff = 0, sumC = 0;
};

} // namespace PGMPlugin

#endif

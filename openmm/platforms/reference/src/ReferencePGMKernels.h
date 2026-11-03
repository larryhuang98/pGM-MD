#ifndef REFERENCE_PGM_KERNELS_H_
#define REFERENCE_PGM_KERNELS_H_

#include "PGMKernels.h"
#include "pgm_core.h"
#include <memory>

namespace PGMPlugin {

/**
 * Reference implementation of PGMForce: pgmcore::Engine (double precision, O(N^2) pair search).
 */
class ReferenceCalcPGMForceKernel : public CalcPGMForceKernel {
public:
    ReferenceCalcPGMForceKernel(std::string name, const OpenMM::Platform& platform) : CalcPGMForceKernel(name, platform) {
    }
    void initialize(const OpenMM::System& system, const PGMForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getPermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    int getInducedDipoleIterations();
    void copyParametersToContext(OpenMM::ContextImpl& context, const PGMForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

private:
    pgmcore::Topology top;
    pgmcore::Settings settings;
    std::unique_ptr<pgmcore::Engine> engine;
    bool computed = false;
};

} // namespace PGMPlugin

#endif

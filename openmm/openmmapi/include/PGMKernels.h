#ifndef PGM_KERNELS_H_
#define PGM_KERNELS_H_

#include "PGMForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <string>
#include <vector>

namespace PGMPlugin {

/**
 * The kernel of PGMForce: energy, forces and induced dipoles.
 */
class CalcPGMForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcPGMForce";
    }
    CalcPGMForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    virtual void initialize(const OpenMM::System& system, const PGMForce& force) = 0;
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    virtual void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles) = 0;
    virtual void getPermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles) = 0;
    virtual int getInducedDipoleIterations() = 0;
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const PGMForce& force) = 0;
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};

} // namespace PGMPlugin

#endif /*PGM_KERNELS_H_*/

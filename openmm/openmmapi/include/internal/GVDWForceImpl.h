#ifndef OPENMM_GVDWFORCEIMPL_H_
#define OPENMM_GVDWFORCEIMPL_H_

#include "GVDWForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace PGMPlugin {

class GVDWForceImpl : public OpenMM::ForceImpl {
public:
    GVDWForceImpl(const GVDWForce& owner);
    ~GVDWForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const GVDWForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>();
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
    /** sum_i sqrt(C6_i), for the dispersion correction */
    static double sumSqrtC6(const GVDWForce& force);

private:
    const GVDWForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace PGMPlugin

#endif /*OPENMM_GVDWFORCEIMPL_H_*/

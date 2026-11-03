#ifndef OPENMM_PGMFORCEIMPL_H_
#define OPENMM_PGMFORCEIMPL_H_

#include "PGMForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace PGMPlugin {

class PGMForceImpl : public OpenMM::ForceImpl {
public:
    PGMForceImpl(const PGMForce& owner);
    ~PGMForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const PGMForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>();
    }
    std::vector<std::string> getKernelNames();
    std::vector<std::pair<int, int> > getBondedParticles() const;
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getPermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    int getInducedDipoleIterations(OpenMM::ContextImpl& context);
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /** Ewald coefficient and grid for a tolerance, cutoff and box (NonbondedForce's rule). */
    static void calcPMEParameters(const OpenMM::System& system, const PGMForce& force, double& alpha, int& xsize, int& ysize, int& zsize);

private:
    const PGMForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace PGMPlugin

#endif /*OPENMM_PGMFORCEIMPL_H_*/

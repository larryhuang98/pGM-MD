#include "TailCorrectionForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include <map>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace std;

namespace PGMPlugin {

/** No kernel: the energy from the box, on the host. */
class TailCorrectionForceImpl : public ForceImpl {
public:
    TailCorrectionForceImpl(const TailCorrectionForce& owner) : owner(owner) {
    }
    void initialize(ContextImpl& context) {
    }
    const TailCorrectionForce& getOwner() const {
        return owner;
    }
    void updateContextState(ContextImpl& context, bool& forcesInvalid) {
    }
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
        if (!includeEnergy || (groups&(1<<owner.getForceGroup())) == 0)
            return 0.0;
        Vec3 a, b, c;
        context.getPeriodicBoxVectors(a, b, c);
        return owner.getCoefficient()/(a[0]*b[1]*c[2]);
    }
    map<string, double> getDefaultParameters() {
        return map<string, double>();
    }
    vector<string> getKernelNames() {
        return vector<string>();
    }

private:
    const TailCorrectionForce& owner;
};

ForceImpl* TailCorrectionForce::createImpl() const {
    return new TailCorrectionForceImpl(*this);
}

} // namespace PGMPlugin

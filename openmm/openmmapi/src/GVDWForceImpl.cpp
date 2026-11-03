#include "internal/GVDWForceImpl.h"
#include "PGMKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <set>

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

GVDWForceImpl::GVDWForceImpl(const GVDWForce& owner) : owner(owner) {
}

GVDWForceImpl::~GVDWForceImpl() {
}

void GVDWForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    int n = owner.getNumParticles();
    if (n != system.getNumParticles())
        throw OpenMMException("GVDWForce must have exactly as many particles as the System it belongs to.");
    set<pair<int, int> > seen;
    for (int i = 0; i < owner.getNumExclusions(); i++) {
        int a, b;
        owner.getExclusionParticles(i, a, b);
        if (a < 0 || a >= n || b < 0 || b >= n || a == b)
            throw OpenMMException("GVDWForce: illegal particle index in an exclusion");
        if (!seen.insert(make_pair(min(a, b), max(a, b))).second)
            throw OpenMMException("GVDWForce: the same pair is excluded twice");
    }
    if (owner.getNonbondedMethod() == GVDWForce::CutoffPeriodic) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("GVDWForce: the cutoff distance cannot be greater than half the periodic box size.");
    }
    kernel = context.getPlatform().createKernel(CalcGVDWForceKernel::Name(), context);
    kernel.getAs<CalcGVDWForceKernel>().initialize(system, owner);
}

double GVDWForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcGVDWForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> GVDWForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcGVDWForceKernel::Name());
    return names;
}

void GVDWForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcGVDWForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

double GVDWForceImpl::sumSqrtC6(const GVDWForce& force) {
    double sum = 0;
    for (int i = 0; i < force.getNumParticles(); i++) {
        double R, a, c, b;
        force.getParticleParameters(i, R, a, c, b);
        sum += c;
    }
    return sum;
}

#include "ReferencePGMKernels.h"
#include "internal/PGMForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

static void fillTopology(const PGMForce& force, pgmcore::Topology& top) {
    int n = force.getNumParticles();
    top.q.resize(n);
    top.radius.resize(n);
    top.alpha.resize(n);
    for (int i = 0; i < n; i++)
        force.getParticleParameters(i, top.q[i], top.radius[i], top.alpha[i]);
    int m = force.getNumCovalentDipoles();
    top.covAtom.resize(m);
    top.covPartner.resize(m);
    top.covStrength.resize(m);
    for (int i = 0; i < m; i++)
        force.getCovalentDipoleParameters(i, top.covAtom[i], top.covPartner[i], top.covStrength[i]);
}

void ReferenceCalcPGMForceKernel::initialize(const System& system, const PGMForce& force) {
    fillTopology(force, top);
    settings.periodic = (force.getNonbondedMethod() == PGMForce::PME);
    settings.cutoff = force.getCutoffDistance();
    settings.order = force.getPMEOrder();
    settings.polarizable = (force.getPolarization() == PGMForce::Mutual);
    settings.tol = force.getInducedDipoleTolerance();
    settings.maxIter = force.getMaxInducedDipoleIterations();
    settings.peek = force.getPeekCoefficient();
    settings.predictor = (int) force.getPredictor();
    settings.ke = force.getCoulombConstant();
    if (settings.periodic) {
        double alpha;
        int nx, ny, nz;
        PGMForceImpl::calcPMEParameters(system, force, alpha, nx, ny, nz);
        settings.beta = alpha;
        settings.grid[0] = nx;
        settings.grid[1] = ny;
        settings.grid[2] = nz;
    }
    engine.reset(new pgmcore::Engine(top, settings));
}

double ReferenceCalcPGMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    int n = pos.size();
    vector<pgmcore::V3> p(n);
    for (int i = 0; i < n; i++)
        p[i] = {pos[i][0], pos[i][1], pos[i][2]};
    pgmcore::V3 box[3];
    if (settings.periodic) {
        Vec3* b = extractBoxVectors(context);
        for (int i = 0; i < 3; i++)
            box[i] = {b[i][0], b[i][1], b[i][2]};
    }
    vector<pgmcore::V3> f(n, pgmcore::V3{0, 0, 0});
    double energy = engine->compute(p, box, includeForces ? &f : nullptr);
    computed = true;
    if (includeForces) {
        vector<Vec3>& force = extractForces(context);
        for (int i = 0; i < n; i++)
            force[i] += Vec3(f[i][0], f[i][1], f[i][2]);
    }
    return energy;
}

void ReferenceCalcPGMForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    if (!computed)
        execute(context, false, true);
    const vector<pgmcore::V3>& mu = engine->inducedDipoles();
    dipoles.resize(mu.size());
    for (size_t i = 0; i < mu.size(); i++)
        dipoles[i] = Vec3(mu[i][0], mu[i][1], mu[i][2]);
}

void ReferenceCalcPGMForceKernel::getPermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    vector<Vec3>& pos = extractPositions(context);
    int n = pos.size();
    vector<pgmcore::V3> p(n);
    for (int i = 0; i < n; i++)
        p[i] = {pos[i][0], pos[i][1], pos[i][2]};
    pgmcore::V3 box[3];
    if (settings.periodic) {
        Vec3* b = extractBoxVectors(context);
        for (int i = 0; i < 3; i++)
            box[i] = {b[i][0], b[i][1], b[i][2]};
    }
    engine->setGeometry(p, box);
    const vector<pgmcore::V3>& perm = engine->permanentDipoles();
    dipoles.resize(n);
    for (int i = 0; i < n; i++)
        dipoles[i] = Vec3(perm[i][0], perm[i][1], perm[i][2]);
}

int ReferenceCalcPGMForceKernel::getInducedDipoleIterations() {
    return engine->iterations();
}

void ReferenceCalcPGMForceKernel::copyParametersToContext(ContextImpl& context, const PGMForce& force) {
    if (force.getNumParticles() != (int) top.q.size())
        throw OpenMMException("updateParametersInContext: the number of particles has changed");
    if (force.getNumCovalentDipoles() != (int) top.covAtom.size())
        throw OpenMMException("updateParametersInContext: the number of covalent dipoles has changed");
    pgmcore::Topology t;
    fillTopology(force, t);
    for (size_t i = 0; i < t.covAtom.size(); i++)
        if (t.covAtom[i] != top.covAtom[i] || t.covPartner[i] != top.covPartner[i])
            throw OpenMMException("updateParametersInContext: the atoms of a covalent dipole have changed");
    top = t;
    engine->setTopology(top);
}

void ReferenceCalcPGMForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = settings.beta;
    nx = settings.grid[0];
    ny = settings.grid[1];
    nz = settings.grid[2];
}

#include "CudaPGMKernels.h"
#include "CommonPGMKernels.h"
#include "internal/PGMForceImpl.h"
#include "pgm_cuda.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <cstdlib>

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

class CudaCalcPGMForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const PGMForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double q1, r1, a1, q2, r2, a2;
        force.getParticleParameters(particle1, q1, r1, a1);
        force.getParticleParameters(particle2, q2, r2, a2);
        return (q1 == q2 && r1 == r2 && a1 == a2);
    }
    int getNumParticleGroups() {
        return force.getNumCovalentDipoles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int a, b;
        double c;
        force.getCovalentDipoleParameters(index, a, b, c);
        particles = {a, b};
    }
    bool areGroupsIdentical(int group1, int group2) {
        int a1, b1, a2, b2;
        double c1, c2;
        force.getCovalentDipoleParameters(group1, a1, b1, c1);
        force.getCovalentDipoleParameters(group2, a2, b2, c2);
        return (c1 == c2);
    }
private:
    const PGMForce& force;
};

class CudaCalcPGMForceKernel::ReorderListener : public ComputeContext::ReorderListener {
public:
    ReorderListener(CudaCalcPGMForceKernel& owner) : owner(owner) {
    }
    void execute() {
        owner.atomsDirty = true;   // the engine gets the new order (its list is renamed, the history is global)
    }
private:
    CudaCalcPGMForceKernel& owner;
};

/**
 * The engine runs on its own stream beside the rest of the force computation (OpenMM nonbonded kernels and
 * other forces); after all forces were launched, before the force buffers are reduced, the host waits for
 * it.  A pair list found out of date then marks the forces invalid: OpenMM computes them again (the list
 * rebuilt).  PGM_OPENMM_DEFER=1: at steps without energies the context stream waits on the GPU instead and
 * the list check comes with the next evaluation (Engine::waitDevice; 1 percent faster, 4096 waters).
 */
class CudaCalcPGMForceKernel::PostComputation : public ComputeContext::ForcePostComputation {
public:
    PostComputation(CudaCalcPGMForceKernel& owner) : owner(owner) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if (!owner.launched)
            return 0.0;
        if (owner.deferred && !owner.wantEnergy && !includeEnergy) {
            owner.launched = false;
            owner.evaluated = true;
            ContextSelector selector(owner.cu);
            owner.engine->waitDevice((void*) owner.cu.getCurrentStream());
            return 0.0;
        }
        return owner.finish(includeEnergy);
    }
private:
    CudaCalcPGMForceKernel& owner;
};

CudaCalcPGMForceKernel::CudaCalcPGMForceKernel(string name, const Platform& platform, CudaContext& cu,
                                               const System& system)
        : CalcPGMForceKernel(name, platform), cu(cu), system(system) {
}

CudaCalcPGMForceKernel::~CudaCalcPGMForceKernel() {
    if (engine) {
        ContextSelector selector(cu);
        engine.reset();
    }
}

void CudaCalcPGMForceKernel::initialize(const System& system, const PGMForce& force) {
    const bool useEngine = cu.getUseMixedPrecision() && force.getNonbondedMethod() == PGMForce::PME
                           && getenv("PGM_OPENMM_COMMON") == nullptr;
    if (!useEngine) {
        common.reset(new CommonCalcPGMForceKernel(getName(), getPlatform(), cu, system));
        common->initialize(system, force);
        return;
    }
    ContextSelector selector(cu);
    numParticles = force.getNumParticles();
    forceGroup = force.getForceGroup();
    polarizable = (force.getPolarization() == PGMForce::Mutual);
    PGMForceImpl::calcPMEParameters(system, force, beta, gridX, gridY, gridZ);
    pgmcuda::Settings s;
    s.cutoff = force.getCutoffDistance();
    s.beta = beta;
    s.grid[0] = gridX;
    s.grid[1] = gridY;
    s.grid[2] = gridZ;
    s.order = force.getPMEOrder();
    s.tol = force.getInducedDipoleTolerance();
    s.maxIter = force.getMaxInducedDipoleIterations();
    s.peek = force.getPeekCoefficient();
    s.predictor = (int) force.getPredictor();   // the same numbering: none, mu3, mu4
    s.ke = force.getCoulombConstant();
    s.device = -1;   // OpenMM's context (current through the ContextSelector)
    if (const char* sk = getenv("PGM_OPENMM_SKIN"))
        s.skin = atof(sk);
    engine.reset(new pgmcuda::Engine(s));
    loadParameters(force);
    info = new ForceInfo(force);
    cu.addForce(info);
    cu.addReorderListener(new ReorderListener(*this));
    cu.addPostComputation(new PostComputation(*this));
    deferred = (getenv("PGM_OPENMM_DEFER") != nullptr);
}

void CudaCalcPGMForceKernel::setAtoms() {
    // local index i holds atom order[i]
    const vector<int>& order = cu.getAtomIndex();
    const int n = numParticles;
    vector<double> ql(n), rl(n), al(n);
    vector<int> local(n);
    for (int i = 0; i < n; i++) {
        const int a = order[i];
        ql[i] = q[a];
        rl[i] = radius[a];
        al[i] = (polarizable ? alpha[a] : 0.0);
        local[a] = i;
    }
    engine->setAtoms(n, ql.data(), rl.data(), al.data(), order.data(), n);
    vector<int> ca(covAtom.size()), cp(covAtom.size());
    for (size_t m = 0; m < covAtom.size(); m++) {
        ca[m] = local[covAtom[m]];
        cp[m] = local[covPartner[m]];
    }
    engine->setCovalent(ca.size(), ca.data(), cp.data(), covStrength.data());
    atomsDirty = false;
}

double CudaCalcPGMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (common)
        return common->execute(context, includeForces, includeEnergy);
    ContextSelector selector(cu);
    if (atomsDirty)
        setAtoms();
    Vec3 a, b, c;
    cu.getPeriodicBoxVectors(a, b, c);
    const double box[3][3] = {{a[0], a[1], a[2]}, {b[0], b[1], b[2]}, {c[0], c[1], c[2]}};
    const long long step = cu.getStepCount();
    const bool newStep = (step != lastStep);
    lastStep = step;
    engine->launchDevice((const void*) cu.getPosq().getDevicePointer(), box, newStep,
                         (long long*) cu.getLongForceBuffer().getDevicePointer(), cu.getPaddedNumAtoms(),
                         (double) 0x100000000LL, (void*) cu.getCurrentStream());
    launched = true;
    wantEnergy = includeEnergy;
    return 0.0;   // the energy: finish()
}

double CudaCalcPGMForceKernel::finish(bool includeEnergy) {
    ContextSelector selector(cu);
    launched = false;
    pgmcuda::Terms t;
    if (!engine->finishDevice(t)) {
        cu.setForcesValid(false);   // computed again, the same step
        return 0.0;
    }
    lastIterations = engine->iterations();
    evaluated = true;
    return includeEnergy ? t.total() : 0.0;
}

void CudaCalcPGMForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    if (common)
        return common->getInducedDipoles(context, dipoles);
    if (!evaluated)
        context.calcForcesAndEnergy(false, true, 1<<forceGroup);
    ContextSelector selector(cu);
    vector<double> d(3*numParticles);
    engine->inducedDipoles(d.data());
    const vector<int>& order = cu.getAtomIndex();
    dipoles.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
}

void CudaCalcPGMForceKernel::getPermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    if (common)
        return common->getPermanentDipoles(context, dipoles);
    // the permanent dipoles depend on the positions only: those of an evaluation at the current positions
    context.calcForcesAndEnergy(false, true, 1<<forceGroup);
    ContextSelector selector(cu);
    vector<double> d(3*numParticles);
    engine->permanentDipoles(d.data());
    const vector<int>& order = cu.getAtomIndex();
    dipoles.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
}

int CudaCalcPGMForceKernel::getInducedDipoleIterations() {
    if (common)
        return common->getInducedDipoleIterations();
    ContextSelector selector(cu);
    return engine->iterations();
}

void CudaCalcPGMForceKernel::loadParameters(const PGMForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: the number of particles has changed");
    q.resize(numParticles);
    radius.resize(numParticles);
    alpha.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        force.getParticleParameters(i, q[i], radius[i], alpha[i]);
    vector<int> ca, cp;
    covStrength.clear();
    for (int m = 0; m < force.getNumCovalentDipoles(); m++) {
        int i, j;
        double s;
        force.getCovalentDipoleParameters(m, i, j, s);
        ca.push_back(i);
        cp.push_back(j);
        covStrength.push_back(s);
    }
    if (loaded && (ca != covAtom || cp != covPartner))
        throw OpenMMException("updateParametersInContext: the covalent dipoles' atoms have changed");
    covAtom = ca;
    covPartner = cp;
    loaded = true;
    atomsDirty = true;
}

void CudaCalcPGMForceKernel::copyParametersToContext(ContextImpl& context, const PGMForce& force) {
    if (common)
        return common->copyParametersToContext(context, force);
    loadParameters(force);
    // as OpenMM's own forces: an exchange of molecules the change made invalid is undone
    cu.invalidateMolecules(info, true, true);
    ContextSelector selector(cu);
    engine->resetHistory();
}

void CudaCalcPGMForceKernel::getPMEParameters(double& a, int& nx, int& ny, int& nz) const {
    if (common)
        return common->getPMEParameters(a, nx, ny, nz);
    a = beta;
    nx = gridX;
    ny = gridY;
    nz = gridZ;
}

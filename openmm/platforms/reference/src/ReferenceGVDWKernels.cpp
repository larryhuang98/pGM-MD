#include "ReferenceGVDWKernels.h"
#include "gvdw_kernel.h"
#include "pgm_math.h"
#include "internal/GVDWForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <cmath>
#include <map>
#include <utility>

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

void ReferenceCalcGVDWForceKernel::setParameters(const GVDWForce& force) {
    int n = force.getNumParticles();
    R2.resize(n);
    sqrtA.resize(n);
    sqrtC6.resize(n);
    b.resize(n);
    active.clear();
    for (int i = 0; i < n; i++) {
        double R;
        force.getParticleParameters(i, R, sqrtA[i], sqrtC6[i], b[i]);
        R2[i] = R*R;
        if (sqrtA[i] != 0 || sqrtC6[i] != 0)
            active.push_back(i);
    }
    sumC = GVDWForceImpl::sumSqrtC6(force);
    // every exception pair leaves the neighbour list; the ones with parameters are evaluated separately
    map<int, int> activeIndex;
    for (int k = 0; k < (int) active.size(); k++)
        activeIndex[active[k]] = k;
    activeExclusions.assign(active.size(), set<int>());
    exceptions.clear();
    exceptionPairs.clear();
    evaluated.clear();
    for (int k = 0; k < force.getNumExceptions(); k++) {
        Exception e;
        force.getExceptionParameters(k, e.i, e.j, e.A, e.C6, e.B);
        exceptionPairs.push_back(make_pair(e.i, e.j));
        if (e.A != 0 || e.C6 != 0)
            evaluated.push_back(k);
        if (activeIndex.count(e.i) && activeIndex.count(e.j)) {
            activeExclusions[activeIndex[e.i]].insert(activeIndex[e.j]);
            activeExclusions[activeIndex[e.j]].insert(activeIndex[e.i]);
        }
        if (e.A != 0 || e.C6 != 0)
            exceptions.push_back(e);
    }
}

void ReferenceCalcGVDWForceKernel::initialize(const System& system, const GVDWForce& force) {
    method = force.getNonbondedMethod();
    rep = force.getRepulsionForm();
    cutoff = force.getCutoffDistance();
    dispersionCorrection = force.getUseDispersionCorrection() && method == GVDWForce::CutoffPeriodic;
    exceptionsPeriodic = force.getExceptionsUsePeriodicBoundaryConditions() && method == GVDWForce::CutoffPeriodic;
    setParameters(force);
}

double ReferenceCalcGVDWForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    const bool periodic = (method == GVDWForce::CutoffPeriodic);
    Vec3* box = (periodic || exceptionsPeriodic ? extractBoxVectors(context) : nullptr);
    double energy = 0;
    auto addPair = [&](int i, int j, const Vec3& d, double r, double A, double C6, double B) {
        double dEdR;
        energy += pgmcore::gvdwPair(r, 1.0/sqrt(2.0*(R2[i]+R2[j])), A, C6, B, rep, dEdR);
        if (includeForces) {
            force[i] += d*dEdR;
            force[j] -= d*dEdR;
        }
    };
    // combination-rule pairs: all pairs of active atoms (exact; OpenMM's voxel-hash list missed pairs near
    // the cutoff in a truncated octahedron with unwrapped positions)
    int na = active.size();
    const bool useCutoff = (method != GVDWForce::NoCutoff);
    const double rc2 = cutoff*cutoff;
    for (int ii = 0; ii < na; ii++)
        for (int jj = ii+1; jj < na; jj++) {
            if (activeExclusions[ii].count(jj))
                continue;
            int i = active[ii], j = active[jj];
            Vec3 d = pos[i]-pos[j];
            if (periodic)
                pgmcore::minImage(d, box);
            double r2 = d.dot(d);
            if (useCutoff && r2 >= rc2)
                continue;
            addPair(i, j, d, sqrt(r2), sqrtA[i]*sqrtA[j], sqrtC6[i]*sqrtC6[j], 0.5*(b[i]+b[j]));
        }
    // exceptions: always evaluated, minimum image only on request
    for (const Exception& e : exceptions) {
        Vec3 d = pos[e.i]-pos[e.j];
        if (exceptionsPeriodic)
            pgmcore::minImage(d, box);
        addPair(e.i, e.j, d, sqrt(d.dot(d)), e.A, e.C6, e.B);
    }
    if (dispersionCorrection)
        energy += pgmcore::gvdwTail(sumC, box[0][0]*box[1][1]*box[2][2], cutoff);
    return energy;
}

void ReferenceCalcGVDWForceKernel::copyParametersToContext(ContextImpl& context, const GVDWForce& force) {
    // the same contract as the Common platforms: the exception pairs, and which of them are evaluated,
    // are fixed when the Context is created
    if (force.getNumParticles() != (int) R2.size())
        throw OpenMMException("GVDWForce.updateParametersInContext: the number of particles has changed");
    if (force.getNumExceptions() != (int) exceptionPairs.size())
        throw OpenMMException("GVDWForce.updateParametersInContext: the number of exceptions has changed");
    vector<int> withParameters;
    for (int k = 0; k < force.getNumExceptions(); k++) {
        int p1, p2;
        double A, C6, B;
        force.getExceptionParameters(k, p1, p2, A, C6, B);
        if (make_pair(p1, p2) != exceptionPairs[k])
            throw OpenMMException("GVDWForce.updateParametersInContext: the particles of an exception have changed");
        if (A != 0 || C6 != 0)
            withParameters.push_back(k);
    }
    if (withParameters != evaluated)
        throw OpenMMException("GVDWForce.updateParametersInContext: an exception cannot change between excluded (A = C6 = 0) and evaluated");
    setParameters(force);
}

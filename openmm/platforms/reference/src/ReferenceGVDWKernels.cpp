#include "ReferenceGVDWKernels.h"
#include "gvdw_kernel.h"
#include "internal/GVDWForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <cmath>

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
}

void ReferenceCalcGVDWForceKernel::initialize(const System& system, const GVDWForce& force) {
    method = force.getNonbondedMethod();
    rep = force.getRepulsionForm();
    cutoff = force.getCutoffDistance();
    dispersionCorrection = force.getUseDispersionCorrection() && method == GVDWForce::CutoffPeriodic;
    setParameters(force);
    exclusions.assign(force.getNumParticles(), set<int>());
    for (int k = 0; k < force.getNumExclusions(); k++) {
        int p1, p2;
        force.getExclusionParticles(k, p1, p2);
        exclusions[p1].insert(p2);
        exclusions[p2].insert(p1);
    }
}

double ReferenceCalcGVDWForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    bool periodic = (method == GVDWForce::CutoffPeriodic);
    bool useCutoff = (method != GVDWForce::NoCutoff);
    Vec3 box[3];
    if (periodic) {
        Vec3* bv = extractBoxVectors(context);
        box[0] = bv[0];
        box[1] = bv[1];
        box[2] = bv[2];
    }
    double rc2 = cutoff*cutoff, energy = 0;
    int na = active.size();
    for (int ii = 0; ii < na; ii++) {
        int i = active[ii];
        for (int jj = ii+1; jj < na; jj++) {
            int j = active[jj];
            if (exclusions[i].count(j))
                continue;
            Vec3 d = pos[i]-pos[j];
            if (periodic) {
                // minimum image in the reduced box (c, then b, then a)
                d -= box[2]*floor(d[2]/box[2][2]+0.5);
                d -= box[1]*floor(d[1]/box[1][1]+0.5);
                d -= box[0]*floor(d[0]/box[0][0]+0.5);
            }
            double r2 = d.dot(d);
            if (useCutoff && r2 >= rc2)
                continue;
            double r = sqrt(r2);
            double beta = 1.0/sqrt(2.0*(R2[i]+R2[j]));
            double dEdR;
            energy += pgmcore::gvdwPair(r, beta, sqrtA[i]*sqrtA[j], sqrtC6[i]*sqrtC6[j], 0.5*(b[i]+b[j]), rep, dEdR);
            if (includeForces) {
                force[i] += d*dEdR;
                force[j] -= d*dEdR;
            }
        }
    }
    if (dispersionCorrection)
        energy += pgmcore::gvdwTail(sumC, box[0][0]*box[1][1]*box[2][2], cutoff);
    return energy;
}

void ReferenceCalcGVDWForceKernel::copyParametersToContext(ContextImpl& context, const GVDWForce& force) {
    if (force.getNumParticles() != (int) R2.size())
        throw OpenMMException("GVDWForce.updateParametersInContext: the number of particles has changed");
    setParameters(force);
}

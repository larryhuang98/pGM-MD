#include "GVDWForce.h"
#include "internal/GVDWForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <set>
#include <sstream>

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

GVDWForce::GVDWForce() : nonbondedMethod(NoCutoff), repulsionForm(Slater), cutoffDistance(0.9), useDispersionCorrection(false),
        exceptionsUsePeriodic(false) {
}

static void checkParticle(double radius, double sqrtA, double sqrtC6, double b) {
    if (!(radius > 0))
        throw OpenMMException("GVDWForce: the Gaussian radius must be positive");
    if (!(sqrtA >= 0) || !(sqrtC6 >= 0))
        throw OpenMMException("GVDWForce: sqrt(A) and sqrt(C6) must not be negative");
    if (!(b >= 0))
        throw OpenMMException("GVDWForce: the repulsion exponent scale b must not be negative");
}

static void checkException(int p1, int p2, double A, double C6, double B) {
    if (p1 == p2)
        throw OpenMMException("GVDWForce: an exception needs two different particles");
    if (!(A >= 0) || !(C6 >= 0) || !(B >= 0))
        throw OpenMMException("GVDWForce: exception parameters must not be negative");
}

int GVDWForce::addParticle(double radius, double sqrtA, double sqrtC6, double b) {
    checkParticle(radius, sqrtA, sqrtC6, b);
    particles.push_back(ParticleInfo(radius, sqrtA, sqrtC6, b));
    return particles.size()-1;
}

void GVDWForce::getParticleParameters(int index, double& radius, double& sqrtA, double& sqrtC6, double& b) const {
    ASSERT_VALID_INDEX(index, particles);
    radius = particles[index].radius;
    sqrtA = particles[index].sqrtA;
    sqrtC6 = particles[index].sqrtC6;
    b = particles[index].b;
}

void GVDWForce::setParticleParameters(int index, double radius, double sqrtA, double sqrtC6, double b) {
    ASSERT_VALID_INDEX(index, particles);
    checkParticle(radius, sqrtA, sqrtC6, b);
    particles[index] = ParticleInfo(radius, sqrtA, sqrtC6, b);
}

int GVDWForce::addException(int particle1, int particle2, double A, double C6, double B, bool replace) {
    checkException(particle1, particle2, A, C6, B);
    pair<int, int> key(min(particle1, particle2), max(particle1, particle2));
    map<pair<int, int>, int>::iterator it = exceptionMap.find(key);
    if (it != exceptionMap.end()) {
        if (!replace) {
            stringstream msg;
            msg << "GVDWForce: there is already an exception for particles " << particle1 << " and " << particle2;
            throw OpenMMException(msg.str());
        }
        exceptions[it->second] = ExceptionInfo(particle1, particle2, A, C6, B);
        return it->second;
    }
    exceptions.push_back(ExceptionInfo(particle1, particle2, A, C6, B));
    exceptionMap[key] = exceptions.size()-1;
    return exceptions.size()-1;
}

void GVDWForce::getExceptionParameters(int index, int& particle1, int& particle2, double& A, double& C6, double& B) const {
    ASSERT_VALID_INDEX(index, exceptions);
    particle1 = exceptions[index].particle1;
    particle2 = exceptions[index].particle2;
    A = exceptions[index].A;
    C6 = exceptions[index].C6;
    B = exceptions[index].B;
}

void GVDWForce::setExceptionParameters(int index, int particle1, int particle2, double A, double C6, double B) {
    ASSERT_VALID_INDEX(index, exceptions);
    checkException(particle1, particle2, A, C6, B);
    pair<int, int> oldKey(min(exceptions[index].particle1, exceptions[index].particle2), max(exceptions[index].particle1, exceptions[index].particle2));
    pair<int, int> key(min(particle1, particle2), max(particle1, particle2));
    if (key != oldKey) {
        if (exceptionMap.count(key))
            throw OpenMMException("GVDWForce: setExceptionParameters would duplicate an existing exception");
        exceptionMap.erase(oldKey);
        exceptionMap[key] = index;
    }
    exceptions[index] = ExceptionInfo(particle1, particle2, A, C6, B);
}

void GVDWForce::createExceptionsFromBonds(const vector<pair<int, int> >& bonds, double scale14A, double scale14C6) {
    int n = particles.size();
    vector<set<int> > neighbours(n);
    for (const pair<int, int>& bond : bonds) {
        if (bond.first < 0 || bond.second < 0 || bond.first >= n || bond.second >= n)
            throw OpenMMException("GVDWForce.createExceptionsFromBonds: illegal particle index in a bond");
        neighbours[bond.first].insert(bond.second);
        neighbours[bond.second].insert(bond.first);
    }
    // 1-2 and 1-3 pairs, then 1-4 pairs that are neither
    set<pair<int, int> > excluded, pairs14;
    for (int i = 0; i < n; i++)
        for (int j : neighbours[i]) {
            if (i < j)
                excluded.insert(make_pair(i, j));
            for (int k : neighbours[j])
                if (k != i)
                    excluded.insert(make_pair(min(i, k), max(i, k)));
        }
    for (int i = 0; i < n; i++)
        for (int j : neighbours[i])
            for (int k : neighbours[j])
                if (k != i)
                    for (int l : neighbours[k])
                        if (l != j && l != i) {
                            pair<int, int> key(min(i, l), max(i, l));
                            if (!excluded.count(key))
                                pairs14.insert(key);
                        }
    for (const pair<int, int>& key : excluded)
        if (!exceptionMap.count(key))
            addException(key.first, key.second, 0.0, 0.0, 0.0);
    for (const pair<int, int>& key : pairs14) {
        if (exceptionMap.count(key))
            continue;
        const ParticleInfo& p = particles[key.first];
        const ParticleInfo& q = particles[key.second];
        addException(key.first, key.second, scale14A*p.sqrtA*q.sqrtA, scale14C6*p.sqrtC6*q.sqrtC6, 0.5*(p.b+q.b));
    }
}

void GVDWForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < NoCutoff || method > CutoffPeriodic)
        throw OpenMMException("GVDWForce: illegal value for the nonbonded method");
    nonbondedMethod = method;
}

void GVDWForce::setCutoffDistance(double distance) {
    if (!(distance > 0))
        throw OpenMMException("GVDWForce: the cutoff must be positive");
    cutoffDistance = distance;
}

void GVDWForce::setRepulsionForm(RepulsionForm form) {
    if (form != Gaussian && form != Slater)
        throw OpenMMException("GVDWForce: illegal value for the repulsion form");
    repulsionForm = form;
}

void GVDWForce::updateParametersInContext(Context& context) {
    dynamic_cast<GVDWForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

ForceImpl* GVDWForce::createImpl() const {
    return new GVDWForceImpl(*this);
}

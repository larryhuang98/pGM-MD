#include "GVDWForce.h"
#include "internal/GVDWForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

GVDWForce::GVDWForce() : nonbondedMethod(NoCutoff), repulsionForm(Slater), cutoffDistance(0.9), useDispersionCorrection(false) {
}

static void checkParameters(double radius, double sqrtA, double sqrtC6, double b) {
    if (radius <= 0)
        throw OpenMMException("GVDWForce: the Gaussian radius must be positive");
    if (sqrtA < 0 || sqrtC6 < 0)
        throw OpenMMException("GVDWForce: sqrt(A) and sqrt(C6) must not be negative");
    if (b < 0)
        throw OpenMMException("GVDWForce: the repulsion exponent scale b must not be negative");
}

int GVDWForce::addParticle(double radius, double sqrtA, double sqrtC6, double b) {
    checkParameters(radius, sqrtA, sqrtC6, b);
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
    checkParameters(radius, sqrtA, sqrtC6, b);
    particles[index] = ParticleInfo(radius, sqrtA, sqrtC6, b);
}

int GVDWForce::addExclusion(int particle1, int particle2) {
    exclusions.push_back(make_pair(particle1, particle2));
    return exclusions.size()-1;
}

void GVDWForce::getExclusionParticles(int index, int& particle1, int& particle2) const {
    ASSERT_VALID_INDEX(index, exclusions);
    particle1 = exclusions[index].first;
    particle2 = exclusions[index].second;
}

void GVDWForce::setExclusionParticles(int index, int particle1, int particle2) {
    ASSERT_VALID_INDEX(index, exclusions);
    exclusions[index] = make_pair(particle1, particle2);
}

void GVDWForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < NoCutoff || method > CutoffPeriodic)
        throw OpenMMException("GVDWForce: illegal value for the nonbonded method");
    nonbondedMethod = method;
}

void GVDWForce::setCutoffDistance(double distance) {
    if (distance <= 0)
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

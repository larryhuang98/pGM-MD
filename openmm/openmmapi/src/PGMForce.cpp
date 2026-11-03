#include "PGMForce.h"
#include "internal/PGMForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/SimTKOpenMMRealType.h"

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

PGMForce::PGMForce() : nonbondedMethod(NoCutoff), polarization(Mutual), predictor(Mu4), cutoffDistance(0.9), alpha(0.0),
        ewaldErrorTol(5e-4), inducedTol(1e-5), peek(0.65), coulomb(ONE_4PI_EPS0), nx(0), ny(0), nz(0), pmeOrder(6), maxIterations(100) {
}

int PGMForce::addParticle(double charge, double radius, double polarizability) {
    if (radius <= 0)
        throw OpenMMException("PGMForce: the Gaussian radius must be positive");
    if (polarizability < 0)
        throw OpenMMException("PGMForce: the polarizability must not be negative");
    particles.push_back(ParticleInfo(charge, radius, polarizability));
    return particles.size()-1;
}

void PGMForce::getParticleParameters(int index, double& charge, double& radius, double& polarizability) const {
    ASSERT_VALID_INDEX(index, particles);
    charge = particles[index].charge;
    radius = particles[index].radius;
    polarizability = particles[index].polarizability;
}

void PGMForce::setParticleParameters(int index, double charge, double radius, double polarizability) {
    ASSERT_VALID_INDEX(index, particles);
    if (radius <= 0)
        throw OpenMMException("PGMForce: the Gaussian radius must be positive");
    if (polarizability < 0)
        throw OpenMMException("PGMForce: the polarizability must not be negative");
    particles[index] = ParticleInfo(charge, radius, polarizability);
}

int PGMForce::addCovalentDipole(int particle, int partner, double strength) {
    if (particle == partner)
        throw OpenMMException("PGMForce: a covalent dipole needs two different atoms");
    covalentDipoles.push_back(CovalentDipoleInfo(particle, partner, strength));
    return covalentDipoles.size()-1;
}

void PGMForce::getCovalentDipoleParameters(int index, int& particle, int& partner, double& strength) const {
    ASSERT_VALID_INDEX(index, covalentDipoles);
    particle = covalentDipoles[index].particle;
    partner = covalentDipoles[index].partner;
    strength = covalentDipoles[index].strength;
}

void PGMForce::setCovalentDipoleParameters(int index, int particle, int partner, double strength) {
    ASSERT_VALID_INDEX(index, covalentDipoles);
    if (particle == partner)
        throw OpenMMException("PGMForce: a covalent dipole needs two different atoms");
    covalentDipoles[index] = CovalentDipoleInfo(particle, partner, strength);
}

PGMForce::NonbondedMethod PGMForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void PGMForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("PGMForce: illegal value for the nonbonded method");
    nonbondedMethod = method;
}

double PGMForce::getCutoffDistance() const {
    return cutoffDistance;
}

void PGMForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

void PGMForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void PGMForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

int PGMForce::getPMEOrder() const {
    return pmeOrder;
}

void PGMForce::setPMEOrder(int order) {
    if (order < 4 || order > 8)
        throw OpenMMException("PGMForce: the PME order must be between 4 and 8");
    pmeOrder = order;
}

double PGMForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void PGMForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

void PGMForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const PGMForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

PGMForce::Polarization PGMForce::getPolarization() const {
    return polarization;
}

void PGMForce::setPolarization(Polarization type) {
    polarization = type;
}

double PGMForce::getInducedDipoleTolerance() const {
    return inducedTol;
}

void PGMForce::setInducedDipoleTolerance(double tol) {
    inducedTol = tol;
}

int PGMForce::getMaxInducedDipoleIterations() const {
    return maxIterations;
}

void PGMForce::setMaxInducedDipoleIterations(int iter) {
    maxIterations = iter;
}

double PGMForce::getPeekCoefficient() const {
    return peek;
}

void PGMForce::setPeekCoefficient(double omega) {
    peek = omega;
}

PGMForce::Predictor PGMForce::getPredictor() const {
    return predictor;
}

void PGMForce::setPredictor(Predictor p) {
    predictor = p;
}

double PGMForce::getCoulombConstant() const {
    return coulomb;
}

void PGMForce::setCoulombConstant(double ke) {
    coulomb = ke;
}

void PGMForce::getInducedDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<PGMForceImpl&>(getImplInContext(context)).getInducedDipoles(getContextImpl(context), dipoles);
}

void PGMForce::getPermanentDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<PGMForceImpl&>(getImplInContext(context)).getPermanentDipoles(getContextImpl(context), dipoles);
}

void PGMForce::getTotalDipoles(Context& context, vector<Vec3>& dipoles) {
    vector<Vec3> induced;
    getPermanentDipoles(context, dipoles);
    getInducedDipoles(context, induced);
    for (int i = 0; i < (int) dipoles.size(); i++)
        dipoles[i] += induced[i];
}

int PGMForce::getInducedDipoleIterations(Context& context) {
    return dynamic_cast<PGMForceImpl&>(getImplInContext(context)).getInducedDipoleIterations(getContextImpl(context));
}

void PGMForce::updateParametersInContext(Context& context) {
    dynamic_cast<PGMForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

ForceImpl* PGMForce::createImpl() const {
    return new PGMForceImpl(*this);
}

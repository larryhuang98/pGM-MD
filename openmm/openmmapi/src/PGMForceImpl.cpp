#include "internal/PGMForceImpl.h"
#include "PGMKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

PGMForceImpl::PGMForceImpl(const PGMForce& owner) : owner(owner) {
}

PGMForceImpl::~PGMForceImpl() {
}

void PGMForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    int n = owner.getNumParticles();
    if (n != system.getNumParticles())
        throw OpenMMException("PGMForce must have exactly as many particles as the System it belongs to.");
    for (int i = 0; i < owner.getNumCovalentDipoles(); i++) {
        int a, b;
        double c;
        owner.getCovalentDipoleParameters(i, a, b, c);
        if (a < 0 || a >= n || b < 0 || b >= n)
            throw OpenMMException("PGMForce: illegal particle index in a covalent dipole");
    }
    if (owner.getNonbondedMethod() == PGMForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("PGMForce: the cutoff distance cannot be greater than half the periodic box size.");
        // the direct sum drops erf(a_ij r)/r - erf(beta r)/r beyond the cutoff, i.e. erfc(a_ij rc)/rc of the
        // Gaussian part (docs/spec.md, section 4): wide Gaussians need a longer cutoff
        double maxRadius = 0;
        for (int i = 0; i < n; i++) {
            double q, R, a;
            owner.getParticleParameters(i, q, R, a);
            maxRadius = max(maxRadius, R);
        }
        double amin = 1.0/sqrt(4.0*maxRadius*maxRadius);
        if (erfc(amin*cutoff) > owner.getEwaldErrorTolerance()) {
            stringstream msg;
            msg << "PGMForce: the largest Gaussian radius (" << maxRadius << " nm) is too wide for the cutoff: "
                << "erfc(a rc) = " << erfc(amin*cutoff) << " exceeds the Ewald error tolerance; increase the cutoff";
            throw OpenMMException(msg.str());
        }
        double alpha;
        int nx, ny, nz;
        calcPMEParameters(system, owner, alpha, nx, ny, nz);
        int order = owner.getPMEOrder();
        if (nx < order || ny < order || nz < order)
            throw OpenMMException("PGMForce: every PME grid dimension must be at least the PME order");
    }
    kernel = context.getPlatform().createKernel(CalcPGMForceKernel::Name(), context);
    kernel.getAs<CalcPGMForceKernel>().initialize(system, owner);
}

double PGMForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcPGMForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> PGMForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcPGMForceKernel::Name());
    return names;
}

vector<pair<int, int> > PGMForceImpl::getBondedParticles() const {
    // The covalent dipoles tie atoms together (molecule detection for atom reordering).
    vector<pair<int, int> > bonds;
    for (int i = 0; i < owner.getNumCovalentDipoles(); i++) {
        int a, b;
        double c;
        owner.getCovalentDipoleParameters(i, a, b, c);
        bonds.push_back(make_pair(a, b));
    }
    return bonds;
}

void PGMForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcPGMForceKernel>().getInducedDipoles(context, dipoles);
}

void PGMForceImpl::getPermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcPGMForceKernel>().getPermanentDipoles(context, dipoles);
}

int PGMForceImpl::getInducedDipoleIterations(ContextImpl& context) {
    return kernel.getAs<CalcPGMForceKernel>().getInducedDipoleIterations();
}

void PGMForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcPGMForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void PGMForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcPGMForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}

static int findLegalFFTDimension(int minimum) {
    // Smallest size >= minimum with factors 2, 3, 5 and 7 only (the FFT libraries' fast sizes).
    if (minimum < 1)
        return 1;
    while (true) {
        int unfactored = minimum;
        for (int factor = 2; factor < 8; factor++) {
            while (unfactored > 1 && unfactored%factor == 0)
                unfactored /= factor;
        }
        if (unfactored == 1)
            return minimum;
        minimum++;
    }
}

void PGMForceImpl::calcPMEParameters(const System& system, const PGMForce& force, double& alpha, int& xsize, int& ysize, int& zsize) {
    force.getPMEParameters(alpha, xsize, ysize, zsize);
    if (alpha == 0.0) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double tol = force.getEwaldErrorTolerance();
        alpha = (1.0/force.getCutoffDistance())*std::sqrt(-log(2.0*tol));
        xsize = (int) ceil(2*alpha*boxVectors[0][0]/(3*pow(tol, 0.2)));
        ysize = (int) ceil(2*alpha*boxVectors[1][1]/(3*pow(tol, 0.2)));
        zsize = (int) ceil(2*alpha*boxVectors[2][2]/(3*pow(tol, 0.2)));
        xsize = findLegalFFTDimension(max(xsize, 6));
        ysize = findLegalFFTDimension(max(ysize, 6));
        zsize = findLegalFFTDimension(max(zsize, 6));
    }
    // A grid given explicitly is used as it is (parity with pmemd / pGM-JAX needs the same grid).
}

#include "CommonGVDWKernels.h"
#include "CommonPGMKernelSources.h"
#include "internal/GVDWForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <map>

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

class CommonCalcGVDWForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const GVDWForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double R1, a1, c1, b1, R2, a2, c2, b2;
        force.getParticleParameters(particle1, R1, a1, c1, b1);
        force.getParticleParameters(particle2, R2, a2, c2, b2);
        return (R1 == R2 && a1 == a2 && c1 == c2 && b1 == b2);
    }
    int getNumParticleGroups() {
        return force.getNumExclusions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int p1, p2;
        force.getExclusionParticles(index, p1, p2);
        particles.resize(2);
        particles[0] = p1;
        particles[1] = p2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const GVDWForce& force;
};

void CommonCalcGVDWForceKernel::uploadParameters(const GVDWForce& force) {
    int n = force.getNumParticles(), padded = cc.getPaddedNumAtoms();
    // in the original atom order, as OpenMM's own forces: the context only exchanges identical
    // molecules (ForceInfo), and invalidateMolecules() undoes an exchange that a change makes invalid
    vector<double> v(4*padded, 0.0);
    for (int i = 0; i < padded; i++)
        v[4*i] = 1.0;   // padding atoms: R^2 = 1, a = c = 0 (no interaction)
    for (int k = 0; k < n; k++) {
        double R, a, c, b;
        force.getParticleParameters(k, R, a, c, b);
        v[4*k] = R*R;
        v[4*k+1] = a;
        v[4*k+2] = c;
        v[4*k+3] = b;
    }
    if (cc.getUseDoublePrecision()) {
        vector<mm_double4> d(padded);
        for (int i = 0; i < padded; i++)
            d[i] = mm_double4(v[4*i], v[4*i+1], v[4*i+2], v[4*i+3]);
        params.upload(d);
    }
    else {
        vector<mm_float4> f(padded);
        for (int i = 0; i < padded; i++)
            f[i] = mm_float4((float) v[4*i], (float) v[4*i+1], (float) v[4*i+2], (float) v[4*i+3]);
        params.upload(f);
    }
    sumC = GVDWForceImpl::sumSqrtC6(force);
}

void CommonCalcGVDWForceKernel::initialize(const System& system, const GVDWForce& force) {
    ContextSelector selector(cc);
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "gvdw"+cc.intToString(forceIndex)+"_";
    int n = force.getNumParticles(), padded = cc.getPaddedNumAtoms();
    bool useDouble = cc.getUseDoublePrecision();
    params.initialize(cc, padded, useDouble ? sizeof(mm_double4) : sizeof(mm_float4), "gvdwParams");
    uploadParameters(force);
    vector<vector<int> > exclusionList(n);
    for (int i = 0; i < n; i++)
        exclusionList[i].push_back(i);
    for (int k = 0; k < force.getNumExclusions(); k++) {
        int p1, p2;
        force.getExclusionParticles(k, p1, p2);
        exclusionList[p1].push_back(p2);
        exclusionList[p2].push_back(p1);
    }
    bool useCutoff = (force.getNonbondedMethod() != GVDWForce::NoCutoff);
    bool usePeriodic = (force.getNonbondedMethod() == GVDWForce::CutoffPeriodic);
    cutoff = force.getCutoffDistance();
    dispersionCorrection = usePeriodic && force.getUseDispersionCorrection();
    map<string, string> replacements;
    replacements["PARAMS1"] = prefix+"params1";   // the parameters of atom1 and atom2 (NonbondedUtilities)
    replacements["PARAMS2"] = prefix+"params2";
    replacements["GVDW_SLATER"] = (force.getRepulsionForm() == GVDWForce::Slater ? "1" : "0");
    replacements["GVDW_C0"] = cc.doubleToString(8.0/(9.0*M_PI));
    replacements["GVDW_2_SQRTPI"] = cc.doubleToString(2.0/sqrt(M_PI));
    string source = cc.replaceStrings(CommonPGMKernelSources::gvdw, replacements);
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    nb.addInteraction(useCutoff, usePeriodic, true, cutoff, exclusionList, source, force.getForceGroup(), n > 2000);
    nb.addParameter(ComputeParameterInfo(params, prefix+"params", useDouble ? "double" : "float", 4));
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcGVDWForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // the pair interaction runs in the nonbonded kernel; only the dispersion correction is here
    if (!dispersionCorrection)
        return 0.0;
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    return -2.0*M_PI*sumC*sumC/(3.0*a[0]*b[1]*c[2]*cutoff*cutoff*cutoff);
}

void CommonCalcGVDWForceKernel::copyParametersToContext(ContextImpl& context, const GVDWForce& force) {
    ContextSelector selector(cc);
    if (force.getNumParticles() != system.getNumParticles())
        throw OpenMMException("GVDWForce.updateParametersInContext: the number of particles has changed");
    uploadParameters(force);
    cc.invalidateMolecules(info, true, false);
}

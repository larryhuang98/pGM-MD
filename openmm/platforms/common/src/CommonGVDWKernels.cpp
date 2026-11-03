#include "CommonGVDWKernels.h"
#include "CommonPGMKernelSources.h"
#include "internal/GVDWForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/BondedUtilities.h"
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
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int p1, p2;
        double A, C6, B;
        force.getExceptionParameters(index, p1, p2, A, C6, B);
        particles.resize(2);
        particles[0] = p1;
        particles[1] = p2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int p1, p2;
        double A1, C1, B1, A2, C2, B2;
        force.getExceptionParameters(group1, p1, p2, A1, C1, B1);
        force.getExceptionParameters(group2, p1, p2, A2, C2, B2);
        return (A1 == A2 && C1 == C2 && B1 == B2);
    }
private:
    const GVDWForce& force;
};

/** The device source of the pair (gvdwPair.cc) inserted into a wrapper, with the model's constants. */
static string gvdwSource(ComputeContext& cc, const string& wrapper, const GVDWForce& force, map<string, string> replacements) {
    map<string, string> pair;
    pair["GVDW_PAIR"] = CommonPGMKernelSources::gvdwPair;
    string source = cc.replaceStrings(wrapper, pair);   // first insert the pair, then fill in its constants
    replacements["GVDW_SLATER"] = (force.getRepulsionForm() == GVDWForce::Slater ? "1" : "0");
    replacements["GVDW_C0"] = cc.doubleToString(8.0/(9.0*M_PI));
    replacements["GVDW_2_SQRTPI"] = cc.doubleToString(2.0/sqrt(M_PI));
    return cc.replaceStrings(source, replacements);
}

template <class T4>
static void uploadReal4(ComputeArray& array, const vector<double>& v) {
    vector<T4> data(v.size()/4);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = T4(v[4*i], v[4*i+1], v[4*i+2], v[4*i+3]);
    array.upload(data);
}

void CommonCalcGVDWForceKernel::uploadParameters(const GVDWForce& force) {
    // In the original atom order, as OpenMM's own forces: the context only exchanges identical
    // molecules (ForceInfo), and invalidateMolecules() undoes an exchange that a change makes invalid.
    int n = force.getNumParticles(), padded = cc.getPaddedNumAtoms();
    vector<double> v(4*padded, 0.0);
    for (int i = n; i < padded; i++)
        v[4*i] = 1.0;   // padding atoms: R^2 = 1, a = c = 0 (no interaction)
    vector<double> R2(n);
    for (int k = 0; k < n; k++) {
        double R, a, c, b;
        force.getParticleParameters(k, R, a, c, b);
        R2[k] = R*R;
        v[4*k] = R2[k];
        v[4*k+1] = a;
        v[4*k+2] = c;
        v[4*k+3] = b;
    }
    bool useDouble = cc.getUseDoublePrecision();
    if (useDouble)
        uploadReal4<mm_double4>(params, v);
    else
        uploadReal4<mm_float4>(params, v);
    if (!exceptionIndex.empty()) {
        vector<double> e(4*exceptionIndex.size());
        for (size_t k = 0; k < exceptionIndex.size(); k++) {
            int p1, p2;
            double A, C6, B;
            force.getExceptionParameters(exceptionIndex[k], p1, p2, A, C6, B);
            e[4*k] = A;
            e[4*k+1] = C6;
            e[4*k+2] = B;
            e[4*k+3] = 1.0/sqrt(2.0*(R2[p1]+R2[p2]));
        }
        if (useDouble)
            uploadReal4<mm_double4>(exceptionParams, e);
        else
            uploadReal4<mm_float4>(exceptionParams, e);
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
    string real4 = (useDouble ? "double4" : "float4");
    int elementSize = (useDouble ? sizeof(mm_double4) : sizeof(mm_float4));

    // exceptions: all of them leave the tile kernel; those with parameters go to BondedUtilities
    vector<vector<int> > exclusionList(n);
    for (int i = 0; i < n; i++)
        exclusionList[i].push_back(i);
    vector<vector<int> > exceptionAtoms;
    for (int k = 0; k < force.getNumExceptions(); k++) {
        int p1, p2;
        double A, C6, B;
        force.getExceptionParameters(k, p1, p2, A, C6, B);
        exceptionPairs.push_back(make_pair(p1, p2));
        exclusionList[p1].push_back(p2);
        exclusionList[p2].push_back(p1);
        if (A != 0 || C6 != 0) {
            exceptionIndex.push_back(k);
            exceptionAtoms.push_back({p1, p2});
        }
    }
    params.initialize(cc, padded, elementSize, "gvdwParams");
    if (!exceptionIndex.empty())
        exceptionParams.initialize(cc, exceptionIndex.size(), elementSize, "gvdwExceptionParams");
    uploadParameters(force);

    bool useCutoff = (force.getNonbondedMethod() != GVDWForce::NoCutoff);
    bool usePeriodic = (force.getNonbondedMethod() == GVDWForce::CutoffPeriodic);
    cutoff = force.getCutoffDistance();
    dispersionCorrection = usePeriodic && force.getUseDispersionCorrection();

    map<string, string> replacements;
    replacements["PARAMS1"] = prefix+"params1";   // the parameters of atom1 and atom2 (NonbondedUtilities)
    replacements["PARAMS2"] = prefix+"params2";
    string source = gvdwSource(cc, CommonPGMKernelSources::gvdwNonbonded, force, replacements);
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    nb.addInteraction(useCutoff, usePeriodic, true, cutoff, exclusionList, source, force.getForceGroup(), n > 2000);
    nb.addParameter(ComputeParameterInfo(params, prefix+"params", useDouble ? "double" : "float", 4));

    if (!exceptionIndex.empty()) {
        map<string, string> excReplacements;
        excReplacements["APPLY_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        excReplacements["PARAMS"] = cc.getBondedUtilities().addArgument(exceptionParams, real4);
        cc.getBondedUtilities().addInteraction(exceptionAtoms, gvdwSource(cc, CommonPGMKernelSources::gvdwExceptions, force, excReplacements),
                force.getForceGroup());
    }
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcGVDWForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // the pairs run in the nonbonded and bonded kernels; only the dispersion correction is here
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
    // the exception pairs, and which of them have parameters, define the kernels: they must not change
    if (force.getNumExceptions() != (int) exceptionPairs.size())
        throw OpenMMException("GVDWForce.updateParametersInContext: the number of exceptions has changed");
    vector<int> withParameters;
    for (int k = 0; k < (int) exceptionPairs.size(); k++) {
        int p1, p2;
        double A, C6, B;
        force.getExceptionParameters(k, p1, p2, A, C6, B);
        if (make_pair(p1, p2) != exceptionPairs[k])
            throw OpenMMException("GVDWForce.updateParametersInContext: the particles of an exception have changed");
        if (A != 0 || C6 != 0)
            withParameters.push_back(k);
    }
    if (withParameters != exceptionIndex)
        throw OpenMMException("GVDWForce.updateParametersInContext: an exception cannot change between excluded (A = C6 = 0) and evaluated");
    uploadParameters(force);
    cc.invalidateMolecules(info, true, true);
}

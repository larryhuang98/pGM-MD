#include "CommonPGMKernels.h"
#include "CommonPGMKernelSources.h"
#include "internal/PGMForceImpl.h"
#include "pgm_constants.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <cmath>
#include <map>

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

// threads of the single-work-group CG kernels: at most this, a power of two within the device limit
static const int MAX_CG_GROUP_SIZE = 512;

class CommonCalcPGMForceKernel::ForceInfo : public ComputeForceInfo {
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
        particles.resize(2);
        particles[0] = a;
        particles[1] = b;
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

class CommonCalcPGMForceKernel::ReorderListener : public ComputeContext::ReorderListener {
public:
    ReorderListener(CommonCalcPGMForceKernel& owner) : owner(owner) {
    }
    void execute() {
        owner.resetHistory();   // the dipole history is in the old atom order
    }
private:
    CommonCalcPGMForceKernel& owner;
};

static vector<double> bsplineModuli(int K, int p) {
    // |sum_k M_p(k+1) exp(2 pi i m k / K)|^2, zeros replaced by the mean of the neighbours
    vector<double> M(p+1, 0.0), tmp(p+1, 0.0);
    M[0] = 1.0;
    for (int k = 2; k <= p; k++) {
        for (int j = 0; j < k; j++) {
            double cur = (j < k-1 ? M[j] : 0.0), prev = (j > 0 ? M[j-1] : 0.0);
            tmp[j] = (j*cur+(k-j)*prev)/(k-1);
        }
        for (int j = 0; j < k; j++)
            M[j] = tmp[j];
    }
    vector<double> data(K, 0.0);
    for (int k = 0; k < p-1 && k < K; k++)
        data[k] = M[k+1];
    vector<double> mod(K);
    for (int m = 0; m < K; m++) {
        double sc = 0, ss = 0;
        for (int k = 0; k < K; k++) {
            double arg = 2*M_PI*m*k/K;
            sc += data[k]*cos(arg);
            ss += data[k]*sin(arg);
        }
        mod[m] = sc*sc+ss*ss;
    }
    for (int m = 0; m < K; m++)
        if (mod[m] < 1e-7)
            mod[m] = 0.5*(mod[(m-1+K)%K]+mod[(m+1)%K]);
    return mod;
}

CommonCalcPGMForceKernel::CommonCalcPGMForceKernel(string name, const Platform& platform, ComputeContext& cc, const System& system) :
        CalcPGMForceKernel(name, platform), cc(cc), system(system), hasInitializedKernels(false), lastIterations(0),
        histCount(0), histHead(0), lastStep(-1) {
}

CommonCalcPGMForceKernel::~CommonCalcPGMForceKernel() {
}

void CommonCalcPGMForceKernel::uploadParameters(const PGMForce& force) {
    int padded = cc.getPaddedNumAtoms();
    vector<double> q(padded, 0.0), rg(padded, 1.0), al(padded, 0.0);
    totalCharge = 0;
    for (int i = 0; i < numParticles; i++) {
        double r;
        force.getParticleParameters(i, q[i], r, al[i]);
        if (al[i] <= pgmcore::MIN_POLARIZABILITY)
            al[i] = 0.0;   // not polarizable (pmemd-pgm's threshold)
        rg[i] = 2*r*r;
        totalCharge += q[i];
    }
    charge.upload(q, true);
    rgauss.upload(rg, true);
    polarizability.upload(al, true);
    // covalent dipoles: grouped by atom (CSR) in the order of the force
    vector<vector<pair<int, double> > > byAtom(numParticles);
    for (int m = 0; m < numCovalent; m++) {
        int a, b;
        double c;
        force.getCovalentDipoleParameters(m, a, b, c);
        byAtom[a].push_back(make_pair(b, c));
    }
    vector<int> start(padded+1, 0), atom, partner;
    vector<double> strength;
    for (int i = 0; i < padded; i++) {
        start[i] = atom.size();
        if (i < numParticles)
            for (auto& e : byAtom[i]) {
                atom.push_back(i);
                partner.push_back(e.first);
                strength.push_back(e.second);
            }
    }
    start[padded] = atom.size();
    if (atom.size() == 0) {
        atom.push_back(0);
        partner.push_back(0);
        strength.push_back(0.0);
    }
    covStart.upload(start);
    covAtom.upload(atom);
    covPartner.upload(partner);
    covStrength.upload(strength, true);
}

void CommonCalcPGMForceKernel::initialize(const System& system, const PGMForce& force) {
    ContextSelector selector(cc);
    numParticles = force.getNumParticles();
    numCovalent = force.getNumCovalentDipoles();
    usePME = (force.getNonbondedMethod() == PGMForce::PME);
    polarizable = (force.getPolarization() == PGMForce::Mutual);
    tol = force.getInducedDipoleTolerance();
    maxIterations = force.getMaxInducedDipoleIterations();
    peek = force.getPeekCoefficient();
    predictor = (int) force.getPredictor();
    ke = force.getCoulombConstant();
    cutoff = force.getCutoffDistance();
    pmeOrder = force.getPMEOrder();
    beta = 0;
    cself = 0;
    gridX = gridY = gridZ = 0;
    if (usePME) {
        PGMForceImpl::calcPMEParameters(system, force, beta, gridX, gridY, gridZ);
        cself = 4*beta*beta*beta/(3*sqrt(M_PI));
    }
    int padded = cc.getPaddedNumAtoms();
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    int mixedSize = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
    charge.initialize(cc, padded, elementSize, "pgmCharge");
    rgauss.initialize(cc, padded, elementSize, "pgmRgauss");
    polarizability.initialize(cc, padded, elementSize, "pgmPolarizability");
    int ncov = max(numCovalent, 1);
    covStart.initialize<int>(cc, padded+1, "pgmCovStart");
    covAtom.initialize<int>(cc, ncov, "pgmCovAtom");
    covPartner.initialize<int>(cc, ncov, "pgmCovPartner");
    covStrength.initialize(cc, ncov, elementSize, "pgmCovStrength");
    uploadParameters(force);
    permDipole.initialize(cc, 3*padded, elementSize, "pgmPermDipole");
    inducedDipole.initialize(cc, 3*padded, elementSize, "pgmInducedDipole");
    totalDipole.initialize(cc, 3*padded, elementSize, "pgmTotalDipole");
    field.initialize(cc, 3*padded, sizeof(long long), "pgmField");
    inducedField.initialize(cc, 3*padded, sizeof(long long), "pgmInducedField");
    totalField.initialize(cc, 3*padded, sizeof(long long), "pgmTotalField");
    cgR.initialize(cc, 3*padded, elementSize, "pgmCgR");
    cgZ.initialize(cc, 3*padded, elementSize, "pgmCgZ");
    cgP.initialize(cc, 3*padded, elementSize, "pgmCgP");
    cgAp.initialize(cc, 3*padded, elementSize, "pgmCgAp");
    for (int k = 0; k < 4; k++)
        hist[k].initialize(cc, 3*padded, elementSize, "pgmHistory");
    histScratch.initialize(cc, 3*padded, elementSize, "pgmHistoryScratch");
    scalars.initialize(cc, 8, mixedSize, "pgmScalars");
    // defined values everywhere, padding atoms included
    for (ComputeArray* a : {&inducedDipole, &permDipole, &totalDipole, &cgR, &cgZ, &cgP, &cgAp, &histScratch, &hist[0], &hist[1], &hist[2], &hist[3]})
        cc.clearBuffer(*a);
    cgGroupSize = 1;
    while (2*cgGroupSize <= min(MAX_CG_GROUP_SIZE, cc.getMaxThreadBlockSize()))
        cgGroupSize *= 2;
    for (int m = 0; m < force.getNumCovalentDipoles(); m++) {
        int a, b;
        double c;
        force.getCovalentDipoleParameters(m, a, b, c);
        covalentPairs.push_back(make_pair(a, b));
    }

    map<string, string> defines;
    defines["NUM_ATOMS"] = cc.intToString(numParticles);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(padded);
    defines["NUM_COVALENT"] = cc.intToString(numCovalent);
    defines["ENERGY_SCALE"] = cc.doubleToString(ke);
    defines["CSELF"] = cc.doubleToString(cself);
    defines["CG_GROUP_SIZE"] = cc.intToString(cgGroupSize);
    defines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
    defines["M_PI"] = cc.doubleToString(M_PI);
    if (usePME) {
        defines["USE_EWALD"] = "1";
        defines["USE_PERIODIC"] = "1";
        defines["USE_CUTOFF"] = "1";
        defines["EWALD_ALPHA"] = cc.doubleToString(beta);
        defines["PME_ORDER"] = cc.intToString(pmeOrder);
        defines["GRID_SIZE_X"] = cc.intToString(gridX);
        defines["GRID_SIZE_Y"] = cc.intToString(gridY);
        defines["GRID_SIZE_Z"] = cc.intToString(gridZ);
    }
    ComputeProgram program = cc.compileProgram(CommonPGMKernelSources::pgmSolver, defines);
    permKernel = program->createKernel("computePermanentDipoles");
    permKernel->addArg(cc.getPosq());
    permKernel->addArg(covStart);
    permKernel->addArg(covPartner);
    permKernel->addArg(covStrength);
    permKernel->addArg(permDipole);
    if (usePME)
        for (int i = 0; i < 5; i++)
            permKernel->addArg();
    guessKernel = program->createKernel("initialGuess");
    guessKernel->addArg(field);
    guessKernel->addArg(permDipole);
    guessKernel->addArg(polarizability);
    for (int k = 0; k < 4; k++)
        guessKernel->addArg(hist[k]);
    guessKernel->addArg(inducedDipole);
    guessKernel->addArg((int) 0);
    cgInitKernel = program->createKernel("cgInit");
    cgInitKernel->addArg(field);
    cgInitKernel->addArg(inducedField);
    cgInitKernel->addArg(permDipole);
    cgInitKernel->addArg(polarizability);
    cgInitKernel->addArg(inducedDipole);
    cgInitKernel->addArg(cgR);
    cgInitKernel->addArg(cgZ);
    cgInitKernel->addArg(cgP);
    cgInitKernel->addArg(scalars);
    cgStepKernel = program->createKernel("cgStep");
    cgStepKernel->addArg(inducedField);
    cgStepKernel->addArg(polarizability);
    cgStepKernel->addArg(inducedDipole);
    cgStepKernel->addArg(cgR);
    cgStepKernel->addArg(cgZ);
    cgStepKernel->addArg(cgP);
    cgStepKernel->addArg(cgAp);
    cgStepKernel->addArg(scalars);
    cgFinishKernel = program->createKernel("cgFinish");
    cgFinishKernel->addArg(inducedDipole);
    cgFinishKernel->addArg(cgR);
    cgFinishKernel->addArg(polarizability);
    cgFinishKernel->addArg(hist[0]);
    if (cc.getUseDoublePrecision())
        cgFinishKernel->addArg(peek);
    else
        cgFinishKernel->addArg((float) peek);
    totalKernel = program->createKernel("totalDipoles");
    totalKernel->addArg(permDipole);
    totalKernel->addArg(inducedDipole);
    totalKernel->addArg(totalDipole);
    selfKernel = program->createKernel("selfEnergy");
    selfKernel->addArg(charge);
    selfKernel->addArg(totalDipole);
    selfKernel->addArg(inducedDipole);
    selfKernel->addArg(polarizability);
    selfKernel->addArg(cc.getEnergyBuffer());
    covForceKernel = program->createKernel("covalentForces");
    covForceKernel->addArg(cc.getPosq());
    covForceKernel->addArg(covAtom);
    covForceKernel->addArg(covPartner);
    covForceKernel->addArg(covStrength);
    covForceKernel->addArg(totalField);
    covForceKernel->addArg(totalDipole);
    covForceKernel->addArg(cc.getLongForceBuffer());
    if (usePME)
        for (int i = 0; i < 5; i++)
            covForceKernel->addArg();

    if (usePME) {
        int gridSize = gridX*gridY*gridZ;
        pmeGridLong.initialize(cc, gridSize, sizeof(long long), "pgmPmeGridLong");
        // FFT3D uses its input as workspace: both arrays must hold complex values
        pmeGrid1.initialize(cc, gridSize, 2*elementSize, "pgmPmeGrid1");
        pmeGrid2.initialize(cc, gridSize, 2*elementSize, "pgmPmeGrid2");
        moduliX.initialize(cc, gridX, elementSize, "pgmModuliX");
        moduliY.initialize(cc, gridY, elementSize, "pgmModuliY");
        moduliZ.initialize(cc, gridZ, elementSize, "pgmModuliZ");
        moduliX.upload(bsplineModuli(gridX, pmeOrder), true);
        moduliY.upload(bsplineModuli(gridY, pmeOrder), true);
        moduliZ.upload(bsplineModuli(gridZ, pmeOrder), true);
        fft = cc.createFFT(gridX, gridY, gridZ, true);
        ComputeProgram pme = cc.compileProgram(CommonPGMKernelSources::pgmPme, defines);
        spreadKernel = pme->createKernel("gridSpread");
        spreadKernel->addArg(cc.getPosq());
        spreadKernel->addArg(charge);
        spreadKernel->addArg(permDipole);
        if (cc.getUseDoublePrecision())
            spreadKernel->addArg(1.0);
        else
            spreadKernel->addArg(1.0f);
        spreadKernel->addArg(pmeGridLong);
        for (int i = 0; i < 6; i++)
            spreadKernel->addArg();
        finishSpreadKernel = pme->createKernel("finishSpread");
        finishSpreadKernel->addArg(pmeGridLong);
        finishSpreadKernel->addArg(pmeGrid1);
        convKernel = pme->createKernel("reciprocalConvolution");
        convKernel->addArg(pmeGrid2);
        convKernel->addArg(moduliX);
        convKernel->addArg(moduliY);
        convKernel->addArg(moduliZ);
        for (int i = 0; i < 4; i++)
            convKernel->addArg();
        convKernel->addArg(cc.getEnergyBuffer());
        convKernel->addArg((int) 0);
        fieldFromGridKernel = pme->createKernel("fieldFromGrid");
        fieldFromGridKernel->addArg(pmeGrid1);
        fieldFromGridKernel->addArg(cc.getPosq());
        fieldFromGridKernel->addArg(field);
        for (int i = 0; i < 6; i++)
            fieldFromGridKernel->addArg();
        forceFromGridKernel = pme->createKernel("forceFromGrid");
        forceFromGridKernel->addArg(pmeGrid1);
        forceFromGridKernel->addArg(cc.getPosq());
        forceFromGridKernel->addArg(charge);
        forceFromGridKernel->addArg(totalDipole);
        forceFromGridKernel->addArg(cc.getLongForceBuffer());
        forceFromGridKernel->addArg(totalField);
        for (int i = 0; i < 6; i++)
            forceFromGridKernel->addArg();
    }

    // Every pair interacts: no exclusions are requested (the vdW forces have their own).
    cc.getNonbondedUtilities().addInteraction(usePME, usePME, false, cutoff, vector<vector<int> >(), "", force.getForceGroup());
    forceGroup = force.getForceGroup();
    info = new ForceInfo(force);
    cc.addForce(info);
    cc.addReorderListener(new ReorderListener(*this));
}

void CommonCalcPGMForceKernel::initKernels() {
    // The pair kernels need the nonbonded utilities, which are initialized after initialize().
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    // The tile kernels need TILE_SIZE cooperating threads; on devices with SIMD width < 32 (CPUs) use
    // work groups of exactly one tile, so that SYNC_WARPS (a barrier there) is reached uniformly.
    pairBlockSize = (cc.getSIMDWidth() < 32 ? ComputeContext::TileSize : nb.getForceThreadBlockSize());
    maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : cc.getNumAtomBlocks()*(cc.getNumAtomBlocks()+1)/2);
    bool dp = cc.getUseDoublePrecision();
    for (int kind = 0; kind < 3; kind++) {
        map<string, string> defines;
        defines["PAIR_KIND"] = cc.intToString(kind);
        defines["NUM_ATOMS"] = cc.intToString(numParticles);
        defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cc.intToString(cc.getNumAtomBlocks());
        defines["TILE_SIZE"] = cc.intToString(ComputeContext::TileSize);
        defines["THREAD_BLOCK_SIZE"] = cc.intToString(pairBlockSize);
        defines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(nb.getExclusionTiles().getSize());
        defines["CUTOFF"] = cc.doubleToString(cutoff);
        defines["CUTOFF_SQUARED"] = cc.doubleToString(cutoff*cutoff);
        defines["ENERGY_SCALE"] = cc.doubleToString(ke);
        defines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
        defines["NSERIES"] = (dp ? "15" : "12");
        defines["DAMP_CUT"] = (dp ? "6.0" : "4.5");
        if (usePME) {
            defines["USE_EWALD"] = "1";
            defines["USE_PERIODIC"] = "1";
            defines["USE_CUTOFF"] = "1";
            defines["EWALD_ALPHA"] = cc.doubleToString(beta);
        }
        ComputeProgram program = cc.compileProgram(CommonPGMKernelSources::pgmPairs, defines);
        ComputeKernel k = program->createKernel("computePairs");
        k->addArg(cc.getPosq());
        k->addArg(nb.getExclusionTiles());
        k->addArg(kind == 0 ? inducedField : (kind == 1 ? field : totalField));
        if (kind == 2) {
            k->addArg(cc.getLongForceBuffer());
            k->addArg(cc.getEnergyBuffer());
        }
        k->addArg(charge);
        k->addArg(rgauss);
        k->addArg(kind == 0 ? cgP : (kind == 1 ? permDipole : totalDipole));
        if (nb.getUseCutoff()) {
            k->addArg(nb.getInteractingTiles());
            k->addArg(nb.getInteractionCount());
            for (int i = 0; i < 5; i++)
                k->addArg();
            k->addArg(maxTiles);
            k->addArg(nb.getBlockCenters());
            k->addArg(nb.getBlockBoundingBoxes());
            k->addArg(nb.getInteractingAtoms());
        }
        else
            k->addArg(maxTiles);
        if (kind == 0)
            pairInducedKernel = k;
        else if (kind == 1)
            pairFixedKernel = k;
        else
            pairForceKernel = k;
    }
    hasInitializedKernels = true;
}

void CommonCalcPGMForceKernel::setBoxArgs(ComputeKernel& kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
}

void CommonCalcPGMForceKernel::setPmeBoxArgs(ComputeKernel& kernel, int index) {
    // periodicBoxVecX/Y/Z, recipBoxVecX/Y/Z
    Vec3 v[3];
    cc.getPeriodicBoxVectors(v[0], v[1], v[2]);
    mm_double4 recip[3];
    cc.computeReciprocalBoxVectors(recip);
    if (cc.getUseDoublePrecision()) {
        for (int i = 0; i < 3; i++)
            kernel->setArg(index+i, mm_double4(v[i][0], v[i][1], v[i][2], 0));
        for (int i = 0; i < 3; i++)
            kernel->setArg(index+3+i, recip[i]);
    }
    else {
        for (int i = 0; i < 3; i++)
            kernel->setArg(index+i, mm_float4((float) v[i][0], (float) v[i][1], (float) v[i][2], 0));
        for (int i = 0; i < 3; i++)
            kernel->setArg(index+3+i, mm_float4((float) recip[i].x, (float) recip[i].y, (float) recip[i].z, 0));
    }
}

void CommonCalcPGMForceKernel::runPairs(ComputeKernel& kernel) {
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    int threads = nb.getNumForceThreadBlocks()*pairBlockSize;
    int maxThreads = (cc.getEnergyBuffer().getSize()/pairBlockSize)*pairBlockSize;   // energyBuffer[GLOBAL_ID]
    kernel->execute(min(threads, maxThreads), pairBlockSize);
}

void CommonCalcPGMForceKernel::pmeField(ComputeArray& dipoles, float chargeScale, ComputeArray& fieldBuffer) {
    spreadKernel->setArg(2, dipoles);
    if (cc.getUseDoublePrecision())
        spreadKernel->setArg(3, (double) chargeScale);
    else
        spreadKernel->setArg(3, chargeScale);
    cc.clearBuffer(pmeGridLong);
    spreadKernel->execute(numParticles);
    finishSpreadKernel->execute(gridX*gridY*gridZ);
    fft->execFFT(pmeGrid1, pmeGrid2, true);
    convKernel->setArg(9, (int) 0);
    convKernel->execute(gridX*gridY*(gridZ/2+1));
    fft->execFFT(pmeGrid2, pmeGrid1, false);
    fieldFromGridKernel->setArg(2, fieldBuffer);
    fieldFromGridKernel->execute(numParticles);
}

vector<double> CommonCalcPGMForceKernel::downloadScalars() {
    vector<double> s(8);
    if (scalars.getElementSize() == sizeof(double))
        scalars.download(s);
    else {
        vector<float> f(8);
        scalars.download(f);
        for (int i = 0; i < 8; i++)
            s[i] = f[i];
    }
    return s;
}

double CommonCalcPGMForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);
    if (!hasInitializedKernels)
        initKernels();
    evaluated = true;
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    if (nb.getUseCutoff() && maxTiles < nb.getInteractingTiles().getSize()) {
        // The neighbor list grew: point the pair kernels at the new arrays.
        maxTiles = nb.getInteractingTiles().getSize();
        for (ComputeKernel* k : {&pairInducedKernel, &pairFixedKernel, &pairForceKernel}) {
            int base = (k == &pairForceKernel ? 8 : 6);
            (*k)->setArg(base, nb.getInteractingTiles());
            (*k)->setArg(base+7, maxTiles);
            (*k)->setArg(base+10, nb.getInteractingAtoms());
        }
    }
    if (usePME) {
        setBoxArgs(permKernel, 5);
        setBoxArgs(covForceKernel, 7);
        setBoxArgs(pairInducedKernel, 8);
        setBoxArgs(pairFixedKernel, 8);
        setBoxArgs(pairForceKernel, 10);
        setPmeBoxArgs(spreadKernel, 5);
        setPmeBoxArgs(fieldFromGridKernel, 3);
        setPmeBoxArgs(forceFromGridKernel, 6);
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        mm_double4 recip[3];
        cc.computeReciprocalBoxVectors(recip);
        if (cc.getUseDoublePrecision()) {
            convKernel->setArg(4, mm_double4(a[0], b[1], c[2], 0));
            for (int i = 0; i < 3; i++)
                convKernel->setArg(5+i, recip[i]);
        }
        else {
            convKernel->setArg(4, mm_float4((float) a[0], (float) b[1], (float) c[2], 0));
            for (int i = 0; i < 3; i++)
                convKernel->setArg(5+i, mm_float4((float) recip[i].x, (float) recip[i].y, (float) recip[i].z, 0));
        }
    }
    int padded = cc.getPaddedNumAtoms();
    permKernel->execute(padded);

    if (polarizable) {
        // permanent field (direct + reciprocal; the self term is added in the solver kernels)
        cc.clearBuffer(field);
        runPairs(pairFixedKernel);
        if (usePME)
            pmeField(permDipole, 1.0f, field);

        // initial guess
        long long step = cc.getStepCount();
        bool sameStep = (histCount > 0 && step == lastStep);
        int mode = 0;
        if (sameStep)
            mode = 1;
        else if (predictor == PGMForce::Mu4 && histCount >= 4)
            mode = 3;
        else if (predictor >= PGMForce::Mu3 && histCount >= 3)
            mode = 2;
        else if (predictor != PGMForce::NoPredictor && histCount >= 1)
            mode = 1;
        for (int k = 0; k < 4; k++)
            guessKernel->setArg(3+k, hist[(histHead-k+4)%4]);
        guessKernel->setArg(8, mode);
        guessKernel->execute(padded);

        // conjugate gradients
        cc.clearBuffer(inducedField);
        pairInducedKernel->setArg(5, inducedDipole);
        runPairs(pairInducedKernel);
        if (usePME)
            pmeField(inducedDipole, 0.0f, inducedField);
        cgInitKernel->execute(cgGroupSize, cgGroupSize);
        pairInducedKernel->setArg(5, cgP);
        double err = downloadScalars()[2];
        int it = 0;
        while (err > tol && it < maxIterations) {
            cc.clearBuffer(inducedField);
            runPairs(pairInducedKernel);
            if (usePME)
                pmeField(cgP, 0.0f, inducedField);
            cgStepKernel->execute(cgGroupSize, cgGroupSize);
            err = downloadScalars()[2];
            it++;
        }
        lastIterations = it;
        // the history holds one solution per step (the first evaluation of the step): a second evaluation
        // at the same step (a barostat trial, getState) starts from it but must not replace it
        if (!sameStep) {
            histHead = (histHead+1)%4;
            histCount = min(histCount+1, 4);
            cgFinishKernel->setArg(3, hist[histHead]);
        }
        else
            cgFinishKernel->setArg(3, histScratch);
        cgFinishKernel->execute(padded);
        lastStep = step;
    }
    else {
        cc.clearBuffer(inducedDipole);
        lastIterations = 0;
    }

    // energy, forces and the total field at fixed dipoles
    totalKernel->execute(3*padded);
    cc.clearBuffer(totalField);
    runPairs(pairForceKernel);
    if (usePME) {
        spreadKernel->setArg(2, totalDipole);
        if (cc.getUseDoublePrecision())
            spreadKernel->setArg(3, 1.0);
        else
            spreadKernel->setArg(3, 1.0f);
        cc.clearBuffer(pmeGridLong);
        spreadKernel->execute(numParticles);
        finishSpreadKernel->execute(gridX*gridY*gridZ);
        fft->execFFT(pmeGrid1, pmeGrid2, true);
        convKernel->setArg(9, (int) (includeEnergy ? 1 : 0));
        convKernel->execute(gridX*gridY*(gridZ/2+1));
        fft->execFFT(pmeGrid2, pmeGrid1, false);
        forceFromGridKernel->execute(numParticles);
    }
    if (includeEnergy)
        selfKernel->execute(numParticles);
    if (numCovalent > 0)
        covForceKernel->execute(numCovalent);
    if (usePME && includeEnergy) {
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        double volume = a[0]*b[1]*c[2];
        return -ke*M_PI*totalCharge*totalCharge/(2*volume*beta*beta);
    }
    return 0.0;
}

void CommonCalcPGMForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    // the dipoles of the last evaluation; before the first one, evaluate this force's group
    if (!evaluated)
        context.calcForcesAndEnergy(false, true, 1<<forceGroup);
    ContextSelector selector(cc);
    vector<double> d;
    inducedDipole.download(d, true);
    const vector<int>& order = cc.getAtomIndex();
    dipoles.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
}

void CommonCalcPGMForceKernel::getPermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    if (usePME)
        setBoxArgs(permKernel, 5);
    permKernel->execute(cc.getPaddedNumAtoms());
    vector<double> d;
    permDipole.download(d, true);
    const vector<int>& order = cc.getAtomIndex();
    dipoles.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
}

void CommonCalcPGMForceKernel::copyParametersToContext(ContextImpl& context, const PGMForce& force) {
    ContextSelector selector(cc);
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: the number of particles has changed");
    if (force.getNumCovalentDipoles() != numCovalent)
        throw OpenMMException("updateParametersInContext: the number of covalent dipoles has changed");
    for (int m = 0; m < numCovalent; m++) {
        int a, b;
        double c;
        force.getCovalentDipoleParameters(m, a, b, c);
        if (make_pair(a, b) != covalentPairs[m])
            throw OpenMMException("updateParametersInContext: the atoms of a covalent dipole have changed");
    }
    // In the original atom order, as OpenMM's own forces: the context only exchanges identical molecules
    // (ForceInfo); if the change makes an exchange invalid, invalidateMolecules() restores the original
    // order (and the reorder listener resets the predictor history).
    uploadParameters(force);
    cc.invalidateMolecules(info, true, true);
    resetHistory();
}

void CommonCalcPGMForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = beta;
    nx = gridX;
    ny = gridY;
    nz = gridZ;
}

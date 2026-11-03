#ifndef COMMON_PGM_KERNELS_H_
#define COMMON_PGM_KERNELS_H_

#include "PGMKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/FFT3D.h"

namespace PGMPlugin {

/**
 * PGMForce on the Common platforms (CUDA, OpenCL, HIP).
 */
class CommonCalcPGMForceKernel : public CalcPGMForceKernel {
public:
    CommonCalcPGMForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system);
    ~CommonCalcPGMForceKernel();
    void initialize(const OpenMM::System& system, const PGMForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getPermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    int getInducedDipoleIterations() {
        return lastIterations;
    }
    void copyParametersToContext(OpenMM::ContextImpl& context, const PGMForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

private:
    class ForceInfo;
    class ReorderListener;
    void initKernels();
    void setBoxArgs(OpenMM::ComputeKernel& kernel, int index);
    void setPmeBoxArgs(OpenMM::ComputeKernel& kernel, int index);
    void runPairs(OpenMM::ComputeKernel& kernel);
    /** Reciprocal-space field of (chargeScale q, dipoles) added to a field buffer. */
    void pmeField(OpenMM::ComputeArray& dipoles, float chargeScale, OpenMM::ComputeArray& fieldBuffer);
    void uploadParameters(const PGMForce& force);
    void resetHistory() {
        histCount = 0;
    }
    std::vector<double> downloadScalars();

    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    bool hasInitializedKernels, usePME, polarizable;
    int numParticles, numCovalent, maxIterations, predictor, lastIterations, histCount, histHead, gridX, gridY, gridZ, pmeOrder;
    long long lastStep;
    double tol, peek, ke, beta, cutoff, totalCharge, cself;
    OpenMM::ComputeArray charge, rgauss, polarizability, covStart, covAtom, covPartner, covStrength;
    OpenMM::ComputeArray permDipole, inducedDipole, totalDipole, field, inducedField, totalField;
    OpenMM::ComputeArray cgR, cgZ, cgP, cgAp, scalars, hist[4], histScratch;
    OpenMM::ComputeArray pmeGridLong, pmeGrid1, pmeGrid2, moduliX, moduliY, moduliZ;
    OpenMM::FFT3D fft;
    OpenMM::ComputeKernel permKernel, guessKernel, cgInitKernel, cgStepKernel, cgFinishKernel, totalKernel, selfKernel, covForceKernel;
    OpenMM::ComputeKernel pairInducedKernel, pairFixedKernel, pairForceKernel;
    OpenMM::ComputeKernel spreadKernel, finishSpreadKernel, convKernel, fieldFromGridKernel, forceFromGridKernel;
    int pairBlockSize, maxTiles;
    int cgGroupSize;                                   // threads of the single-group CG kernels (power of 2)
    ForceInfo* info = nullptr;
    bool evaluated = false;
    int forceGroup = 0;
    std::vector<std::pair<int, int> > covalentPairs;   // fixed after initialize
};

} // namespace PGMPlugin

#endif

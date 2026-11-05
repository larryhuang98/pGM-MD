#ifndef CUDA_PGM_KERNELS_H_
#define CUDA_PGM_KERNELS_H_

#include "PGMKernels.h"
#include "openmm/cuda/CudaContext.h"
#include <memory>
#include <vector>

namespace pgmcuda {
class Engine;
}

namespace PGMPlugin {

class CommonCalcPGMForceKernel;

/**
 * PGMForce on the CUDA platform.  In mixed precision with PME the whole force runs in the engine of
 * core/cuda/pgm_cuda (its own pair list, PME with cuFFT, the CG on the device, one CUDA graph per step), on
 * OpenMM's CUDA context: positions from OpenMM's posq, forces added to OpenMM's force buffer.  Otherwise (single
 * or double precision, no cutoff, PGM_OPENMM_COMMON=1) the Common kernels.
 */
class CudaCalcPGMForceKernel : public CalcPGMForceKernel {
public:
    CudaCalcPGMForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu,
                           const OpenMM::System& system);
    ~CudaCalcPGMForceKernel();
    void initialize(const OpenMM::System& system, const PGMForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getPermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    int getInducedDipoleIterations();
    void copyParametersToContext(OpenMM::ContextImpl& context, const PGMForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

private:
    class ForceInfo;
    class ReorderListener;
    class PostComputation;
    double finish(bool includeEnergy);   // the evaluation launched by execute (PostComputation)
    void setAtoms();   // the parameters in the context's current atom order
    void loadParameters(const PGMForce& force);
    OpenMM::CudaContext& cu;
    const OpenMM::System& system;
    std::unique_ptr<CommonCalcPGMForceKernel> common;   // the fallback
    std::unique_ptr<pgmcuda::Engine> engine;
    int numParticles = 0, forceGroup = 0, gridX = 0, gridY = 0, gridZ = 0, lastIterations = 0;
    double beta = 0;
    bool polarizable = true, atomsDirty = true, evaluated = false, loaded = false, launched = false;
    bool wantEnergy = false, deferred = false;   // deferred: no host wait at steps without energies
    long long lastStep = -1;
    std::vector<double> q, radius, alpha;   // the original order
    std::vector<int> covAtom, covPartner;
    std::vector<double> covStrength;
    ForceInfo* info = nullptr;
};

} // namespace PGMPlugin

#endif

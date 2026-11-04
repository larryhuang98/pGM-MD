#include "CommonGVDWKernels.h"
#include "CommonPGMKernels.h"
#include "CudaPGMKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/windowsExport.h"
#include <exception>

using namespace PGMPlugin;
using namespace OpenMM;

namespace PGMPlugin {
class CudaPGMKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
        CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
        if (name == CalcPGMForceKernel::Name())
            return new CudaCalcPGMForceKernel(name, platform, cu, context.getSystem());
        if (name == CalcGVDWForceKernel::Name())
            return new CommonCalcGVDWForceKernel(name, platform, cu, context.getSystem());
        throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
    }
};
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaPGMKernelFactory* factory = new CudaPGMKernelFactory();
        platform.registerKernelFactory(CalcPGMForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcGVDWForceKernel::Name(), factory);
    }
    catch (std::exception& ex) {
        // CUDA is not available
    }
}

extern "C" OPENMM_EXPORT void registerPGMCudaKernelFactories() {
    registerKernelFactories();
}

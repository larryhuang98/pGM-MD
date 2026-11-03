#include "CommonPGMKernels.h"
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
            return new CommonCalcPGMForceKernel(name, platform, cu, context.getSystem());
        throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
    }
};
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        platform.registerKernelFactory(CalcPGMForceKernel::Name(), new CudaPGMKernelFactory());
    }
    catch (std::exception& ex) {
        // CUDA is not available
    }
}

extern "C" OPENMM_EXPORT void registerPGMCudaKernelFactories() {
    registerKernelFactories();
}

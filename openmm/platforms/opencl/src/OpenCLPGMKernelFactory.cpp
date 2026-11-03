#include "CommonGVDWKernels.h"
#include "CommonPGMKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLPlatform.h"
#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/windowsExport.h"
#include <exception>

using namespace PGMPlugin;
using namespace OpenMM;

namespace PGMPlugin {
class OpenCLPGMKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
        OpenCLContext& cl = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
        if (name == CalcPGMForceKernel::Name())
            return new CommonCalcPGMForceKernel(name, platform, cl, context.getSystem());
        if (name == CalcGVDWForceKernel::Name())
            return new CommonCalcGVDWForceKernel(name, platform, cl, context.getSystem());
        throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
    }
};
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("OpenCL");
        OpenCLPGMKernelFactory* factory = new OpenCLPGMKernelFactory();
        platform.registerKernelFactory(CalcPGMForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcGVDWForceKernel::Name(), factory);
    }
    catch (std::exception& ex) {
        // OpenCL is not available
    }
}

extern "C" OPENMM_EXPORT void registerPGMOpenCLKernelFactories() {
    registerKernelFactories();
}

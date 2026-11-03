#include "ReferenceGVDWKernels.h"
#include "ReferencePGMKernels.h"
#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace PGMPlugin;
using namespace OpenMM;

namespace PGMPlugin {
class ReferencePGMKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
        if (name == CalcPGMForceKernel::Name())
            return new ReferenceCalcPGMForceKernel(name, platform);
        if (name == CalcGVDWForceKernel::Name())
            return new ReferenceCalcGVDWForceKernel(name, platform);
        throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
    }
};
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferencePGMKernelFactory* factory = new ReferencePGMKernelFactory();
            platform.registerKernelFactory(CalcPGMForceKernel::Name(), factory);
            platform.registerKernelFactory(CalcGVDWForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerPGMReferenceKernelFactories() {
    registerKernelFactories();
}

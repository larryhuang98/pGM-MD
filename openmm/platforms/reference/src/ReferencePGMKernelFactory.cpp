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
        throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
    }
};
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL)
            platform.registerKernelFactory(CalcPGMForceKernel::Name(), new ReferencePGMKernelFactory());
    }
}

extern "C" OPENMM_EXPORT void registerPGMReferenceKernelFactories() {
    registerKernelFactories();
}

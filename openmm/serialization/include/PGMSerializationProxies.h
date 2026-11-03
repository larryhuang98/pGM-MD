#ifndef OPENMM_PGM_SERIALIZATION_PROXIES_H_
#define OPENMM_PGM_SERIALIZATION_PROXIES_H_

#include "internal/windowsExportPGM.h"
#include "openmm/serialization/SerializationProxy.h"

namespace PGMPlugin {

/** XML serialization of PGMForce (XmlSerializer, System files, Context checkpoints of Systems). */
class OPENMM_EXPORT_PGM PGMForceProxy : public OpenMM::SerializationProxy {
public:
    PGMForceProxy();
    void serialize(const void* object, OpenMM::SerializationNode& node) const;
    void* deserialize(const OpenMM::SerializationNode& node) const;
};

/** XML serialization of GVDWForce. */
class OPENMM_EXPORT_PGM GVDWForceProxy : public OpenMM::SerializationProxy {
public:
    GVDWForceProxy();
    void serialize(const void* object, OpenMM::SerializationNode& node) const;
    void* deserialize(const OpenMM::SerializationNode& node) const;
};

} // namespace PGMPlugin

#endif

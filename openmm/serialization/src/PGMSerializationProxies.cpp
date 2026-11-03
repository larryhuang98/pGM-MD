#include "PGMSerializationProxies.h"
#include "GVDWForce.h"
#include "PGMForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"
#include <typeinfo>

using namespace PGMPlugin;
using namespace OpenMM;
using namespace std;

// Version 1 of both formats.  A later version must keep reading version 1 (new properties with
// defaults), so that saved Systems stay usable.

PGMForceProxy::PGMForceProxy() : SerializationProxy("PGMForce") {
}

void PGMForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const PGMForce& force = *reinterpret_cast<const PGMForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setIntProperty("method", (int) force.getNonbondedMethod());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("alpha", alpha).setIntProperty("nx", nx).setIntProperty("ny", ny).setIntProperty("nz", nz);
    node.setIntProperty("pmeOrder", force.getPMEOrder());
    node.setDoubleProperty("ewaldTolerance", force.getEwaldErrorTolerance());
    node.setIntProperty("polarization", (int) force.getPolarization());
    node.setDoubleProperty("inducedTolerance", force.getInducedDipoleTolerance());
    node.setIntProperty("maxIterations", force.getMaxInducedDipoleIterations());
    node.setDoubleProperty("peek", force.getPeekCoefficient());
    node.setIntProperty("predictor", (int) force.getPredictor());
    node.setDoubleProperty("coulomb", force.getCoulombConstant());
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double q, R, a;
        force.getParticleParameters(i, q, R, a);
        particles.createChildNode("Particle").setDoubleProperty("q", q).setDoubleProperty("radius", R).setDoubleProperty("alpha", a);
    }
    SerializationNode& dipoles = node.createChildNode("CovalentDipoles");
    for (int i = 0; i < force.getNumCovalentDipoles(); i++) {
        int p, j;
        double c;
        force.getCovalentDipoleParameters(i, p, j, c);
        dipoles.createChildNode("Dipole").setIntProperty("particle", p).setIntProperty("partner", j).setDoubleProperty("strength", c);
    }
}

void* PGMForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 1)
        throw OpenMMException("Unsupported version number of PGMForce");
    PGMForce* force = new PGMForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setNonbondedMethod((PGMForce::NonbondedMethod) node.getIntProperty("method"));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setPMEParameters(node.getDoubleProperty("alpha"), node.getIntProperty("nx"), node.getIntProperty("ny"), node.getIntProperty("nz"));
        force->setPMEOrder(node.getIntProperty("pmeOrder"));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldTolerance"));
        force->setPolarization((PGMForce::Polarization) node.getIntProperty("polarization"));
        force->setInducedDipoleTolerance(node.getDoubleProperty("inducedTolerance"));
        force->setMaxInducedDipoleIterations(node.getIntProperty("maxIterations"));
        force->setPeekCoefficient(node.getDoubleProperty("peek"));
        force->setPredictor((PGMForce::Predictor) node.getIntProperty("predictor"));
        force->setCoulombConstant(node.getDoubleProperty("coulomb"));
        for (const SerializationNode& p : node.getChildNode("Particles").getChildren())
            force->addParticle(p.getDoubleProperty("q"), p.getDoubleProperty("radius"), p.getDoubleProperty("alpha"));
        for (const SerializationNode& d : node.getChildNode("CovalentDipoles").getChildren())
            force->addCovalentDipole(d.getIntProperty("particle"), d.getIntProperty("partner"), d.getDoubleProperty("strength"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}

GVDWForceProxy::GVDWForceProxy() : SerializationProxy("GVDWForce") {
}

void GVDWForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const GVDWForce& force = *reinterpret_cast<const GVDWForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setIntProperty("method", (int) force.getNonbondedMethod());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setIntProperty("repulsion", (int) force.getRepulsionForm());
    node.setBoolProperty("dispersionCorrection", force.getUseDispersionCorrection());
    node.setBoolProperty("exceptionsUsePeriodic", force.getExceptionsUsePeriodicBoundaryConditions());
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double R, a, c, b;
        force.getParticleParameters(i, R, a, c, b);
        particles.createChildNode("Particle").setDoubleProperty("radius", R).setDoubleProperty("sqrtA", a)
                .setDoubleProperty("sqrtC6", c).setDoubleProperty("b", b);
    }
    SerializationNode& exceptions = node.createChildNode("Exceptions");
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int p1, p2;
        double A, C6, B;
        force.getExceptionParameters(i, p1, p2, A, C6, B);
        exceptions.createChildNode("Exception").setIntProperty("p1", p1).setIntProperty("p2", p2).setDoubleProperty("A", A)
                .setDoubleProperty("C6", C6).setDoubleProperty("B", B);
    }
}

void* GVDWForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 1)
        throw OpenMMException("Unsupported version number of GVDWForce");
    GVDWForce* force = new GVDWForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setNonbondedMethod((GVDWForce::NonbondedMethod) node.getIntProperty("method"));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setRepulsionForm((GVDWForce::RepulsionForm) node.getIntProperty("repulsion"));
        force->setUseDispersionCorrection(node.getBoolProperty("dispersionCorrection"));
        force->setExceptionsUsePeriodicBoundaryConditions(node.getBoolProperty("exceptionsUsePeriodic"));
        for (const SerializationNode& p : node.getChildNode("Particles").getChildren())
            force->addParticle(p.getDoubleProperty("radius"), p.getDoubleProperty("sqrtA"), p.getDoubleProperty("sqrtC6"), p.getDoubleProperty("b"));
        for (const SerializationNode& e : node.getChildNode("Exceptions").getChildren())
            force->addException(e.getIntProperty("p1"), e.getIntProperty("p2"), e.getDoubleProperty("A"), e.getDoubleProperty("C6"), e.getDoubleProperty("B"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}

// Registered when the library is loaded (as the AMOEBA plugin does).
#if defined(WIN32)
#include <windows.h>
extern "C" OPENMM_EXPORT_PGM void registerPGMSerializationProxies();
BOOL WINAPI DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    if (ul_reason_for_call == DLL_PROCESS_ATTACH)
        registerPGMSerializationProxies();
    return TRUE;
}
#else
extern "C" void __attribute__((constructor)) registerPGMSerializationProxies();
#endif

extern "C" OPENMM_EXPORT_PGM void registerPGMSerializationProxies() {
    SerializationProxy::registerProxy(typeid(PGMForce), new PGMForceProxy());
    SerializationProxy::registerProxy(typeid(GVDWForce), new GVDWForceProxy());
}

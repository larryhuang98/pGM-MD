#ifndef OPENMM_TAILCORRECTIONFORCE_H_
#define OPENMM_TAILCORRECTIONFORCE_H_

#include "openmm/Force.h"
#include "internal/windowsExportPGM.h"

namespace PGMPlugin {

/**
 * The energy coefficient / V of the periodic box (volume V), no forces: a long-range dispersion correction
 * such as Amber vdwmeth = 1 (coefficient -2 pi/(3 rc^3) sum_ij B_ij), which depends on the box only.
 * Evaluated on the host from the box vectors, on every platform, without kernels or copies (OpenMM
 * CustomVolumeForce computes on the host from downloaded positions and uploads forces, every step).  The
 * coefficient is read at every evaluation.  Its virial, -dE/dV V = E, enters the pressure through the
 * energy (MonteCarloBarostat).
 */
class OPENMM_EXPORT_PGM TailCorrectionForce : public OpenMM::Force {
public:
    explicit TailCorrectionForce(double coefficient = 0.0) : coefficient(coefficient) {
    }
    double getCoefficient() const {
        return coefficient;
    }
    void setCoefficient(double c) {
        coefficient = c;
    }
    bool usesPeriodicBoundaryConditions() const {
        return true;
    }

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    double coefficient;
};

} // namespace PGMPlugin

#endif

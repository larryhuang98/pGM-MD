#ifndef OPENMM_GVDWFORCE_H_
#define OPENMM_GVDWFORCE_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM pGM plugin                                   *
 *                                                                            *
 * GVDW, the Gaussian-density van der Waals of Huang, Luo and Duan            *
 * (pmemd-pgm igvdw = 1, pGM-JAX vdw.py), as a native OpenMM force.           *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "internal/windowsExportPGM.h"
#include <vector>

namespace PGMPlugin {

/**
 * GVDW: a finite van der Waals potential of Gaussian atomic densities.
 *
 * Every atom has the Gaussian radius R of the pGM electrostatics and three GVDW parameters
 * a = sqrt(A), c = sqrt(C6), b.  A pair interacts through
 *
 *     U_ij = A_ij exp(-B_ij y^2)  (Gaussian repulsion)  or  A_ij exp(-B_ij y)  (Slater repulsion)
 *            - C6_ij F(y) / r^6,      y = b_ij r,  b_ij = 1/sqrt(2 (R_i^2 + R_j^2)),
 *
 * F(y) = A(y)^2 + g(y)^2/2 the damping of the dispersion between two Gaussian clouds (F -> 1 for
 * large y, F/r^6 finite at r = 0), A_ij = a_i a_j, C6_ij = c_i c_j, B_ij = (b_i + b_j)/2.
 * Atoms with a = c = 0 do not interact.  pmemd-pgm's global gvdw_arep, gvdw_c6, b_rep_scale are
 * a = sqrt(gvdw_arep), c = sqrt(gvdw_c6), b = b_rep_scale on every atom that carries GVDW.
 *
 * The potential is cut (not shifted) at the cutoff; the optional dispersion correction is Amber's
 * vdwmeth = 1 for the C6 term, E = -2 pi (sum_i c_i)^2 / (3 V rc^3).  Excluded pairs (bonded
 * neighbours) do not interact, as with Lennard-Jones.
 *
 * Units: nm, kJ/mol; a [sqrt(kJ/mol)], c [sqrt(kJ/mol nm^6)], b dimensionless.
 */
class OPENMM_EXPORT_PGM GVDWForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        /** All pairs, no periodic boundary conditions. */
        NoCutoff = 0,
        /** Pairs within the cutoff, no periodic boundary conditions. */
        CutoffNonPeriodic = 1,
        /** Pairs within the cutoff, minimum image convention. */
        CutoffPeriodic = 2
    };
    enum RepulsionForm {
        /** A exp(-B y^2): pmemd-pgm gvdw_rep_form = 0. */
        Gaussian = 0,
        /** A exp(-B y) (Born-Mayer): pmemd-pgm gvdw_rep_form = 1. */
        Slater = 1
    };

    GVDWForce();

    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Add an atom.
     *
     * @param radius  pGM Gaussian radius R [nm] (> 0), the same as in PGMForce
     * @param sqrtA   square root of the repulsion amplitude [sqrt(kJ/mol)]
     * @param sqrtC6  square root of the dispersion coefficient [sqrt(kJ/mol nm^6)]
     * @param b       repulsion exponent scale (dimensionless)
     * @return the index of the atom
     */
    int addParticle(double radius, double sqrtA, double sqrtC6, double b);
    void getParticleParameters(int index, double& radius, double& sqrtA, double& sqrtC6, double& b) const;
    void setParticleParameters(int index, double radius, double sqrtA, double sqrtC6, double b);

    int getNumExclusions() const {
        return exclusions.size();
    }
    /** Exclude a pair of atoms from the interaction; returns the index of the exclusion. */
    int addExclusion(int particle1, int particle2);
    void getExclusionParticles(int index, int& particle1, int& particle2) const;
    void setExclusionParticles(int index, int particle1, int particle2);

    NonbondedMethod getNonbondedMethod() const {
        return nonbondedMethod;
    }
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const {
        return cutoffDistance;
    }
    void setCutoffDistance(double distance);
    RepulsionForm getRepulsionForm() const {
        return repulsionForm;
    }
    void setRepulsionForm(RepulsionForm form);
    /** Dispersion correction (periodic systems only). */
    bool getUseDispersionCorrection() const {
        return useDispersionCorrection;
    }
    void setUseDispersionCorrection(bool use) {
        useDispersionCorrection = use;
    }
    /**
     * Copy changed particle parameters (radius, a, c, b) into a Context.  The exclusions, the
     * method and the repulsion form cannot be changed this way.
     */
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == CutoffPeriodic;
    }

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    class ParticleInfo;
    NonbondedMethod nonbondedMethod;
    RepulsionForm repulsionForm;
    double cutoffDistance;
    bool useDispersionCorrection;
    std::vector<ParticleInfo> particles;
    std::vector<std::pair<int, int> > exclusions;
};

class GVDWForce::ParticleInfo {
public:
    double radius, sqrtA, sqrtC6, b;
    ParticleInfo() : radius(0), sqrtA(0), sqrtC6(0), b(1) {
    }
    ParticleInfo(double radius, double sqrtA, double sqrtC6, double b) : radius(radius), sqrtA(sqrtA), sqrtC6(sqrtC6), b(b) {
    }
};

} // namespace PGMPlugin

#endif /*OPENMM_GVDWFORCE_H_*/

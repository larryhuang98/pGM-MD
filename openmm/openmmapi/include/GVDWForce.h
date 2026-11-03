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
#include <map>
#include <utility>
#include <vector>

namespace PGMPlugin {

/**
 * GVDW: a finite van der Waals potential of Gaussian atomic densities.
 *
 * Every atom has a Gaussian radius R (the pGM radius of PGMForce, set here separately) and three GVDW
 * parameters a = sqrt(A), c = sqrt(C6), b.  Two atoms interact through
 *
 *     U_ij = A_ij exp(-B_ij y^2)  (Gaussian repulsion)  or  A_ij exp(-B_ij y)  (Slater repulsion)
 *            - C6_ij F(y) / r^6,      y = b_ij r,  b_ij = 1/sqrt(2 (R_i^2 + R_j^2)),
 *
 * F(y) = A(y)^2 + g(y)^2/2 the damping of the dispersion between two Gaussian clouds (F -> 1 for large
 * y, F/r^6 finite at r = 0).  The pair parameters come from the combination rule
 * A_ij = a_i a_j, C6_ij = c_i c_j, B_ij = (b_i + b_j)/2, unless the pair is an exception.
 * pmemd-pgm's global gvdw_arep, gvdw_c6, b_rep_scale are a = sqrt(gvdw_arep), c = sqrt(gvdw_c6),
 * b = b_rep_scale on every atom that carries GVDW; atoms with a = c = 0 do not interact.
 *
 * Exceptions (as in NonbondedForce) give a pair its own A_ij, C6_ij, B_ij: A_ij = C6_ij = 0 excludes it
 * (bonded neighbours), other values scale or replace it (1-4 pairs, pair-specific fits).  An exception
 * is evaluated for every step regardless of the cutoff, with the minimum image only if
 * setExceptionsUsePeriodicBoundaryConditions(true); b_ij always comes from the two radii.
 *
 * The potential is cut (not shifted) at the cutoff.  The optional dispersion correction is Amber's
 * vdwmeth = 1 for the C6 term, E = -2 pi (sum_i c_i)^2 / (3 V rc^3), over all ordered pairs with the
 * combination rule (exceptions do not change it, as in Amber).
 *
 * Units: nm, kJ/mol; a [sqrt(kJ/mol)], c [sqrt(kJ/mol nm^6)], b dimensionless; exception A_ij
 * [kJ/mol], C6_ij [kJ/mol nm^6], B_ij dimensionless.
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

    // ------------------------------------------------------------------ particles
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Add an atom.
     *
     * @param radius  Gaussian radius R [nm] (> 0), normally the pGM radius of PGMForce
     * @param sqrtA   a = sqrt of the repulsion amplitude [sqrt(kJ/mol)] (>= 0)
     * @param sqrtC6  c = sqrt of the dispersion coefficient [sqrt(kJ/mol nm^6)] (>= 0)
     * @param b       repulsion exponent scale (>= 0, dimensionless)
     * @return the index of the atom
     */
    int addParticle(double radius, double sqrtA, double sqrtC6, double b);
    void getParticleParameters(int index, double& radius, double& sqrtA, double& sqrtC6, double& b) const;
    void setParticleParameters(int index, double radius, double sqrtA, double sqrtC6, double b);

    // ------------------------------------------------------------------ exceptions
    int getNumExceptions() const {
        return exceptions.size();
    }
    /**
     * Give a pair its own parameters (A = C6 = 0: exclude it).
     *
     * @param particle1, particle2  the two atoms
     * @param A        repulsion amplitude A_ij [kJ/mol]
     * @param C6       dispersion coefficient C6_ij [kJ/mol nm^6]
     * @param B        repulsion exponent scale B_ij
     * @param replace  if the pair is already an exception, replace it (otherwise that is an error)
     * @return the index of the exception
     */
    int addException(int particle1, int particle2, double A, double C6, double B, bool replace = false);
    void getExceptionParameters(int index, int& particle1, int& particle2, double& A, double& C6, double& B) const;
    void setExceptionParameters(int index, int particle1, int particle2, double A, double C6, double B);
    /**
     * Exclude the 1-2 and 1-3 pairs of a bond graph and scale the combination-rule parameters of the 1-4
     * pairs: A_ij *= scale14A, C6_ij *= scale14C6 (0, 0: exclude them too; NonbondedForce's
     * createExceptionsFromBonds), with the particle parameters at the time of the call.  Pairs that are
     * already exceptions are left alone.
     */
    void createExceptionsFromBonds(const std::vector<std::pair<int, int> >& bonds, double scale14A, double scale14C6);
    bool getExceptionsUsePeriodicBoundaryConditions() const {
        return exceptionsUsePeriodic;
    }
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic) {
        exceptionsUsePeriodic = periodic;
    }

    // ------------------------------------------------------------------ settings
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
     * Copy changed particle parameters and exception parameters into a Context.  The set of exception
     * pairs (and of pairs with A_ij = C6_ij = 0 among them), the method, the cutoff and the repulsion form
     * cannot change this way.
     */
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == CutoffPeriodic;
    }

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    class ParticleInfo;
    class ExceptionInfo;
    NonbondedMethod nonbondedMethod;
    RepulsionForm repulsionForm;
    double cutoffDistance;
    bool useDispersionCorrection, exceptionsUsePeriodic;
    std::vector<ParticleInfo> particles;
    std::vector<ExceptionInfo> exceptions;
    std::map<std::pair<int, int>, int> exceptionMap;
};

class GVDWForce::ParticleInfo {
public:
    double radius, sqrtA, sqrtC6, b;
    ParticleInfo() : radius(1), sqrtA(0), sqrtC6(0), b(0) {
    }
    ParticleInfo(double radius, double sqrtA, double sqrtC6, double b) : radius(radius), sqrtA(sqrtA), sqrtC6(sqrtC6), b(b) {
    }
};

class GVDWForce::ExceptionInfo {
public:
    int particle1, particle2;
    double A, C6, B;
    ExceptionInfo() : particle1(-1), particle2(-1), A(0), C6(0), B(0) {
    }
    ExceptionInfo(int particle1, int particle2, double A, double C6, double B) :
        particle1(particle1), particle2(particle2), A(A), C6(C6), B(B) {
    }
};

} // namespace PGMPlugin

#endif /*OPENMM_GVDWFORCE_H_*/

#ifndef OPENMM_PGMFORCE_H_
#define OPENMM_PGMFORCE_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM pGM plugin                                   *
 *                                                                            *
 * The polarizable Gaussian multipole (pGM) electrostatics of pmemd-pgm and   *
 * pGM-JAX as a native OpenMM force.  See docs/spec.md.                       *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportPGM.h"
#include <vector>

namespace PGMPlugin {

/**
 * pGM electrostatics: Gaussian charges, covalent permanent dipoles and Gaussian induced dipoles.
 *
 * Every atom i has a Gaussian charge q_i, a Gaussian radius R_i and an isotropic polarizability
 * alpha_i.  Its permanent dipole is the sum over its covalent dipoles m = (i, j) of
 * c_m (r_j - r_i)/|r_j - r_i| (no local frames).  Two atoms interact through the Gaussian Coulomb
 * kernel erf(a_ij r)/r with a_ij = 1/sqrt(2 (R_i^2 + R_j^2)), for EVERY pair: this force has no
 * exclusions and no 1-4 scaling (van der Waals terms, which do have exclusions, belong to other
 * forces).  Induced dipoles mu solve (alpha^-1 - T) mu = E_perm and the energy is
 *
 *     E = KE [U(q, p + mu) + sum |mu|^2 / (2 alpha)]
 *
 * with U the Gaussian multipole energy (smooth PME for periodic systems).  Atoms with
 * alpha = 0 are not polarizable.
 *
 * Units: nm, e, e nm (covalent dipole strengths), nm^3 (polarizabilities), kJ/mol.
 */
class OPENMM_EXPORT_PGM PGMForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        /** No cutoff, no periodic boundary conditions: every pair through erf(a_ij r)/r. */
        NoCutoff = 0,
        /** Smooth particle-mesh Ewald for charges and dipoles; direct sum cut at the cutoff. */
        PME = 1
    };
    enum Polarization {
        /** Mutually induced Gaussian dipoles (full pGM). */
        Mutual = 0,
        /** No induced dipoles (Gaussian charges and covalent dipoles only). */
        NoPolarization = 1
    };
    enum Predictor {
        /** Start every solve from mu = alpha E_perm. */
        NoPredictor = 0,
        /** Quadratic extrapolation 3 mu_1 - 3 mu_2 + mu_3 of the previous steps' dipoles. */
        Mu3 = 1,
        /** Cubic extrapolation 4 mu_1 - 6 mu_2 + 4 mu_3 - mu_4 (pmemd-pgm GPU's default). */
        Mu4 = 2
    };

    PGMForce();

    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Add an atom.
     *
     * @param charge          Gaussian charge [e]
     * @param radius          pGM Gaussian radius R [nm] (> 0)
     * @param polarizability  isotropic polarizability [nm^3] (0: not polarizable)
     * @return the index of the atom
     */
    int addParticle(double charge, double radius, double polarizability);
    void getParticleParameters(int index, double& charge, double& radius, double& polarizability) const;
    void setParticleParameters(int index, double charge, double radius, double polarizability);

    int getNumCovalentDipoles() const {
        return covalentDipoles.size();
    }
    /**
     * Add a covalent dipole: p_particle += strength * unit(r_partner - r_particle).
     *
     * @param particle  the atom that carries the dipole
     * @param partner   the atom it points to (usually a bonded neighbour)
     * @param strength  c [e nm]
     * @return the index of the covalent dipole
     */
    int addCovalentDipole(int particle, int partner, double strength);
    void getCovalentDipoleParameters(int index, int& particle, int& partner, double& strength) const;
    void setCovalentDipoleParameters(int index, int particle, int partner, double strength);

    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    /** Cutoff of the direct-space sum [nm] (PME only). */
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    /**
     * PME parameters.  alpha = 0 (the default) selects alpha and the grid from the Ewald error
     * tolerance, as NonbondedForce does.
     *
     * @param alpha  Ewald coefficient beta [1/nm]
     * @param nx, ny, nz  grid points along the three box vectors
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /** B-spline order of PME (4 to 8; default 6 as pGM-JAX and pmemd-pgm inputs). */
    int getPMEOrder() const;
    void setPMEOrder(int order);
    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);
    /** The PME parameters a Context actually uses (after automatic selection). */
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;

    Polarization getPolarization() const;
    void setPolarization(Polarization type);
    /** Convergence of the induced dipoles: max|alpha r| / mean|alpha E_perm| <= tol (componentwise). */
    double getInducedDipoleTolerance() const;
    void setInducedDipoleTolerance(double tol);
    int getMaxInducedDipoleIterations() const;
    void setMaxInducedDipoleIterations(int iter);
    /** Final step mu += omega alpha r after convergence (pmemd scf_sor_coefficient; 0 disables). */
    double getPeekCoefficient() const;
    void setPeekCoefficient(double omega);
    Predictor getPredictor() const;
    void setPredictor(Predictor predictor);
    /** Coulomb constant [kJ mol^-1 nm e^-2]; default ONE_4PI_EPS0 (pmemd-pgm: 332.05382 * 0.4184). */
    double getCoulombConstant() const;
    void setCoulombConstant(double ke);

    /** Induced dipoles of the last evaluation [e nm]. */
    void getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /** Permanent (covalent) dipoles at the current positions [e nm]. */
    void getPermanentDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /** Permanent + induced dipoles [e nm]. */
    void getTotalDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /** Conjugate-gradient iterations of the last solve. */
    int getInducedDipoleIterations(OpenMM::Context& context);

    /** Copy changed particle and covalent-dipole parameters (not the topology) to a Context. */
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == PME;
    }

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    class ParticleInfo;
    class CovalentDipoleInfo;
    NonbondedMethod nonbondedMethod;
    Polarization polarization;
    Predictor predictor;
    double cutoffDistance, alpha, ewaldErrorTol, inducedTol, peek, coulomb;
    int nx, ny, nz, pmeOrder, maxIterations;
    std::vector<ParticleInfo> particles;
    std::vector<CovalentDipoleInfo> covalentDipoles;
};

class PGMForce::ParticleInfo {
public:
    double charge, radius, polarizability;
    ParticleInfo() : charge(0), radius(1), polarizability(0) {
    }
    ParticleInfo(double charge, double radius, double polarizability) :
        charge(charge), radius(radius), polarizability(polarizability) {
    }
};

class PGMForce::CovalentDipoleInfo {
public:
    int particle, partner;
    double strength;
    CovalentDipoleInfo() : particle(-1), partner(-1), strength(0) {
    }
    CovalentDipoleInfo(int particle, int partner, double strength) :
        particle(particle), partner(partner), strength(strength) {
    }
};

} // namespace PGMPlugin

#endif /*OPENMM_PGMFORCE_H_*/

/*
 * pGM electrostatics for GROMACS (coulombtype = pGM).
 *
 * Gaussian charges, covalent permanent dipoles p_i = sum c unit(r_j - r_i) and Gaussian induced dipoles,
 * with smooth PME of point charges and point dipoles (docs/spec.md of pgm-native).  The direct-space
 * sum runs over the nbnxm cluster pair lists, which keep all pairs within the cut-off, excluded pairs
 * included (pGM electrostatics have no exclusions; the nbnxm kernels compute Lennard-Jones only).
 *
 * Phase G1: scalar CPU implementation (double-precision kernels), single rank, the reference of the SIMD and
 * GPU kernels.  Forces go to the ForceWithVirial buffer with the analytical strain derivative as the
 * virial.
 *
 * GVDW (Gaussian-density van der Waals, pmemd-pgm igvdw = 1), if the topology has [ pgm_gvdw ]: pairs
 * within rvdw over the same pair lists, the topology's exclusions applied, [ pgm_gvdw_pairs ] overrides,
 * the C6 dispersion tail with DispCorr; energy in LJ (SR) / "GVDW pairs" / Disper. corr.
 */
#ifndef GMX_PGM_PGM_H
#define GMX_PGM_PGM_H

#include <cstdio>

#include <memory>

#include "gromacs/utility/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

struct gmx_enerdata_t;
struct t_inputrec;
class InteractionDefinitions;

namespace gmx
{
class ForceWithVirial;
class StepWorkload;
template<typename>
class ListOfLists;
struct nonbonded_verlet_t;

class PgmElectrostatics
{
public:
    /*! \brief Set up from the input record (cut-off, Ewald tolerance, PME grid and order, solver options).
     * \param fplog log file for the settings (may be nullptr) */
    PgmElectrostatics(const t_inputrec& ir, FILE* fplog);
    ~PgmElectrostatics();

    /*! \brief Energies, forces and virial of the pGM electrostatics.
     *
     * Solves for the induced dipoles, adds the forces to \p forceWithVirial together with their virial,
     * the direct-space energy to the Coulomb (SR) group pair term, reciprocal + self + background to the
     * Coulomb (recip.) term and the polarization energy sum |mu|^2/(2 alpha) to the PgmAtom term.
     *
     * \param x              local coordinates (as used by the nbnxm pair search)
     * \param box            the box
     * \param numHomeAtoms   number of home atoms
     * \param charges        atom charges (the pGM Gaussian monopoles)
     * \param nbv            the nonbonded setup with the current local CPU pair lists
     * \param idef           local topology: the PgmAtom, PgmCovalentDipole, PgmGvdwAtom, PgmGvdwPair lists
     * \param exclusions     local exclusions (they act on GVDW as on LJ; pGM electrostatics have none)
     * \param shiftVectors   the periodic shift vectors of the pair lists
     * \param globalIndex    global index of each local atom (empty: local = global), for the predictor
     */
    void calculate(ArrayRef<const RVec>      x,
                   const matrix              box,
                   int                       numHomeAtoms,
                   ArrayRef<const real>      charges,
                   const nonbonded_verlet_t& nbv,
                   const InteractionDefinitions& idef,
                   const ListOfLists<int>&   exclusions,
                   ArrayRef<const RVec>      shiftVectors,
                   ArrayRef<const int>       globalIndex,
                   ForceWithVirial*          forceWithVirial,
                   gmx_enerdata_t*           enerd,
                   const StepWorkload&       stepWork);

    //! The induced dipoles [e nm] of the last calculation (local atom order)
    ArrayRef<const DVec> inducedDipoles() const;
    //! The permanent (covalent) dipoles [e nm] of the last calculation
    ArrayRef<const DVec> permanentDipoles() const;
    //! Conjugate-gradient iterations of the last calculation
    int iterations() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif

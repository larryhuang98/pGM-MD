#ifndef PGM_CONSTANTS_H_
#define PGM_CONSTANTS_H_

/* Constants shared by every pGM implementation (pgm_core, OpenMM plugin, GROMACS module). */

namespace pgmcore {

/** Polarizabilities at or below this [nm^3] are not polarizable (pmemd-pgm: 1e-6 A^3). */
constexpr double MIN_POLARIZABILITY = 1e-9;

} // namespace pgmcore

#endif

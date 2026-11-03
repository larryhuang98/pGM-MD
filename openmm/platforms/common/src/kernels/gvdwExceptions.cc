// GVDW exceptions with their own parameters, a BondedUtilities interaction (always evaluated).
// PARAMS[index] (real4): (A_ij, C6_ij, B_ij, b_ij).
const real4 gExc = PARAMS[index];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(delta)
#endif
const real r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
const real gR = SQRT(r2);
const real gInvR = (r2 > 0 ? RSQRT(r2) : (real) 0);
const real gA = gExc.x;
const real gC6 = gExc.y;
const real gB = gExc.z;
const real gBeta = gExc.w;
real gEnergy, gDEdR;
GVDW_PAIR
energy += gEnergy;
delta *= gDEdR;
real3 force1 = -delta;
real3 force2 = delta;

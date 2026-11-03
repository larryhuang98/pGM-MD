/**
 * Smooth PME of point charges and point dipoles for pGM (reciprocal part; the Gaussian widths enter only
 * the direct sum).  Grid conventions as OpenMM's AMOEBA PME: atom spline points igrid..igrid+PME_ORDER-1
 * with igrid = floor(w) - PME_ORDER + 1, theta[i] = (M, M', M'', 0) of the B-spline (the third
 * derivative of AMOEBA's version is not needed: dipole forces use M'' at most).
 * A dipole d is spread as d . grad of its charge spline: fractional dipole e_a = K_a sum_k d_k recip_k[a].
 */

#define ARRAY(x,y) array[(x)-1+((y)-1)*PME_ORDER]

DEVICE void computeBSplinePoint(real4* thetai, real w, real* array) {
    ARRAY(2,2) = w;
    ARRAY(2,1) = 1 - w;
    ARRAY(3,3) = 0.5f * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5f * ((1+w)*ARRAY(2,1)+(2-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5f * (1-w) * ARRAY(2,1);
    for (int i = 4; i <= PME_ORDER; i++) {
        int k = i - 1;
        real denom = RECIP((real) k);
        ARRAY(i,i) = denom * w * ARRAY(k,k);
        for (int j = 1; j <= i-2; j++)
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(k,i-j-1)+(i-j-w)*ARRAY(k,i-j));
        ARRAY(i,1) = denom * (1-w) * ARRAY(k,1);
    }
    int k = PME_ORDER - 1;
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    k = PME_ORDER - 2;
    ARRAY(k,PME_ORDER-1) = ARRAY(k,PME_ORDER-2);
    for (int i = PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    for (int i = 1; i <= PME_ORDER; i++)
        thetai[i-1] = make_real4(ARRAY(PME_ORDER,i), ARRAY(PME_ORDER-1,i), ARRAY(PME_ORDER-2,i), 0);   // M''' unused
}

/**
 * Splines and the first grid index of one atom along the three axes.
 */
DEVICE void atomSplines(real4 pos, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, real4* theta1, real4* theta2, real4* theta3,
        int* igrid1, int* igrid2, int* igrid3, real* array) {
    pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
    pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.y+0.5f);
    pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.x+0.5f);
    real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
    real fr = GRID_SIZE_X*(w-floor(w));
    int ifr = min((int) floor(fr), GRID_SIZE_X-1);
    computeBSplinePoint(theta1, fr-ifr, array);
    *igrid1 = ifr-PME_ORDER+1;
    w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
    fr = GRID_SIZE_Y*(w-floor(w));
    ifr = min((int) floor(fr), GRID_SIZE_Y-1);
    computeBSplinePoint(theta2, fr-ifr, array);
    *igrid2 = ifr-PME_ORDER+1;
    w = pos.z*recipBoxVecZ.z;
    fr = GRID_SIZE_Z*(w-floor(w));
    ifr = min((int) floor(fr), GRID_SIZE_Z-1);
    computeBSplinePoint(theta3, fr-ifr, array);
    *igrid3 = ifr-PME_ORDER+1;
    *igrid1 += (*igrid1 < 0 ? GRID_SIZE_X : 0);
    *igrid2 += (*igrid2 < 0 ? GRID_SIZE_Y : 0);
    *igrid3 += (*igrid3 < 0 ? GRID_SIZE_Z : 0);
}

DEVICE real3 fractionalDipole(real3 d, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    return make_real3(GRID_SIZE_X*(d.x*recipBoxVecX.x+d.y*recipBoxVecY.x+d.z*recipBoxVecZ.x),
                      GRID_SIZE_Y*(d.y*recipBoxVecY.y+d.z*recipBoxVecZ.y),
                      GRID_SIZE_Z*(d.z*recipBoxVecZ.z));
}

/** Cartesian gradient of a fractional gradient g (g_a = d/dw_a). */
DEVICE real3 cartesianGradient(real3 g, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    return make_real3(GRID_SIZE_X*recipBoxVecX.x*g.x,
                      GRID_SIZE_X*recipBoxVecY.x*g.x+GRID_SIZE_Y*recipBoxVecY.y*g.y,
                      GRID_SIZE_X*recipBoxVecZ.x*g.x+GRID_SIZE_Y*recipBoxVecZ.y*g.y+GRID_SIZE_Z*recipBoxVecZ.z*g.z);
}

/**
 * Spread chargeScale*q and the dipoles onto the grid (fixed point).
 */
KERNEL void gridSpread(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT charge, GLOBAL const real* RESTRICT dipole,
        real chargeScale, GLOBAL mm_ulong* RESTRICT pmeGrid,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real array[PME_ORDER*PME_ORDER];
    real4 theta1[PME_ORDER], theta2[PME_ORDER], theta3[PME_ORDER];
    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        int igrid1, igrid2, igrid3;
        atomSplines(posq[m], periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, recipBoxVecX, recipBoxVecY, recipBoxVecZ,
                theta1, theta2, theta3, &igrid1, &igrid2, &igrid3, array);
        real q = chargeScale*charge[m];
        real3 e = fractionalDipole(make_real3(dipole[3*m], dipole[3*m+1], dipole[3*m+2]), recipBoxVecX, recipBoxVecY, recipBoxVecZ);
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = igrid1+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real4 t = theta1[ix];
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = igrid2+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real4 u = theta2[iy];
                real term0 = q*t.x*u.x + e.x*t.y*u.x + e.y*t.x*u.y;
                real term1 = e.z*t.x*u.x;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = igrid3+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    real4 v = theta3[iz];
                    real add = term0*v.x + term1*v.y;
                    ATOMIC_ADD(&pmeGrid[ybase+zindex], (mm_ulong) ((mm_long) (add*GRID_FIXED_SCALE)));
                }
            }
        }
    }
}

KERNEL void finishSpread(GLOBAL const mm_long* RESTRICT pmeGridLong, GLOBAL real* RESTRICT pmeGrid) {
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real scale = 1/(real) GRID_FIXED_SCALE;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE)
        pmeGrid[index] = scale*pmeGridLong[index];
}

/**
 * Multiply the half spectrum by the influence function; with computeEnergy, add
 * E = (1/2) sum_m G(m) |F(m)|^2 (times the Coulomb constant) to the energy buffer first.
 */
KERNEL void reciprocalConvolution(GLOBAL real2* RESTRICT pmeGrid, GLOBAL const real* RESTRICT pmeBsplineModuliX,
        GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ, real4 periodicBoxSize,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL mixed* RESTRICT energyBuffer, int computeEnergy) {
    const unsigned int zsize = GRID_SIZE_Z/2+1;
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*zsize;
    real expFactor = M_PI*M_PI/(EWALD_ALPHA*EWALD_ALPHA);
    real scaleFactor = RECIP(M_PI*periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    mixed energy = 0;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        int kx = index/(GRID_SIZE_Y*zsize);
        int remainder = index-kx*GRID_SIZE_Y*zsize;
        int ky = remainder/zsize;
        int kz = remainder-ky*zsize;
        if (kx == 0 && ky == 0 && kz == 0) {
            pmeGrid[index] = make_real2(0, 0);
            continue;
        }
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = kz;
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real2 grid = pmeGrid[index];
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        real denom = m2*bx*by*bz;
        real eterm = scaleFactor*EXP(-expFactor*m2)/denom;
        if (computeEnergy) {
            real weight = (kz == 0 || (GRID_SIZE_Z%2 == 0 && kz == GRID_SIZE_Z/2)) ? 1 : 2;
            energy += 0.5f*weight*eterm*(grid.x*grid.x+grid.y*grid.y);
        }
        pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
    }
    if (computeEnergy)
        energyBuffer[GLOBAL_ID] += ENERGY_SCALE*energy;
}

/**
 * Add the reciprocal-space field -grad phi at every atom to a fixed-point field buffer.
 */
KERNEL void fieldFromGrid(GLOBAL const real* RESTRICT pmeGrid, GLOBAL const real4* RESTRICT posq, GLOBAL mm_long* RESTRICT fieldBuffer,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real array[PME_ORDER*PME_ORDER];
    real4 theta1[PME_ORDER], theta2[PME_ORDER], theta3[PME_ORDER];
    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        int igrid1, igrid2, igrid3;
        atomSplines(posq[m], periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, recipBoxVecX, recipBoxVecY, recipBoxVecZ,
                theta1, theta2, theta3, &igrid1, &igrid2, &igrid3, array);
        real3 g = make_real3(0);
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int i = igrid1+ix-(igrid1+ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            real4 t = theta1[ix];
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int j = igrid2+iy-(igrid2+iy >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                real4 u = theta2[iy];
                real3 tu = make_real3(0);
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int k = igrid3+iz-(igrid3+iz >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    real ph = pmeGrid[i*GRID_SIZE_Y*GRID_SIZE_Z+j*GRID_SIZE_Z+k];
                    real4 v = theta3[iz];
                    tu.x += ph*v.x;
                    tu.y += ph*v.y;
                }
                g.x += t.y*u.x*tu.x;
                g.y += t.x*u.y*tu.x;
                g.z += t.x*u.x*tu.y;
            }
        }
        real3 E = -cartesianGradient(g, recipBoxVecX, recipBoxVecY, recipBoxVecZ);
        fieldBuffer[m] += realToFixedPoint(E.x);
        fieldBuffer[m+PADDED_NUM_ATOMS] += realToFixedPoint(E.y);
        fieldBuffer[m+2*PADDED_NUM_ATOMS] += realToFixedPoint(E.z);
    }
}

/**
 * Reciprocal-space forces -(q grad phi + (grad grad phi) . d) [times the Coulomb constant] and field.
 */
KERNEL void forceFromGrid(GLOBAL const real* RESTRICT pmeGrid, GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT charge,
        GLOBAL const real* RESTRICT dipole, GLOBAL mm_ulong* RESTRICT forceBuffer, GLOBAL mm_long* RESTRICT fieldBuffer,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real array[PME_ORDER*PME_ORDER];
    real4 theta1[PME_ORDER], theta2[PME_ORDER], theta3[PME_ORDER];
    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        int igrid1, igrid2, igrid3;
        atomSplines(posq[m], periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, recipBoxVecX, recipBoxVecY, recipBoxVecZ,
                theta1, theta2, theta3, &igrid1, &igrid2, &igrid3, array);
        real gx = 0, gy = 0, gz = 0, gxx = 0, gyy = 0, gzz = 0, gxy = 0, gxz = 0, gyz = 0;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int i = igrid1+ix-(igrid1+ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            real4 t = theta1[ix];
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int j = igrid2+iy-(igrid2+iy >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                real4 u = theta2[iy];
                real tu0 = 0, tu1 = 0, tu2 = 0;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int k = igrid3+iz-(igrid3+iz >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    real ph = pmeGrid[i*GRID_SIZE_Y*GRID_SIZE_Z+j*GRID_SIZE_Z+k];
                    real4 v = theta3[iz];
                    tu0 += ph*v.x;
                    tu1 += ph*v.y;
                    tu2 += ph*v.z;
                }
                gx += t.y*u.x*tu0;
                gy += t.x*u.y*tu0;
                gz += t.x*u.x*tu1;
                gxx += t.z*u.x*tu0;
                gyy += t.x*u.z*tu0;
                gzz += t.x*u.x*tu2;
                gxy += t.y*u.y*tu0;
                gxz += t.y*u.x*tu1;
                gyz += t.x*u.y*tu1;
            }
        }
        real q = charge[m];
        real3 e = fractionalDipole(make_real3(dipole[3*m], dipole[3*m+1], dipole[3*m+2]), recipBoxVecX, recipBoxVecY, recipBoxVecZ);
        real3 c = make_real3(q*gx + e.x*gxx + e.y*gxy + e.z*gxz,
                             q*gy + e.x*gxy + e.y*gyy + e.z*gyz,
                             q*gz + e.x*gxz + e.y*gyz + e.z*gzz);
        real3 F = -cartesianGradient(c, recipBoxVecX, recipBoxVecY, recipBoxVecZ)*((real) ENERGY_SCALE);
        ATOMIC_ADD(&forceBuffer[m], (mm_ulong) realToFixedPoint(F.x));
        ATOMIC_ADD(&forceBuffer[m+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(F.y));
        ATOMIC_ADD(&forceBuffer[m+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(F.z));
        real3 E = -cartesianGradient(make_real3(gx, gy, gz), recipBoxVecX, recipBoxVecY, recipBoxVecZ);
        fieldBuffer[m] += realToFixedPoint(E.x);
        fieldBuffer[m+PADDED_NUM_ATOMS] += realToFixedPoint(E.y);
        fieldBuffer[m+2*PADDED_NUM_ATOMS] += realToFixedPoint(E.z);
    }
}

#include "pGM/copyright_pgm.i"
!*******************************************************************************
!
! Module: pgm_gvdw_mod
!
! Gaussian van der Waals (igvdw = 1) for pairs handled outside the pGM direct-space
! loop, i.e. the 1-4 pairs of the AMBER nb14 list.  Same functional form as the
! direct-space GVDW pairs (pGM/pairs_calc_PGM.i):
!
!   U = A exp(-b y^2) - C6 F(y) / r^6         (gvdw_rep_form = 0)
!   U = A exp(-b y)   - C6 F(y) / r^6         (gvdw_rep_form = 1)
!   y = kappa r,  kappa = (r_gauss_i + r_gauss_j)^(-1/2),  b = b_rep_scale
!   F(y) = A(y)^2 + g(y)^2/2,  A = erf(y) - 2 y exp(-y^2)/sqrt(pi) - g,
!   g(y) = 4 y^3 exp(-y^2) / (3 sqrt(pi))
!
! A = ACOEF and C6 = BCOEF of the topology.  1-4 VDW is not scaled with pGM.
!*******************************************************************************
module pgm_gvdw_mod

  implicit none

  private
  public :: pgm_gvdw_pair
#ifdef CUDA
  public :: pgm_gvdw_gpu_nb14_setup
#endif

contains

!*******************************************************************************
! pgm_gvdw_pair: energy e and df = -(dU/dr)/r of one GVDW pair
!*******************************************************************************
subroutine pgm_gvdw_pair(r2, rgsum, a, c6, e, df)

  use pol_gauss_mdin_dat_mod, only : gvdw_rep_form, b_rep_scale

  implicit none

  double precision, intent(in)  :: r2, rgsum, a, c6
  double precision, intent(out) :: e, df

  double precision, parameter :: isp = 0.56418958354775628695d0
  double precision, parameter :: pi  = 3.14159265358979323846d0
  double precision :: rinv, r2inv, r6, kap, y, y2, ey, gx, ax, fx, apx, gpx, fpx, erep

  rinv  = 1.d0 / sqrt(r2)
  r2inv = rinv * rinv
  r6    = r2inv * r2inv * r2inv
  kap   = 1.d0 / sqrt(rgsum)
  y     = kap / rinv
  y2    = y * y
  ey    = exp(-y2)
  gx    = (4.d0 / 3.d0) * isp * y * y2 * ey
  ax    = erf(y) - 2.d0 * isp * y * ey - gx
  fx    = ax * ax + 0.5d0 * gx * gx
  apx   = (8.d0 / 3.d0) * isp * y2 * y2 * ey
  gpx   = 4.d0 * isp * y2 * (1.d0 - (2.d0 / 3.d0) * y2) * ey
  fpx   = 2.d0 * ax * apx + gx * gpx
  if (gvdw_rep_form .eq. 0) then
    erep = a * exp(-b_rep_scale * y2)
    df   = 2.d0 * b_rep_scale * kap * kap * erep
  else
    erep = a * exp(-b_rep_scale * y)
    df   = b_rep_scale * kap * erep * rinv
  end if
  if (y .gt. 0.01d0) then
    e  = erep - c6 * fx * r6
    df = df + c6 * (kap * fpx * rinv - 6.d0 * fx * r2inv) * r6
  else
    e  = erep - c6 * (8.d0 / (9.d0 * pi)) * kap**6
  end if

end subroutine pgm_gvdw_pair

#ifdef CUDA
!*******************************************************************************
! pgm_gvdw_gpu_nb14_setup: with igvdw = 1, replace the 1-4 Lennard-Jones of the
! GPU bonded work units by GVDW.  Called from nb14_setup after gpu_nb14_setup.
!*******************************************************************************
subroutine pgm_gvdw_gpu_nb14_setup(nb14_cnt, nb14, ntypes, iac, ico, cn1, cn2)

  use pol_gauss_mdin_dat_mod, only : igvdw
  use pol_gauss_multipoles_mod, only : r_gauss

  implicit none

  integer, intent(in)          :: nb14_cnt, ntypes
  integer, intent(in)          :: nb14(3, *), iac(*), ico(*)
  double precision, intent(in) :: cn1(*), cn2(*)

  integer                       :: n, i, j, ic, ibig, isml
  integer, allocatable          :: ij(:,:)
  double precision, allocatable :: coef(:,:)

  if (igvdw .ne. 1) return
  allocate(ij(2, max(nb14_cnt, 1)), coef(3, max(nb14_cnt, 1)))
  do n = 1, nb14_cnt
    i = abs(nb14(1, n))
    j = abs(nb14(2, n))
    ! same type-pair index as the CPU 1-4 code (get_nb14_energy)
    ibig = max(iac(i), iac(j))
    isml = min(iac(i), iac(j))
    ic = ibig * (ibig - 1) / 2 + isml
    ij(1, n) = i
    ij(2, n) = j
    coef(1, n) = cn1(ic)
    coef(2, n) = cn2(ic)
    coef(3, n) = 1.d0 / sqrt(r_gauss(i) + r_gauss(j))
  end do
  call pgm_gpu_nb14_gvdw_setup(nb14_cnt, ij, coef)
  deallocate(ij, coef)

end subroutine pgm_gvdw_gpu_nb14_setup
#endif

end module pgm_gvdw_mod

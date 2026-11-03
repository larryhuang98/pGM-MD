#include "copyright_pgm.i"
#define PGM_SUB_RAW(a, b, c, d, e, f, g) a##b##c##d##e##f##g
#define PGM_SUB_NAME(a, b, c, d, e, f, g) PGM_SUB_RAW(a, b, c, d, e, f, g)
#define PGM_HEAD pairs_calc_PGM
#ifdef _NEED_B3_
#undef _NEED_B3_
#endif
#if defined(_USE_SAVED_BFACT_)
#define _BFACT _use_saved_B
#elif defined(_SAVE_BFACT_)
#define _BFACT _save_B
#else
#define _BFACT
#endif
#ifndef _FIELD_ONLY_
!
! --- A. NON-FIELD_ONLY Calculations (_ef, _efv, _fv, _f) ---
!
#define _NEED_B3_
   ! Define TYPE
#if defined(BUILD_PAIRS_CALC_EFV)
#define _TYPE _efv
#elif defined(BUILD_PAIRS_CALC_EF)
#define _TYPE _ef
#elif defined(BUILD_PAIRS_CALC_FV)
#define _TYPE _fv
#elif defined(BUILD_PAIRS_CALC_F)
#define _TYPE _f
#else
#define _TYPE
#endif
#ifdef NEED_ENE
#undef NEED_ENE
#endif
#if defined(BUILD_PAIRS_CALC_EFV) || defined(BUILD_PAIRS_CALC_EF)
#define NEED_ENE
#define _EFTBL_ ef_tbl
#else
#define _EFTBL_ f_tbl
#endif
#ifdef NEED_VIR
#undef NEED_VIR
#endif
#if defined(BUILD_PAIRS_CALC_EFV) || defined(BUILD_PAIRS_CALC_FV)
#define NEED_VIR
#endif
   ! Define SWITCH/VACUUM/DEVDW suffixes
#define _VDW
#define _VAC
#define _SWITCH

#ifdef FSWITCH
#undef _SWITCH
#define _SWITCH _fs
#else
#ifdef VACUUM
#undef _VAC
#define _VAC _vac
#endif
#ifdef NEED_GVDW
#undef _VDW
#define _VDW _gvdw
#endif
#endif
#else /* IS _FIELD_ONLY_ */
!
! --- B. _FIELD_ONLY_ Calculations (_fld) ---
!
#define _TYPE _fld
#ifdef _SAVE_BFACT_
#define _NEED_B3_
#endif
   ! _fld routines don not use FSWITCH or DEVDW
#define _SWITCH
#define _VDW
   ! Define VACUUM suffix
#ifdef VACUUM
#define _VAC _vac
#else
#define _VAC
#endif
#endif /* _FIELD_ONLY_ */
! --- Define CUT suffix (common to both types) ---
#ifdef BUILD_PAIRS_CALC_NOVEC_2CUT
#define _CUT _2cut
#else
#define _CUT
#endif
! --- Finally, Build the Subroutine Name ---
#define PGM_SUBS PGM_SUB_NAME(PGM_HEAD, _TYPE, _VDW, _CUT, _VAC, _SWITCH, _BFACT)
!*******************************************************************************
!
! Adopted from non-pGM pmemd code by YD
!
! Internal Subroutine:  PGM_SUBS
!
! Description:  Direct force computation on one atom.  Different implementations
!               are covered by conditional compilation in this file.
!*******************************************************************************
#ifndef _FIELD_ONLY_
subroutine PGM_SUBS &
                   (img_frc, img_crd, img_chg, img_dip, _EFTBL_, &
                    eed_cub, phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#else /* IS _FIELD_ONLY_ */
subroutine PGM_SUBS &
                  (img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
                   gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                   dipole_dipole_list_short, dipole_dipole_tensor_short)
#endif /* _FIELD_ONLY_ */

  implicit none

! Formal arguments:
  double precision, intent(in)  :: img_crd(3, *)
  double precision, intent(in)  :: img_chg(*)
  double precision, intent(in)  :: img_dip(3,*)
#ifndef _FIELD_ONLY_
  double precision              :: img_frc(3, *)
  double precision              :: phi(10, *)
#ifdef NEED_ENE
  double precision, intent(in)  :: ef_tbl(*)
#else
  double precision, intent(in)  :: f_tbl(*)
#endif
#ifdef NEED_VIR
#endif

  integer, intent(in)           :: ico(*)
  integer, intent(in)           :: img_iac(*)
  double precision, intent(in)  :: cn1(*), cn2(*)
#else  /* _FIELD_ONLY_ */
  double precision              :: gradphi(3,*)
  double precision              :: dipole_dipole_tensor(6,*)
  double precision              :: dipole_dipole_tensor_short(6,*)
  integer                       :: dipole_dipole_list(*)
  integer                       :: dipole_dipole_list_short(*)
  integer                       :: num_tensor, num_tensor_short
  double precision, intent(in)  :: ef_tbl(*)
#endif /* _FIELD_ONLY_ */
  double precision, intent(in)  :: eed_cub(*)
  integer                       :: ipairs_sublst(*)
  double precision, intent(in)  :: x_tran(1:3, 0:17)

! Local variables:

  integer, parameter            :: mask27 = int(Z"07FFFFFF")
  double precision, parameter   :: half   = 1.d0/2.d0
  double precision, parameter   :: third  = 1.d0/3.d0
  double precision, parameter   :: zero   = 0.d0
  double precision, parameter   :: one    = 1.d0
  double precision, parameter   :: two    = 2.d0
  double precision, parameter   :: three  = 3.d0
  double precision, parameter   :: five   = 5.d0
  double precision      :: dumx, dumy, dumz
  double precision      :: cgi

  double precision      :: cgj

  double precision      :: cgi_cgj
  double precision      :: b0, b1
#ifndef _FIELD_ONLY_
  double precision      :: df
  double precision      :: dfx, dfy, dfz
  double precision      :: f6, r6, f12
#ifdef HAS_10_12
  double precision      :: f10, r10
#endif
#endif /* _FIELD_ONLY_ */
  double precision      :: du, du2, du3 ! 'u' is another name for delr2
  double precision      :: del_efs
  double precision      :: dens_efs
  double precision      :: lowest_efs_u
  ! Variables used with erfc switch table; name are historical:
  double precision      :: switch
  double precision      :: d_switch_dx
  double precision      :: x, dx, e3dx, e4dx2
  double precision      :: delr, delrinv
! PGM variables
  double precision      :: betaij, switchij, d_switchij_dx, y
  double precision      :: fac, fact, facij
  double precision      :: B(0:3), Bij(0:3)
! Among these, only Rn_3, Rn_5 are used
  double precision      :: Rn_1(4), Rn_2(10), Rn_3(20), Rn_4(35), Rn_5(56)
  double precision      :: gphi_i(3), gphi_j(3)
  double precision      :: phi_i(10), phi_j(10)
  double precision      :: t_x, t_y, t_z
  double precision      :: tmiq, tmix, tmiy, tmiz, tmjq, tmjx, tmjy, tmjz
!  double precision      :: gmj(10), gmi(10)
!  double precision      :: BD(5)
  double precision      :: tmi(10), tmj(10)
!  double precision      :: g_ind(3), i_di(3), i_dj(3), i_mi(3), i_mj(3)
 
#ifdef _FIELD_ONLY_
  double precision      :: scf_local_cut2
#else  /* _FIELD_ONLY_ */
#ifdef FSWITCH
  double precision      :: delr3inv, delr12inv
  double precision      :: cut, cut6, cut3, cutinv, cut2inv, cut3inv, cut6inv
  double precision      :: fswitch2, fswitch3, fswitch6
  double precision      :: p12, p6, a_energy, b_energy, df12, df6

  double precision      :: invfswitch6cut6, invfswitch3cut3
#endif /*FSWITCH*/

#endif /* _FIELD_ONLY_ */
  integer               :: iaci
  integer               :: ic
  integer               :: ind
  integer               :: nxt_img_j, img_j, jj
  integer               :: itran
  integer               :: sublst_idx
  integer               :: saved_pairlist_val

  double precision      :: nxt_delx, nxt_dely, nxt_delz
  double precision      :: delx, dely, delz, delr2, delr2inv
  double precision      :: delx2, dely2, delz2
  double precision      :: delxd, delyd, delzd

  integer               :: i,j,l

#ifdef _USE_SAVED_BFACT_
  integer               :: idx
  double precision      :: fswitchinv
#endif /* _USE_SAVED_BFACT_ */

#ifdef NEED_GVDW
  double precision      :: yg, yg2, expy2g, erfyg, kapg, exprep, gx, bx, ax, fx, apx, gpx, fpx
  double precision, parameter :: inv_sqrtpi = 0.56418958354775628695d0
  double precision, parameter :: pi_gvdw = 3.14159265358979323846d0
#endif

!  i = img_atm_map(img_i)  ! inherited from get_direct_fld_PGM/get_nb_energy_PGM, i points to atom i
                           ! img_i is image of atom_i

  dens_efs = efs_tbl_dens
  del_efs = 1.d0 / dens_efs
  lowest_efs_u = lowest_efs_delr2

!  fac = two * ewaldcof * ewaldcof 
  fac  = two * dxdr * dxdr 
! Precompute some constants

#ifndef _FIELD_ONLY_
#ifdef FSWITCH
#ifdef _USE_SAVED_BFACT_
  fswitchinv = 1.d0/fswitch
#endif /* _USE_SAVED_BFACT_ */
  fswitch2 = fswitch * fswitch
  fswitch3 = fswitch2 * fswitch
  fswitch6 = fswitch3 * fswitch3
  cut = sqrt(max_nb_cut2)
  cut3 = cut*cut*cut
  cut6 = cut3*cut3
  cutinv = 1/cut
  cut2inv = cutinv*cutinv
  cut3inv = cut2inv*cutinv
  cut6inv = cut3inv*cut3inv

  invfswitch6cut6 = cut6inv / fswitch6
  invfswitch3cut3 = cut3inv / fswitch3
#endif /*FSWITCH*/

! First loop over the ee evaluation-only pairs:

  dumx = 0.d0
  dumy = 0.d0
  dumz = 0.d0

#ifndef _USE_SAVED_BFACT_
!  cgi = img_qterm(img_i)
  cgi = img_chg(img_i)
#endif /* _USE_SAVED_BFACT_ */

      tmiq   = img_chg(img_i)
      tmix   = img_dip(1, img_i)
      tmiy   = img_dip(2, img_i)
      tmiz   = img_dip(3, img_i)

#else  /* _FIELD_ONLY_ */
  scf_local_cut2    = scf_local_cut * scf_local_cut
!  dip_dip_cnt       = 0
!  dip_dip_cnt_short = 0
#endif /* _FIELD_ONLY_ */

#ifndef _USE_SAVED_BFACT_
  ! The pairlist must have one dummy end entry to cover reading past the
  ! end of the list...
  saved_pairlist_val = ipairs_sublst(ee_eval_cnt + full_eval_cnt + 1)

  ipairs_sublst(ee_eval_cnt + full_eval_cnt + 1) = &
    ipairs_sublst(ee_eval_cnt + full_eval_cnt)

  if (common_tran .eq. 1) then
    nxt_img_j = ipairs_sublst(1)
    itran = 13
  else
    nxt_img_j = iand(ipairs_sublst(1), mask27)
    itran = ishft(ipairs_sublst(1), -27)
  end if

  nxt_delx = img_crd(1, nxt_img_j) + x_tran(1, itran)
  nxt_dely = img_crd(2, nxt_img_j) + x_tran(2, itran)
  nxt_delz = img_crd(3, nxt_img_j) + x_tran(3, itran)

  do sublst_idx = 2, ee_eval_cnt + 1
    img_j = nxt_img_j
    delx = nxt_delx
    dely = nxt_dely
    delz = nxt_delz

    if (common_tran .eq. 1) then
      nxt_img_j = ipairs_sublst(sublst_idx)
      itran = 13
    else
      nxt_img_j = iand(ipairs_sublst(sublst_idx), mask27)
      itran = ishft(ipairs_sublst(sublst_idx), -27)
    end if

    nxt_delx = img_crd(1, nxt_img_j) + x_tran(1, itran)
    nxt_dely = img_crd(2, nxt_img_j) + x_tran(2, itran)
    nxt_delz = img_crd(3, nxt_img_j) + x_tran(3, itran)
    delx2 = delx * delx
    dely2 = dely * dely
    delz2 = delz * delz
!    delr2 = delx * delx + dely * dely + delz * delz
    delr2 = delx2 + dely2 + delz2


#ifdef BUILD_PAIRS_CALC_NOVEC_2CUT
    if (delr2 .lt. es_cut2) then
#else
    if (delr2 .lt. max_nb_cut2) then
#endif /* BUILD_PAIRS_CALC_NOVEC_2CUT */
#else /* _USE_SAVED_BFACT_ */
  do idx = dipole_dipole_ptrs(img_i), last_ee_ptrs(img_i)
    img_j = dipole_dipole_list(idx)

    delx  = delvec_save(1, idx)
    dely  = delvec_save(2, idx)
    delz  = delvec_save(3, idx)
    B(0)  = Bfact_save(1, idx)
    B(1)  = Bfact_save(2, idx)
    B(2)  = Bfact_save(3, idx)
    B(3)  = Bfact_save(4, idx)
    delx2         = delx * delx
    dely2         = dely * dely
    delz2         = delz * delz

    delrinv  = delvec_save(4, idx)
    delr2inv = delrinv * delrinv

#endif /* _USE_SAVED_BFACT_ */

#ifndef _USE_SAVED_BFACT_
#if 0
! The monopole-only code

      cgi_cgj = cgi * img_qterm(img_j)
! Following if-block has been disabled. Will enable later  YD
      if (delr2 .ge. lowest_efs_u) then

        ! Do the Coulomb part of the direct sum using efs: 

        ind = int(dens_efs * delr2)
        du = delr2 - dble(ind) * del_efs
        du2 = du * du
        du3 = du * du2
#ifdef NEED_ENE
        ind = ishft(ind, 3)             ! 8 * ind

        b0 = cgi_cgj * (ef_tbl(1 + ind) + du * ef_tbl(2 + ind) + &
             du2 * ef_tbl(3 + ind) + du3 * ef_tbl(4 + ind))

        df = cgi_cgj * (ef_tbl(5 + ind) + du * ef_tbl(6 + ind) + &
             du2 * ef_tbl(7 + ind) + du3 * ef_tbl(8 + ind))
#else
        ind = ishft(ind, 2)             ! 4 * ind

        df = cgi_cgj * (f_tbl(1 + ind) + du * f_tbl(2 + ind) + &
             du2 * f_tbl(3 + ind) + du3 * f_tbl(4 + ind))
#endif /* NEED_ENE */

#ifdef NEED_ENE
        eed_stk = eed_stk + b0
#endif /* NEED_ENE */

#ifdef NEED_VIR
        eedvir_stk = eedvir_stk - df * delr2 
#endif /* NEED_VIR */
      else
! monopole-only code ends here
#endif /* 0 */

        ! Do the Coulomb part of the direct sum using erfc spline table: 

        delr = sqrt(delr2)
        delrinv = 1.0 / delr
        delr2inv = delrinv * delrinv

#ifdef VACUUM
        switch = 1.0
        d_switch_dx = 1.0
#else  /* VACUUM */
        x = dxdr * delr
        ind = int(eedtbdns_stk * x)
!        dx = x - (dble(ind)-one)*del    ! from pGM_direct.F90
        dx = x - dble(ind) * del
        ind = ishft(ind, 2)             ! 4 * ind

        e3dx  = dx * eed_cub(3 + ind)
        e4dx2 = dx * dx *  eed_cub(4 + ind)

        switch = eed_cub(1 + ind) + &
                 dx * (eed_cub(2 + ind) + half * (e3dx + third * e4dx2))

        d_switch_dx = eed_cub(2 + ind) + e3dx + half * e4dx2

        B(0) = (one - switch)*delr*delr2inv                                                                      
        fact = -d_switch_dx*dxdr     ! fact = -d_switch_dx*beta
        B(1) = (B(0) - fact)*delr2inv
        fact = fac*fact                                                                                          
        B(2) = (three*B(1) - fact)*delr2inv
#ifdef _NEED_B3_
        fact = fac*fact
        B(3) = (five*B(2) - fact)*delr2inv
#endif /* _NEED_B3_ */

#endif /* VACUUM */
!  j = img_atm_map(img_j) ! j points to atom j

        ! compute erfc=one-erf function and its derivative for pGM gaussians.
        betaij  = one/(r_gauss_img(img_i) + r_gauss_img(img_j))
        facij = two*betaij
        betaij  = sqrt(betaij)
        y = betaij*delr
        !if ( y < ee_gauss_cut ) then 
!        ind = int(eedtbdns_stk * y) + 1
        ind = int(eedtbdns_stk * y)
!        dx = y - (dble(ind)-one)*del
        dx = y - (dble(ind))*del
        ind = ishft(ind, 2)             ! 4 * ind

        switchij = eed_cub(1+ind) + dx * (eed_cub(2+ind) + &
             dx * (eed_cub(3+ind) + dx *  eed_cub(4+ind) * third) * half)

        d_switchij_dx =                   eed_cub(2+ind) + &
             dx * (eed_cub(3+ind) + dx *  eed_cub(4+ind) * half)

        !call get_ee_func(x,switchij,d_switchij_dx,1)

        Bij(0) = (one - switchij)*delr*delr2inv ! erf/r
        fact = -d_switchij_dx*betaij
        Bij(1) = (Bij(0) - fact)*delr2inv
        fact = facij*fact
        Bij(2) = (three*Bij(1) - fact)*delr2inv
#ifdef _NEED_B3_
        fact = facij*fact
        Bij(3) = (five*Bij(2) - fact)*delr2inv
#endif /* _NEED_B3_ */

#ifdef VACUUM
        B(0)  = Bij(0)
        B(1)  = -Bij(1)
        B(2)  = Bij(2)
#ifdef _NEED_B3_
        B(3)  = -Bij(3)
#endif
#else  /* VACUUM */
        B(0)  = Bij(0) - B(0)
        B(1)  = B(1) - Bij(1)
        B(2)  = Bij(2) - B(2)
#ifdef _NEED_B3_
        B(3)  = B(3) - Bij(3)
#endif
#endif /* VACUUM */
#endif /* _USE_SAVED_BFACT_ */

! B_BLOCK_ENDS_HERE

#ifdef _FIELD_ONLY_

      ! multipole interaction tensor elements

      Rn_3(Ind_100) = delx*B(1)
      Rn_3(Ind_010) = dely*B(1)
      Rn_3(Ind_001) = delz*B(1)
      Rn_3(Ind_200) = B(1) + delx2*B(2)
      Rn_3(Ind_020) = B(1) + dely2*B(2)
      Rn_3(Ind_002) = B(1) + delz2*B(2)
      Rn_3(Ind_110) = delx*dely*B(2)
      Rn_3(Ind_101) = delx*delz*B(2)
      Rn_3(Ind_011) = dely*delz*B(2)

#if 0
      Rn_1(Ind_000) = B(2)
      Rn_2(Ind_000) = B(1)
      Rn_2(Ind_100) = delx*Rn_1(Ind_000)
      Rn_2(Ind_010) = dely*Rn_1(Ind_000)
      Rn_2(Ind_001) = delz*Rn_1(Ind_000)
      Rn_3(Ind_100) = delx*Rn_2(Ind_000)
      Rn_3(Ind_010) = dely*Rn_2(Ind_000)
      Rn_3(Ind_001) = delz*Rn_2(Ind_000)
      Rn_3(Ind_200) = Rn_2(Ind_000) + delx*Rn_2(Ind_100)
      Rn_3(Ind_020) = Rn_2(Ind_000) + dely*Rn_2(Ind_010)
      Rn_3(Ind_002) = Rn_2(Ind_000) + delz*Rn_2(Ind_001)
      Rn_3(Ind_110) = delx*Rn_2(Ind_010)
      Rn_3(Ind_101) = delx*Rn_2(Ind_001)
      Rn_3(Ind_011) = dely*Rn_2(Ind_001)
#endif
      ! store the dipole-dipole component
      num_tensor = num_tensor+1
#ifdef _SAVE_BFACT_
      delvec_save(1,num_tensor) = delx
      delvec_save(2,num_tensor) = dely
      delvec_save(3,num_tensor) = delz
      delvec_save(4,num_tensor) = delrinv
      Bfact_save(1, num_tensor) = B(0)
      Bfact_save(2, num_tensor) = B(1)
      Bfact_save(3, num_tensor) = B(2)
      Bfact_save(4, num_tensor) = B(3)
#endif
      dipole_dipole_list(num_tensor) = img_j
      dipole_dipole_tensor(1,num_tensor) = Rn_3(Ind_200)
      dipole_dipole_tensor(2,num_tensor) = Rn_3(Ind_110)
      dipole_dipole_tensor(3,num_tensor) = Rn_3(Ind_101)
      dipole_dipole_tensor(4,num_tensor) = Rn_3(Ind_020)
      dipole_dipole_tensor(5,num_tensor) = Rn_3(Ind_011)
      dipole_dipole_tensor(6,num_tensor) = Rn_3(Ind_002)

      if ( delr2 .le. scf_local_cut2 ) then
      ! store the short dipole-dipole component
      num_tensor_short = num_tensor_short+1
      dipole_dipole_list_short(num_tensor_short) = img_j
      dipole_dipole_tensor_short(1,num_tensor_short) = -Bij(1) + delx2*Bij(2) ! != Rn_3(Ind_200)
      dipole_dipole_tensor_short(2,num_tensor_short) = delx*dely*Bij(2)       ! != Rn_3(Ind_110)
      dipole_dipole_tensor_short(3,num_tensor_short) = delx*delz*Bij(2)       ! != Rn_3(Ind_101)
      dipole_dipole_tensor_short(4,num_tensor_short) = -Bij(1) + dely2*Bij(2) ! != Rn_3(Ind_020)
      dipole_dipole_tensor_short(5,num_tensor_short) = dely*delz*Bij(2)       ! != Rn_3(Ind_011)
      dipole_dipole_tensor_short(6,num_tensor_short) = -Bij(1) + delz2*Bij(2) ! != Rn_3(Ind_002)
      end if

      gphi_i(1) = Rn_3(Ind_100)*img_chg(img_j) + &
                  Rn_3(Ind_200)*img_dip(1,img_j) + &
                  Rn_3(Ind_110)*img_dip(2,img_j) + &
                  Rn_3(Ind_101)*img_dip(3,img_j)
      gphi_i(2) = Rn_3(Ind_010)*img_chg(img_j) + &
                  Rn_3(Ind_110)*img_dip(1,img_j) + &
                  Rn_3(Ind_020)*img_dip(2,img_j) + &
                  Rn_3(Ind_011)*img_dip(3,img_j)
      gphi_i(3) = Rn_3(Ind_001)*img_chg(img_j) + &
                  Rn_3(Ind_101)*img_dip(1,img_j) + &
                  Rn_3(Ind_011)*img_dip(2,img_j) + &
                  Rn_3(Ind_002)*img_dip(3,img_j)

      gradphi(1,img_i) = gradphi(1,img_i) - gphi_i(1) ! R. Luo:
      gradphi(2,img_i) = gradphi(2,img_i) - gphi_i(2) ! We are returning to the convention
      gradphi(3,img_i) = gradphi(3,img_i) - gphi_i(3) ! gradphi is grad of phi, not field

      gphi_j(1) = -Rn_3(Ind_100)*img_chg(img_i) + &
                   Rn_3(Ind_200)*img_dip(1,img_i) + &
                   Rn_3(Ind_110)*img_dip(2,img_i) + &
                   Rn_3(Ind_101)*img_dip(3,img_i)
      gphi_j(2) = -Rn_3(Ind_010)*img_chg(img_i) + &
                   Rn_3(Ind_110)*img_dip(1,img_i) + &
                   Rn_3(Ind_020)*img_dip(2,img_i) + &
                   Rn_3(Ind_011)*img_dip(3,img_i)
      gphi_j(3) = -Rn_3(Ind_001)*img_chg(img_i) + &
                   Rn_3(Ind_101)*img_dip(1,img_i) + &
                   Rn_3(Ind_011)*img_dip(2,img_i) + &
                   Rn_3(Ind_002)*img_dip(3,img_i)
      gradphi(1,img_j) = gradphi(1,img_j) - gphi_j(1) ! R.Luo
      gradphi(2,img_j) = gradphi(2,img_j) - gphi_j(2) ! we are returning to the convention
      gradphi(3,img_j) = gradphi(3,img_j) - gphi_j(3) ! gradphi is grad of phi, not field
#else  /* _FIELD_ONLY_ */

      Rn_5(Ind_000) = B(0)
      Rn_5(Ind_100) = delx*B(1)
      Rn_5(Ind_010) = dely*B(1)
      Rn_5(Ind_001) = delz*B(1)
      Rn_5(Ind_200) = B(1) + delx2*B(2)
      Rn_5(Ind_020) = B(1) + dely2*B(2)
      Rn_5(Ind_002) = B(1) + delz2*B(2)
      Rn_5(Ind_110) = delx*dely*B(2)
      Rn_5(Ind_101) = delx*delz*B(2)
      Rn_5(Ind_011) = dely*delz*B(2)
      delxd         = B(2) + delx2*B(3)
      delyd         = B(2) + dely2*B(3)
      delzd         = B(2) + delz2*B(3)
      Rn_5(Ind_300) = delx*(2.d0*B(2) + delxd)
      Rn_5(Ind_030) = dely*(2.d0*B(2) + delyd)
      Rn_5(Ind_003) = delz*(2.d0*B(2) + delzd)
      Rn_5(Ind_210) = dely*delxd
      Rn_5(Ind_201) = delz*delxd
      Rn_5(Ind_120) = delx*delyd
      Rn_5(Ind_021) = delz*delyd
      Rn_5(Ind_102) = delx*delzd
      Rn_5(Ind_012) = dely*delzd
      Rn_5(Ind_111) = delx*dely*delz*B(3)
#if 0
      Rn_2(Ind_000) = B(3)
      Rn_3(Ind_000) = B(2)
      Rn_3(Ind_100) = delx*Rn_2(Ind_000)
      Rn_3(Ind_010) = dely*Rn_2(Ind_000)
      Rn_3(Ind_001) = delz*Rn_2(Ind_000)
      Rn_4(Ind_000) = B(1)
      Rn_4(Ind_100) = delx*Rn_3(Ind_000)
      Rn_4(Ind_010) = dely*Rn_3(Ind_000)
      Rn_4(Ind_001) = delz*Rn_3(Ind_000)
      Rn_4(Ind_200) = Rn_3(Ind_000) + delx*Rn_3(Ind_100)
      Rn_4(Ind_020) = Rn_3(Ind_000) + dely*Rn_3(Ind_010)
      Rn_4(Ind_002) = Rn_3(Ind_000) + delz*Rn_3(Ind_001)
      Rn_5(Ind_000) = B(0)
      Rn_5(Ind_100) = delx*Rn_4(Ind_000)
      Rn_5(Ind_010) = dely*Rn_4(Ind_000)
      Rn_5(Ind_001) = delz*Rn_4(Ind_000)
      Rn_5(Ind_200) = Rn_4(Ind_000) + delx*Rn_4(Ind_100)
      Rn_5(Ind_020) = Rn_4(Ind_000) + dely*Rn_4(Ind_010)
      Rn_5(Ind_002) = Rn_4(Ind_000) + delz*Rn_4(Ind_001)
      Rn_5(Ind_110) = delx*Rn_4(Ind_010)
      Rn_5(Ind_101) = delx*Rn_4(Ind_001)
      Rn_5(Ind_011) = dely*Rn_4(Ind_001)
      Rn_5(Ind_300) = 2.d0*Rn_4(Ind_100) + delx*Rn_4(Ind_200)
      Rn_5(Ind_030) = 2.d0*Rn_4(Ind_010) + dely*Rn_4(Ind_020)
      Rn_5(Ind_003) = 2.d0*Rn_4(Ind_001) + delz*Rn_4(Ind_002)
      Rn_5(Ind_210) = dely*Rn_4(Ind_200)
      Rn_5(Ind_201) = delz*Rn_4(Ind_200)
      Rn_5(Ind_120) = delx*Rn_4(Ind_020)
      Rn_5(Ind_021) = delz*Rn_4(Ind_020)
      Rn_5(Ind_102) = delx*Rn_4(Ind_002)
      Rn_5(Ind_012) = dely*Rn_4(Ind_002)
      Rn_5(Ind_111) = delx*dely*delz*Rn_2(Ind_000)
#endif

!      do jj = 1,4!10
!        gmi(jj) = img_mpole(jj,img_i) ! local perm moments: charges/dipoles
!        gmj(jj) = img_mpole(jj,img_j) ! local perm moments: charges/dipoles
!      enddo
!      do jj = 1,3
!        g_ind(jj) = 0.d0 ! induced force components
!        i_di(jj) = ind_dip(jj,img_i) ! local induced dipoles
!        i_dj(jj) = ind_dip(jj,img_j) ! local induced dipoles
!        i_mi(jj) = ind_dip(jj,img_i) + gmi(jj+1) ! local total dipoles
!        i_mj(jj) = ind_dip(jj,img_j) + gmj(jj+1) ! local total dipoles
!      enddo

      ! electrostatic potential at i due to permanent + induced mpoles at j
      ! and derivatives of that with respect to r_i
!      tmj(1) = gmj(1)
!      tmj(2:4) = i_mj(1:3)

#if 0
      tmj(1)  = img_chg(img_j)
      tmj(2:4)= img_dip(1:3,img_j)

      phi_i(Ind_000)=  Rn_5(Ind_000)*tmj(Ind_000)+Rn_5(Ind_100)*tmj(Ind_100)+ &
                       Rn_5(Ind_010)*tmj(Ind_010)+Rn_5(Ind_001)*tmj(Ind_001)
      phi_i(Ind_100)=-(Rn_5(Ind_100)*tmj(Ind_000)+Rn_5(Ind_200)*tmj(Ind_100)+ &
                       Rn_5(Ind_110)*tmj(Ind_010)+Rn_5(Ind_101)*tmj(Ind_001))
      phi_i(Ind_010)=-(Rn_5(Ind_010)*tmj(Ind_000)+Rn_5(Ind_110)*tmj(Ind_100)+ &
                       Rn_5(Ind_020)*tmj(Ind_010)+Rn_5(Ind_011)*tmj(Ind_001))
      phi_i(Ind_001)=-(Rn_5(Ind_001)*tmj(Ind_000)+Rn_5(Ind_101)*tmj(Ind_100)+ &
                       Rn_5(Ind_011)*tmj(Ind_010)+Rn_5(Ind_002)*tmj(Ind_001))
      phi_i(Ind_200)=  Rn_5(Ind_200)*tmj(Ind_000)+Rn_5(Ind_300)*tmj(Ind_100)+ &
                       Rn_5(Ind_210)*tmj(Ind_010)+Rn_5(Ind_201)*tmj(Ind_001)
      phi_i(Ind_020)=  Rn_5(Ind_020)*tmj(Ind_000)+Rn_5(Ind_120)*tmj(Ind_100)+ &
                       Rn_5(Ind_030)*tmj(Ind_010)+Rn_5(Ind_021)*tmj(Ind_001)
      phi_i(Ind_002)=  Rn_5(Ind_002)*tmj(Ind_000)+Rn_5(Ind_102)*tmj(Ind_100)+ &
                       Rn_5(Ind_012)*tmj(Ind_010)+Rn_5(Ind_003)*tmj(Ind_001)
      phi_i(Ind_110)=  Rn_5(Ind_110)*tmj(Ind_000)+Rn_5(Ind_210)*tmj(Ind_100)+ &
                       Rn_5(Ind_120)*tmj(Ind_010)+Rn_5(Ind_111)*tmj(Ind_001)
      phi_i(Ind_101)=  Rn_5(Ind_101)*tmj(Ind_000)+Rn_5(Ind_201)*tmj(Ind_100)+ &
                       Rn_5(Ind_111)*tmj(Ind_010)+Rn_5(Ind_102)*tmj(Ind_001)
      phi_i(Ind_011)=  Rn_5(Ind_011)*tmj(Ind_000)+Rn_5(Ind_111)*tmj(Ind_100)+ &
                       Rn_5(Ind_021)*tmj(Ind_010)+Rn_5(Ind_012)*tmj(Ind_001)
#endif

      tmjq    = img_chg(img_j)
      tmjx    = img_dip(1, img_j)
      tmjy    = img_dip(2, img_j)
      tmjz    = img_dip(3, img_j)

#ifdef NEED_VIR
      phi_i(Ind_000)=  Rn_5(Ind_000)*tmjq+Rn_5(Ind_100)*tmjx+Rn_5(Ind_010)*tmjy+Rn_5(Ind_001)*tmjz
      phi_i(Ind_100)=-(Rn_5(Ind_100)*tmjq+Rn_5(Ind_200)*tmjx+Rn_5(Ind_110)*tmjy+Rn_5(Ind_101)*tmjz)
      phi_i(Ind_010)=-(Rn_5(Ind_010)*tmjq+Rn_5(Ind_110)*tmjx+Rn_5(Ind_020)*tmjy+Rn_5(Ind_011)*tmjz)
      phi_i(Ind_001)=-(Rn_5(Ind_001)*tmjq+Rn_5(Ind_101)*tmjx+Rn_5(Ind_011)*tmjy+Rn_5(Ind_002)*tmjz)
      phi_i(Ind_200)=  Rn_5(Ind_200)*tmjq+Rn_5(Ind_300)*tmjx+Rn_5(Ind_210)*tmjy+Rn_5(Ind_201)*tmjz
      phi_i(Ind_020)=  Rn_5(Ind_020)*tmjq+Rn_5(Ind_120)*tmjx+Rn_5(Ind_030)*tmjy+Rn_5(Ind_021)*tmjz
      phi_i(Ind_002)=  Rn_5(Ind_002)*tmjq+Rn_5(Ind_102)*tmjx+Rn_5(Ind_012)*tmjy+Rn_5(Ind_003)*tmjz
      phi_i(Ind_110)=  Rn_5(Ind_110)*tmjq+Rn_5(Ind_210)*tmjx+Rn_5(Ind_120)*tmjy+Rn_5(Ind_111)*tmjz
      phi_i(Ind_101)=  Rn_5(Ind_101)*tmjq+Rn_5(Ind_201)*tmjx+Rn_5(Ind_111)*tmjy+Rn_5(Ind_102)*tmjz
      phi_i(Ind_011)=  Rn_5(Ind_011)*tmjq+Rn_5(Ind_111)*tmjx+Rn_5(Ind_021)*tmjy+Rn_5(Ind_012)*tmjz

      phi(Ind_000,img_i)=phi(Ind_000,img_i) + phi_i(Ind_000)
      phi(Ind_100,img_i)=phi(Ind_100,img_i) + phi_i(Ind_100)
      phi(Ind_010,img_i)=phi(Ind_010,img_i) + phi_i(Ind_010)
      phi(Ind_001,img_i)=phi(Ind_001,img_i) + phi_i(Ind_001)
      phi(Ind_200,img_i)=phi(Ind_200,img_i) + phi_i(Ind_200)
      phi(Ind_020,img_i)=phi(Ind_020,img_i) + phi_i(Ind_020)
      phi(Ind_002,img_i)=phi(Ind_002,img_i) + phi_i(Ind_002)
      phi(Ind_110,img_i)=phi(Ind_110,img_i) + phi_i(Ind_110)
      phi(Ind_101,img_i)=phi(Ind_101,img_i) + phi_i(Ind_101)
      phi(Ind_011,img_i)=phi(Ind_011,img_i) + phi_i(Ind_011)
#else
      phi(Ind_000,img_i)=phi(Ind_000,img_i) + &
                         Rn_5(Ind_000)*tmjq+Rn_5(Ind_100)*tmjx+Rn_5(Ind_010)*tmjy+Rn_5(Ind_001)*tmjz
      phi(Ind_100,img_i)=phi(Ind_100,img_i) - &
                        (Rn_5(Ind_100)*tmjq+Rn_5(Ind_200)*tmjx+Rn_5(Ind_110)*tmjy+Rn_5(Ind_101)*tmjz)
      phi(Ind_010,img_i)=phi(Ind_010,img_i) - &
                        (Rn_5(Ind_010)*tmjq+Rn_5(Ind_110)*tmjx+Rn_5(Ind_020)*tmjy+Rn_5(Ind_011)*tmjz)
      phi(Ind_001,img_i)=phi(Ind_001,img_i) - &
                        (Rn_5(Ind_001)*tmjq+Rn_5(Ind_101)*tmjx+Rn_5(Ind_011)*tmjy+Rn_5(Ind_002)*tmjz)
      phi(Ind_200,img_i)=phi(Ind_200,img_i) + &
                         Rn_5(Ind_200)*tmjq+Rn_5(Ind_300)*tmjx+Rn_5(Ind_210)*tmjy+Rn_5(Ind_201)*tmjz
      phi(Ind_020,img_i)=phi(Ind_020,img_i) + &
                         Rn_5(Ind_020)*tmjq+Rn_5(Ind_120)*tmjx+Rn_5(Ind_030)*tmjy+Rn_5(Ind_021)*tmjz
      phi(Ind_002,img_i)=phi(Ind_002,img_i) + &
                         Rn_5(Ind_002)*tmjq+Rn_5(Ind_102)*tmjx+Rn_5(Ind_012)*tmjy+Rn_5(Ind_003)*tmjz
      phi(Ind_110,img_i)=phi(Ind_110,img_i) + &
                         Rn_5(Ind_110)*tmjq+Rn_5(Ind_210)*tmjx+Rn_5(Ind_120)*tmjy+Rn_5(Ind_111)*tmjz
      phi(Ind_101,img_i)=phi(Ind_101,img_i) + &
                         Rn_5(Ind_101)*tmjq+Rn_5(Ind_201)*tmjx+Rn_5(Ind_111)*tmjy+Rn_5(Ind_102)*tmjz
      phi(Ind_011,img_i)=phi(Ind_011,img_i) + &
                         Rn_5(Ind_011)*tmjq+Rn_5(Ind_111)*tmjx+Rn_5(Ind_021)*tmjy+Rn_5(Ind_012)*tmjz

#endif

#if 0
      tmi(1) = img_chg(img_i)
      tmi(2:4)= img_dip(1:3,img_i)

      Rn_5(2:4) = -Rn_5(2:4) ! now these should be derivatives against i
      Rn_5(11:20) = -Rn_5(11:20) ! now these should be derivatives against i

      phi_j(Ind_000)=  Rn_5(Ind_000)*tmi(Ind_000)+Rn_5(Ind_100)*tmi(Ind_100)+ &
                       Rn_5(Ind_010)*tmi(Ind_010)+Rn_5(Ind_001)*tmi(Ind_001)
      phi_j(Ind_100)=-(Rn_5(Ind_100)*tmi(Ind_000)+Rn_5(Ind_200)*tmi(Ind_100)+ &
                       Rn_5(Ind_110)*tmi(Ind_010)+Rn_5(Ind_101)*tmi(Ind_001))
      phi_j(Ind_010)=-(Rn_5(Ind_010)*tmi(Ind_000)+Rn_5(Ind_110)*tmi(Ind_100)+ &
                       Rn_5(Ind_020)*tmi(Ind_010)+Rn_5(Ind_011)*tmi(Ind_001))
      phi_j(Ind_001)=-(Rn_5(Ind_001)*tmi(Ind_000)+Rn_5(Ind_101)*tmi(Ind_100)+ &
                       Rn_5(Ind_011)*tmi(Ind_010)+Rn_5(Ind_002)*tmi(Ind_001))
      phi_j(Ind_200)=  Rn_5(Ind_200)*tmi(Ind_000)+Rn_5(Ind_300)*tmi(Ind_100)+ &
                       Rn_5(Ind_210)*tmi(Ind_010)+Rn_5(Ind_201)*tmi(Ind_001)
      phi_j(Ind_020)=  Rn_5(Ind_020)*tmi(Ind_000)+Rn_5(Ind_120)*tmi(Ind_100)+ &
                       Rn_5(Ind_030)*tmi(Ind_010)+Rn_5(Ind_021)*tmi(Ind_001)
      phi_j(Ind_002)=  Rn_5(Ind_002)*tmi(Ind_000)+Rn_5(Ind_102)*tmi(Ind_100)+ &
                       Rn_5(Ind_012)*tmi(Ind_010)+Rn_5(Ind_003)*tmi(Ind_001)
      phi_j(Ind_110)=  Rn_5(Ind_110)*tmi(Ind_000)+Rn_5(Ind_210)*tmi(Ind_100)+ &
                       Rn_5(Ind_120)*tmi(Ind_010)+Rn_5(Ind_111)*tmi(Ind_001)
      phi_j(Ind_101)=  Rn_5(Ind_101)*tmi(Ind_000)+Rn_5(Ind_201)*tmi(Ind_100)+ &
                       Rn_5(Ind_111)*tmi(Ind_010)+Rn_5(Ind_102)*tmi(Ind_001)
      phi_j(Ind_011)=  Rn_5(Ind_011)*tmi(Ind_000)+Rn_5(Ind_111)*tmi(Ind_100)+ &
                       Rn_5(Ind_021)*tmi(Ind_010)+Rn_5(Ind_012)*tmi(Ind_001)

#endif

#if 0
      tmiq   = img_chg(img_i)
      tmix   = img_dip(1, img_i)
      tmiy   = img_dip(2, img_i)
      tmiz   = img_dip(3, img_i)
#endif

#ifdef NEED_VIR
      phi_j(Ind_000)=  Rn_5(Ind_000)*tmiq-(Rn_5(Ind_100)*tmix+Rn_5(Ind_010)*tmiy+Rn_5(Ind_001)*tmiz)
      phi_j(Ind_100)=  Rn_5(Ind_100)*tmiq-(Rn_5(Ind_200)*tmix+Rn_5(Ind_110)*tmiy+Rn_5(Ind_101)*tmiz)
      phi_j(Ind_010)=  Rn_5(Ind_010)*tmiq-(Rn_5(Ind_110)*tmix+Rn_5(Ind_020)*tmiy+Rn_5(Ind_011)*tmiz)
      phi_j(Ind_001)=  Rn_5(Ind_001)*tmiq-(Rn_5(Ind_101)*tmix+Rn_5(Ind_011)*tmiy+Rn_5(Ind_002)*tmiz)
      phi_j(Ind_200)=  Rn_5(Ind_200)*tmiq-(Rn_5(Ind_300)*tmix+Rn_5(Ind_210)*tmiy+Rn_5(Ind_201)*tmiz)
      phi_j(Ind_020)=  Rn_5(Ind_020)*tmiq-(Rn_5(Ind_120)*tmix+Rn_5(Ind_030)*tmiy+Rn_5(Ind_021)*tmiz)
      phi_j(Ind_002)=  Rn_5(Ind_002)*tmiq-(Rn_5(Ind_102)*tmix+Rn_5(Ind_012)*tmiy+Rn_5(Ind_003)*tmiz)
      phi_j(Ind_110)=  Rn_5(Ind_110)*tmiq-(Rn_5(Ind_210)*tmix+Rn_5(Ind_120)*tmiy+Rn_5(Ind_111)*tmiz)
      phi_j(Ind_101)=  Rn_5(Ind_101)*tmiq-(Rn_5(Ind_201)*tmix+Rn_5(Ind_111)*tmiy+Rn_5(Ind_102)*tmiz)
      phi_j(Ind_011)=  Rn_5(Ind_011)*tmiq-(Rn_5(Ind_111)*tmix+Rn_5(Ind_021)*tmiy+Rn_5(Ind_012)*tmiz)

      phi(Ind_000,img_j)=phi(Ind_000,img_j) + phi_j(Ind_000)
      phi(Ind_100,img_j)=phi(Ind_100,img_j) + phi_j(Ind_100)
      phi(Ind_010,img_j)=phi(Ind_010,img_j) + phi_j(Ind_010)
      phi(Ind_001,img_j)=phi(Ind_001,img_j) + phi_j(Ind_001)
      phi(Ind_200,img_j)=phi(Ind_200,img_j) + phi_j(Ind_200)
      phi(Ind_020,img_j)=phi(Ind_020,img_j) + phi_j(Ind_020)
      phi(Ind_002,img_j)=phi(Ind_002,img_j) + phi_j(Ind_002)
      phi(Ind_110,img_j)=phi(Ind_110,img_j) + phi_j(Ind_110)
      phi(Ind_101,img_j)=phi(Ind_101,img_j) + phi_j(Ind_101)
      phi(Ind_011,img_j)=phi(Ind_011,img_j) + phi_j(Ind_011)
#else
      phi(Ind_000,img_j)=phi(Ind_000,img_j) + &
                         Rn_5(Ind_000)*tmiq-(Rn_5(Ind_100)*tmix+Rn_5(Ind_010)*tmiy+Rn_5(Ind_001)*tmiz)
      phi(Ind_100,img_j)=phi(Ind_100,img_j) + &
                         Rn_5(Ind_100)*tmiq-(Rn_5(Ind_200)*tmix+Rn_5(Ind_110)*tmiy+Rn_5(Ind_101)*tmiz)
      phi(Ind_010,img_j)=phi(Ind_010,img_j) + &
                         Rn_5(Ind_010)*tmiq-(Rn_5(Ind_110)*tmix+Rn_5(Ind_020)*tmiy+Rn_5(Ind_011)*tmiz)
      phi(Ind_001,img_j)=phi(Ind_001,img_j) + &
                         Rn_5(Ind_001)*tmiq-(Rn_5(Ind_101)*tmix+Rn_5(Ind_011)*tmiy+Rn_5(Ind_002)*tmiz)
      phi(Ind_200,img_j)=phi(Ind_200,img_j) + &
                         Rn_5(Ind_200)*tmiq-(Rn_5(Ind_300)*tmix+Rn_5(Ind_210)*tmiy+Rn_5(Ind_201)*tmiz)
      phi(Ind_020,img_j)=phi(Ind_020,img_j) + &
                         Rn_5(Ind_020)*tmiq-(Rn_5(Ind_120)*tmix+Rn_5(Ind_030)*tmiy+Rn_5(Ind_021)*tmiz)
      phi(Ind_002,img_j)=phi(Ind_002,img_j) + &
                         Rn_5(Ind_002)*tmiq-(Rn_5(Ind_102)*tmix+Rn_5(Ind_012)*tmiy+Rn_5(Ind_003)*tmiz)
      phi(Ind_110,img_j)=phi(Ind_110,img_j) + &
                         Rn_5(Ind_110)*tmiq-(Rn_5(Ind_210)*tmix+Rn_5(Ind_120)*tmiy+Rn_5(Ind_111)*tmiz)
      phi(Ind_101,img_j)=phi(Ind_101,img_j) + &
                         Rn_5(Ind_101)*tmiq-(Rn_5(Ind_201)*tmix+Rn_5(Ind_111)*tmiy+Rn_5(Ind_102)*tmiz)
      phi(Ind_011,img_j)=phi(Ind_011,img_j) + &
                         Rn_5(Ind_011)*tmiq-(Rn_5(Ind_111)*tmix+Rn_5(Ind_021)*tmiy+Rn_5(Ind_012)*tmiz)
#endif

#ifdef NEED_VIR
! img_disp() is inherited from get_nb_energy_PGM()

      delxd = delx-(img_disp(1,img_j)-img_disp(1,img_i))
      delyd = dely-(img_disp(2,img_j)-img_disp(2,img_i))
      delzd = delz-(img_disp(3,img_j)-img_disp(3,img_i))

! multiply half later, at the end of the subroutine

      t_x  =  (tmjq*phi_j(Ind_100) + &
               tmjx*phi_j(Ind_200) + tmjy*phi_j(Ind_110) + tmjz*phi_j(Ind_101)) &
             -(tmiq*phi_i(Ind_100) + &
               tmix*phi_i(Ind_200) + tmiy*phi_i(Ind_110) + tmiz*phi_i(Ind_101))

      t_y  =  (tmjq*phi_j(Ind_010) + &
               tmjx*phi_j(Ind_110) + tmjy*phi_j(Ind_020) + tmjz*phi_j(Ind_011)) &
             -(tmiq*phi_i(Ind_010) + &
               tmix*phi_i(Ind_110) + tmiy*phi_i(Ind_020) + tmiz*phi_i(Ind_011))

      t_z  =  (tmjq*phi_j(Ind_001) + &
               tmjx*phi_j(Ind_101) + tmjy*phi_j(Ind_011) + tmjz*phi_j(Ind_002)) &
             -(tmiq*phi_i(Ind_001) + &
               tmix*phi_i(Ind_101) + tmiy*phi_i(Ind_011) + tmiz*phi_i(Ind_002))

      eed_virial(1,1) = eed_virial(1,1) + t_x * delxd
      eed_virial(1,2) = eed_virial(1,2) + t_x * delyd
      eed_virial(1,3) = eed_virial(1,3) + t_x * delzd

      eed_virial(2,1) = eed_virial(2,1) + t_y * delxd
      eed_virial(2,2) = eed_virial(2,2) + t_y * delyd
      eed_virial(2,3) = eed_virial(2,3) + t_y * delzd

      eed_virial(3,1) = eed_virial(3,1) + t_z * delxd
      eed_virial(3,2) = eed_virial(3,2) + t_z * delyd
      eed_virial(3,3) = eed_virial(3,3) + t_z * delzd

#endif /* NEED_VIR */

! multipole block ends here

#ifdef NEED_ENE
!        eed_stk = eed_stk + b0
#endif /* NEED_ENE */

!        df = b1 * delrinv * delrinv

#if 0
      end if
#endif /* 0 */

!      dfx = delx * df
!      dfy = dely * df
!      dfz = delz * df

#ifdef NEED_VIR
!      vxx = vxx - delx * dfx
!      vxy = vxy - delx * dfy
!      vxz = vxz - delx * dfz
!      vyy = vyy - dely * dfy
!      vyz = vyz - dely * dfz
!      vzz = vzz - delz * dfz
#endif /* NEED_VIR */

!      dumx = dumx + dfx
!      dumy = dumy + dfy
!      dumz = dumz + dfz

!      img_frc(1, img_j) = img_frc(1, img_j) + dfx
!      img_frc(2, img_j) = img_frc(2, img_j) + dfy
!      img_frc(3, img_j) = img_frc(3, img_j) + dfz
#endif /* _FIELD_ONLY_ */
#ifndef _USE_SAVED_BFACT_
    end if
#endif /* _USE_SAVED_BFACT_ */ 

  end do

#if defined(_FIELD_ONLY_) && defined(_SAVE_BFACT_)
! mark the end of ee_only pairs
  last_ee_ptrs(img_i) = num_tensor
#endif  /* _FIELD_ONLY_ && _SAVE_BFACT_ */

#ifndef _FIELD_ONLY_
  iaci = ntypes_stk * (img_iac(img_i) - 1)
#endif /* _FIELD_ONLY_ */

#ifndef _USE_SAVED_BFACT_
  do sublst_idx = ee_eval_cnt + 2, ee_eval_cnt + full_eval_cnt + 1

    img_j = nxt_img_j

    delx = nxt_delx
    dely = nxt_dely
    delz = nxt_delz

    if (common_tran .eq. 1) then
      nxt_img_j = ipairs_sublst(sublst_idx)
      itran = 13
    else
      nxt_img_j = iand(ipairs_sublst(sublst_idx), mask27)
      itran = ishft(ipairs_sublst(sublst_idx), -27)
    end if

    nxt_delx = img_crd(1, nxt_img_j) + x_tran(1, itran)
    nxt_dely = img_crd(2, nxt_img_j) + x_tran(2, itran)
    nxt_delz = img_crd(3, nxt_img_j) + x_tran(3, itran)

    delx2         = delx * delx
    dely2         = dely * dely
    delz2         = delz * delz
!    delr2 = delx * delx + dely * dely + delz * delz
    delr2 = delx2 + dely2 + delz2

    if (delr2 .lt. max_nb_cut2) then

#else /* _USE_SAVED_BFACT_ */
  do idx = last_ee_ptrs(img_i)+1, dipole_dipole_ptrs(img_i+1) - 1
    img_j = dipole_dipole_list(idx)

    delx  = delvec_save(1, idx)
    dely  = delvec_save(2, idx)
    delz  = delvec_save(3, idx)
    B(0)  = Bfact_save(1, idx)
    B(1)  = Bfact_save(2, idx)
    B(2)  = Bfact_save(3, idx)
    B(3)  = Bfact_save(4, idx)
    delx2         = delx * delx
    dely2         = dely * dely
    delz2         = delz * delz

    delrinv  = delvec_save(4, idx)
    delr2inv = delrinv * delrinv

#endif /* _USE_SAVED_BFACT_ */
#ifndef _FIELD_ONLY_
      ic = ico(iaci + img_iac(img_j))
#endif /* _FIELD_ONLY_ */

#ifndef _USE_SAVED_BFACT_
#ifndef _FIELD_ONLY_
#ifndef _SAVE_BFACT_
! disable es-vdw two cut option for now which requires yet one
! more counter separation of es from vdw terms. YD
#ifdef BUILD_PAIRS_CALC_NOVEC_2CUT
      if (delr2 .lt. es_cut2) then
#endif /* BUILD_PAIRS_CALC_NOVEC_2CUT */

#endif /* _SAVE_BFACT_ */
#endif /* _FIELD_ONLY_ */
#if 0

! beginning of monopole code

      cgi_cgj = cgi * img_qterm(img_j)
!write(0,*)"Test",cgi_cgj,cgi,delr2
      if (delr2 .ge. lowest_efs_u) then

        ! Do the Coulomb part of the direct sum using efs: 

        ind = int(dens_efs * delr2)
        du = delr2 - dble(ind) * del_efs
        du2 = du * du
        du3 = du * du2

#ifdef NEED_ENE
        ind = ishft(ind, 3)             ! 8 * ind

        b0 = cgi_cgj * (ef_tbl(1 + ind) + du * ef_tbl(2 + ind) + &
             du2 * ef_tbl(3 + ind) + du3 * ef_tbl(4 + ind))

        df = cgi_cgj * (ef_tbl(5 + ind) + du * ef_tbl(6 + ind) + &
             du2 * ef_tbl(7 + ind) + du3 * ef_tbl(8 + ind))
#else
        ind = ishft(ind, 2)             ! 4 * ind

        df = cgi_cgj * (f_tbl(1 + ind) + du * f_tbl(2 + ind) + &
             du2 * f_tbl(3 + ind) + du3 * f_tbl(4 + ind))
#endif /* NEED_ENE */
#ifdef NEED_ENE
        eed_stk = eed_stk + b0
#endif /* NEED_ENE */

#ifdef NEED_VIR
        eedvir_stk = eedvir_stk - df * delr2
#endif /* NEED_VIR */
        delr2inv = 1.0 / delr2
      else    ! (delr2 .ge. lowest_efs_u)
        ! Do the Coulomb part of the direct sum using erfc spline table: 

        delr = sqrt(delr2)
        delrinv = 1.0 / delr

        x = dxdr * delr
        ind = int(eedtbdns_stk * x)
        dx = x - dble(ind) * del
        ind = ishft(ind, 2)             ! 4 * ind

        e3dx  = dx * eed_cub(3 + ind)
        e4dx2 = dx * dx *  eed_cub(4 + ind)
#ifdef VACUUM
        switch = 1.0
        d_switch_dx = 1.0
#else
        switch = eed_cub(1 + ind) + &
                 dx * (eed_cub(2 + ind) + half * (e3dx + third * e4dx2))

        d_switch_dx = eed_cub(2 + ind) + e3dx + half * e4dx2
#endif

        b0 = cgi_cgj * delrinv * switch
!write(0,*)b0,cgi_cgj,delrinv,switch
        b1 = (b0 - cgi_cgj * d_switch_dx * dxdr)

#ifdef NEED_ENE
        eed_stk = eed_stk + b0
#endif /* NEED_ENE */
#ifdef NEED_VIR
        eedvir_stk = eedvir_stk - b1
#endif /* NEED_VIR */

        delr2inv = delrinv * delrinv
        df = b1 * delrinv * delrinv

      end if    ! (delr2 .ge. lowest_efs_u)

! monopole code ends here

#endif /* 0 */

! multipole block starts here
        ! Do the Coulomb part of the direct sum using erfc spline table: 

        delr = sqrt(delr2)
        delrinv = 1.0 / delr
        delr2inv = delrinv * delrinv

#ifdef VACUUM
        switch = 1.0
        d_switch_dx = 1.0
#else  /* VACUUM */
        x = dxdr * delr
        ind = int(eedtbdns_stk * x)
!        dx = x - (dble(ind)-one)*del    ! from pGM_direct.F90
        dx = x - dble(ind) * del
        ind = ishft(ind, 2)             ! 4 * ind

        e3dx  = dx * eed_cub(3 + ind)
        e4dx2 = dx * dx *  eed_cub(4 + ind)

        switch = eed_cub(1 + ind) + &
                 dx * (eed_cub(2 + ind) + half * (e3dx + third * e4dx2))

        d_switch_dx = eed_cub(2 + ind) + e3dx + half * e4dx2

        B(0) = (one - switch)*delr*delr2inv                                                                      
        fact = -d_switch_dx*dxdr     ! fact = -d_switch_dx*beta
        B(1) = (B(0) - fact)*delr2inv
        fact = fac*fact                                                                                          
        B(2) = (three*B(1) - fact)*delr2inv
        fact = fac*fact
        B(3) = (five*B(2) - fact)*delr2inv

#endif /* VACUUM */
!  j = img_atm_map(img_j)  ! j points to atom j

        ! compute erfc=one-erf function and its derivative for pGM gaussians.
        betaij  = one/(r_gauss_img(img_i) + r_gauss_img(img_j))
        facij = two*betaij
        betaij  = sqrt(betaij)
        y = betaij*delr
        !if ( y < ee_gauss_cut ) then 
!        ind = int(eedtbdns_stk * y) + 1
        ind = int(eedtbdns_stk * y)
!        dx = y - (dble(ind)-one)*del
        dx = y - (dble(ind))*del
        ind = ishft(ind, 2)             ! 4 * ind

        switchij = eed_cub(1+ind) + dx * (eed_cub(2+ind) + &
             dx * (eed_cub(3+ind) + dx *  eed_cub(4+ind) * third) * half)

        d_switchij_dx =                   eed_cub(2+ind) + &
             dx * (eed_cub(3+ind) + dx *  eed_cub(4+ind) * half)

        !call get_ee_func(x,switchij,d_switchij_dx,1)

        Bij(0) = (one - switchij)*delr*delr2inv ! erf/r
        fact = -d_switchij_dx*betaij
        Bij(1) = (Bij(0) - fact)*delr2inv
        fact = facij*fact
        Bij(2) = (three*Bij(1) - fact)*delr2inv
        fact = facij*fact
        Bij(3) = (five*Bij(2) - fact)*delr2inv

#ifdef VACUUM
        B(0)  = Bij(0)
        B(1)  = -Bij(1)
        B(2)  = Bij(2)
        B(3)  = -Bij(3)
#else  /* VACUUM */
        B(0)  = Bij(0) - B(0)
        B(1)  = B(1) - Bij(1)
        B(2)  = Bij(2) - B(2)
        B(3)  = B(3) - Bij(3)
#endif /* VACUUM */

#endif /* _USE_SAVED_BFACT_ */

! B_BLOCK_ENDS_HERE

#ifdef _FIELD_ONLY_

      ! multipdelz*Rn_1(Ind_000)ole interaction tensor elements

      Rn_3(Ind_100) = delx*B(1)
      Rn_3(Ind_010) = dely*B(1)
      Rn_3(Ind_001) = delz*B(1)
      Rn_3(Ind_200) = B(1) + delx2*B(2)
      Rn_3(Ind_020) = B(1) + dely2*B(2)
      Rn_3(Ind_002) = B(1) + delz2*B(2)
      Rn_3(Ind_110) = delx*dely*B(2)
      Rn_3(Ind_101) = delx*delz*B(2)
      Rn_3(Ind_011) = dely*delz*B(2)

      ! store the dipole-dipole component
      num_tensor = num_tensor+1
#ifdef _SAVE_BFACT_
      delvec_save(1,num_tensor) = delx
      delvec_save(2,num_tensor) = dely
      delvec_save(3,num_tensor) = delz
      delvec_save(4,num_tensor) = delrinv
      Bfact_save(1, num_tensor) = B(0)
      Bfact_save(2, num_tensor) = B(1)
      Bfact_save(3, num_tensor) = B(2)
      Bfact_save(4, num_tensor) = B(3)
#endif
      dipole_dipole_list(num_tensor) = img_j
      dipole_dipole_tensor(1,num_tensor) = Rn_3(Ind_200)
      dipole_dipole_tensor(2,num_tensor) = Rn_3(Ind_110)
      dipole_dipole_tensor(3,num_tensor) = Rn_3(Ind_101)
      dipole_dipole_tensor(4,num_tensor) = Rn_3(Ind_020)
      dipole_dipole_tensor(5,num_tensor) = Rn_3(Ind_011)
      dipole_dipole_tensor(6,num_tensor) = Rn_3(Ind_002)

      if ( delr2 .le. scf_local_cut2 ) then
      ! store the short dipole-dipole component
      num_tensor_short = num_tensor_short+1
      dipole_dipole_list_short(num_tensor_short) = img_j
      dipole_dipole_tensor_short(1,num_tensor_short) = -Bij(1) + delx2*Bij(2) != Rn_3(Ind_200)
      dipole_dipole_tensor_short(2,num_tensor_short) = delx*dely*Bij(2)           != Rn_3(Ind_110)
      dipole_dipole_tensor_short(3,num_tensor_short) = delx*delz*Bij(2)           != Rn_3(Ind_101)
      dipole_dipole_tensor_short(4,num_tensor_short) = -Bij(1) + dely2*Bij(2) != Rn_3(Ind_020)
      dipole_dipole_tensor_short(5,num_tensor_short) = dely*delz*Bij(2)           != Rn_3(Ind_011)
      dipole_dipole_tensor_short(6,num_tensor_short) = -Bij(1) + delz2*Bij(2) != Rn_3(Ind_002)
      end if

      gphi_i(1) = Rn_3(Ind_100)*img_chg(img_j) + &
                  Rn_3(Ind_200)*img_dip(1,img_j) + &
                  Rn_3(Ind_110)*img_dip(2,img_j) + &
                  Rn_3(Ind_101)*img_dip(3,img_j)
      gphi_i(2) = Rn_3(Ind_010)*img_chg(img_j) + &
                  Rn_3(Ind_110)*img_dip(1,img_j) + &
                  Rn_3(Ind_020)*img_dip(2,img_j) + &
                  Rn_3(Ind_011)*img_dip(3,img_j)
      gphi_i(3) = Rn_3(Ind_001)*img_chg(img_j) + &
                  Rn_3(Ind_101)*img_dip(1,img_j) + &
                  Rn_3(Ind_011)*img_dip(2,img_j) + &
                  Rn_3(Ind_002)*img_dip(3,img_j)

      gradphi(1,img_i) = gradphi(1,img_i) - gphi_i(1) ! R. Luo:
      gradphi(2,img_i) = gradphi(2,img_i) - gphi_i(2) ! We are returning to the convention
      gradphi(3,img_i) = gradphi(3,img_i) - gphi_i(3) ! gradphi is grad of phi, not field

      gphi_j(1) = -Rn_3(Ind_100)*img_chg(img_i) + &
                   Rn_3(Ind_200)*img_dip(1,img_i) + &
                   Rn_3(Ind_110)*img_dip(2,img_i) + &
                   Rn_3(Ind_101)*img_dip(3,img_i)
      gphi_j(2) = -Rn_3(Ind_010)*img_chg(img_i) + &
                   Rn_3(Ind_110)*img_dip(1,img_i) + &
                   Rn_3(Ind_020)*img_dip(2,img_i) + &
                   Rn_3(Ind_011)*img_dip(3,img_i)
      gphi_j(3) = -Rn_3(Ind_001)*img_chg(img_i) + &
                   Rn_3(Ind_101)*img_dip(1,img_i) + &
                   Rn_3(Ind_011)*img_dip(2,img_i) + &
                   Rn_3(Ind_002)*img_dip(3,img_i)
      gradphi(1,img_j) = gradphi(1,img_j) - gphi_j(1) ! R.Luo
      gradphi(2,img_j) = gradphi(2,img_j) - gphi_j(2) ! we are returning to the convention
      gradphi(3,img_j) = gradphi(3,img_j) - gphi_j(3) ! gradphi is grad of phi, not field
! _FIELD_ONLY_ BLOCK ENDS HERE
#else  /* _FIELD_ONLY_ */
! NON_FIELD_ONLY_ BLOCK STARTS HERE

      Rn_5(Ind_000) = B(0)
      Rn_5(Ind_100) = delx*B(1)
      Rn_5(Ind_010) = dely*B(1)
      Rn_5(Ind_001) = delz*B(1)
      Rn_5(Ind_200) = B(1) + delx2*B(2)
      Rn_5(Ind_020) = B(1) + dely2*B(2)
      Rn_5(Ind_002) = B(1) + delz2*B(2)
      Rn_5(Ind_110) = delx*dely*B(2)
      Rn_5(Ind_101) = delx*delz*B(2)
      Rn_5(Ind_011) = dely*delz*B(2)
      delxd         = B(2) + delx2*B(3)
      delyd         = B(2) + dely2*B(3)
      delzd         = B(2) + delz2*B(3)
      Rn_5(Ind_300) = delx*(2.d0*B(2) + delxd)
      Rn_5(Ind_030) = dely*(2.d0*B(2) + delyd)
      Rn_5(Ind_003) = delz*(2.d0*B(2) + delzd)
      Rn_5(Ind_210) = dely*delxd
      Rn_5(Ind_201) = delz*delxd
      Rn_5(Ind_120) = delx*delyd
      Rn_5(Ind_021) = delz*delyd
      Rn_5(Ind_102) = delx*delzd
      Rn_5(Ind_012) = dely*delzd
      Rn_5(Ind_111) = delx*dely*delz*B(3)

      tmjq    = img_chg(img_j)
      tmjx    = img_dip(1, img_j)
      tmjy    = img_dip(2, img_j)
      tmjz    = img_dip(3, img_j)

#ifdef NEED_VIR
      phi_i(Ind_000)=  Rn_5(Ind_000)*tmjq+Rn_5(Ind_100)*tmjx+Rn_5(Ind_010)*tmjy+Rn_5(Ind_001)*tmjz
      phi_i(Ind_100)=-(Rn_5(Ind_100)*tmjq+Rn_5(Ind_200)*tmjx+Rn_5(Ind_110)*tmjy+Rn_5(Ind_101)*tmjz)
      phi_i(Ind_010)=-(Rn_5(Ind_010)*tmjq+Rn_5(Ind_110)*tmjx+Rn_5(Ind_020)*tmjy+Rn_5(Ind_011)*tmjz)
      phi_i(Ind_001)=-(Rn_5(Ind_001)*tmjq+Rn_5(Ind_101)*tmjx+Rn_5(Ind_011)*tmjy+Rn_5(Ind_002)*tmjz)
      phi_i(Ind_200)=  Rn_5(Ind_200)*tmjq+Rn_5(Ind_300)*tmjx+Rn_5(Ind_210)*tmjy+Rn_5(Ind_201)*tmjz
      phi_i(Ind_020)=  Rn_5(Ind_020)*tmjq+Rn_5(Ind_120)*tmjx+Rn_5(Ind_030)*tmjy+Rn_5(Ind_021)*tmjz
      phi_i(Ind_002)=  Rn_5(Ind_002)*tmjq+Rn_5(Ind_102)*tmjx+Rn_5(Ind_012)*tmjy+Rn_5(Ind_003)*tmjz
      phi_i(Ind_110)=  Rn_5(Ind_110)*tmjq+Rn_5(Ind_210)*tmjx+Rn_5(Ind_120)*tmjy+Rn_5(Ind_111)*tmjz
      phi_i(Ind_101)=  Rn_5(Ind_101)*tmjq+Rn_5(Ind_201)*tmjx+Rn_5(Ind_111)*tmjy+Rn_5(Ind_102)*tmjz
      phi_i(Ind_011)=  Rn_5(Ind_011)*tmjq+Rn_5(Ind_111)*tmjx+Rn_5(Ind_021)*tmjy+Rn_5(Ind_012)*tmjz

      phi(Ind_000,img_i)=phi(Ind_000,img_i) + phi_i(Ind_000)
      phi(Ind_100,img_i)=phi(Ind_100,img_i) + phi_i(Ind_100)
      phi(Ind_010,img_i)=phi(Ind_010,img_i) + phi_i(Ind_010)
      phi(Ind_001,img_i)=phi(Ind_001,img_i) + phi_i(Ind_001)
      phi(Ind_200,img_i)=phi(Ind_200,img_i) + phi_i(Ind_200)
      phi(Ind_020,img_i)=phi(Ind_020,img_i) + phi_i(Ind_020)
      phi(Ind_002,img_i)=phi(Ind_002,img_i) + phi_i(Ind_002)
      phi(Ind_110,img_i)=phi(Ind_110,img_i) + phi_i(Ind_110)
      phi(Ind_101,img_i)=phi(Ind_101,img_i) + phi_i(Ind_101)
      phi(Ind_011,img_i)=phi(Ind_011,img_i) + phi_i(Ind_011)
#else  /* NEED_VIR */
      phi(Ind_000,img_i)=phi(Ind_000,img_i) + &
                         Rn_5(Ind_000)*tmjq+Rn_5(Ind_100)*tmjx+Rn_5(Ind_010)*tmjy+Rn_5(Ind_001)*tmjz
      phi(Ind_100,img_i)=phi(Ind_100,img_i) - &
                        (Rn_5(Ind_100)*tmjq+Rn_5(Ind_200)*tmjx+Rn_5(Ind_110)*tmjy+Rn_5(Ind_101)*tmjz)
      phi(Ind_010,img_i)=phi(Ind_010,img_i) - &
                        (Rn_5(Ind_010)*tmjq+Rn_5(Ind_110)*tmjx+Rn_5(Ind_020)*tmjy+Rn_5(Ind_011)*tmjz)
      phi(Ind_001,img_i)=phi(Ind_001,img_i) - &
                        (Rn_5(Ind_001)*tmjq+Rn_5(Ind_101)*tmjx+Rn_5(Ind_011)*tmjy+Rn_5(Ind_002)*tmjz)
      phi(Ind_200,img_i)=phi(Ind_200,img_i) + &
                         Rn_5(Ind_200)*tmjq+Rn_5(Ind_300)*tmjx+Rn_5(Ind_210)*tmjy+Rn_5(Ind_201)*tmjz
      phi(Ind_020,img_i)=phi(Ind_020,img_i) + &
                         Rn_5(Ind_020)*tmjq+Rn_5(Ind_120)*tmjx+Rn_5(Ind_030)*tmjy+Rn_5(Ind_021)*tmjz
      phi(Ind_002,img_i)=phi(Ind_002,img_i) + &
                         Rn_5(Ind_002)*tmjq+Rn_5(Ind_102)*tmjx+Rn_5(Ind_012)*tmjy+Rn_5(Ind_003)*tmjz
      phi(Ind_110,img_i)=phi(Ind_110,img_i) + &
                         Rn_5(Ind_110)*tmjq+Rn_5(Ind_210)*tmjx+Rn_5(Ind_120)*tmjy+Rn_5(Ind_111)*tmjz
      phi(Ind_101,img_i)=phi(Ind_101,img_i) + &
                         Rn_5(Ind_101)*tmjq+Rn_5(Ind_201)*tmjx+Rn_5(Ind_111)*tmjy+Rn_5(Ind_102)*tmjz
      phi(Ind_011,img_i)=phi(Ind_011,img_i) + &
                         Rn_5(Ind_011)*tmjq+Rn_5(Ind_111)*tmjx+Rn_5(Ind_021)*tmjy+Rn_5(Ind_012)*tmjz

#endif  /* NEED_VIR */

#ifdef NEED_VIR
      phi_j(Ind_000)=  Rn_5(Ind_000)*tmiq-(Rn_5(Ind_100)*tmix+Rn_5(Ind_010)*tmiy+Rn_5(Ind_001)*tmiz)
      phi_j(Ind_100)=  Rn_5(Ind_100)*tmiq-(Rn_5(Ind_200)*tmix+Rn_5(Ind_110)*tmiy+Rn_5(Ind_101)*tmiz)
      phi_j(Ind_010)=  Rn_5(Ind_010)*tmiq-(Rn_5(Ind_110)*tmix+Rn_5(Ind_020)*tmiy+Rn_5(Ind_011)*tmiz)
      phi_j(Ind_001)=  Rn_5(Ind_001)*tmiq-(Rn_5(Ind_101)*tmix+Rn_5(Ind_011)*tmiy+Rn_5(Ind_002)*tmiz)
      phi_j(Ind_200)=  Rn_5(Ind_200)*tmiq-(Rn_5(Ind_300)*tmix+Rn_5(Ind_210)*tmiy+Rn_5(Ind_201)*tmiz)
      phi_j(Ind_020)=  Rn_5(Ind_020)*tmiq-(Rn_5(Ind_120)*tmix+Rn_5(Ind_030)*tmiy+Rn_5(Ind_021)*tmiz)
      phi_j(Ind_002)=  Rn_5(Ind_002)*tmiq-(Rn_5(Ind_102)*tmix+Rn_5(Ind_012)*tmiy+Rn_5(Ind_003)*tmiz)
      phi_j(Ind_110)=  Rn_5(Ind_110)*tmiq-(Rn_5(Ind_210)*tmix+Rn_5(Ind_120)*tmiy+Rn_5(Ind_111)*tmiz)
      phi_j(Ind_101)=  Rn_5(Ind_101)*tmiq-(Rn_5(Ind_201)*tmix+Rn_5(Ind_111)*tmiy+Rn_5(Ind_102)*tmiz)
      phi_j(Ind_011)=  Rn_5(Ind_011)*tmiq-(Rn_5(Ind_111)*tmix+Rn_5(Ind_021)*tmiy+Rn_5(Ind_012)*tmiz)

      phi(Ind_000,img_j)=phi(Ind_000,img_j) + phi_j(Ind_000)
      phi(Ind_100,img_j)=phi(Ind_100,img_j) + phi_j(Ind_100)
      phi(Ind_010,img_j)=phi(Ind_010,img_j) + phi_j(Ind_010)
      phi(Ind_001,img_j)=phi(Ind_001,img_j) + phi_j(Ind_001)
      phi(Ind_200,img_j)=phi(Ind_200,img_j) + phi_j(Ind_200)
      phi(Ind_020,img_j)=phi(Ind_020,img_j) + phi_j(Ind_020)
      phi(Ind_002,img_j)=phi(Ind_002,img_j) + phi_j(Ind_002)
      phi(Ind_110,img_j)=phi(Ind_110,img_j) + phi_j(Ind_110)
      phi(Ind_101,img_j)=phi(Ind_101,img_j) + phi_j(Ind_101)
      phi(Ind_011,img_j)=phi(Ind_011,img_j) + phi_j(Ind_011)
#else
      phi(Ind_000,img_j)=phi(Ind_000,img_j) + &
                         Rn_5(Ind_000)*tmiq-(Rn_5(Ind_100)*tmix+Rn_5(Ind_010)*tmiy+Rn_5(Ind_001)*tmiz)
      phi(Ind_100,img_j)=phi(Ind_100,img_j) + &
                         Rn_5(Ind_100)*tmiq-(Rn_5(Ind_200)*tmix+Rn_5(Ind_110)*tmiy+Rn_5(Ind_101)*tmiz)
      phi(Ind_010,img_j)=phi(Ind_010,img_j) + &
                         Rn_5(Ind_010)*tmiq-(Rn_5(Ind_110)*tmix+Rn_5(Ind_020)*tmiy+Rn_5(Ind_011)*tmiz)
      phi(Ind_001,img_j)=phi(Ind_001,img_j) + &
                         Rn_5(Ind_001)*tmiq-(Rn_5(Ind_101)*tmix+Rn_5(Ind_011)*tmiy+Rn_5(Ind_002)*tmiz)
      phi(Ind_200,img_j)=phi(Ind_200,img_j) + &
                         Rn_5(Ind_200)*tmiq-(Rn_5(Ind_300)*tmix+Rn_5(Ind_210)*tmiy+Rn_5(Ind_201)*tmiz)
      phi(Ind_020,img_j)=phi(Ind_020,img_j) + &
                         Rn_5(Ind_020)*tmiq-(Rn_5(Ind_120)*tmix+Rn_5(Ind_030)*tmiy+Rn_5(Ind_021)*tmiz)
      phi(Ind_002,img_j)=phi(Ind_002,img_j) + &
                         Rn_5(Ind_002)*tmiq-(Rn_5(Ind_102)*tmix+Rn_5(Ind_012)*tmiy+Rn_5(Ind_003)*tmiz)
      phi(Ind_110,img_j)=phi(Ind_110,img_j) + &
                         Rn_5(Ind_110)*tmiq-(Rn_5(Ind_210)*tmix+Rn_5(Ind_120)*tmiy+Rn_5(Ind_111)*tmiz)
      phi(Ind_101,img_j)=phi(Ind_101,img_j) + &
                         Rn_5(Ind_101)*tmiq-(Rn_5(Ind_201)*tmix+Rn_5(Ind_111)*tmiy+Rn_5(Ind_102)*tmiz)
      phi(Ind_011,img_j)=phi(Ind_011,img_j) + &
                         Rn_5(Ind_011)*tmiq-(Rn_5(Ind_111)*tmix+Rn_5(Ind_021)*tmiy+Rn_5(Ind_012)*tmiz)
#endif

#ifdef NEED_VIR

      delxd = delx-(img_disp(1,img_j)-img_disp(1,img_i))
      delyd = dely-(img_disp(2,img_j)-img_disp(2,img_i))
      delzd = delz-(img_disp(3,img_j)-img_disp(3,img_i))
! multiply half later, at the end of the subroutine

      t_x  =  (tmjq*phi_j(Ind_100) + &
               tmjx*phi_j(Ind_200) + tmjy*phi_j(Ind_110) + tmjz*phi_j(Ind_101)) &
             -(tmiq*phi_i(Ind_100) + &
               tmix*phi_i(Ind_200) + tmiy*phi_i(Ind_110) + tmiz*phi_i(Ind_101))

      t_y  =  (tmjq*phi_j(Ind_010) + &
               tmjx*phi_j(Ind_110) + tmjy*phi_j(Ind_020) + tmjz*phi_j(Ind_011)) &
             -(tmiq*phi_i(Ind_010) + &
               tmix*phi_i(Ind_110) + tmiy*phi_i(Ind_020) + tmiz*phi_i(Ind_011))

      t_z  =  (tmjq*phi_j(Ind_001) + &
               tmjx*phi_j(Ind_101) + tmjy*phi_j(Ind_011) + tmjz*phi_j(Ind_002)) &
             -(tmiq*phi_i(Ind_001) + &
               tmix*phi_i(Ind_101) + tmiy*phi_i(Ind_011) + tmiz*phi_i(Ind_002))

      eed_virial(1,1) = eed_virial(1,1) + t_x * delxd
      eed_virial(1,2) = eed_virial(1,2) + t_x * delyd
      eed_virial(1,3) = eed_virial(1,3) + t_x * delzd

      eed_virial(2,1) = eed_virial(2,1) + t_y * delxd
      eed_virial(2,2) = eed_virial(2,2) + t_y * delyd
      eed_virial(2,3) = eed_virial(2,3) + t_y * delzd

      eed_virial(3,1) = eed_virial(3,1) + t_z * delxd
      eed_virial(3,2) = eed_virial(3,2) + t_z * delyd
      eed_virial(3,3) = eed_virial(3,3) + t_z * delzd

#endif /* NEED_VIR */

! multipole block ends here

#ifndef _USE_SAVED_BFACT_
#ifndef _FIELD_ONLY_
#ifndef _SAVE_BFACT_
! disable es-vdw two cut option for now which requires yet one
! more counter separation of es from vdw terms. YD

#ifdef BUILD_PAIRS_CALC_NOVEC_2CUT
      else   ! (delr2 .lt. es_cut2)
#ifdef NEED_VIR
        delxd = delx-(img_disp(1,img_j)-img_disp(1,img_i))
        delyd = dely-(img_disp(2,img_j)-img_disp(2,img_i))
        delzd = delz-(img_disp(3,img_j)-img_disp(3,img_i))
#endif /* NEED_VIR */
        delr2inv = 1.0 / delr2
!        df = 0.0
      end if   ! (delr2 .lt. es_cut2)
#endif /* BUILD_PAIRS_CALC_NOVEC_2CUT */

#endif /* _SAVE_BFACT_ */
#endif /* _FIELD_ONLY_ */
#endif /* _USE_SAVED_BFACT_ */

#ifdef HAS_10_12
      if (ic .gt. 0) then
#endif

#ifndef NEED_GVDW
! LJ 12-6
        r6 = delr2inv * delr2inv * delr2inv
#ifdef FSWITCH
#ifdef _USE_SAVED_BFACT_
          delr12inv=r6*r6
            
          if(delrinv .lt. fswitchinv) then ! r < ron
          delr3inv=delr2inv * delrinv
#else /* _USE_SAVED_BFACT_ */
          delr2inv=1.d0 / delr2
          delr12inv=r6*r6
            
          if(delr2 .gt. fswitch2) then ! r < ron
          delr3inv=delr2inv * sqrt(delr2inv)
#endif /* _USE_SAVED_BFACT_ */
            p12=cut6/(cut6-fswitch6)
            p6= cut3/(cut3-fswitch3)
            f12=cn1(ic)*p12*(r6-cut6inv)*(r6-cut6inv)
            f6=cn2(ic)*p6*(delr3inv-cut3inv)*(delr3inv-cut3inv)
            df12=-12.d0*p12*delr2inv*r6*(r6-cut6inv)
            df6=-6.d0*p6*(delr3inv-cut3inv)*delr3inv*delr2inv
          else  ! r<fswitch
            f12=cn1(ic)*delr12inv-cn1(ic)*invfswitch6cut6
            f6=cn2(ic)*r6-cn2(ic)*invfswitch3cut3
            df12=-12.d0*delr2inv*delr12inv
            df6=-6.d0*delr2inv*r6
          end if
#ifdef NEED_ENE
          evdw_stk = evdw_stk + f12 - f6
#endif /* NEED ENE */
!          df = df - cn1(ic)*df12 + cn2(ic)*df6
          df = - cn1(ic)*df12 + cn2(ic)*df6
#else  /* FSWITCH */
          f6 = cn2(ic) * r6
          f12 = cn1(ic) * (r6 * r6)
#ifdef NEED_ENE
          evdw_stk = evdw_stk + f12 - f6
#endif /* NEED_ENE */
!          df = df + (12.0 * f12 - 6.0 * f6) * delr2inv
          df = (12.0 * f12 - 6.0 * f6) * delr2inv
#endif /* FSWITCH */

#else /* NEED_GVDW */
! Gaussian VDW (GVDW):
!   U = A_ij exp(-b kappa_ij^2 r^2) - C6_ij F(kappa_ij r) / r^6     (gvdw_rep_form = 0)
!   U = A_ij exp(-b kappa_ij r)     - C6_ij F(kappa_ij r) / r^6     (gvdw_rep_form = 1)
!   F(x) = A(x)^2 + g(x)^2/2,  A(x) = erf(x) - 2x exp(-x^2)/sqrt(pi) - g(x),
!   g(x) = 4 x^3 exp(-x^2)/(3 sqrt(pi)),  kappa_ij = (r_i^2 + r_j^2)^(-1/2) from the pGM Gaussian radii.
!   A_ij = cn1(ic), C6_ij = cn2(ic) (topology LJ arrays hold GVDW coefficients); b = b_rep_scale.
        delrinv = sqrt(delr2inv)
        kapg   = sqrt(one / (r_gauss_img(img_i) + r_gauss_img(img_j)))
        yg     = kapg / delrinv
        yg2    = yg * yg
        expy2g = exp(-yg2)
        erfyg  = erf(yg)
        r6     = delr2inv * delr2inv * delr2inv
        gx  = (4.0d0 / 3.0d0) * inv_sqrtpi * yg * yg2 * expy2g
        bx  = erfyg - two * inv_sqrtpi * yg * expy2g
        ax  = bx - gx
        fx  = ax * ax + 0.5d0 * gx * gx
        apx = (8.0d0 / 3.0d0) * inv_sqrtpi * yg2 * yg2 * expy2g
        gpx = 4.0d0 * inv_sqrtpi * yg2 * (one - (two / three) * yg2) * expy2g
        fpx = two * ax * apx + gx * gpx
        if (gvdw_rep_form .eq. 0) then
          exprep = cn1(ic) * exp(-b_rep_scale * yg2)
          df = two * b_rep_scale * kapg * kapg * exprep
        else
          exprep = cn1(ic) * exp(-b_rep_scale * yg)
          df = b_rep_scale * kapg * exprep * delrinv
        end if
        if (yg .gt. 0.01d0) then
#ifdef NEED_ENE
          evdw_stk = evdw_stk + exprep - cn2(ic) * fx * r6
#endif /* NEED_ENE */
          df = df + cn2(ic) * (kapg * fpx * delrinv - 6.0d0 * fx * delr2inv) * r6
        else
#ifdef NEED_ENE
          ! F(x)/x^6 -> 8/(9 pi) as x -> 0
          evdw_stk = evdw_stk + exprep - cn2(ic) * (8.0d0 / (9.0d0 * pi_gvdw)) * kapg**6
#endif /* NEED_ENE */
        end if
#endif /* NEED_GVDW */

#ifdef HAS_10_12
      else if( ic.lt.0 ) then
        ! This code allows 10-12 terms; in many (most?) (all?) cases, the
        ! only "nominal" 10-12 terms are on waters, where the asol and bsol
        ! parameters are always zero; hence we can skip this part.
        ic = - ic
        r10 = delr2inv * delr2inv * delr2inv * delr2inv * delr2inv
        f10 = gbl_bsol(ic) * r10
        f12 = gbl_asol(ic) * (r10 * delr2inv)
#ifdef NEED_ENE
        ehb_stk = ehb_stk + f12 - f10
      end if  ! ( ic.lt.0 )    !! Should this line be removed? YD
#endif /* NEED_ENE */
!        df = df + (12.0 * f12 - 10.0 * f10) * delr2inv
        df = (12.0 * f12 - 10.0 * f10) * delr2inv
      end if  ! (ic .gt. 0)
#endif /* HAS_10_12 */
      dfx = delx * df
      dfy = dely * df
      dfz = delz * df

#ifdef NEED_VIR

      vxx = vxx - dfx * delxd
      vxy = vxy - dfx * delyd
      vxz = vxz - dfx * delzd
      vyx = vyx - dfy * delxd
      vyy = vyy - dfy * delyd
      vyz = vyz - dfy * delzd
      vzx = vzx - dfz * delxd
      vzy = vzy - dfz * delyd
      vzz = vzz - dfz * delzd
#endif /* NEED_VIR */

      dumx = dumx + dfx
      dumy = dumy + dfy
      dumz = dumz + dfz

      img_frc(1, img_j) = img_frc(1, img_j) + dfx
      img_frc(2, img_j) = img_frc(2, img_j) + dfy
      img_frc(3, img_j) = img_frc(3, img_j) + dfz
! NON_FIELD_ONLY_ BLOCK ENDS HERE
#endif /* _FIELD_ONLY_ */
#ifndef _USE_SAVED_BFACT_
    end if
#endif /* _USE_SAVED_BFACT_ */

  end do

#ifndef _FIELD_ONLY_
  img_frc(1, img_i) = img_frc(1, img_i) - dumx
  img_frc(2, img_i) = img_frc(2, img_i) - dumy
  img_frc(3, img_i) = img_frc(3, img_i) - dumz
#else
!  dipole_dipole_ptrs(img_i+1) = num_tensor + 1
!  dipole_dipole_short_ptrs(img_i+1) = num_tensor_short + 1
#endif

#ifndef _USE_SAVED_BFACT_
  ipairs_sublst(ee_eval_cnt + full_eval_cnt + 1) = saved_pairlist_val
#endif

  return

  end subroutine PGM_SUBS

#undef _TYPE
#undef _CUT
#undef _VDW
#undef _VAC
#undef _SWITCH
#undef _BFACT
#undef PGM_SUBS
#undef PGM_SUB_RAW
#undef PGM_SUB_NAME
#undef PGM_HEAD

#ifdef NEED_ENE
#undef NEED_ENE
#endif
#ifdef NEED_VIR
#undef NEED_VIR
#endif
#ifdef _NEED_B3_
#undef _NEED_B3_
#endif
#ifdef _EFTBL_
#undef _EFTBL_
#endif
#ifdef NEED_GVDW
#undef NEED_GVDW
#endif

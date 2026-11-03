#include "copyright_pgm.i"
!*******************************************************************************
!
! Adopted from non-pGM pmemd code by YD
!
! Subroutine:  grad_sum
!
! Description: <TBS>
!              
!*******************************************************************************

#ifdef _PGM_
#ifdef _NOT_PGM_
#undef _NOT_PGM_
#endif
#ifdef FIELD_ONLY
#ifdef NOT_FIELD_ONLY
#undef NOT_FIELD_ONLY
#endif
subroutine grad_sum_fld_PGM(q, theta, dtheta, d2theta, order, img_chg, img_frac_dip, img_fld, ifracts, &
                    my_chgs, my_chg_cnt)
#else /* FIELD_ONLY */

#ifndef NOT_FIELD_ONLY
#define NOT_FIELD_ONLY
#endif

subroutine grad_sum_PGM(q, theta, dtheta, d2theta, order, img_chg, img_frac_dip, F_dip_fld, ifracts, &
                    my_chgs, my_chg_cnt)
#endif /* FIELD_ONLY */
#else /* PGM */
#ifndef _NOT_PGM_
#define _NOT_PGM_
#endif
subroutine grad_sum(q, theta, dtheta, order, img_qterm, img_frc, ifracts, &
                    my_chgs, my_chg_cnt)
#endif /* PGM */

  use img_mod
  use mdin_ewald_dat_mod
  use prmtop_dat_mod
  use pbc_mod
  use pme_blk_fft_mod
  use ti_mod

  implicit none

! Formal arguments:

  double precision      :: q(2 * fft_x_dim, &
                             fft_y_cnts1(my_grid_idx1), &
                             fft_z_cnts(my_grid_idx2))
  integer               :: order
  double precision      :: theta(order, 3, *)
  double precision      :: dtheta(order, 3, *)
#ifdef _PGM_
#include "pGM_mpole_index.h"
  double precision      :: img_chg(*)
  double precision      :: img_frac_dip(3, *)
  double precision      :: d2theta(order, 3, *)
#else
  double precision      :: img_qterm(*)
#endif
#ifdef _PGM_
#ifdef FIELD_ONLY
  double precision      :: img_fld(3, *)
#else
  double precision      :: F_dip_fld(10, *)
#endif
#else
  double precision      :: img_frc(3, *)
#endif
  integer               :: ifracts(3, *)

  integer               :: my_chgs(*), my_chg_cnt

! Local variables:

  integer               :: i, j, k
  integer               :: i_base, j_base, k_base
  integer               :: ith1, ith2, ith3
  integer               :: my_chg_idx
  double precision      :: f1, f2, f3
  double precision      :: charge, qterm
  double precision      :: f1_term, f2_term, f3_term
  double precision      :: dfx, dfy, dfz
  double precision      :: recip_11, recip_22, recip_33
  double precision      :: dnfft1, dnfft2, dnfft3
  integer               :: img_i
  integer               :: y_cnt, y_off
  integer               :: z_cnt, z_off
  integer               :: jbot, jtop
  integer               :: kbot, ktop
  integer               :: i_tbl(-order : nfft1)
  integer               :: j_tbl(-order : nfft2)
  integer               :: k_tbl(-order : nfft3)
  ! For TI:
  integer               :: iatom
#ifdef _PGM_
  double precision      :: tx(3), txy(6), txyz(10), tyz(6), dip(3)
#ifndef FIELD_ONLY
  double precision      :: xform11(3,3),xform22(3,3),xform33(3,3),xform12(3,3),xform13(3,3),xform23(3,3)
  double precision      :: xform(6,6), xform3x3(3,3,6)
#endif
#endif
#ifdef RANGE_DBG
  integer               :: dbg_img_map(natom)
  integer               :: dbg_img_map_res(natom)

  dbg_img_map(:) = 0
#endif /* RANGE_DBG */

  y_cnt = fft_y_cnts1(my_grid_idx1)
  y_off = fft_y_offs1(my_grid_idx1)
  z_cnt = fft_z_cnts(my_grid_idx2)
  z_off = fft_z_offs(my_grid_idx2)

#ifdef _PGM_
#ifndef FIELD_ONLY
  do j = 1, 3
    do i = 1, 3
      xform11(i, j) = field_xform_3x3(1, i) * field_xform_3x3(1, j)
      xform22(i, j) = field_xform_3x3(2, i) * field_xform_3x3(2, j)
      xform33(i, j) = field_xform_3x3(3, i) * field_xform_3x3(3, j)
      xform12(i, j) = field_xform_3x3(1, i) * field_xform_3x3(2, j)
      xform13(i, j) = field_xform_3x3(1, i) * field_xform_3x3(3, j)
      xform23(i, j) = field_xform_3x3(2, i) * field_xform_3x3(3, j)
      xform3x3(i,j, 1) = xform11(i, j)
      xform3x3(i,j, 2) = xform22(i, j)
      xform3x3(i,j, 3) = xform33(i, j)
      xform3x3(i,j, 4) = xform12(i, j)
      xform3x3(i,j, 5) = xform13(i, j)
      xform3x3(i,j, 6) = xform23(i, j)
    end do
  end do
  do i = 1, 6
     xform(1,i) = xform3x3(1,1,i)
     xform(2,i) = xform3x3(2,2,i)
     xform(3,i) = xform3x3(3,3,i)
     xform(4,i) = xform3x3(1,2,i) + xform3x3(2,1,i)
     xform(5,i) = xform3x3(1,3,i) + xform3x3(3,1,i)
     xform(6,i) = xform3x3(2,3,i) + xform3x3(3,2,i)
  end do
#endif
#endif

  ! Initialize the indexing tables.  It actually produces faster code to
  ! do this here rather than caching the info remotely.
  do i = 0, nfft1
    i_tbl(i) = i + 1
  end do
  do i = -order, -1
    i_tbl(i) = i + nfft1 + 1
  end do

  jbot = 1
  jtop = y_cnt
  do j = 0, nfft2
    j_tbl(j) = j - y_off + 1
    if (j_tbl(j) .lt. jbot .or. j_tbl(j) .gt. jtop) j_tbl(j) = 0
  end do
  do j = -order, -1
    j_tbl(j) = j + nfft2 - y_off + 1
    if (j_tbl(j) .lt. jbot .or. j_tbl(j) .gt. jtop) j_tbl(j) = 0
  end do

  kbot = 1
  ktop = z_cnt
  do k = 0, nfft3
    k_tbl(k) = k - z_off + 1
    if (k_tbl(k) .lt. kbot .or. k_tbl(k) .gt. ktop) k_tbl(k) = 0
  end do
  do k = -order, -1
    k_tbl(k) = k + nfft3 - z_off + 1
    if (k_tbl(k) .lt. kbot .or. k_tbl(k) .gt. ktop) k_tbl(k) = 0
  end do

  recip_11 = recip(1, 1)
  recip_22 = recip(2, 2)
  recip_33 = recip(3, 3)

  dnfft1 = dble(nfft1)
  dnfft2 = dble(nfft2)
  dnfft3 = dble(nfft3)

  do my_chg_idx = 1, my_chg_cnt
    img_i = my_chgs(my_chg_idx)

#ifndef _PGM_
    if (ti_mode .eq. 1 .or. ti_mode .eq. 2) then
      !Determine atom # that we are dealing with 
      iatom = gbl_img_atm_map(img_i) ! get atom number from index
      if(ti_lst(ti_mask_piece,iatom) .ne. 0) cycle
    end if

    charge = img_qterm(img_i)
#else
#ifndef FIELD_ONLY
    charge = img_chg(img_i)
    dip(1:3) = img_frac_dip(1:3,img_i)
#endif
#endif

    i_base = ifracts(1, my_chg_idx)
    j_base = ifracts(2, my_chg_idx)
    k_base = ifracts(3, my_chg_idx)

#if defined(FIELD_ONLY) || defined(_NOT_PGM_)
    f1 = 0.d0
    f2 = 0.d0
    f3 = 0.d0
#else
    txyz = 0.d0
#endif

    ! We special-case order 4, the default.

    if (order .eq. 4) then
      do ith3 = 1, 4

        k = k_tbl(k_base + ith3)

        if (k .ne. 0) then

#ifdef RANGE_DBG
          dbg_img_map(my_chgs(my_chg_idx)) = dbg_img_map(my_chgs(my_chg_idx))+1
#endif /* RANGE_DBG */

          do ith2 = 1, 4

            j = j_tbl(j_base + ith2)

            if (j .ne. 0) then
#if defined(FIELD_ONLY) || defined(_NOT_PGM_)
              f1_term = theta(ith2, 2, my_chg_idx) * theta(ith3, 3, my_chg_idx)
              f2_term = dtheta(ith2, 2, my_chg_idx) * theta(ith3, 3, my_chg_idx)
              f3_term = theta(ith2, 2, my_chg_idx) * dtheta(ith3, 3, my_chg_idx)

! Force is negative of grad:

              qterm = q(i_tbl(i_base+1), j, k)
              f1 = f1 - qterm * dtheta(1, 1, my_chg_idx) * f1_term
              f2 = f2 - qterm * theta(1, 1, my_chg_idx) * f2_term
              f3 = f3 - qterm * theta(1, 1, my_chg_idx) * f3_term
              qterm = q(i_tbl(i_base+2), j, k)
              f1 = f1 - qterm * dtheta(2, 1, my_chg_idx) * f1_term
              f2 = f2 - qterm * theta(2, 1, my_chg_idx) * f2_term
              f3 = f3 - qterm * theta(2, 1, my_chg_idx) * f3_term
              qterm = q(i_tbl(i_base+3), j, k)
              f1 = f1 - qterm * dtheta(3, 1, my_chg_idx) * f1_term
              f2 = f2 - qterm * theta(3, 1, my_chg_idx) * f2_term
              f3 = f3 - qterm * theta(3, 1, my_chg_idx) * f3_term
              qterm = q(i_tbl(i_base+4), j, k)
              f1 = f1 - qterm * dtheta(4, 1, my_chg_idx) * f1_term
              f2 = f2 - qterm * theta(4, 1, my_chg_idx) * f2_term
              f3 = f3 - qterm * theta(4, 1, my_chg_idx) * f3_term
#else
            tyz(Ind_00) =  -theta(ith2, 2, my_chg_idx) *   theta(ith3, 3, my_chg_idx)
            tyz(Ind_10) = -dtheta(ith2, 2, my_chg_idx) *   theta(ith3, 3, my_chg_idx)
            tyz(Ind_01) =  -theta(ith2, 2, my_chg_idx) *  dtheta(ith3, 3, my_chg_idx)
            tyz(Ind_20) =-d2theta(ith2, 2, my_chg_idx) *   theta(ith3, 3, my_chg_idx)
            tyz(Ind_02) =  -theta(ith2, 2, my_chg_idx) * d2theta(ith3, 3, my_chg_idx)
            tyz(Ind_11) = -dtheta(ith2, 2, my_chg_idx) *  dtheta(ith3, 3, my_chg_idx)

            qterm = q(i_tbl(i_base+1), j, k)
            txyz(Ind_000) = txyz(Ind_000) +   theta(1, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_100) = txyz(Ind_100) +  dtheta(1, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_010) = txyz(Ind_010) +   theta(1, 1, my_chg_idx) * tyz(Ind_10) * qterm
            txyz(Ind_001) = txyz(Ind_001) +   theta(1, 1, my_chg_idx) * tyz(Ind_01) * qterm
            txyz(Ind_200) = txyz(Ind_200) + d2theta(1, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_020) = txyz(Ind_020) +   theta(1, 1, my_chg_idx) * tyz(Ind_20) * qterm
            txyz(Ind_002) = txyz(Ind_002) +   theta(1, 1, my_chg_idx) * tyz(Ind_02) * qterm
            txyz(Ind_110) = txyz(Ind_110) +  dtheta(1, 1, my_chg_idx) * tyz(Ind_10) * qterm
            txyz(Ind_101) = txyz(Ind_101) +  dtheta(1, 1, my_chg_idx) * tyz(Ind_01) * qterm
            txyz(Ind_011) = txyz(Ind_011) +   theta(1, 1, my_chg_idx) * tyz(Ind_11) * qterm

            qterm = q(i_tbl(i_base+2), j, k)
            txyz(Ind_000) = txyz(Ind_000) +   theta(2, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_100) = txyz(Ind_100) +  dtheta(2, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_010) = txyz(Ind_010) +   theta(2, 1, my_chg_idx) * tyz(Ind_10) * qterm
            txyz(Ind_001) = txyz(Ind_001) +   theta(2, 1, my_chg_idx) * tyz(Ind_01) * qterm
            txyz(Ind_200) = txyz(Ind_200) + d2theta(2, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_020) = txyz(Ind_020) +   theta(2, 1, my_chg_idx) * tyz(Ind_20) * qterm
            txyz(Ind_002) = txyz(Ind_002) +   theta(2, 1, my_chg_idx) * tyz(Ind_02) * qterm
            txyz(Ind_110) = txyz(Ind_110) +  dtheta(2, 1, my_chg_idx) * tyz(Ind_10) * qterm
            txyz(Ind_101) = txyz(Ind_101) +  dtheta(2, 1, my_chg_idx) * tyz(Ind_01) * qterm
            txyz(Ind_011) = txyz(Ind_011) +   theta(2, 1, my_chg_idx) * tyz(Ind_11) * qterm

            qterm = q(i_tbl(i_base+3), j, k)
            txyz(Ind_000) = txyz(Ind_000) +   theta(3, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_100) = txyz(Ind_100) +  dtheta(3, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_010) = txyz(Ind_010) +   theta(3, 1, my_chg_idx) * tyz(Ind_10) * qterm
            txyz(Ind_001) = txyz(Ind_001) +   theta(3, 1, my_chg_idx) * tyz(Ind_01) * qterm
            txyz(Ind_200) = txyz(Ind_200) + d2theta(3, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_020) = txyz(Ind_020) +   theta(3, 1, my_chg_idx) * tyz(Ind_20) * qterm
            txyz(Ind_002) = txyz(Ind_002) +   theta(3, 1, my_chg_idx) * tyz(Ind_02) * qterm
            txyz(Ind_110) = txyz(Ind_110) +  dtheta(3, 1, my_chg_idx) * tyz(Ind_10) * qterm
            txyz(Ind_101) = txyz(Ind_101) +  dtheta(3, 1, my_chg_idx) * tyz(Ind_01) * qterm
            txyz(Ind_011) = txyz(Ind_011) +   theta(3, 1, my_chg_idx) * tyz(Ind_11) * qterm

            qterm = q(i_tbl(i_base+4), j, k)
            txyz(Ind_000) = txyz(Ind_000) +   theta(4, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_100) = txyz(Ind_100) +  dtheta(4, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_010) = txyz(Ind_010) +   theta(4, 1, my_chg_idx) * tyz(Ind_10) * qterm
            txyz(Ind_001) = txyz(Ind_001) +   theta(4, 1, my_chg_idx) * tyz(Ind_01) * qterm
            txyz(Ind_200) = txyz(Ind_200) + d2theta(4, 1, my_chg_idx) * tyz(Ind_00) * qterm
            txyz(Ind_020) = txyz(Ind_020) +   theta(4, 1, my_chg_idx) * tyz(Ind_20) * qterm
            txyz(Ind_002) = txyz(Ind_002) +   theta(4, 1, my_chg_idx) * tyz(Ind_02) * qterm
            txyz(Ind_110) = txyz(Ind_110) +  dtheta(4, 1, my_chg_idx) * tyz(Ind_10) * qterm
            txyz(Ind_101) = txyz(Ind_101) +  dtheta(4, 1, my_chg_idx) * tyz(Ind_01) * qterm
            txyz(Ind_011) = txyz(Ind_011) +   theta(4, 1, my_chg_idx) * tyz(Ind_11) * qterm
#endif
            end if

          end do

        end if
      end do
    else
      do ith3 = 1, order

        k = k_tbl(k_base + ith3)

        if (k .ne. 0) then

#ifdef RANGE_DBG
          dbg_img_map(my_chgs(my_chg_idx)) = dbg_img_map(my_chgs(my_chg_idx))+1
#endif /* RANGE_DBG */

#if defined(NOT_FIELD_ONLY) && defined(_PGM_)
          txy = 0.d0
#endif
          do ith2 = 1, order

            j = j_tbl(j_base + ith2)

            if (j .ne. 0) then
#if defined(FIELD_ONLY) || defined(_NOT_PGM_)
              f1_term = theta(ith2, 2, my_chg_idx) * theta(ith3, 3, my_chg_idx)
              f2_term = dtheta(ith2, 2, my_chg_idx) * theta(ith3, 3, my_chg_idx)
              f3_term = theta(ith2, 2, my_chg_idx) * dtheta(ith3, 3, my_chg_idx)
#else
             tx(3) = 0.d0
             tx(2) = 0.d0
             tx(1) = 0.d0
#endif

              do ith1 = 1, order

                qterm = q(i_tbl(i_base+ith1), j, k)

#if defined(FIELD_ONLY) || defined(_NOT_PGM_)
! Force is negative of grad:

                f1 = f1 - qterm * dtheta(ith1, 1, my_chg_idx) * f1_term
                f2 = f2 - qterm * theta(ith1, 1, my_chg_idx) * f2_term
                f3 = f3 - qterm * theta(ith1, 1, my_chg_idx) * f3_term
#else
              tx(1) = tx(1) - qterm *   theta(ith1, 1, my_chg_idx)
              tx(2) = tx(2) - qterm *  dtheta(ith1, 1, my_chg_idx)
              tx(3) = tx(3) - qterm * d2theta(ith1, 1, my_chg_idx)
#endif
  
              end do
#if defined(NOT_FIELD_ONLY) && defined(_PGM_)
            txy(Ind_00) = txy(Ind_00) +   theta(ith2, 2, my_chg_idx) * tx(1)
            txy(Ind_10) = txy(Ind_10) +   theta(ith2, 2, my_chg_idx) * tx(2)
            txy(Ind_01) = txy(Ind_01) +  dtheta(ith2, 2, my_chg_idx) * tx(1)
            txy(Ind_20) = txy(Ind_20) +   theta(ith2, 2, my_chg_idx) * tx(3)
            txy(Ind_02) = txy(Ind_02) + d2theta(ith2, 2, my_chg_idx) * tx(1)
            txy(Ind_11) = txy(Ind_11) +  dtheta(ith2, 2, my_chg_idx) * tx(2)
#endif
            end if

          end do
#if defined(NOT_FIELD_ONLY) && defined(_PGM_)
          txyz(Ind_100) = txyz(Ind_100) +   theta(ith3, 3, my_chg_idx) * txy(Ind_10)
          txyz(Ind_010) = txyz(Ind_010) +   theta(ith3, 3, my_chg_idx) * txy(Ind_01)
          txyz(Ind_001) = txyz(Ind_001) +  dtheta(ith3, 3, my_chg_idx) * txy(Ind_00)
          txyz(Ind_000) = txyz(Ind_000) +   theta(ith3, 3, my_chg_idx) * txy(Ind_00)
          txyz(Ind_200) = txyz(Ind_200) +   theta(ith3, 3, my_chg_idx) * txy(Ind_20)
          txyz(Ind_020) = txyz(Ind_020) +   theta(ith3, 3, my_chg_idx) * txy(Ind_02)
          txyz(Ind_002) = txyz(Ind_002) + d2theta(ith3, 3, my_chg_idx) * txy(Ind_00)
          txyz(Ind_110) = txyz(Ind_110) +   theta(ith3, 3, my_chg_idx) * txy(Ind_11)
          txyz(Ind_101) = txyz(Ind_101) +  dtheta(ith3, 3, my_chg_idx) * txy(Ind_10)
          txyz(Ind_011) = txyz(Ind_011) +  dtheta(ith3, 3, my_chg_idx) * txy(Ind_01)
#endif

        end if
      end do
    end if

#if defined(FIELD_ONLY)
    f1 = -f1 * dnfft1
    f2 = -f2 * dnfft2
    f3 = -f3 * dnfft3
#else
#ifndef _PGM_
    f1 = f1 * dnfft1 * charge
    f2 = f2 * dnfft2 * charge
    f3 = f3 * dnfft3 * charge
#endif
#endif

#if defined(FIELD_ONLY) || defined(_NOT_PGM_)

    if (is_orthog .ne. 0) then
      dfx = recip_11 * f1
      dfy = recip_22 * f2
      dfz = recip_33 * f3
    else
      dfx = recip(1, 1) * f1 + recip(1, 2) * f2 + recip(1, 3) * f3
      dfy = recip(2, 1) * f1 + recip(2, 2) * f2 + recip(2, 3) * f3
      dfz = recip(3, 1) * f1 + recip(3, 2) * f2 + recip(3, 3) * f3
    end if
#endif

#ifdef FIELD_ONLY
    img_fld(1, img_i) = img_fld(1, img_i) + dfx
    img_fld(2, img_i) = img_fld(2, img_i) + dfy
    img_fld(3, img_i) = img_fld(3, img_i) + dfz
#else /* FIELD_ONLY */
#ifndef _PGM_
    img_frc(1, img_i) = img_frc(1, img_i) + dfx
    img_frc(2, img_i) = img_frc(2, img_i) + dfy
    img_frc(3, img_i) = img_frc(3, img_i) + dfz
#endif /* _PGM_ */
#endif /* FIELD_ONLY */

#if defined(_PGM_) && defined(NOT_FIELD_ONLY)

! negative signs are due to the fact that the original code was written for force

!  potentials
    F_dip_fld(1, img_i) = F_dip_fld(1, img_i) - txyz(1)

!  1st gradients
    F_dip_fld(2, img_i) = F_dip_fld(2, img_i) - sum(field_xform_3x3(1, 1:3) * txyz(2:4))
    F_dip_fld(3, img_i) = F_dip_fld(3, img_i) - sum(field_xform_3x3(2, 1:3) * txyz(2:4))
    F_dip_fld(4, img_i) = F_dip_fld(4, img_i) - sum(field_xform_3x3(3, 1:3) * txyz(2:4))

!  2nd gradients
    F_dip_fld( 5, img_i) = F_dip_fld( 5, img_i) - sum(xform(1:6, 1) * txyz(5:10))
    F_dip_fld( 6, img_i) = F_dip_fld( 6, img_i) - sum(xform(1:6, 2) * txyz(5:10))
    F_dip_fld( 7, img_i) = F_dip_fld( 7, img_i) - sum(xform(1:6, 3) * txyz(5:10))
    F_dip_fld( 8, img_i) = F_dip_fld( 8, img_i) - sum(xform(1:6, 4) * txyz(5:10))
    F_dip_fld( 9, img_i) = F_dip_fld( 9, img_i) - sum(xform(1:6, 5) * txyz(5:10))
    F_dip_fld(10, img_i) = F_dip_fld(10, img_i) - sum(xform(1:6, 6) * txyz(5:10))
#endif
  end do

#ifdef RANGE_DBG
  call mpi_reduce(dbg_img_map, dbg_img_map_res, size(dbg_img_map), &
                  mpi_integer, mpi_sum, 0, pmemd_comm, err_code_mpi)
  if (master) then
    do i = 1, natom
      if (dbg_img_map_res(i) .ne. order) then
        write(0,*)'DBG: IMAGE ', i, 'processed ', &
                  dbg_img_map_res(i), 'times in grad_sum code!!!'
      end if
    end do
  end if
#endif /* RANGE_DBG */

  return

#ifdef _PGM_
#ifdef FIELD_ONLY
end subroutine grad_sum_fld_PGM
#else
end subroutine grad_sum_PGM
#endif
#else
end subroutine grad_sum
#endif


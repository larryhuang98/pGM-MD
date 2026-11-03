#include "copyright_pgm.i"
!*******************************************************************************
!
!   Adopted from relevant non-pGM pmemd code by YD
!
! Subroutine:  fill_charge_grid
! 
! Description: <TBS>
!
! INPUT:
!
! theta1, theta2, theta3:       Spline coeff arrays.
! ifracts:                      int(scaled fractional coords).
! nfft1, nfft2, nfft3:          Logical charge grid dimensions.
!
! fft_x_dim, fft_y_dim, fft_z_dim: Physical charge grid dims.
!
! order:                    Order of spline interpolation.
!
! OUTPUT:
!
! q:                            Charge grid.
!              
!*******************************************************************************

#ifdef _PGM_
! NOTE: gbl_mpole replaces img_qterm.
! eventually, should use img_qterm and img_dip.
! img_qterm is gbl_mpole have different units.
#ifdef DIPGRID
subroutine fill_charge_grid_dip_only_PGM(q, theta, dtheta, order, img_frac_dip, ifracts, &
                            my_chgs, my_chg_cnt)
#else
#define NOT_DIPGRID
subroutine fill_charge_grid_PGM(q, theta, dtheta, order, img_chg, img_frac_dip, ifracts, &
                            my_chgs, my_chg_cnt)
#endif
#else /* _PGM_ */
#define _NOT_PGM_
subroutine fill_charge_grid(q, theta, order, img_qterm, ifracts, &
                            my_chgs, my_chg_cnt)
#endif  /* _PGM_ */

  use img_mod
  use mdin_ewald_dat_mod
  use pme_blk_fft_mod
  use parallel_dat_mod
  use prmtop_dat_mod
! BEGIN DBG
  use pmemd_lib_mod
! END DBG
  use ti_mod

  implicit none

! Formal arguments:

  integer               :: order
  double precision      :: theta(order, 3, *)
#ifdef _PGM_
  double precision      :: dtheta(order, 3, *)
  double precision      :: img_frac_dip(3,*)
#ifndef DIPGRID
  double precision      :: img_chg(*)
#endif
#else
  double precision      :: img_qterm(*)
#endif
  integer               :: ifracts(3, *)
  double precision      :: q(2 * fft_x_dim, &
                             fft_y_cnts1(my_grid_idx1), &
                             fft_z_cnts(my_grid_idx2))
  integer               :: my_chgs(*), my_chg_cnt

! Local variables:

  integer               :: i, j, k
  integer               :: i_base, j_base, k_base
  integer               :: ith1, ith2, ith3
  integer               :: my_chg_idx
  double precision      :: charge
  double precision      :: k_term, j_term
  double precision      :: dip(3)
  double precision      :: k_term1, k_term2, k_term3, j_term1
! integer               :: kbot0
  integer               :: y_off, z_off
  integer               :: y_cnt, z_cnt
  integer               :: jbot, jtop
  integer               :: kbot, ktop
  integer               :: i_tbl(-order : nfft1)
  integer               :: j_tbl(-order : nfft2)
  integer               :: k_tbl(-order : nfft3)
  ! For TI
  integer               :: iatom
#ifdef RANGE_DBG
  integer               :: dbg_img_map(natom)
  integer               :: dbg_img_map_res(natom)

  dbg_img_map(:) = 0
#endif /* RANGE_DBG */

! Old usages; kept for documentation during code conversion...
! kbot0 = my_xy_slab_start
! kbot = kbot0 + 1
! ktop = kbot0 + my_xy_slab_cnt

  y_off = fft_y_offs1(my_grid_idx1)
  y_cnt = fft_y_cnts1(my_grid_idx1)

  z_off = fft_z_offs(my_grid_idx2)
  z_cnt = fft_z_cnts(my_grid_idx2)

! Zero the Charge grids:

  call zero_q_array(q, 2 * fft_x_dim * y_cnt * z_cnt)

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

  ! We special-case order 4, the default.

  if (order .eq. 4) then
    do my_chg_idx = 1, my_chg_cnt

#ifndef _PGM_
! PGM has not implemented for TI, comment this out for now.
      if (ti_mode .eq. 1 .or. ti_mode .eq. 2) then
        !Determine atom # that we are dealing with 
        iatom = gbl_img_atm_map(my_chgs(my_chg_idx)) ! get atm number from idx
        if(ti_lst(ti_mask_piece,iatom) .ne. 0) cycle
      end if

      charge = img_qterm(my_chgs(my_chg_idx))
#else
#ifndef DIPGRID
      charge = img_chg(my_chgs(my_chg_idx))
#endif
      dip(1:3) = img_frac_dip(1:3,my_chgs(my_chg_idx))
#endif /* _PGM_ */

      i_base = ifracts(1, my_chg_idx)
      j_base = ifracts(2, my_chg_idx)
      k_base = ifracts(3, my_chg_idx)

      do ith3 = 1, 4

        k = k_tbl(k_base + ith3)

        if (k .ne. 0) then

#ifdef RANGE_DBG
          dbg_img_map(my_chgs(my_chg_idx)) = dbg_img_map(my_chgs(my_chg_idx))+1
#endif /* RANGE_DBG */
#ifndef _PGM_
          k_term =  theta(ith3, 3, my_chg_idx) * charge
#else
#ifndef DIPGRID
          k_term =  theta(ith3, 3, my_chg_idx) * charge + &
                   dtheta(ith3, 3, my_chg_idx) * dip(3)
#else
          k_term = dtheta(ith3, 3, my_chg_idx) * dip(3) 
#endif
          k_term1=  theta(ith3, 3, my_chg_idx) * dip(1)
          k_term2=  theta(ith3, 3, my_chg_idx) * dip(2)
#endif
          do ith2 = 1, 4
            j = j_tbl(j_base + ith2)
            if (j .ne. 0) then
#ifndef _PGM_
              j_term = k_term * theta(ith2, 2, my_chg_idx)
#else
              j_term = k_term * theta(ith2, 2, my_chg_idx) + &
                       k_term2*dtheta(ith2, 2, my_chg_idx)
              j_term1= k_term1* theta(ith2, 2, my_chg_idx)
#endif
#ifndef _PGM_
              q(i_tbl(i_base+1), j, k) = q(i_tbl(i_base+1), j, k) + &
                                         theta(1, 1, my_chg_idx) * j_term
              q(i_tbl(i_base+2), j, k) = q(i_tbl(i_base+2), j, k) + &
                                         theta(2, 1, my_chg_idx) * j_term
              q(i_tbl(i_base+3), j, k) = q(i_tbl(i_base+3), j, k) + &
                                         theta(3, 1, my_chg_idx) * j_term
              q(i_tbl(i_base+4), j, k) = q(i_tbl(i_base+4), j, k) + &
                                         theta(4, 1, my_chg_idx) * j_term
#else
            q(i_tbl(i_base+1), j, k) = q(i_tbl(i_base+1), j, k) + &
                                       theta(1, 1, my_chg_idx) * j_term + &
                                      dtheta(1, 1, my_chg_idx) * j_term1
            q(i_tbl(i_base+2), j, k) = q(i_tbl(i_base+2), j, k) + &
                                       theta(2, 1, my_chg_idx) * j_term + &
                                      dtheta(2, 1, my_chg_idx) * j_term1
            q(i_tbl(i_base+3), j, k) = q(i_tbl(i_base+3), j, k) + &
                                       theta(3, 1, my_chg_idx) * j_term + &
                                      dtheta(3, 1, my_chg_idx) * j_term1
            q(i_tbl(i_base+4), j, k) = q(i_tbl(i_base+4), j, k) + &
                                       theta(4, 1, my_chg_idx) * j_term + &
                                      dtheta(4, 1, my_chg_idx) * j_term1
#endif
            end if
          end do
        end if
      end do
    end do
  else
    do my_chg_idx = 1, my_chg_cnt

#ifndef _PGM_
      if (ti_mode .eq. 1 .or. ti_mode .eq. 2) then
        !Determine atom # that we are dealing with 
        iatom = gbl_img_atm_map(my_chgs(my_chg_idx)) ! get atm number from idx
        if(ti_lst(ti_mask_piece,iatom) .ne. 0) cycle
      end if

      charge = img_qterm(my_chgs(my_chg_idx))
#else
#ifndef DIPGRID
      charge = img_chg(my_chgs(my_chg_idx))
#endif
      dip(1:3) = img_frac_dip(1:3,my_chgs(my_chg_idx))
#endif /* _PGM_ */

      i_base = ifracts(1, my_chg_idx)
      j_base = ifracts(2, my_chg_idx)
      k_base = ifracts(3, my_chg_idx)

      do ith3 = 1, order

        k = k_tbl(k_base + ith3)

        if (k .ne. 0) then

#ifdef RANGE_DBG
          dbg_img_map(my_chgs(my_chg_idx)) = dbg_img_map(my_chgs(my_chg_idx))+1
#endif /* RANGE_DBG */
#ifndef _PGM_
          k_term =  theta(ith3, 3, my_chg_idx) * charge
#else
#ifndef DIPGRID
          k_term =  theta(ith3, 3, my_chg_idx) * charge + &
                   dtheta(ith3, 3, my_chg_idx) * dip(3)
#else
          k_term = dtheta(ith3, 3, my_chg_idx) * dip(3)
#endif
          k_term1=  theta(ith3, 3, my_chg_idx) * dip(1)
          k_term2=  theta(ith3, 3, my_chg_idx) * dip(2)
#endif
          do ith2 = 1, order
            j = j_tbl(j_base + ith2)
            if (j .ne. 0) then
#ifndef _PGM_
              j_term = k_term * theta(ith2, 2, my_chg_idx)
              do ith1 = 1, order
                i = i_tbl(i_base + ith1)
                q(i, j, k) = q(i, j, k) + theta(ith1, 1, my_chg_idx) * j_term
              end do
#else
              j_term = k_term * theta(ith2, 2, my_chg_idx) + &
                       k_term2*dtheta(ith2, 2, my_chg_idx)
              j_term1= k_term1* theta(ith2, 2, my_chg_idx)
              do ith1 = 1, order
                i = i_tbl(i_base + ith1)
                q(i, j, k) = q(i, j, k) + theta(ith1, 1, my_chg_idx) * j_term + &
                                         dtheta(ith1, 1, my_chg_idx) * j_term1
              end do
#endif
            end if
          end do
        end if
      end do
    end do
  end if

#ifdef RANGE_DBG
  call mpi_reduce(dbg_img_map, dbg_img_map_res, size(dbg_img_map), &
                  mpi_integer, mpi_sum, 0, pmemd_comm, err_code_mpi)
  if (master) then
    do i = 1, natom
      if (dbg_img_map_res(i) .ne. order) then
        write(0,*)'DBG: IMAGE ', i, 'processed ', &
                  dbg_img_map_res(i), 'times in fill_charge_grid code!!!'
      end if
    end do
  end if
#endif /* RANGE_DBG */

  return

#ifdef _PGM_
#ifdef DIPGRID
end subroutine fill_charge_grid_dip_only_PGM
#else
end subroutine fill_charge_grid_PGM
#undef NOT_DIPGRIP
#endif
#else
#undef _NOT_PRM_
end subroutine fill_charge_grid
#endif

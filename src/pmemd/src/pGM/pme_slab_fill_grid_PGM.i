#include "copyright_pgm.i"
#include "copyright.i"
!*******************************************************************************
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
#ifdef MPI
subroutine fill_charge_grid_dip_only_PGM(q, theta, dtheta, order, img_frac_dip, ifracts, &
                            my_chgs, my_chg_cnt)
#else
subroutine fill_charge_grid_dip_only_PGM(q, theta, dtheta, order, img_frac_dip, ifracts)
#endif
#else

#define NOT_DIPGRID

#ifdef MPI
subroutine fill_charge_grid_PGM(q, theta, dtheta, order, img_chg, img_frac_dip, ifracts, &
                            my_chgs, my_chg_cnt)
#else
subroutine fill_charge_grid_PGM(q, theta, dtheta, order, img_chg, img_frac_dip, ifracts)
#endif
#endif

#else /* _PGM_ */

#define _NOT_PGM_

#ifdef MPI
subroutine fill_charge_grid(q, theta, order, img_qterm, ifracts, &
                            my_chgs, my_chg_cnt)
#else
subroutine fill_charge_grid(q, theta, order, img_qterm, ifracts)
#endif
#endif

  use img_mod
  use mdin_ewald_dat_mod
  use pme_slab_fft_mod
  use prmtop_dat_mod
  use ti_mod
#if defined(MPI) && defined(_PGM_)
  use parallel_dat_mod, only : mytaskid
#endif

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
  double precision      :: q(2 * fft_x_dim, fft_y_dim, my_xy_slab_cnt)

#ifdef MPI
  integer               :: my_chgs(*), my_chg_cnt  ! These are reused in PGM, defined in the module
#endif

! Local variables:

  integer               :: i, j, k
  integer               :: i_base, j_base, k_base
  integer               :: ith1, ith2, ith3
  integer               :: my_chg_idx
  double precision      :: charge
  double precision      :: dip(3)
  double precision      :: k_term, j_term
  double precision      :: k_term1, k_term2, k_term3, j_term1
#ifdef MPI
! integer               :: kbot0
  integer               :: kbot, ktop
#endif
  integer               :: i_tbl(-order : nfft1)
  integer               :: j_tbl(-order : nfft2)
  integer               :: k_tbl(-order : nfft3)
  ! For TI:
  integer               :: iatom
#ifdef MPI
#ifdef RANGE_DBG
  integer               :: dbg_img_map(natom)
  integer               :: dbg_img_map_res(natom)

  dbg_img_map(:) = 0
#endif /* RANGE_DBG */
#endif

dip = 0.d0

#ifdef MPI
! Old usages; kept for documentation during code conversion...
! kbot0 = my_xy_slab_start
! kbot = kbot0 + 1
! ktop = kbot0 + my_xy_slab_cnt
#endif

! Zero the Charge grids:

#ifdef MPI
  call zero_q_array(q, 2 * fft_x_dim * fft_y_dim * my_xy_slab_cnt)
#else
  call zero_q_array(q, 2 * fft_x_dim * fft_y_dim * fft_z_dim)
#endif

  ! Initialize the indexing tables.  It actually produces faster code to
  ! do this here rather than caching the info remotely.

  do i = 0, nfft1
    i_tbl(i) = i + 1
  end do
  do i = -order, -1
    i_tbl(i) = i + nfft1 + 1
  end do

  do j = 0, nfft2
    j_tbl(j) = j + 1
  end do
  do j = -order, -1
    j_tbl(j) = j + nfft2 + 1
  end do

#ifdef MPI
  kbot = 1
  ktop = my_xy_slab_cnt
  do k = 0, nfft3
    k_tbl(k) = k - my_xy_slab_start + 1
    if (k_tbl(k) .lt. kbot .or. k_tbl(k) .gt. ktop) k_tbl(k) = 0
  end do
  do k = -order, -1
    k_tbl(k) = k + nfft3 - my_xy_slab_start + 1
    if (k_tbl(k) .lt. kbot .or. k_tbl(k) .gt. ktop) k_tbl(k) = 0
  end do
#else
  do k = 0, nfft3
    k_tbl(k) = k + 1
  end do
  do k = -order, -1
    k_tbl(k) = k + nfft3 + 1
  end do
#endif

  ! We special-case order 4, the default.

  if (order .eq. 4) then
#ifdef MPI
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
#else
! PGM has not implemented for TI, comment this out for now.
    do my_chg_idx = 1, natom
#ifndef _PGM_
      if (ti_mode .eq. 1 .or. ti_mode .eq. 2) then
        !Determine atom # that we are dealing with 
        iatom = gbl_img_atm_map(my_chg_idx) ! get atm number from idx
        if(ti_lst(ti_mask_piece,iatom) .ne. 0) cycle
      end if
      charge = img_qterm(my_chg_idx)
#else
#ifndef DIPGRID
      charge = img_chg(my_chg_idx)
#endif
      dip(1:3) = img_frac_dip(1:3,my_chg_idx)
#endif /* _PGM_ */
#endif

      i_base = ifracts(1, my_chg_idx)
      j_base = ifracts(2, my_chg_idx)
      k_base = ifracts(3, my_chg_idx)

      do ith3 = 1, 4

        k = k_tbl(k_base + ith3)

#ifdef MPI
        if (k .ne. 0) then

#ifdef RANGE_DBG
          dbg_img_map(my_chgs(my_chg_idx)) = dbg_img_map(my_chgs(my_chg_idx))+1
#endif /* RANGE_DBG */
#endif
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
          end do
#ifdef MPI
        end if
#endif
      end do
    end do
  else if (order .eq. 6) then
#ifdef MPI
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
#else
    do my_chg_idx = 1, natom
#ifndef _PGM_
      if (ti_mode .eq. 1 .or. ti_mode .eq. 2) then
        !Determine atom # that we are dealing with 
        iatom = gbl_img_atm_map(my_chg_idx) ! get atm number from idx
        if(ti_lst(ti_mask_piece,iatom) .ne. 0) cycle
      end if
      charge = img_qterm(my_chg_idx)
#else
#ifndef DIPGRID
      charge = img_chg(my_chg_idx)
#endif
      dip(1:3) = img_frac_dip(1:3,my_chg_idx)
#endif /* _PGM_ */
#endif

      i_base = ifracts(1, my_chg_idx)
      j_base = ifracts(2, my_chg_idx)
      k_base = ifracts(3, my_chg_idx)

      do ith3 = 1, 6

        k = k_tbl(k_base + ith3)

#ifdef MPI
        if (k .ne. 0) then

#ifdef RANGE_DBG
          dbg_img_map(my_chgs(my_chg_idx)) = dbg_img_map(my_chgs(my_chg_idx))+1
#endif /* RANGE_DBG */
#endif /* MPI */
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
          do ith2 = 1, 6
            j = j_tbl(j_base + ith2)
#ifndef _PGM_
            j_term = k_term * theta(ith2, 2, my_chg_idx)
            q(i_tbl(i_base+1), j, k) = q(i_tbl(i_base+1), j, k) + &
                                       theta(1, 1, my_chg_idx) * j_term
            q(i_tbl(i_base+2), j, k) = q(i_tbl(i_base+2), j, k) + &
                                       theta(2, 1, my_chg_idx) * j_term
            q(i_tbl(i_base+3), j, k) = q(i_tbl(i_base+3), j, k) + &
                                       theta(3, 1, my_chg_idx) * j_term
            q(i_tbl(i_base+4), j, k) = q(i_tbl(i_base+4), j, k) + &
                                       theta(4, 1, my_chg_idx) * j_term
            q(i_tbl(i_base+5), j, k) = q(i_tbl(i_base+5), j, k) + &
                                       theta(5, 1, my_chg_idx) * j_term
            q(i_tbl(i_base+6), j, k) = q(i_tbl(i_base+6), j, k) + &
                                       theta(6, 1, my_chg_idx) * j_term
#else
            j_term = k_term * theta(ith2, 2, my_chg_idx) + &
                     k_term2*dtheta(ith2, 2, my_chg_idx)
            j_term1= k_term1* theta(ith2, 2, my_chg_idx)
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

            q(i_tbl(i_base+5), j, k) = q(i_tbl(i_base+5), j, k) + &
                                       theta(5, 1, my_chg_idx) * j_term + &
                                      dtheta(5, 1, my_chg_idx) * j_term1

            q(i_tbl(i_base+6), j, k) = q(i_tbl(i_base+6), j, k) + &
                                       theta(6, 1, my_chg_idx) * j_term + &
                                      dtheta(6, 1, my_chg_idx) * j_term1

!if ((2 * fft_x_dim * (fft_y_dim * (k-1) + j -1) + i_base) .gt. 32 .and. & 
!    (2 * fft_x_dim * (fft_y_dim * (k-1) + j -1) + i_base+6)  .le. 32) then
!end if

#endif
          end do
#ifdef MPI
        end if
#endif
      end do
    end do
  else
#ifdef MPI
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
#else
    do my_chg_idx = 1, natom
#ifndef _PGM_
      if (ti_mode .eq. 1 .or. ti_mode .eq. 2) then
        !Determine atom # that we are dealing with 
        iatom = gbl_img_atm_map(my_chg_idx) ! get atm number from idx
        if(ti_lst(ti_mask_piece,iatom) .ne. 0) cycle
      end if
      charge = img_qterm(my_chg_idx)
#else
#ifndef DIPGRID
      charge = img_chg(my_chg_idx)
#endif
      dip(1:3) = img_frac_dip(1:3,my_chg_idx)
#endif /* _PGM_ */
#endif

      i_base = ifracts(1, my_chg_idx)
      j_base = ifracts(2, my_chg_idx)
      k_base = ifracts(3, my_chg_idx)

      do ith3 = 1, order

        k = k_tbl(k_base + ith3)

#ifdef MPI
        if (k .ne. 0) then

#ifdef RANGE_DBG
          dbg_img_map(my_chgs(my_chg_idx)) = dbg_img_map(my_chgs(my_chg_idx))+1
#endif /* RANGE_DBG */
#endif /* MPI */

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
          end do
#ifdef MPI
        end if
#endif
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
#endif
#else
end subroutine fill_charge_grid
#endif

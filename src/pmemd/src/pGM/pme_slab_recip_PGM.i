#include "copyright_pgm.i"

!*******************************************************************************
!
! Adopted from non-pGM pmemd code by YD
!
! Module: pme_slab_recip_mod
!
! Description: <TBS>
!
! NOTE NOTE NOTE: This code assumes frc_int .eq. 0 and should only be used
!                 under these conditions!!!
!*******************************************************************************

module pme_slab_recip_PGM_mod

  use pme_recip_dat_mod

!precision setting for SPDP, DPDP, SPSP
#include "include_precision.i"
  implicit none

  private       ! Everything here is private unless specified public

  double precision, allocatable, save :: theta(:,:,:)
  double precision, allocatable, save :: dtheta(:,:,:)
  double precision, allocatable, save :: d2theta(:,:,:)
  double precision, allocatable, save :: d3theta(:,:,:)  ! d3theta is not used, needed for Q-poles
  double precision,              save :: mpole_xform_3x3(3,3)
  double precision,              save :: field_xform_3x3(3,3)
  double precision, allocatable, save :: G(:)  ! stores the weighting function to speed up dipole iteration
  integer,          allocatable, save :: ifracts(:, :)
#ifdef MPI
  integer,                       save :: my_chg_cnt
  integer,          allocatable, save :: my_chgs(:)

  public        :: pme_slab_recip_setup_PGM, &
                   do_slab_pmesh_kspace_PGM, &
                   claim_slab_recip_imgs_PGM, &
                   claim_slab_recip_imgs_nonorthog_PGM
#else
  public        :: pme_slab_recip_setup_PGM, &
                   do_slab_pmesh_kspace_PGM
#endif /* MPI */
contains

!*******************************************************************************
!
! Subroutine:  pme_slab_recip_setup
!
! Description:
!              
!*******************************************************************************

subroutine pme_slab_recip_setup_PGM(num_ints, num_reals)

  use gbl_constants_mod
  use loadbal_mod
  use pmemd_lib_mod
  use pme_slab_fft_mod
  use mdin_ctrl_dat_mod
  use mdin_ewald_dat_mod
  use parallel_dat_mod
  use prmtop_dat_mod
  use pbc_mod

  implicit none

! Formal arguments:

  ! num_ints and num_reals are used to return allocation counts. Don't zero.

  integer, intent(in out)       :: num_ints, num_reals

! original pme_slab_recip_setup code, done in pme_slab_recip_mod
! This is the non-PGM pme_slab_recip_setup part
#ifdef MPI
  if (nrespa .eq. 1 .and. imin .eq. 0 .and. numtasks * bspl_order * 2 .gt. nfft3) then
    fft_redist_enabled = .true.
    fft_redist_needed = .true.
  end if

  call slab_fft_setup(recip_numtasks, fft_redist_enabled, &
                      num_ints, num_reals)

#else
  call slab_fft_setup(num_ints, num_reals)
#endif

! This is the extra part for pGM setup
  if (ipgm .ne. 0 .or. ipol .eq. 5) then  ! NOTE ipgm and ipol are defined in mdin_ctrl_dat_mod
    if (.not. allocated(theta)) allocate(theta(bspl_order, 3, natom))
    if (.not. allocated(dtheta)) allocate(dtheta(bspl_order, 3, natom))
    if (.not. allocated(d2theta)) allocate(d2theta(bspl_order, 3, natom))
! uncomment this if quadrupoles are included
!    if (.not. allocated(d3theta)) allocate(d3theta(bspl_order, 3, natom))
#ifdef MPI
    if (.not. allocated(G)) allocate(G(max(xy_slab_dbl_cnt * max_xy_slab_cnt, &
                                           zx_slab_dbl_cnt * max_zx_slab_cnt, 1)))
    num_reals = num_reals + bspl_order * 3 * natom * 3 + &
                          max(xy_slab_dbl_cnt * max_xy_slab_cnt, zx_slab_dbl_cnt * max_zx_slab_cnt)
#else
    if (.not. allocated(G)) allocate(G(fft_x_dim * fft_y_dim * fft_z_dim))
    num_reals = num_reals + bspl_order * 3 * natom * 3 + &
                          fft_x_dim * fft_y_dim * fft_z_dim
#endif
    if (.not. allocated(ifracts)) allocate(ifracts(3, natom))
    num_ints = num_ints + natom*3
#ifdef MPI
    if (.not. allocated(my_chgs)) allocate(my_chgs(natom))
    num_ints = num_ints + natom

#endif
  end if

  return

end subroutine pme_slab_recip_setup_PGM

!*******************************************************************************
!
! Subroutine:  do_slab_pmesh_kspace and do_slab_pmesh_kspace_PGM
!
! Description:  <TBS>
!
! INPUT:
!
! OUTPUT:
!
! img_frc:      Forces incremented by k-space sum.
! virial:       Virial due to k-space sum (valid for atomic scaling;
!               rigid molecule virial needs a correction term not computed here.
!
! Additional parameters for do_slab_pmesh_kspace_PGM
!
! img_vec:      output array, either field or field gradient tensor
! img_chg:      imaged charges
! img_dip:      imaged dipoles
! need_frc:     .false. field only; .true. calculate field gradient tensor
! dip_select:    = 1   img_chg/img_dip are perm_chg_img and perm_dip_img
!                      need to calculate bspline coefficients
!                = 2   img_chg/img_dip are perm_chg_img and ind_dip_img
!                      use the saved bspline coefficients without update
!
!*******************************************************************************

subroutine do_slab_pmesh_kspace_PGM(img_vec, need_pot_enes, need_virials, need_frc, dip_select, img_dip, &
                                    img_chg, virial, vir_self, img_disp, eer, ee_chg, ee_dip, img_perm_dip)

  use img_mod
  use inpcrd_dat_mod
  use mdin_ctrl_dat_mod
  use mdin_ewald_dat_mod
  use parallel_dat_mod
  use pbc_mod
  use pme_slab_fft_mod
  use prmtop_dat_mod
  use timers_mod

  implicit none

! Formal arguments:
  logical               :: need_pot_enes
  logical               :: need_virials
  double precision      :: img_vec(*)
  logical               :: need_frc
  integer               :: dip_select
  double precision      :: img_dip(3, *)
  double precision, optional :: virial(3, 3)
  double precision, optional :: vir_self(3, 3)
  double precision, optional :: img_chg(*)
  double precision, optional :: eer
  double precision, optional :: ee_chg
  double precision, optional :: ee_dip
  double precision, optional :: img_disp(3,*)
  double precision, optional :: img_perm_dip(3,*)

! Local variables:

! NOTE - For FFTW, *_q need to be 16 byte aligned if they are moved off
!        the stack (and the stack must be aligned if they are on the stack).

  integer  :: i, j, l, m, ier
  double precision, allocatable      :: img_frac_dip(:,:)
  double precision, allocatable      :: dip(:,:)
  double precision, allocatable      :: xyz_q(:)
  double precision, allocatable      :: zxy_q(:)

#ifdef MPI
  allocate(xyz_q(max(xy_slab_dbl_cnt * max_xy_slab_cnt, &
                     zx_slab_dbl_cnt * max_zx_slab_cnt, 1)))
  allocate(zxy_q(max(xy_slab_dbl_cnt * max_xy_slab_cnt, &
                     zx_slab_dbl_cnt * max_zx_slab_cnt, 1)))
#else
  allocate(xyz_q(2 * fft_x_dim * fft_y_dim * fft_z_dim))
  allocate(zxy_q(2 * fft_x_dim * fft_y_dim * fft_z_dim))
#endif
  allocate(img_frac_dip(3,natom))
  allocate(dip(3,natom))

  ! Exponential tables are only practical to use for orthogonal unit cells, and
  ! only need updating when constant pressure runs are being done.
 
  if (ntp .gt. 0 .and. is_orthog .ne. 0) call load_m_exp_tbls

if (ipgm .ge. 1 .or. ipol .eq. 5) then
! theta, dtheta, d2theta, d3theta are calculated only once per step 
    if (dip_select .eq. 1) then     
#ifdef MPI
  if (is_orthog .ne. 0) then
! use optional variables to control different levels
    call get_grid_weights_PGM(natom, gbl_img_crd, ifracts, theta, dtheta, &
                          bspl_order, gbl_used_recip_img_cnt, &
                          gbl_used_recip_img_lst, my_chgs, my_chg_cnt, d2theta, d3theta, ipgm)
  else
    call get_grid_weights_nonorthog_PGM(natom, atm_crd, gbl_img_atm_map, &
                                    ifracts, theta, dtheta, bspl_order, &
                                    gbl_used_recip_img_cnt, &
                                    gbl_used_recip_img_lst, &
                                    my_chgs, my_chg_cnt, d2theta, d3theta, ipgm)
  end if
#else  /* MPI */
  if (is_orthog .ne. 0) then
    call get_grid_weights_PGM(natom, gbl_img_crd, ifracts, theta, dtheta, &
                          bspl_order, d2theta, d3theta, ipgm)
  else
    call get_grid_weights_nonorthog_PGM(natom, atm_crd, gbl_img_atm_map, ifracts, &
                                    theta, dtheta, bspl_order, d2theta, d3theta, ipgm)
  end if
#endif /* MPI */
    end if   ! dip_select
else

#ifdef MPI

  if (is_orthog .ne. 0) then
    call get_grid_weights_PGM(natom, gbl_img_crd, ifracts, theta, dtheta, &
                          bspl_order, gbl_used_recip_img_cnt, &
                          gbl_used_recip_img_lst, my_chgs, my_chg_cnt)
  else
    call get_grid_weights_nonorthog_PGM(natom, atm_crd, gbl_img_atm_map, &
                                    ifracts, theta, dtheta, bspl_order, &
                                    gbl_used_recip_img_cnt, &
                                    gbl_used_recip_img_lst, &
                                    my_chgs, my_chg_cnt)
  end if
#else
  if (is_orthog .ne. 0) then
    call get_grid_weights_PGM(natom, gbl_img_crd, ifracts, theta, dtheta, &
                          bspl_order, d2theta, d3theta, ipgm)
  else
    call get_grid_weights_nonorthog_PGM(natom, atm_crd, gbl_img_atm_map, ifracts, &
                                    theta, dtheta, bspl_order, d2theta, d3theta, ipgm)
  end if
#endif
end if

  call update_pme_time(bspline_timer)

if (ipgm .ge. 1 .or. ipol .eq. 5) then

    if (dip_select .eq. 1) then     

#ifdef MPI
      img_frac_dip(1:3, 1:natom) = 0.d0  ! always a good idea to clean the array
#endif

! Dipoles need to be transformed to lattice fractional coordinates

  call pGM_RECIP_xform_matrices(nfft1, nfft2, nfft3, recip, mpole_xform_3x3, field_xform_3x3)

    end if    ! dip_select


#ifdef MPI

  call pGM_RECIP_dip_to_frac_dip(my_chg_cnt, my_chgs, img_dip, img_frac_dip, mpole_xform_3x3, is_orthog)
#else
  call pGM_RECIP_dip_to_frac_dip(natom, img_dip, img_frac_dip, mpole_xform_3x3, is_orthog)
#endif
! Fill Charge Grid.  Charges are approximated on an even grid.

    if (dip_select .eq. 1) then

#ifdef MPI
  call fill_charge_grid_PGM(xyz_q, theta, dtheta, bspl_order, img_chg, img_frac_dip, ifracts, &
                        my_chgs, my_chg_cnt)
#else /*MPI*/
  call fill_charge_grid_PGM(xyz_q, theta, dtheta, bspl_order, img_chg, img_frac_dip, ifracts)
#endif /*MPI*/

    else if (dip_select .eq. 2) then

#ifdef MPI
  call fill_charge_grid_dip_only_PGM(xyz_q, theta, dtheta, bspl_order, img_frac_dip, ifracts, &
                        my_chgs, my_chg_cnt)
#else /*MPI*/
  call fill_charge_grid_dip_only_PGM(xyz_q, theta, dtheta, bspl_order, img_frac_dip, ifracts)
#endif /*MPI*/

    else if (dip_select .ge. 3) then
! Here, we choose to recalculate the charge + dipole, which is perm + induced dipoles
! alternatively, we can reuse the xyz_q of the perm_charge and perm_dipoles. In which case,
! we would only need to calculate the induced dipoles. The computational cost should be similar since
! xyz_q array is usually much larger than the charge array, often by an order of magnitude.
#ifdef MPI
  call fill_charge_grid_PGM(xyz_q, theta, dtheta, bspl_order, img_chg, img_frac_dip, ifracts, &
                        my_chgs, my_chg_cnt)
#else /*MPI*/
  call fill_charge_grid_PGM(xyz_q, theta, dtheta, bspl_order, img_chg, img_frac_dip, ifracts)
#endif /*MPI*/
    end if
else   ! not pGM

! Fill Charge Grid.  Charges are approximated on an even grid.

#ifdef MPI
  call fill_charge_grid(xyz_q, theta, bspl_order, gbl_img_qterm, ifracts, &
                        my_chgs, my_chg_cnt)
#else /*MPI*/
  call fill_charge_grid(xyz_q, theta, bspl_order, gbl_img_qterm, ifracts)
#endif /*MPI*/
end if  ! ipgm .lt. 1 .and. ipol .ne. 5

  call update_pme_time(grid_charges_timer)

! either xyz_q or zxyz_q needs to be saved for final virial and force calculation

  call slab_fft3drc_forward(xyz_q, zxy_q, fft_x_dim, fft_y_dim, fft_z_dim)

  call update_pme_time(fft_timer)

  if (is_orthog .ne. 0) then
if (ipgm .le. 0 .and. ipol .ne. 5) then
! pGM virial is calculated in grad_sum()
    if (need_virials) then
      call scalar_sumrc_uv(zxy_q, ew_coeff, uc_volume, &
                           gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                           nfft1, nfft2, nfft3, &
                           fft_x_dim, fft_y_dim, fft_z_dim, eer, virial)
    else if (need_pot_enes) then
!      call scalar_sumrc_u(zxy_q, ew_coeff, uc_volume, &
!                          gbl_prefac1, gbl_prefac2, gbl_prefac3, &
!                          nfft1, nfft2, nfft3, &
!                          fft_x_dim, fft_y_dim, fft_z_dim, eer)
! Ross Walker - 7/8/2014 - bug bug - The PME Error estimate requires the vir
! since it compares virial vs energy. 
! pme_err_est =   vir_vs_ene = vir%elec_recip(1, 1) + &
!               vir%elec_recip(2, 2) + &
!               vir%elec_recip(3, 3) + &
!               vir%eedvir + &
!               vir%elec_nb_adjust(1, 1) + &
!               vir%elec_nb_adjust(2, 2) + &
!               vir%elec_nb_adjust(3, 3)
! So we can't just naively skip it here unless we choose not to print the 
! Ewald error estimate anymore (like in the GPU code). For now we will calculate
! the reciprocal virial whenever need_pot_enes is true.
      call scalar_sumrc_uv(zxy_q, ew_coeff, uc_volume, &
                           gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                           nfft1, nfft2, nfft3, &
                           fft_x_dim, fft_y_dim, fft_z_dim, eer, virial)
    else
      call scalar_sumrc(zxy_q, ew_coeff, uc_volume, &
                        gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                        nfft1, nfft2, nfft3, &
                        fft_x_dim, fft_y_dim, fft_z_dim)
    end if
else
    if (dip_select .eq. 1) then
      call scalar_sumrc_save_eterm(zxy_q, ew_coeff, uc_volume, &
                        gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                        nfft1, nfft2, nfft3, &
                        fft_x_dim, fft_y_dim, fft_z_dim, G)
    else if (dip_select .ge. 2) then

      if (need_virials) then
        call scalar_sumrc_use_saved_eterm_uv(zxy_q, ew_coeff, uc_volume, &
                        gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                        nfft1, nfft2, nfft3, &
                        fft_x_dim, fft_y_dim, fft_z_dim, G, eer, virial)
      else if (need_pot_enes) then
        call scalar_sumrc_use_saved_eterm_u(zxy_q, ew_coeff, uc_volume, &
                        gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                        nfft1, nfft2, nfft3, &
                        fft_x_dim, fft_y_dim, fft_z_dim, G, eer)
      else
        call scalar_sumrc_use_saved_eterm(zxy_q, ew_coeff, uc_volume, &
                        gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                        nfft1, nfft2, nfft3, &
                        fft_x_dim, fft_y_dim, fft_z_dim, G)
      end if
    end if
endif
  else
if (ipgm .le. 0 .and. ipol .ne. 5) then
! pGM virial is calculated in grad_sum()
    if (need_virials) then
      call scalar_sumrc_nonorthog_uv(zxy_q, ew_coeff, uc_volume, &
                                     gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                                     nfft1, nfft2, nfft3, &
                                     fft_x_dim, fft_y_dim, fft_z_dim, &
                                     eer, virial)
    else if (need_pot_enes) then
!      call scalar_sumrc_nonorthog_u(zxy_q, ew_coeff, uc_volume, &
!                                    gbl_prefac1, gbl_prefac2, gbl_prefac3, &
!                                    nfft1, nfft2, nfft3, &
!                                    fft_x_dim, fft_y_dim, fft_z_dim, &
!                                    eer)
! Ross Walker - 7/8/2014 - bug bug - The PME Error estimate requires the vir
! since it compares virial vs energy. 
! pme_err_est =   vir_vs_ene = vir%elec_recip(1, 1) + &
!               vir%elec_recip(2, 2) + &
!               vir%elec_recip(3, 3) + &
!               vir%eedvir + &
!               vir%elec_nb_adjust(1, 1) + &
!               vir%elec_nb_adjust(2, 2) + &
!               vir%elec_nb_adjust(3, 3)
! So we can't just naively skip it here unless we choose not to print the 
! Ewald error estimate anymore (like in the GPU code). For now we will calculate
! the reciprocal virial whenever need_pot_enes is true.
      call scalar_sumrc_nonorthog_uv(zxy_q, ew_coeff, uc_volume, &
                                     gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                                     nfft1, nfft2, nfft3, &
                                     fft_x_dim, fft_y_dim, fft_z_dim, &
                                     eer, virial)
    else
      call scalar_sumrc_nonorthog(zxy_q, ew_coeff, uc_volume, &
                                  gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  fft_x_dim, fft_y_dim, fft_z_dim)
    end if
else
    if (dip_select .eq. 1) then
      call scalar_sumrc_nonorthog_save_eterm(zxy_q, ew_coeff, uc_volume, &
                                  gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  fft_x_dim, fft_y_dim, fft_z_dim, G)
    else if (dip_select .ge. 2) then
      if (need_virials) then
        call scalar_sumrc_nonorthog_use_saved_eterm_uv(zxy_q, ew_coeff, uc_volume, &
                                  gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  fft_x_dim, fft_y_dim, fft_z_dim, G, eer, virial)
      else if (need_pot_enes) then
        call scalar_sumrc_nonorthog_use_saved_eterm_u(zxy_q, ew_coeff, uc_volume, &
                                  gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  fft_x_dim, fft_y_dim, fft_z_dim, G, eer)
      else
        call scalar_sumrc_nonorthog_use_saved_eterm(zxy_q, ew_coeff, uc_volume, &
                                  gbl_prefac1, gbl_prefac2, gbl_prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  fft_x_dim, fft_y_dim, fft_z_dim, G)
      end if
    end if
end if
  end if
  call update_pme_time(scalar_sum_timer)

  call slab_fft3drc_back(zxy_q, xyz_q, fft_x_dim, fft_y_dim, fft_z_dim)

  call update_pme_time(fft_timer)

  if (ipgm .ge. 1 .or. ipol .eq. 5) then
! transformation from fractional to cartesian coordinates are performed in grad_sum_PGM()
! output field vectors and gradient tensors are in img_vec.
! It might make more sense to move reciprocal space virial to here.
    if (.not. need_frc) then
#ifdef MPI
      call grad_sum_fld_PGM(xyz_q, theta, dtheta, d2theta, bspl_order, img_chg, img_frac_dip, &
                img_vec, ifracts, my_chgs, my_chg_cnt, &
                fft_x_dim, fft_y_dim, fft_z_dim)
#else /*MPI*/
      call grad_sum_fld_PGM(xyz_q, theta, dtheta, d2theta, bspl_order, img_chg, img_frac_dip, &
                img_vec, ifracts, fft_x_dim, fft_y_dim, fft_z_dim)
#endif /* MPI */
    else
#ifdef MPI
      call grad_sum_PGM(xyz_q, theta, dtheta, d2theta, bspl_order, img_chg, img_frac_dip, &
                img_vec, ifracts, my_chgs, my_chg_cnt, &
                fft_x_dim, fft_y_dim, fft_z_dim)
#else /* MPI */
      call grad_sum_PGM(xyz_q, theta, dtheta, d2theta, bspl_order, img_chg, img_frac_dip, &
                img_vec, ifracts, fft_x_dim, fft_y_dim, fft_z_dim)
#endif /* MPI */
      if (need_virials .or. need_pot_enes) then
#ifdef MPI
        call pGM_recip_ene_virial(my_chg_cnt, img_vec, img_chg, img_dip, need_pot_enes, need_virials, &
                                       ee_chg, ee_dip, img_perm_dip, img_disp, vir_self)
#else
        call pGM_recip_ene_virial(natom, img_vec, img_chg, img_dip, need_pot_enes, need_virials, &
                                       ee_chg, ee_dip, img_perm_dip, img_disp, vir_self)
#endif
      end if
    end if
  else  ! ipgm .and. ipol

#ifdef MPI

    call grad_sum(xyz_q, theta, dtheta, bspl_order, gbl_img_qterm, &
                img_vec, ifracts, my_chgs, my_chg_cnt, &
                fft_x_dim, fft_y_dim, fft_z_dim)

#else /*MPI*/
    call grad_sum(xyz_q, theta, dtheta, bspl_order, gbl_img_qterm, &
                img_vec, ifracts, fft_x_dim, fft_y_dim, fft_z_dim)
#endif /* MPI */

  end if  ! ipgm .and. ipol

  call update_pme_time(grad_sum_timer)

  return

end subroutine do_slab_pmesh_kspace_PGM

#define POT_ENES 1
#define VIRIAL 1
#include "scalar_sum_slab.i"
#undef VIRIAL
#define VIRIAL 0
#include "scalar_sum_slab.i"
#undef POT_ENES
#define POT_ENES 0
#include "scalar_sum_slab.i"
#undef POT_ENES
#undef VIRIAL

#ifdef DIPGRID
#undef DIPGRID
#endif /* DIPGRID */

#ifdef FIELD_ONLY
#undef FIELD_ONLY
#endif /* FIELD_ONLY */

#ifdef _PGM_
#undef _PGM_
#endif /* _PGM_ */

#include "pme_slab_fill_grid_PGM.i"
#include "pme_slab_grad_sum_PGM.i"
!#define _PGM_ 1
#define _PGM_
#define DIPGRID
#include "pme_slab_fill_grid_PGM.i"
#undef DIPGRID
#include "pme_slab_fill_grid_PGM.i"
#define FIELD_ONLY
#include "pme_slab_grad_sum_PGM.i"
#undef FIELD_ONLY
#include "pme_slab_grad_sum_PGM.i"
#undef _PGM_

#undef SAVE_ETERM
#define SAVE_ETERM 1
#include "scalar_sum_slab_PGM.i"
#undef SAVE_ETERM
#define SAVE_ETERM 0
#ifdef USE_SAVED_ETERM
#undef USE_SAVED_ETERM
#endif
#define USE_SAVED_ETERM 1
#define POT_ENES 1
#define VIRIAL 1
#include "scalar_sum_slab_PGM.i"
#undef VIRIAL
#define VIRIAL 0
#include "scalar_sum_slab_PGM.i"
#undef POT_ENES
#define POT_ENES 0
#include "scalar_sum_slab_PGM.i"
#undef POT_ENES
#undef VIRIAL
#undef USE_SAVED_ETERM
#define USE_SAVED_ETERM 0

!*******************************************************************************
!
! Subroutine:  get_grid_weights
!
! Description: <TBS>
!              
!*******************************************************************************

#ifdef MPI
subroutine get_grid_weights_PGM(img_cnt, img_crd, ifracts, theta, dtheta, order, &
                            used_recip_img_cnt, used_recip_img_lst, &
                            my_chgs, my_chg_cnt, d2theta, d3theta, ipgm)
#else
subroutine get_grid_weights_PGM(img_cnt, img_crd, ifracts, theta, dtheta, order, d2theta, d3theta, ipgm)
#endif

  use bspline_mod
  use pme_slab_fft_mod
  use gbl_datatypes_mod
  use img_mod
  use mdin_ewald_dat_mod
  use parallel_dat_mod
  use pbc_mod

  implicit none

! Formal arguments:

  integer               :: img_cnt
  double precision      :: img_crd(3, *)
  integer               :: ifracts(3, *)
  integer               :: order
  double precision      :: theta(order, 3, *)
  double precision      :: dtheta(order, 3, *)
  integer, optional     :: ipgm
  double precision, optional :: d2theta(order, 3, *)
  double precision, optional :: d3theta(order, 3, *)
#ifdef MPI
  integer               :: used_recip_img_cnt
  integer               :: used_recip_img_lst(*)
  integer               :: my_chgs(*), my_chg_cnt
#endif

! Local variables:

  double precision      :: box_x, box_y, box_z
  double precision      :: crd_x, crd_y, crd_z
  double precision      :: factor1, factor2, factor3
  double precision      :: fract(3)
  double precision      :: weight(3)
  integer               :: i, k

#ifdef MPI
  integer               :: kbot0, ktop1
  integer               :: lst_idx
  logical               :: kwraps
#endif /* MPI */

#ifdef MPI
  my_chg_cnt = 0

  kbot0 = xy_slab_start(mytaskid)
  ktop1 = kbot0 + my_xy_slab_cnt + order - 2
  kwraps = (ktop1 .ge. nfft3)

#endif /* MPI */

  box_x = ucell(1, 1)
  box_y = ucell(2, 2)
  box_z = ucell(3, 3)

  ! Scaling factors to get from cit table indexes to nfft indexes:

  factor1 = dble(nfft1) * recip(1, 1)
  factor2 = dble(nfft2) * recip(2, 2)
  factor3 = dble(nfft3) * recip(3, 3)

#ifdef MPI
  do lst_idx = 1, used_recip_img_cnt
     i = used_recip_img_lst(lst_idx)
#else /* MPI */
  do i = 1, img_cnt
#endif
    crd_x = img_crd(1, i)
    crd_y = img_crd(2, i)
    crd_z = img_crd(3, i)

    if (crd_x .ge. box_x) then
      crd_x = crd_x - box_x
    else if (crd_x .lt. 0.d0) then
      crd_x = crd_x + box_x
    end if

    if (crd_y .ge. box_y) then
      crd_y = crd_y - box_y
    else if (crd_y .lt. 0.d0) then
      crd_y = crd_y + box_y
    end if

    if (crd_z .ge. box_z) then
      crd_z = crd_z - box_z
    else if (crd_z .lt. 0.d0) then
      crd_z = crd_z + box_z
    end if

    fract(1) = factor1 * crd_x
    fract(2) = factor2 * crd_y
    fract(3) = factor3 * crd_z
#ifdef MPI
    k = int(fract(3))
#endif

#ifdef MPI

    if (kwraps) then
      if (k .lt. kbot0 .and. k .gt. ktop1 - nfft3) cycle
    else
      if (k .lt. kbot0 .or. k .gt. ktop1) cycle
    end if

#endif /* MPI */

#ifdef MPI
    my_chg_cnt = my_chg_cnt + 1
    my_chgs(my_chg_cnt) = i
#endif /* MPI */

    weight(:) = fract(:) - int(fract(:))

#ifdef MPI
    ifracts(:, my_chg_cnt) = int(fract(:)) - order

    if (.not.present(ipgm)) then
      ! for charge-only fill
      call fill_bspline_1_3d(weight, order, &
                           theta(1, 1, my_chg_cnt), dtheta(1, 1, my_chg_cnt))
    else
      if (ipgm .eq. 1) then
        call fill_bspline_2_3d(weight, order, &
                           theta(1, 1, my_chg_cnt), dtheta(1, 1, my_chg_cnt), d2theta(1, 1, my_chg_cnt))
!      else if (ipgm .eq. 2) then
! use fill_bspline_3_3d if Q-poles are included, untested code
!        call fill_bspline_3_3d(weight, order, &
!                           theta(1, 1, my_chg_cnt), dtheta(1, 1, my_chg_cnt), d2theta(1,1,my_chg_cnt), &
!                           d3theta(1,1,my_chg_cnt))
      else
        write(6,*)' grid weights error: use fill_bspline_3_3d if Q-poles are included, untested code'
        stop
      end if
    end if
#else
    ifracts(:, i) = int(fract(:)) - order

    if (.not.present(ipgm)) then
      ! for charge-only fill
      call fill_bspline_1_3d(weight, order, theta(1, 1, i), dtheta(1, 1, i))
    else
      if (ipgm .eq. 1) then
        call fill_bspline_2_3d(weight, order, theta(1, 1, i), dtheta(1, 1, i), d2theta(1,1,i))
!      else if (ipgm .eq. 2) then
! use fill_bspline_3_3d if Q-poles are included, untested code
!        call fill_bspline_3_3d(weight, order, theta(1, 1, i), dtheta(1, 1, i), d2theta(1,1,i), d3theta(1,1,i))
      else
        write(6,*)' grid weights error: use fill_bspline_3_3d if Q-poles are included, untested code'
        stop
      end if
    end if
#endif

  end do

  return

end subroutine get_grid_weights_PGM

#ifdef MPI
!*******************************************************************************
!
! Subroutine:  claim_slab_recip_imgs
!
! Description: <TBS>
!              
!*******************************************************************************
subroutine claim_slab_recip_imgs_PGM(img_cnt, fraction, box, crd_idx_tbl,  &
#ifdef pmepack
                                 crd_q_pack, img_iac, mapped_img_cnt, &
#else
                                 img_crd, img_qterm, img_iac, mapped_img_cnt, &
#endif
                                 mapped_img_lst, img_atm_map, used_img_map, &
                                 used_img_cnt, used_img_lst)

  use cit_mod
  use pme_slab_fft_mod
  use gbl_datatypes_mod
  use img_mod
  use mdin_ewald_dat_mod
  use parallel_dat_mod
  use prmtop_dat_mod

!precision setting for SPDP, DPDP, SPSP
#include "include_precision.i"
  implicit none

! Formal arguments:

  integer               :: img_cnt
  double precision      :: fraction(3, img_cnt)
  double precision      :: box(3)
  type(cit_tbl_rec)     :: crd_idx_tbl(0 : cit_tbl_x_dim - 1, &
                                       0 : cit_tbl_y_dim - 1, &
                                       0 : cit_tbl_z_dim - 1)
  double precision      :: img_crd(3, img_cnt)
  double precision      :: img_qterm(img_cnt)
  integer               :: img_iac(img_cnt)
  integer               :: mapped_img_cnt
  integer               :: mapped_img_lst(img_cnt)
  integer               :: img_atm_map(img_cnt)
  integer(byte)         :: used_img_map(img_cnt)
  integer               :: used_img_cnt
  integer               :: used_img_lst(*)

! Local variables:

  integer               :: k_bot0, k_top1
  integer               :: i, j, k
  integer               :: atm_id, img_id
  integer               :: z_bkt_lo, z_bkt_hi
  logical               :: z_wraps
  double precision      :: x_box, y_box, z_box
  double precision      :: z_lo, z_hi, z_cur

  gbl_used_recip_img_cnt = 0

  if (.not. i_do_recip) return    ! Map null range; not actually ever used...

  x_box = box(1)
  y_box = box(2)
  z_box = box(3)

  ! If there really is only one task doing reciprocal force calcs, we have to
  ! catch it here, because the code further below assumes that one process does
  ! not need to map the entire range of images.  Besides, this is more
  ! efficient for the special case...

  if (my_xy_slab_cnt .eq. fft_z_dim) then
    do img_id = 1, img_cnt
      if (used_img_map(img_id) .eq. 0) then
        used_img_map(img_id) = 1
        used_img_cnt = used_img_cnt + 1
        used_img_lst(used_img_cnt) = img_id
        if (img_iac(img_id) .eq. 0) then
          atm_id = img_atm_map(img_id)
          img_crd(1, img_id) = fraction(1, atm_id) * x_box
          img_crd(2, img_id) = fraction(2, atm_id) * y_box
          img_crd(3, img_id) = fraction(3, atm_id) * z_box
          img_qterm(img_id) = atm_qterm(atm_id)
          img_iac(img_id) = atm_iac(atm_id)
          mapped_img_cnt = mapped_img_cnt + 1
          mapped_img_lst(mapped_img_cnt) = img_id
        end if
      end if
      gbl_used_recip_img_cnt = gbl_used_recip_img_cnt + 1
      gbl_used_recip_img_lst(gbl_used_recip_img_cnt) = img_id
    end do
    return
  end if

  k_bot0 = xy_slab_start(mytaskid)
  k_top1 = k_bot0 + my_xy_slab_cnt + bspl_order - 2

  ! We have to allow for motion of skinnb, plus we have to allow for fact that
  ! atom z crds are rounded down to the grid point.  Hence, the + 1 for z_hi

  z_lo = dble(k_bot0) * z_box / dble(nfft3) - (0.5d0 * skinnb)
  z_hi = dble(k_top1 + 1) * z_box / dble(nfft3) + (0.5d0 * skinnb)

!  if (z_lo .lt. 0.d0) then
!    z_wraps = .true.
!    z_lo = z_lo + z_box
!  else if (z_hi .ge. z_box) then
!    z_wraps = .true.
!    z_hi = z_hi - z_box
!  else
!    z_wraps = .false.
!  end if

!  z_bkt_lo = int(z_lo * dble(cit_tbl_z_dim) / z_box)
!  z_bkt_hi = int(z_hi * dble(cit_tbl_z_dim) / z_box)

!  if (z_bkt_lo .eq. z_bkt_hi) z_wraps = .false.

  call wrap_hi_lo(z_lo, z_hi, cit_tbl_z_dim, z_bkt_lo, z_bkt_hi, z_wraps, z_box)

  if (z_wraps) then

    do k = z_bkt_lo, cit_tbl_z_dim - 1
      do j = 0, cit_tbl_y_dim - 1
        do i = 0, cit_tbl_x_dim - 1
          do img_id = crd_idx_tbl(i,j,k)%img_lo, crd_idx_tbl(i,j,k)%img_hi
            atm_id = img_atm_map(img_id)
            z_cur = fraction(3, atm_id) * z_box
            if (z_cur .ge. z_lo) then
              if (used_img_map(img_id) .eq. 0) then
                used_img_map(img_id) = 1
                used_img_cnt = used_img_cnt + 1
                used_img_lst(used_img_cnt) = img_id
                if (img_iac(img_id) .eq. 0) then
                  img_crd(1, img_id) = fraction(1, atm_id) * x_box
                  img_crd(2, img_id) = fraction(2, atm_id) * y_box
                  img_crd(3, img_id) = z_cur
                  img_qterm(img_id) = atm_qterm(atm_id)
                  img_iac(img_id) = atm_iac(atm_id)
                  mapped_img_cnt = mapped_img_cnt + 1
                  mapped_img_lst(mapped_img_cnt) = img_id
                end if
              end if
              gbl_used_recip_img_cnt = gbl_used_recip_img_cnt + 1
              gbl_used_recip_img_lst(gbl_used_recip_img_cnt) = img_id
            end if
          end do
        end do
      end do
    end do

    do k = 0, z_bkt_hi
      do j = 0, cit_tbl_y_dim - 1
        do i = 0, cit_tbl_x_dim - 1
          do img_id = crd_idx_tbl(i,j,k)%img_lo, crd_idx_tbl(i,j,k)%img_hi
            atm_id = img_atm_map(img_id)
            z_cur = fraction(3, atm_id) * z_box
            if (z_cur .lt. z_hi) then
              if (used_img_map(img_id) .eq. 0) then
                used_img_map(img_id) = 1
                used_img_cnt = used_img_cnt + 1
                used_img_lst(used_img_cnt) = img_id
                if (img_iac(img_id) .eq. 0) then
                  img_crd(1, img_id) = fraction(1, atm_id) * x_box
                  img_crd(2, img_id) = fraction(2, atm_id) * y_box
                  img_crd(3, img_id) = z_cur
                  img_qterm(img_id) = atm_qterm(atm_id)
                  img_iac(img_id) = atm_iac(atm_id)
                  mapped_img_cnt = mapped_img_cnt + 1
                  mapped_img_lst(mapped_img_cnt) = img_id
                end if
              end if
              gbl_used_recip_img_cnt = gbl_used_recip_img_cnt + 1
              gbl_used_recip_img_lst(gbl_used_recip_img_cnt) = img_id
            end if
          end do
        end do
      end do
    end do

  else

    do k = z_bkt_lo, z_bkt_hi
      do j = 0, cit_tbl_y_dim - 1
        do i = 0, cit_tbl_x_dim - 1
          do img_id = crd_idx_tbl(i,j,k)%img_lo, crd_idx_tbl(i,j,k)%img_hi
            atm_id = img_atm_map(img_id)
            z_cur = fraction(3, atm_id) * z_box
            if (z_cur .ge. z_lo .and. z_cur .lt. z_hi) then
              if (used_img_map(img_id) .eq. 0) then
                used_img_map(img_id) = 1
                used_img_cnt = used_img_cnt + 1
                used_img_lst(used_img_cnt) = img_id
                if (img_iac(img_id) .eq. 0) then
                  img_crd(1, img_id) = fraction(1, atm_id) * x_box
                  img_crd(2, img_id) = fraction(2, atm_id) * y_box
                  img_crd(3, img_id) = z_cur
                  img_qterm(img_id) = atm_qterm(atm_id)
                  img_iac(img_id) = atm_iac(atm_id)
                  mapped_img_cnt = mapped_img_cnt + 1
                  mapped_img_lst(mapped_img_cnt) = img_id
                end if
              end if
              gbl_used_recip_img_cnt = gbl_used_recip_img_cnt + 1
              gbl_used_recip_img_lst(gbl_used_recip_img_cnt) = img_id
            end if
          end do
        end do
      end do
    end do

  end if

  return

end subroutine claim_slab_recip_imgs_PGM
#endif /* MPI */

#ifdef MPI
!*******************************************************************************
!
! Subroutine:  claim_slab_recip_imgs_nonorthog
!
! Description: <TBS>
!              
!*******************************************************************************

subroutine claim_slab_recip_imgs_nonorthog_PGM(img_cnt, fraction, crd_idx_tbl, &
                                           img_crd, img_qterm, img_iac, &
                                           mapped_img_cnt, mapped_img_lst, &
                                           img_atm_map, used_img_map, &
                                           used_img_cnt, used_img_lst)

  use cit_mod
  use pme_slab_fft_mod
  use gbl_datatypes_mod
  use img_mod
  use mdin_ewald_dat_mod
  use parallel_dat_mod
  use prmtop_dat_mod
  use pbc_mod

!precision setting for SPDP, DPDP, SPSP
#include "include_precision.i"

  implicit none

! Formal arguments:

  integer               :: img_cnt
  double precision      :: fraction(3, img_cnt)
  type(cit_tbl_rec)     :: crd_idx_tbl(0 : cit_tbl_x_dim - 1, &
                                       0 : cit_tbl_y_dim - 1, &
                                       0 : cit_tbl_z_dim - 1)
  double precision      :: img_crd(3, img_cnt)
  double precision      :: img_qterm(img_cnt)
  integer               :: img_iac(img_cnt)
  integer               :: mapped_img_cnt
  integer               :: mapped_img_lst(img_cnt)
  integer               :: img_atm_map(img_cnt)
  integer(byte)         :: used_img_map(img_cnt)
  integer               :: used_img_cnt
  integer               :: used_img_lst(*)

! Local variables:

  integer               :: k_bot0, k_top1
  integer               :: i, j, k
  integer               :: atm_id, img_id
  integer               :: z_bkt_lo, z_bkt_hi
  logical               :: z_wraps
  double precision      :: ucell_stk(3, 3)
  double precision      :: f1, f2, f3
  double precision      :: z_lo, z_hi

  ucell_stk(:,:) = ucell(:,:)

  gbl_used_recip_img_cnt = 0

  if (.not. i_do_recip) return    ! Map null range; not actually ever used...

  ! If there really is only one task doing reciprocal force calcs, we have to
  ! catch it here, because the code further below assumes that one process does
  ! not need to map the entire range of images.  Besides, this is more
  ! efficient for the special case...

  if (my_xy_slab_cnt .eq. fft_z_dim) then
    do img_id = 1, img_cnt
      if (used_img_map(img_id) .eq. 0) then
        used_img_map(img_id) = 1
        used_img_cnt = used_img_cnt + 1
        used_img_lst(used_img_cnt) = img_id
        if (img_iac(img_id) .eq. 0) then
          atm_id = img_atm_map(img_id)
          f1 = fraction(1, atm_id)
          f2 = fraction(2, atm_id)
          f3 = fraction(3, atm_id)
          ! ucell(2,1), ucell(3,1), and ucell(3,2) are always 0.d0, so
          ! we can simplify the expression in this critical inner loop
          img_crd(1, img_id) = f1 * ucell_stk(1, 1) + &
                               f2 * ucell_stk(1, 2) + &
                               f3 * ucell_stk(1, 3)
          img_crd(2, img_id) = f2 * ucell_stk(2, 2) + &
                               f3 * ucell_stk(2, 3)
          img_crd(3, img_id) = f3 * ucell_stk(3, 3)
          img_qterm(img_id) = atm_qterm(atm_id)
          img_iac(img_id) = atm_iac(atm_id)
          mapped_img_cnt = mapped_img_cnt + 1
          mapped_img_lst(mapped_img_cnt) = img_id
        end if
      end if
      gbl_used_recip_img_cnt = gbl_used_recip_img_cnt + 1
      gbl_used_recip_img_lst(gbl_used_recip_img_cnt) = img_id
    end do
    return
  end if

  k_bot0 = xy_slab_start(mytaskid)
  k_top1 = k_bot0 + my_xy_slab_cnt + bspl_order - 2

  ! We have to allow for motion of skinnb, plus we have to allow for fact that
  ! atom z crds are rounded down to the grid point.  Hence, the + 1 for z_hi

  z_lo = dble(k_bot0) / dble(nfft3) - &
         (0.5d0 * cut_factor(3) * skinnb) / ucell_stk(3,3)      ! a fractional

  z_hi = dble(k_top1 + 1) / dble(nfft3) + &
         (0.5d0 * cut_factor(3) * skinnb) / ucell_stk(3,3)      ! a fractional

  call wrap_hi_lo(z_lo, z_hi, cit_tbl_z_dim, z_bkt_lo, z_bkt_hi, z_wraps)

!  if (z_lo .lt. 0.d0) then
!    z_wraps = .true.
!    z_lo = z_lo + 1.d0
!  else if (z_hi .ge. 1.d0) then
!    z_wraps = .true.
!    z_hi = z_hi - 1.d0
!  else
!    z_wraps = .false.
!  end if

!  z_bkt_lo = int(z_lo * dble(cit_tbl_z_dim))
!  z_bkt_hi = int(z_hi * dble(cit_tbl_z_dim))

!  if (z_bkt_lo .eq. z_bkt_hi) z_wraps = .false.

  if (z_wraps) then

    do k = z_bkt_lo, cit_tbl_z_dim - 1
      do j = 0, cit_tbl_y_dim - 1
        do i = 0, cit_tbl_x_dim - 1
          do img_id = crd_idx_tbl(i,j,k)%img_lo, crd_idx_tbl(i,j,k)%img_hi
            atm_id = img_atm_map(img_id)
            f3 = fraction(3, atm_id)
            if (f3 .ge. z_lo) then
              if (used_img_map(img_id) .eq. 0) then
                used_img_map(img_id) = 1
                used_img_cnt = used_img_cnt + 1
                used_img_lst(used_img_cnt) = img_id
                if (img_iac(img_id) .eq. 0) then
                  f1 = fraction(1, atm_id)
                  f2 = fraction(2, atm_id)
                  ! ucell(2,1), ucell(3,1), and ucell(3,2) are always 0.d0, so
                  ! we can simplify the expression in this critical inner loop
                  img_crd(1, img_id) = f1 * ucell_stk(1, 1) + &
                                       f2 * ucell_stk(1, 2) + &
                                       f3 * ucell_stk(1, 3)
                  img_crd(2, img_id) = f2 * ucell_stk(2, 2) + &
                                       f3 * ucell_stk(2, 3)
                  img_crd(3, img_id) = f3 * ucell_stk(3, 3)
                  img_qterm(img_id) = atm_qterm(atm_id)
                  img_iac(img_id) = atm_iac(atm_id)
                  mapped_img_cnt = mapped_img_cnt + 1
                  mapped_img_lst(mapped_img_cnt) = img_id
                end if
              end if
              gbl_used_recip_img_cnt = gbl_used_recip_img_cnt + 1
              gbl_used_recip_img_lst(gbl_used_recip_img_cnt) = img_id
            end if
          end do
        end do
      end do
    end do

    do k = 0, z_bkt_hi
      do j = 0, cit_tbl_y_dim - 1
        do i = 0, cit_tbl_x_dim - 1
          do img_id = crd_idx_tbl(i,j,k)%img_lo, crd_idx_tbl(i,j,k)%img_hi
            atm_id = img_atm_map(img_id)
            f3 = fraction(3, atm_id)
            if (f3 .lt. z_hi) then
              if (used_img_map(img_id) .eq. 0) then
                used_img_map(img_id) = 1
                used_img_cnt = used_img_cnt + 1
                used_img_lst(used_img_cnt) = img_id
                if (img_iac(img_id) .eq. 0) then
                  f1 = fraction(1, atm_id)
                  f2 = fraction(2, atm_id)
                  ! ucell(2,1), ucell(3,1), and ucell(3,2) are always 0.d0, so
                  ! we can simplify the expression in this critical inner loop
                  img_crd(1, img_id) = f1 * ucell_stk(1, 1) + &
                                       f2 * ucell_stk(1, 2) + &
                                       f3 * ucell_stk(1, 3)
                  img_crd(2, img_id) = f2 * ucell_stk(2, 2) + &
                                       f3 * ucell_stk(2, 3)
                  img_crd(3, img_id) = f3 * ucell_stk(3, 3)
                  img_qterm(img_id) = atm_qterm(atm_id)
                  img_iac(img_id) = atm_iac(atm_id)
                  mapped_img_cnt = mapped_img_cnt + 1
                  mapped_img_lst(mapped_img_cnt) = img_id
                end if
              end if
              gbl_used_recip_img_cnt = gbl_used_recip_img_cnt + 1
              gbl_used_recip_img_lst(gbl_used_recip_img_cnt) = img_id
            end if
          end do
        end do
      end do
    end do

  else

    do k = z_bkt_lo, z_bkt_hi
      do j = 0, cit_tbl_y_dim - 1
        do i = 0, cit_tbl_x_dim - 1
          do img_id = crd_idx_tbl(i,j,k)%img_lo, crd_idx_tbl(i,j,k)%img_hi
            atm_id = img_atm_map(img_id)
            f3 = fraction(3, atm_id)
            if (f3 .ge. z_lo .and. f3 .lt. z_hi) then
              if (used_img_map(img_id) .eq. 0) then
                used_img_map(img_id) = 1
                used_img_cnt = used_img_cnt + 1
                used_img_lst(used_img_cnt) = img_id
                if (img_iac(img_id) .eq. 0) then
                  f1 = fraction(1, atm_id)
                  f2 = fraction(2, atm_id)
                  ! ucell(2,1), ucell(3,1), and ucell(3,2) are always 0.d0, so
                  ! we can simplify the expression in this critical inner loop
                  img_crd(1, img_id) = f1 * ucell_stk(1, 1) + &
                                       f2 * ucell_stk(1, 2) + &
                                       f3 * ucell_stk(1, 3)
                  img_crd(2, img_id) = f2 * ucell_stk(2, 2) + &
                                       f3 * ucell_stk(2, 3)
                  img_crd(3, img_id) = f3 * ucell_stk(3, 3)
                  img_qterm(img_id) = atm_qterm(atm_id)
                  img_iac(img_id) = atm_iac(atm_id)
                  mapped_img_cnt = mapped_img_cnt + 1
                  mapped_img_lst(mapped_img_cnt) = img_id
                end if
              end if
              gbl_used_recip_img_cnt = gbl_used_recip_img_cnt + 1
              gbl_used_recip_img_lst(gbl_used_recip_img_cnt) = img_id
            end if
          end do
        end do
      end do
    end do

  end if

  return

end subroutine claim_slab_recip_imgs_nonorthog_PGM

subroutine wrap_hi_lo(lo, hi, cit_tbl_dim, bkt_lo, bkt_hi, wraps, box)

  implicit none

  double precision, intent(inout) :: lo, hi
  integer, intent(out) :: bkt_lo, bkt_hi
  integer, intent(in)  :: cit_tbl_dim
  logical, intent(out) :: wraps
  double precision, intent(in), optional :: box

  logical              :: wrap_lo, wrap_hi

  wrap_lo = .false.
  wrap_hi = .false.

  if (present(box)) then
    lo = lo / box
    hi = hi / box
  end if

  bkt_lo = int(lo * dble(cit_tbl_dim))
  bkt_hi = int(hi * dble(cit_tbl_dim))

  if (lo .lt. 0.d0) then
    wrap_lo = .true.
    lo = lo + 1.d0
    bkt_lo = int(lo * dble(cit_tbl_dim))
    if (bkt_lo .eq. bkt_hi) then
      bkt_lo = bkt_lo + 1
      if (bkt_lo .ge. cit_tbl_dim) then
        bkt_lo = 0
        lo = lo - 1.d0
        wrap_lo = .false.
      end if
    end if
  end if

  if (hi .ge. 1.d0) then
    wrap_hi = .true.
    hi = hi - 1.d0
    bkt_hi = int(hi * dble(cit_tbl_dim))
    if (bkt_lo .eq. bkt_hi) then
      bkt_hi = bkt_hi - 1
      if (bkt_hi .lt. 0) then
        bkt_hi = cit_tbl_dim - 1
        hi = hi + 1.d0
        wrap_hi = .false.
      end if
    end if
  end if

  wraps = wrap_lo .or. wrap_hi
  if (present(box)) then
    lo = lo * box
    hi = hi * box
  end if

end subroutine wrap_hi_lo

#endif /* MPI */

!*******************************************************************************
!
! Subroutine:  get_grid_weights_nonorthog
!
! Description: <TBS>
!              
!*******************************************************************************

#ifdef MPI
subroutine get_grid_weights_nonorthog_PGM(img_cnt, crd, img_atm_map, &
                                      ifracts, theta, dtheta, order, &
                                      used_recip_img_cnt, used_recip_img_lst, &
                                      my_chgs, my_chg_cnt, d2theta, d3theta, ipgm)
#else
subroutine get_grid_weights_nonorthog_PGM(img_cnt, crd, img_atm_map, ifracts, &
                                      theta, dtheta, order, d2theta, d3theta, ipgm)
#endif

  use bspline_mod
  use gbl_datatypes_mod
  use pme_slab_fft_mod
  use mdin_ewald_dat_mod
  use parallel_dat_mod
  use pbc_mod

  implicit none

! Formal arguments:

  integer               :: img_cnt
  double precision      :: crd(3, *)
  integer               :: img_atm_map(*)
  integer               :: ifracts(3, *)
  integer               :: order
  double precision      :: theta(order, 3, *)
  double precision      :: dtheta(order, 3, *)
  integer, optional     :: ipgm
  double precision, optional :: d2theta(order, 3, *)
  double precision, optional :: d3theta(order, 3, *)
#ifdef MPI
  integer               :: used_recip_img_cnt
  integer               :: used_recip_img_lst(*)
  integer               :: my_chgs(*), my_chg_cnt
#endif

! Local variables:

  double precision      :: crd_x, crd_y, crd_z
  double precision      :: fract(3)
  double precision      :: weight(3)
  double precision      :: recip_stk(3, 3)
  integer               :: atm_id
  integer               :: i, k

#ifdef MPI
  integer               :: kbot0, ktop1
  logical               :: kwraps
  integer               :: lst_idx
#endif /* MPI */

#ifdef MPI
  my_chg_cnt = 0

  kbot0 = xy_slab_start(mytaskid)
  ktop1 = kbot0 + my_xy_slab_cnt + order - 2
  kwraps = (ktop1 .ge. nfft3)
#endif /* MPI */

  recip_stk(:,:) = recip(:,:)


#ifdef MPI
  do lst_idx = 1, used_recip_img_cnt
    i = used_recip_img_lst(lst_idx)
#else
  do i = 1, img_cnt
#endif

! Unfortunately we need fractional coords that are not available without going
! back to atm_crd().  This is the case because the coordinates in crd
! were based on fractional coords at list build time, but as movement occurs
! you can't reconstruct the fractional.

    atm_id = img_atm_map(i)
    crd_x = crd(1, atm_id)
    crd_y = crd(2, atm_id)
    crd_z = crd(3, atm_id)

    fract(1) = crd_x * recip_stk(1,1) + &
               crd_y * recip_stk(2,1) +  &
               crd_z * recip_stk(3,1)

    fract(1) = dble(nfft1) * (fract(1) - dnint(fract(1)) + 0.5d0)

    fract(2) = crd_x * recip_stk(1,2) + &
               crd_y * recip_stk(2,2) + &
               crd_z * recip_stk(3,2)

    fract(2) = dble(nfft2) * (fract(2) - dnint(fract(2)) + 0.5d0)

    fract(3) = crd_x * recip_stk(1,3) + &
               crd_y * recip_stk(2,3) + &
               crd_z * recip_stk(3,3)

    fract(3) = dble(nfft3) * (fract(3) - dnint(fract(3)) + 0.5d0)

#ifdef MPI
    k = int(fract(3))

    if (kwraps) then
      if (k .lt. kbot0 .and. k .gt. ktop1 - nfft3) cycle
    else
      if (k .lt. kbot0 .or. k .gt. ktop1) cycle
    end if

    my_chg_cnt = my_chg_cnt + 1
    my_chgs(my_chg_cnt) = i
#endif

    weight(:) = fract(:) - int(fract(:))

#ifdef MPI
    ifracts(:, my_chg_cnt) = int(fract(:)) - order
    if (.not.present(ipgm)) then
      call fill_bspline_1_3d(weight, order, &
                           theta(1, 1, my_chg_cnt), dtheta(1, 1, my_chg_cnt))
    else
      if (ipgm .eq. 1) then   ! fill charge and dipole
        call fill_bspline_2_3d(weight, order, &
                           theta(1, 1, my_chg_cnt), dtheta(1, 1, my_chg_cnt), d2theta(1, 1, my_chg_cnt))
!      else if (ipgm .eq. 2) then
! use fill_bspline_3_3d if Q-poles are included, untested code
!        call fill_bspline_3_3d(weight, order, &
!                           theta(1, 1, my_chg_cnt), dtheta(1, 1, my_chg_cnt), d2theta(1, 1, my_chg_cnt), &
!                           d3theta(1, 1, my_chg_cnt))
      else
        write(6,*)' grid weights error: use fill_bspline_3_3d if Q-poles are included, untested code'
        stop
      end if
    end if
#else
    ifracts(:, i) = int(fract(:)) - order

    if (.not.present(ipgm)) then
      call fill_bspline_1_3d(weight, order, theta(1, 1, i), dtheta(1, 1, i))
    else
      if (ipgm .eq. 1) then   ! fill charge and dipole
        call fill_bspline_2_3d(weight, order, theta(1, 1, i), dtheta(1, 1, i), d2theta(1,1,i))
!      else if (ipgm .eq. 2) then
! use fill_bspline_3_3d if Q-poles are included, untested code
!        call fill_bspline_3_3d(weight, order, theta(1, 1, i), dtheta(1, 1, i), d2theta(1,1,i), d3theta(1,1,i))
      else
        write(6,*)' grid weights error: use fill_bspline_3_3d if Q-poles are included, untested code'
        stop
      end if
    end if
#endif

  end do

  return

end subroutine get_grid_weights_nonorthog_PGM

!-----------------------------------------------------------------------
subroutine pGM_RECIP_xform_matrices(nfft1, nfft2, nfft3, recip, mpole_xform, field_xform)

  integer,intent(in) :: nfft1,nfft2,nfft3

  double precision,intent(in) :: recip(3,3)
  double precision,intent(out) :: mpole_xform(3,3),field_xform(3,3)

  double precision du1_dx,du1_dy,du1_dz,du2_dx,du2_dy,du2_dz,du3_dx,du3_dy,du3_dz

  du1_dx = nfft1*recip(1,1)
  du1_dy = nfft1*recip(2,1)
  du1_dz = nfft1*recip(3,1)
  du2_dx = nfft2*recip(1,2)
  du2_dy = nfft2*recip(2,2)
  du2_dz = nfft2*recip(3,2)
  du3_dx = nfft3*recip(1,3)
  du3_dy = nfft3*recip(2,3)
  du3_dz = nfft3*recip(3,3)

  field_xform(1,1) = du1_dx
  field_xform(1,2) = du2_dx
  field_xform(1,3) = du3_dx
  field_xform(2,1) = du1_dy
  field_xform(2,2) = du2_dy
  field_xform(2,3) = du3_dy
  field_xform(3,1) = du1_dz
  field_xform(3,2) = du2_dz
  field_xform(3,3) = du3_dz

  mpole_xform(1,1) = du1_dx
  mpole_xform(1,2) = du1_dy
  mpole_xform(1,3) = du1_dz
  mpole_xform(2,1) = du2_dx
  mpole_xform(2,2) = du2_dy
  mpole_xform(2,3) = du2_dz
  mpole_xform(3,1) = du3_dx
  mpole_xform(3,2) = du3_dy
  mpole_xform(3,3) = du3_dz
end subroutine pGM_RECIP_xform_matrices
!--------------------------------------------------------------
! Use this if incoming multipole data is in single array
subroutine pGM_RECIP_mpole_to_frac_dip(natom, mpole, f_dip, mpole_xform, is_orthog)

  implicit none

  integer,intent(in) :: natom
  integer,intent(in) :: is_orthog
  double precision,intent(in) :: mpole(10,*)
  double precision,intent(out) :: f_dip(3,*)

  integer :: n
  double precision :: mpole_xform(3,3)

  if (is_orthog .eq. 0) then
  do n = 1, natom
! Assuming mpol_xform is a triangular matrix
     f_dip(1,n) = mpole_xform(1,1)*mpole(2,n)+mpole_xform(1,2)*mpole(3,n)+mpole_xform(1,3)*mpole(4,n)
     f_dip(2,n) =                             mpole_xform(2,2)*mpole(3,n)+mpole_xform(2,3)*mpole(4,n)
     f_dip(3,n) =                                                         mpole_xform(3,3)*mpole(4,n)
  enddo
  else
  do n = 1, natom
     f_dip(1,n) = mpole_xform(1,1)*mpole(2,n)
     f_dip(2,n) = mpole_xform(2,2)*mpole(3,n)
     f_dip(3,n) = mpole_xform(3,3)*mpole(4,n)
  enddo
  endif
end subroutine pGM_RECIP_mpole_to_frac_dip

!--------------------------------------------------------------
#ifdef MPI
subroutine pGM_RECIP_dip_to_frac_dip(my_chg_cnt, my_chgs, dip, f_dip, mpole_xform, is_orthog)
use parallel_dat_mod
use prmtop_dat_mod, only : natom
#else
subroutine pGM_RECIP_dip_to_frac_dip(natom, dip, f_dip, mpole_xform, is_orthog)
#endif

  implicit none

#ifdef MPI
  integer, intent(in) :: my_chg_cnt
  integer, intent(in) :: my_chgs(*)
  integer             :: idx
#else
  integer, intent(in) :: natom
#endif
  integer, intent(in) :: is_orthog
  double precision, intent(in) :: dip(3,*)
  double precision, intent(out) :: f_dip(3,*)

  integer :: n
  double precision :: mpole_xform(3,3)

  if (is_orthog .eq. 0) then
#ifdef MPI
  do idx = 1, my_chg_cnt
     n = my_chgs(idx)
#else
  do n = 1, natom
#endif
! Assuming mpol_xform is a triangular matrix
     f_dip(1,n) = mpole_xform(1,1)*dip(1,n)+mpole_xform(1,2)*dip(2,n)+mpole_xform(1,3)*dip(3,n)
     f_dip(2,n) =                           mpole_xform(2,2)*dip(2,n)+mpole_xform(2,3)*dip(3,n)
     f_dip(3,n) =                                                     mpole_xform(3,3)*dip(3,n)
  enddo
  else
#ifdef MPI
  do idx = 1, my_chg_cnt
     n = my_chgs(idx)
#else
  do n = 1, natom
#endif
     f_dip(1,n) = mpole_xform(1,1)*dip(1,n)
     f_dip(2,n) = mpole_xform(2,2)*dip(2,n)
     f_dip(3,n) = mpole_xform(3,3)*dip(3,n)
  enddo
  endif
end subroutine pGM_RECIP_dip_to_frac_dip

subroutine pGM_recip_ene_virial(atm_cnt, phi_rec, img_chg, img_dip, need_ene, need_vir, &
                                eer_chg, eer_dip, img_perm_dip, img_disp, virial)

implicit none

integer, intent(in)           :: atm_cnt
double precision, intent(in)  :: phi_rec(10,*)
double precision, intent(in)  :: img_chg(*)
double precision, intent(in)  :: img_dip(3,*)
logical, intent(in)           :: need_ene
logical, intent(in)           :: need_vir

double precision, optional, intent(in)    :: img_perm_dip(3,*)
double precision, optional, intent(inout) :: eer_chg
double precision, optional, intent(inout) :: eer_dip
double precision, optional, intent(in)    :: img_disp(3,*)
double precision, optional, intent(inout) :: virial(3,3)

double precision                :: eer_dip_stk
double precision                :: eer_chg_stk
double precision                :: vir_stk(3,3)
double precision                :: tt

#ifdef MPI
!  integer               :: my_chgs(*), my_chg_cnt   ! defined in the module 
integer               :: my_chg_idx
#endif
integer               :: i

eer_dip_stk = 0.d0
eer_chg_stk = 0.d0
vir_stk = 0.d0

if (need_ene .and. need_vir) then
#ifdef MPI
  do my_chg_idx = 1, atm_cnt
    i = my_chgs(my_chg_idx)
#else
  do i = 1, atm_cnt
#endif
    eer_chg_stk = eer_chg_stk + phi_rec(1, i) * img_chg(i)
    eer_dip_stk = eer_dip_stk + sum(phi_rec(2:4, i) * img_perm_dip(1:3, i))
                      

    tt = phi_rec(5,i)*img_dip(1,i) + phi_rec(8,i)*img_dip(2,i) + phi_rec(9,i)*img_dip(3,i)

    vir_stk(1,1) = vir_stk(1,1) + phi_rec(2,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(1,2) = vir_stk(1,2) + phi_rec(2,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(1,3) = vir_stk(1,3) + phi_rec(2,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)

    tt = phi_rec(8,i)*img_dip(1,i) + phi_rec(6,i)*img_dip(2,i) + phi_rec(10,i)*img_dip(3,i)

    vir_stk(2,1) = vir_stk(2,1) + phi_rec(3,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(2,2) = vir_stk(2,2) + phi_rec(3,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(2,3) = vir_stk(2,3) + phi_rec(3,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)
             
    tt = phi_rec(9,i)*img_dip(1,i) + phi_rec(10,i)*img_dip(2,i) + phi_rec(7,i)*img_dip(3,i)

    vir_stk(3,1) = vir_stk(3,1) + phi_rec(4,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(3,2) = vir_stk(3,2) + phi_rec(4,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(3,3) = vir_stk(3,3) + phi_rec(4,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)
             
  end do
  eer_chg = eer_chg_stk * 0.5d0
  eer_dip = eer_dip_stk * 0.5d0
  virial = virial - vir_stk
else if (need_ene) then
#ifdef MPI
  do my_chg_idx = 1, atm_cnt
    i = my_chgs(my_chg_idx)
#else
  do i = 1, atm_cnt
#endif
    eer_chg_stk = eer_chg_stk + phi_rec(1, i) * img_chg(i)
    eer_dip_stk = eer_dip_stk + sum(phi_rec(2:4, i) * img_perm_dip(1:3, i))
  end do
  eer_chg = eer_chg_stk * 0.5d0
  eer_dip = eer_dip_stk * 0.5d0
else if (need_vir) then

#ifdef MPI
  do my_chg_idx = 1, atm_cnt
    i = my_chgs(my_chg_idx)
#else
  do i = 1, atm_cnt
#endif
    tt = phi_rec(5,i)*img_dip(1,i) + phi_rec(8,i)*img_dip(2,i) + phi_rec(9,i)*img_dip(3,i)

    vir_stk(1,1) = vir_stk(1,1) + phi_rec(2,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(1,2) = vir_stk(1,2) + phi_rec(2,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(1,3) = vir_stk(1,3) + phi_rec(2,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)

    tt = phi_rec(8,i)*img_dip(1,i) + phi_rec(6,i)*img_dip(2,i) + phi_rec(10,i)*img_dip(3,i)

    vir_stk(2,1) = vir_stk(2,1) + phi_rec(3,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(2,2) = vir_stk(2,2) + phi_rec(3,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(2,3) = vir_stk(2,3) + phi_rec(3,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)
             
    tt = phi_rec(9,i)*img_dip(1,i) + phi_rec(10,i)*img_dip(2,i) + phi_rec(7,i)*img_dip(3,i)

    vir_stk(3,1) = vir_stk(3,1) + phi_rec(4,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(3,2) = vir_stk(3,2) + phi_rec(4,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(3,3) = vir_stk(3,3) + phi_rec(4,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)
             
  end do

  virial = virial - vir_stk

end if

end subroutine pGM_recip_ene_virial

subroutine pGM_recip_virial(atm_cnt, phi_rec, img_disp, img_chg, img_dip, virial)

implicit none

integer, intent(in)           :: atm_cnt
double precision, intent(in)  :: phi_rec(10,*)
double precision, intent(in)  :: img_disp(3,*)
double precision, intent(in)  :: img_chg(*)
double precision, intent(in)  :: img_dip(3,*)
double precision, intent(inout) :: virial(3,3)

double precision                :: vir_stk(3,3)
double precision                :: tt

#ifdef MPI
!  integer               :: my_chgs(*), my_chg_cnt   ! defined in the module 
integer               :: my_chg_idx
#endif
integer               :: i

vir_stk = 0.d0

#ifdef MPI
  do my_chg_idx = 1, atm_cnt
    i = my_chgs(my_chg_idx)
#else
  do i = 1, atm_cnt
#endif

    tt = phi_rec(5,i)*img_dip(1,i) + phi_rec(8,i)*img_dip(2,i) + phi_rec(9,i)*img_dip(3,i)

    vir_stk(1,1) = vir_stk(1,1) + phi_rec(2,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(1,2) = vir_stk(1,2) + phi_rec(2,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(1,3) = vir_stk(1,3) + phi_rec(2,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)

    tt = phi_rec(8,i)*img_dip(1,i) + phi_rec(6,i)*img_dip(2,i) + phi_rec(10,i)*img_dip(3,i)

    vir_stk(2,1) = vir_stk(2,1) + phi_rec(3,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(2,2) = vir_stk(2,2) + phi_rec(3,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(2,3) = vir_stk(2,3) + phi_rec(3,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)
             
    tt = phi_rec(9,i)*img_dip(1,i) + phi_rec(10,i)*img_dip(2,i) + phi_rec(7,i)*img_dip(3,i)

    vir_stk(3,1) = vir_stk(3,1) + phi_rec(4,i)*(img_dip(1,i)+img_chg(i)*img_disp(1,i)) &
                                + tt * img_disp(1,i)
    vir_stk(3,2) = vir_stk(3,2) + phi_rec(4,i)*(img_dip(2,i)+img_chg(i)*img_disp(2,i)) &
                                + tt * img_disp(2,i)
    vir_stk(3,3) = vir_stk(3,3) + phi_rec(4,i)*(img_dip(3,i)+img_chg(i)*img_disp(3,i)) &
                                + tt * img_disp(3,i)
             
  end do
  virial = virial - vir_stk

end subroutine pGM_recip_virial

end module pme_slab_recip_PGM_mod



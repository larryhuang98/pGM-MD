#include "pGM/copyright_pgm.i"
#include "../include/assert.fh"

!
! Adopted from sander pGM code by YD
!
subroutine pGM_keep_nsp(nspmin, atm_nspin)

  use pgm_interface_data_mod
  use pol_gauss_mdin_dat_mod, only : nspm_lcl
#ifdef MPI
  use prmtop_dat_mod
  use parallel_dat_mod
#endif

implicit none

integer, intent(in) :: nspmin
integer, intent(in) :: atm_nspin(nspmin)
integer             :: ier
integer :: i, j

!  allocate(atm_nsp_lcl(nspmin))
  allocate(mol_atm_ptr(nspmin+1), &
           mol_atm_cnt(nspmin), &
           stat=ier)

  REQUIRE(ier==0)

  mol_atm_cnt(1:nspmin) = atm_nspin(1:nspmin)
  nspm_lcl = nspmin

  mol_atm_ptr(1) = 1
  do i = 2, nspm_lcl+1
    mol_atm_ptr(i) = mol_atm_ptr(i-1) + mol_atm_cnt(i-1)
  end do

#ifdef MPI

  allocate(pGM_gbl_atm_mol_map(natom), &
           my_used_mol_mask(nspmin), &
           my_used_mols_lst(nspmin), &
           stat=ier)

  REQUIRE(ier==0)

  do i = 1, nspmin
    do j = mol_atm_ptr(i), mol_atm_ptr(i+1) - 1
      pGM_gbl_atm_mol_map(j) = i
    end do
  end do

  my_used_mol_mask = 0
  my_used_mols_cnt = 0
  my_used_mols_lst = 0

#endif

end subroutine pGM_keep_nsp
!-------------------------------------------------------------------------------
subroutine pGM_NonBond_perm_fields(atm_cnt, img_crd, ipairs, perm_field_img, ene_perm_dip)

  use pol_gauss_self, only : pGM_SELF_dipole_field
  use pol_gauss_multipoles_mod, only : perm_chg_img, perm_dip_img
  use constants_module, only : coulomb_const_kcal_per_mole
!  use constants_module, only : COULOMB_CONST_E
!                                        my_mpole_cnt, start_multipoles, end_multipoles

!  use pol_gauss_induced_data, only : ind_dip
! pmemd module
  use img_mod
  use pme_direct_mod, only : gbl_eed_cub

  use pme_slab_recip_PGM_mod
  use pme_blk_recip_PGM_mod
!  use pme_slab_recip_mod
!  use pme_blk_recip_mod
  use mdin_ewald_dat_mod, only : block_fft

  use timers_mod
#ifdef MPI
  use parallel_dat_mod
  use pme_recip_dat_mod
  use loadbal_mod
#endif
  use prmtop_dat_mod, only: natom
  use pol_gauss_direct_data, only : dipole_dipole_ptrs

  implicit none

  integer, intent(in)           :: atm_cnt
  integer, intent(in)           :: ipairs(*)

  double precision, intent(in)  :: img_crd(3,*)
  double precision, intent(inout):: ene_perm_dip
  double precision, intent(out) :: perm_field_img(3,atm_cnt)
  integer, parameter            :: dip_select = 1  ! First step in this induced dipole iteration

  integer :: i
  integer :: ierr
  logical :: need_frc
  logical :: need_pot_enes
  logical :: need_virials

  need_frc      = .false.
  need_pot_enes = .false.
  need_virials  = .false.

  ! All computed fields are actually grad phi
#ifdef MPI
  if (i_do_recip) then
    if (block_fft .eq. 0) then
#endif
      call do_slab_pmesh_kspace_PGM(perm_field_img, need_pot_enes, need_virials, need_frc, &
                                dip_select, perm_dip_img, perm_chg_img)
#ifdef MPI
    else
      call do_blk_pmesh_kspace_PGM(perm_field_img, need_pot_enes, need_virials, need_frc, &
                                dip_select, perm_dip_img, perm_chg_img)
    end if
  end if
  call update_loadbal_timer(elapsed_100usec_recipfrc)
#endif

  call update_time(nonbond_time)

  call pGM_DIRECT_perm_field(ipairs, perm_chg_img, perm_dip_img, img_crd, gbl_eed_cub, gbl_tranvec, &
                             perm_field_img)
#ifdef MPI
  call update_loadbal_timer(elapsed_100usec_dirfrc)
#endif

  call update_pme_time(dir_frc_sum_timer)

#ifdef MPI
  call pGM_SELF_dipole_field(my_img_hi-my_img_lo+1, perm_dip_img(1, my_img_lo), &
                             perm_field_img(1, my_img_lo))
  call update_loadbal_timer(elapsed_100usec_atmowner)
#else
  call pGM_SELF_dipole_field(atm_cnt, perm_dip_img, perm_field_img)
#endif

! permanent dipole elecstrostatic potential
! This is partial, needs to be reduced.

! NOTE: in MPI code, ene_perm_dip is reduced in pme_force().
!  ene_perm_dip = -0.5d0 * coulomb_const_kcal_per_mole &
  ene_perm_dip = -coulomb_const_kcal_per_mole &
                 * sum(perm_dip_img(1:3, my_img_lo:my_img_hi) &
                     * perm_field_img(1:3, my_img_lo:my_img_hi))  

  call update_pme_time(pme_misc_timer)
  call update_time(nonbond_time)

end subroutine pGM_NonBond_perm_fields

!*******************************************************************************
!
!
!*******************************************************************************
subroutine pGM_DIRECT_perm_field(ipairs, img_chg, img_dip, img_crd, eed_cub, tranvec, perm_dip_field)

! PGM modules
  use pol_gauss_mdin_dat_mod, only : ee_dsum_cut,ee_damped_cut,scf_local_cut
  use pol_gauss_direct_data
!, only : num_tensor, num_tensor_short, &                   ! # of tensors
!                                    dipole_dipole_list, dipole_dipole_list_short, &   ! pointers to the atm pair
!                                    dipole_dipole_tensor, dipole_dipole_tensor_short  ! tensor

! pmemd modules
  use pme_direct_mod
  use nb_pairlist_mod, only : list_tot, list_short_tot
  use img_mod, only : gbl_atm_img_map, gbl_img_atm_map    ! atm to img, img to atm
  use prmtop_dat_mod, only: natom
#ifdef MPI
  use parallel_dat_mod
#endif

  implicit none

  integer,intent(in)               :: ipairs(*)
  double precision, intent(in)     :: img_crd(3,*)
  double precision, intent(in)     :: eed_cub(*)
  double precision, intent(in)     :: tranvec(1:3, 0:17)
  double precision, intent(in)     :: img_chg(*)
  double precision, intent(in)     :: img_dip(3,*)
  double precision, intent(out)    :: perm_dip_field(3, *)

!  double precision,intent(inout) :: gradphi(3,*)  ! gradphi = cart_dipole_field

  integer, save :: list_tot_kp = 0
  integer, save :: list_short_tot_kp = 0

  if (list_tot .gt. list_tot_kp .or. list_short_tot .gt. list_short_tot_kp) then
     call pGM_alloc_dip_dip_tensors(list_tot, list_short_tot)
     list_tot_kp = list_tot
     list_short_tot_kp = list_short_tot
  end if

  call get_direct_fld_PGM_save_B(ipairs, img_crd, img_chg, img_dip, eed_cub, tranvec, perm_dip_field)
#if 0
  call get_direct_fld_PGM(ipairs, img_crd, img_chg, img_dip, eed_cub, tranvec, perm_dip_field)
#endif

! What if ind_dip is undefined yet?

! pGM_DIRECT_dip_dip_field is a wrapper of pGM_DIRECT_calc_dipdip_field
!  call pGM_DIRECT_dip_dip_field(ind_dip,dip_field)    

!  call pGM_DIRECT_calc_dipdip_field(num_tensor, dipole_dipole_list, dipole_dipole_tensor, &
!            ind_dip, dip_field)

end subroutine pGM_DIRECT_perm_field
!*******************************************************************************
!
!
!*******************************************************************************
subroutine pGM_alloc_dip_dip_tensors(list_tot, list_short_tot)

  use pol_gauss_mdin_dat_mod, only : ee_dsum_cut,ee_damped_cut,scf_local_cut
  use pol_gauss_direct_data
  use prmtop_dat_mod, only : natom

  implicit none

  integer :: list_tot, list_short_tot

    if ( allocated(dipole_dipole_tensor) ) deallocate(dipole_dipole_tensor)
    if ( allocated(dipole_dipole_list) ) deallocate(dipole_dipole_list)
    if ( allocated(dipole_dipole_tensor_short) ) deallocate(dipole_dipole_tensor_short)
    if ( allocated(dipole_dipole_list_short) ) deallocate(dipole_dipole_list_short)
    if ( allocated(delvec_save) ) deallocate(delvec_save)
    if ( allocated(Bfact_save) ) deallocate(Bfact_save)

    allocate(dipole_dipole_tensor(6,list_tot))
    allocate(dipole_dipole_list(list_tot))
    allocate(dipole_dipole_tensor_short(6,list_short_tot))
    allocate(dipole_dipole_list_short(list_short_tot))

    allocate(delvec_save(4,list_tot))
    allocate(Bfact_save(4,list_tot))

    dipole_dipole_list = 0
    dipole_dipole_list_short = 0
    if (.not.allocated(dipole_dipole_ptrs)) then
       allocate(dipole_dipole_ptrs(natom+1))
       dipole_dipole_ptrs(1:natom+1) = 0
    end if
    if (.not.allocated(dipole_dipole_short_ptrs)) then
       allocate(dipole_dipole_short_ptrs(natom+1))
       dipole_dipole_short_ptrs(1:natom+1) = 0
    end if
    if (.not.allocated(last_ee_ptrs)) then
       allocate(last_ee_ptrs(natom))
       last_ee_ptrs(1:natom)=0
    end if

end subroutine pGM_alloc_dip_dip_tensors

subroutine pGM_img_arrays(atm_cnt, img_atm_map)

  use pol_gauss_induced, only : pGM_INDUCED_img
  use pol_gauss_multipoles_mod, only : pGM_MPOLE_img

  implicit none

  integer, intent(in) :: atm_cnt
  integer, intent(in) :: img_atm_map(atm_cnt)

  call pGM_MPOLE_img(atm_cnt, img_atm_map)    ! imaging multiples and associated parameters
  call pGM_INDUCED_img(atm_cnt, img_atm_map)  ! imaging polarizabilities

end subroutine pGM_img_arrays
!*******************************************************************************
subroutine pGM_img_update(atm_cnt, img_atm_map)

  use pol_gauss_induced, only : pGM_INDUCED_img_update
  use pol_gauss_multipoles_mod, only : pGM_MPOLE_img_update

  implicit none

  integer, intent(in) :: atm_cnt
  integer, intent(in) :: img_atm_map(atm_cnt)

  call pGM_MPOLE_img_update(atm_cnt, img_atm_map)    ! imaging multiples and associated parameters
  call pGM_INDUCED_img_update(atm_cnt, img_atm_map)  ! imaging polarizabilities

end subroutine pGM_img_update

!*******************************************************************************
!
! The main interface for force, energy, virials.
!
!*******************************************************************************
subroutine pGM_NonBond_eval(atm_cnt, img_frc, &          ! natom, img_frc
                       crd, img_crd, eed_cub, &          ! crd, bspline for erfc()
                       ipairs, tranvec, new_list, &      ! NB pair list, PBC transfer vectors
                       need_pot_enes, need_virials, &    !
                       epolar, ee_dir, &                 ! epol, direct ele, recip ele, self ele
                       ee_recip, ee_self, &              ! recip ele, self ele
                       evdw, ehb, &                      ! VDW, HBond
                       eedvir, vir_eedir, vdw_virial, &  ! eedvir (for PME error estimation)
                       vir_eerecip, vir_eeself, &        ! vir recip ele, vir self ele
                       diprms_lcl, dipiter_lcl)
           
  use pol_gauss_mdin_dat_mod, only : do_pol_gauss_nonbond,pol_gauss_verbose, &
                                     dipole_print, nspm_lcl, pgm_curr_nstep
  use pol_gauss_multipoles_mod, only : perm_chg_img, &
                                       perm_dip_img, pGM_MPOLE_local_to_global, &
                                       covalent_ptr, covalent_atm, pGM_MPOL_update_my_covalent_lst
  use constants_module, only : coulomb_const_kcal_per_mole

!  use constants_module, only : COULOMB_CONST_E

#ifdef MPI
  use pol_gauss_induced, only : pGM_INDUCED_eval, ind_dip_img, init_scf_iter_comm, & !, scf_iter_comm
                                pGM_reduce_full_distr_full_used_img, &
!                                my_full_img_cnt, &
!                                my_full_img_lst, &
                                send_taskmap,recv_taskmap
!  use pgm_interface_data_mod, only : my_full_img_cnt, my_full_img_lst
#else
  use pol_gauss_induced, only : pGM_INDUCED_eval, ind_dip_img
#endif
  use pgm_interface_data_mod

#ifdef CUDA
#ifndef MPI
  use pGM_gpu_interface_mod
  use pol_gauss_mdin_dat_mod, only : dipole_scf_tol, scf_cg_niter, &
      scf_solv_opt, scf_local_niter, scf_local_cut, scf_sor_coefficient, &
      scf_gpu_pred, scf_gpu_check, scf_gpu_overlap, pgm_power, igvdw, gvdw_rep_form, b_rep_scale
  use pol_gauss_multipoles_mod, only : covalent_dip, ncovalent, gbl_perm_chg, r_gauss
  use pol_gauss_induced, only : polarizability
#endif
#endif

  use pbc_mod, only : recip             ! from pbc.F90
  use mdin_ctrl_dat_mod, only : ntb, imin, igb     ! from mdin_ctrl_dat.F90
  use prmtop_dat_mod
  use img_mod
  use inpcrd_dat_mod, only : atm_crd

!  use gbl_datatypes_mod
  use mol_list_mod

  use timers_mod

  use energy_records_mod, only : pme_pot_ene_rec, pme_virial_rec
#ifdef MPI
  use parallel_dat_mod
  use parallel_mod, only : mpi_allgathervec
  use multipmemd_mod, only : free_comms
  use nb_pairlist_mod, only : list_tot, list_short_tot 
#endif

  implicit none

!  use nblist, only: adjust_imagcrds, map_coords

  integer, intent(in)             :: atm_cnt
  double precision, intent(inout) :: img_frc(3,atm_cnt)
  double precision, intent(in)    :: crd(3,atm_cnt)
  double precision, intent(in)    :: img_crd(3,atm_cnt)
  double precision, intent(in)    :: eed_cub(*)
  integer, intent(in)             :: ipairs(*)
  double precision, intent(in)    :: tranvec(*)
  logical, intent(in)             :: need_pot_enes
  logical, intent(in)             :: need_virials
  logical, intent(in)             :: new_list
!  type(pme_pot_ene_rec), intent(inout)   :: pot_ene_rec
!  type(pme_virial_rec), intent(inout)    :: vir_rec

  double precision                :: epolar
  double precision                :: ee_dir
  double precision                :: ee_recip
  double precision                :: ee_self
  double precision                :: evdw
  double precision                :: ehb
  double precision                :: eedvir
  double precision                :: vir_eedir(3,3)
  double precision                :: vir_eerecip(3,3)
  double precision                :: vir_eeself(3,3)
  double precision                :: vdw_virial(3,3)
  double precision                :: tot_virial(3,3)
  double precision                :: virial_cor(3,3)
  double precision                :: net_frc(3)
  double precision                :: diprms_lcl
  double precision                :: dipiter_lcl

  double precision                :: vir_tensor(3,3)
  double precision                :: eelt
  double precision                :: ene_perm_dip
  double precision                :: ene_chg
#if 0
  double precision                :: evdw_14
  double precision                :: eelt_14
#endif

  type(nb_uv_rec)                 :: ene_vir
  double precision, allocatable   :: img_disp(:,:)
  double precision :: mass
  double precision :: massmol
  double precision :: xmol
  double precision :: ymol
  double precision :: zmol

  integer :: i, n, isave, num, imol, j, k
  integer :: ier, ier0
  character(800) :: str
  double precision, save, allocatable :: frc_tmp(:,:)
  integer :: called = 0
  ! GPU-specific variables
  real(8) :: gpu_virial(9)
  real(8) :: gpu_ene_chg, gpu_ene_dip, gpu_ene_vdw
  logical, save :: gpu_test_done = .false.
#if 0
  integer :: ich, ichf
#endif

#ifdef MPI
  integer, save              :: my_mols_cnt
  integer, save, allocatable :: my_mols_lst(:)
  double precision, save     :: vir_tmp(3,3)

  if (.not. allocated(frc_tmp)) allocate(frc_tmp(3, atm_cnt))
  if (.not. allocated(my_mols_lst)) allocate(my_mols_lst(nspm_lcl))
  if (.not. allocated(my_used_mols_lst)) then
     allocate(my_used_mols_lst(nspm_lcl))
     my_used_mols_lst(:) = 0
  end if
  if (.not. allocated(my_used_mol_mask)) then
     allocate(my_used_mol_mask(nspm_lcl))
     my_used_mol_mask(:) = 0
  end if
#endif
  if (.not. allocated(img_disp)) allocate(img_disp(3, atm_cnt))

#if 0
#ifdef MPI
  ich = 60+mytaskid
  ichf= 65+mytaskid
#else
  ich = 70
  ichf= 75
#endif
#endif

!# include "box.h"

!  double precision :: vir_tensor(3,3), vdw_virial(3,3)

  called = called + 1

  ene_vir    = null_nb_uv_rec
  
    ! R. Luo: molecular moments can be read in the global frame for development
    ! the conversion to local is done here after crd is obtained

! There are two types of images that need to be updated. The static parameters, mass, charges,
! polarizabilities, radius, should be imaged after nb-list update. Other dynamic parameters,
! cartesian permanent dipoles, coordinates, and induced dipoles, need to be imaged every time step.
!


#ifdef MPI
!  if (need_virials) then
! displacement and molecular virial need this
! Now, it is done when atm_crd are distributed and updated in pme_direct().
!    call mpi_allgathervec(atm_cnt, atm_crd)
!  end if
#endif

#if !defined(CUDA) || defined(MPI)
  if (new_list) then
#ifdef MPI
     call pGM_MPOL_update_my_covalent_lst(gbl_used_img_cnt, gbl_used_img_lst, &
        gbl_img_atm_map, gbl_atm_img_map)
#else
     call pGM_MPOL_update_my_covalent_lst(gbl_img_atm_map, gbl_atm_img_map)
#endif
  end if

  ! Convert local-frame bond dipoles to global Cartesian dipoles (CPU side)
  ! (GPU path does this in kLocalToGlobal_PGM kernel)
  call pGM_MPOLE_local_to_global(atm_crd)

! Now, we convert everything to images. We can now just work on images and produce image forces too.

  ! Update image arrays for CPU path
  ! (GPU path manages per-atom data via pgm_gpu_setup_after_nl)
  if (new_list) then

#ifdef MPI

! initiate scf iteration communicators
    call init_scf_iter_comm

!! Somehow this was commented out. YD

!    call pGM_add_my_used_mols(my_full_img_cnt, my_full_img_lst, mol_atm_ptr, my_mols_cnt, my_mols_lst)
! This should be done in pme_list() instead, such that coordinates can be
! updated and distributed properly.
!    call pGM_add_my_used_mols(my_full_img_cnt, my_full_img_lst, my_used_img_mask)
#endif
    call pGM_img_arrays(atm_cnt, gbl_img_atm_map)

  else
    call pGM_img_update(atm_cnt, gbl_img_atm_map)
  end if
#endif

! =============================================================================
! GPU CODE PATH (CUDA without MPI)
! =============================================================================
#if defined(CUDA) && !defined(MPI)

  ! One-time GPU initialization only.
  ! All per-step computation is handled by pgm_gpu_pme_ene/pgm_gpu_pme_force
  ! called from pme_force.F90, matching the standard gpu_pme_ene/gpu_pme_force pattern.
  if (.not. gpu_test_done) then
    gpu_test_done = .true.
    call pGM_GPU_init(atm_cnt)
    call pGM_GPU_upload_bonds(atm_cnt, covalent_ptr, covalent_atm, &
                               covalent_dip, ncovalent)
    call pGM_GPU_upload_prmtop(atm_cnt, gbl_perm_chg, r_gauss, polarizability, &
                                dipole_scf_tol, scf_cg_niter)
    call pGM_GPU_set_pcg_params(scf_solv_opt, scf_local_niter, &
                                 scf_local_cut, scf_sor_coefficient)
    call pgm_gpu_set_options(scf_gpu_pred, scf_gpu_check, scf_gpu_overlap, pgm_power, &
                             igvdw, gvdw_rep_form, b_rep_scale)
  end if
  return  ! Everything else handled by pgm_gpu_pme_ene/force from pme_force.F90

#else

! =============================================================================
! CPU CODE PATH (original implementation)
! =============================================================================

  pgm_curr_nstep = pgm_curr_nstep + 1   ! RL: 1-based step index for diagnostic dumps (pGM_NonBond_eval is called once per MD step)

#ifdef MPI
  call pGM_INDUCED_eval(atm_cnt, img_crd, ind_dip_img, ipairs, ene_perm_dip, diprms_lcl, dipiter_lcl)
#else
  call pGM_INDUCED_eval(atm_cnt, img_crd, ind_dip_img, ipairs, ene_perm_dip, diprms_lcl, dipiter_lcl)
#endif

! nspm_lcl:     nummols

  if (need_virials) then
#ifdef MPI
    call img_displacement(my_used_mols_cnt, mol_atm_ptr, img_disp, atm_mass, atm_crd, my_used_mols_lst)
#else
    call img_displacement(nspm_lcl, mol_atm_ptr, img_disp, atm_mass, atm_crd)
#endif
  end if

#ifdef MPI
  call pGM_NonBond_ene_frc(atm_cnt, atm_crd, img_crd, perm_chg_img, perm_dip_img, &
                     ind_dip_img, img_disp, ipairs, &
                     ntypes, gbl_img_iac, typ_ico, gbl_cn1, gbl_cn2, & ! R. Luo: vdw
                     tranvec, need_pot_enes, need_virials, &
                     img_frc, ene_vir, new_list)
!    call mpi_allreduce(img_frc, frc_tmp, 3*atm_cnt, MPI_DOUBLE_PRECISION, MPI_SUM, pmemd_comm, ier)
#else
  call pGM_NonBond_ene_frc(atm_cnt, atm_crd, img_crd, perm_chg_img, perm_dip_img, &
                     ind_dip_img, img_disp, ipairs, &
                     ntypes, gbl_img_iac, typ_ico, gbl_cn1, gbl_cn2, & ! R. Luo: vdw
                     tranvec, need_pot_enes, need_virials, &
                     img_frc, ene_vir, new_list)
#endif

#endif  /* CUDA && !MPI */

  ee_dir     = ene_vir%ene_dir_chg+ene_vir%ene_dir_dip
  ee_recip   = ene_vir%ene_rec_chg+ene_vir%ene_rec_dip
  ee_self    = ene_vir%ene_self_chg+ene_vir%ene_self_dip
  evdw       = ene_vir%ene_dir_vdw
  ehb        = ene_vir%ene_dir_hb
  vir_eedir  = ene_vir%vireedir
  eedvir     = ene_vir%vireedir(1,1) + ene_vir%vireedir(2,2) + ene_vir%vireedir(3,3) 
  vir_eerecip= ene_vir%vireerec
  vir_eeself = ene_vir%vireeself
  vdw_virial = ene_vir%virvdw

#if 0
  write(ich,*)'vir_eedir   ',vir_eedir
  write(ich,*)'eedvir      ',eedvir
  write(ich,*)'vir_eerecip ',vir_eerecip
  write(ich,*)'vir_eeself  ',vir_eeself 
  write(ich,*)'vdw_virial  ',vdw_virial 
  flush(ich)
#endif

!  Electrostatic and polarization energies should be partitioned
!  This is one of the ways.
!  Here, eelt counts for contributions from all permanent multipoles (charges and perm dipoles)
!  and epolar counts for all terms involving induced dipoles.

  epolar = ene_vir%ene_tot_dip
  eelt   = ene_vir%ene_tot_chg
! to be consistent with sander_pGM, will remove later

!  pot_ene_rec%epol = epolar
!  pot_ene_rec%elec_dir = eelt

  epolar  = epolar + eelt   ! for now, to be consistent with sander.pGM
  eelt    = 0.d0            !
  ee_dir  = 0.d0
  ee_recip= 0.d0
  ee_self = 0.d0
#ifdef MPI
  if (.not.master) then
    diprms_lcl  = 0.d0
    dipiter_lcl = 0.d0
  end if
#endif

!  if (need_virials) then
!  if ( ntb > 0 ) then
    ! the factor 0.5 is due to the fact that sander uses molecular kinetic energy as the
    ! kinetic part of virial, which is actually half of the real virial
    ! Virials are scaled by 0.5 in pme_force.F90.
    ! Unit conversion is done together with energy unit conversion in pGM_NonBond_ene_frc().
    ! Eventually, should all be done when parameters are read.

!    vir_tensor  = 0.5 * coulomb_const_kcal_per_mole * vir_tensor 
!    vir_eedir   = 0.5 * coulomb_const_kcal_per_mole * vir_eedir 
!    vir_eerecip = 0.5 * coulomb_const_kcal_per_mole * vir_eerecip
!    vir_eeself  = 0.5 * coulomb_const_kcal_per_mole * vir_eeself 

!    vdw_virial  = 0.5 * vdw_virial

!    eedvir      = vir_eedir(1,1)+vir_eedir(2,2)+vir_eedir(3,3)

!  end if

  ene_vir    = null_nb_uv_rec

end subroutine pGM_NonBond_eval
!-------------------------------------------------------------------------------
!
!  The main force, energy, virial calculation driver.
!
!-------------------------------------------------------------------------------
subroutine pGM_NonBond_ene_frc(numatoms, crd, img_crd, img_chg, img_perm_dip, &
                       img_ind_dip, img_disp, ipairs, &
                       ntypes, iac, ico, cn1, cn2, & ! R. Luo: vdw
                       tranvec, need_pot_enes, need_virials, &
                       img_frc, ene_vir, new_list)

  use pol_gauss_self, only : pGM_SELF_ene_frc
  use pgm_interface_data_mod
  use pol_gauss_direct_data

  use pol_gauss_mdin_dat_mod, only : do_pol_gauss_nonbond, pol_gauss_verbose, &
                                     dipole_print, saved_nprint, nspm_lcl, &
                                     pgm_dump_enable, pgm_curr_nstep
  use constants_module, only : coulomb_const_kcal_per_mole
!  use constants_module, only : COULOMB_CONST_E
!                                       my_mpole_cnt, start_multipoles, end_multipoles
  use pol_gauss_induced, only : pGM_INDUCED_eval, ind_dip_img, ind_dip, pgm_dump_frame

  use pbc_mod, only : recip             ! from pbc.F90
  use mdin_ctrl_dat_mod, only : ntb, imin, igb     ! from mdin_ctrl_dat.F90
  use prmtop_dat_mod, only : natom
  use img_mod
  use file_io_dat_mod, only : mdout
  use pme_direct_mod, only : gbl_eed_cub, get_nb_energy_PGM, &
                             get_nb_energy_PGM_use_saved_B
  use pme_slab_recip_PGM_mod
  use pme_blk_recip_PGM_mod
!  use pme_slab_recip_mod
!  use pme_blk_recip_mod
  use mdin_ewald_dat_mod, only : block_fft
#ifdef MPI
  use parallel_dat_mod
  use pme_recip_dat_mod
  use loadbal_mod
!  use pgm_interface_data_mod, only : my_full_img_cnt, my_full_img_lst
!gbl_atm_owner_map
#endif
  use nb_exclusions_mod, only : atm_nb_maskdata, atm_nb_mask

  use timers_mod

  implicit none

  integer, intent(in) :: numatoms
  double precision, intent(in)    :: img_chg(*)
  double precision, intent(in)    :: img_perm_dip(3,*)
  double precision, intent(in)    :: img_ind_dip(3,*)
  double precision, intent(in)    :: crd(3,*)
  double precision, intent(in)    :: img_crd(3,*)
  double precision, intent(in)    :: img_disp(3,*)
  integer, intent(in)             :: ipairs(*)
  integer, intent(in)             :: ntypes
  integer, intent(in)             :: iac(*)
  integer, intent(in)             :: ico(*) ! R. Luo: vdw
  double precision, intent(in)    :: cn1(*)
  double precision, intent(in)    :: cn2(*) ! R. Luo: vdw
  double precision, intent(in)    :: tranvec(1:3, 0:17)
  logical, intent(in)             :: need_pot_enes
  logical, intent(in)             :: need_virials
  double precision, intent(inout) :: img_frc(3,numatoms)
  type(nb_uv_rec),  intent(inout) :: ene_vir

  double precision :: e_recip  
  double precision :: e_rec_chg
  double precision :: e_rec_dip
  double precision :: e_dir_chg
  double precision :: e_dir_dip
  double precision :: e_self_chg
  double precision :: e_self_dip
  double precision :: ene_chg
  double precision :: ene_dip
  double precision :: e_dir_vdw
  double precision :: e_dir_hb
  
  double precision, save, allocatable :: pGM_force_img(:,:)
  double precision, save, allocatable :: phi(:,:)
  double precision, save, allocatable :: img_tot_dip(:,:)

  double precision :: vir_eedir(3,3)
  double precision :: vir_eerecip(3,3)
  double precision :: vir_eeself(3,3)
  double precision :: vdw_virial(3,3)
  double precision :: eedvir
 
  integer :: ier
  integer :: ierr
  integer :: imol, num, isave, iatom, i, j, img
  logical :: need_frc_rec = .true.
  logical :: need_ene_rec = .true.
  logical :: need_vir_rec = .true.
  logical :: new_list
  integer, save :: called = 0
  logical, save :: opened100 = .false.

  ! RL: Turning on dipole printing
  double precision :: xmolt, ymolt, zmolt, momt

  ! electric potentials and derivatives
  if (.not.allocated(pGM_force_img)) allocate(pGM_force_img(3, numatoms))
  if (.not.allocated(img_tot_dip)) allocate(img_tot_dip(3, numatoms))
  if (.not.allocated(phi)) allocate(phi(10, numatoms))

  ene_chg    = 0.d0
  ene_dip    = 0.d0
  e_recip    = 0.d0
  e_rec_chg  = 0.d0
  e_rec_dip  = 0.d0
  e_dir_chg  = 0.d0
  e_dir_dip  = 0.d0
  e_self_chg = 0.d0
  e_self_dip = 0.d0
  e_dir_vdw  = 0.d0
  e_dir_hb   = 0.d0

  vir_eedir  = 0.d0
  vir_eerecip= 0.d0
  vir_eeself = 0.d0
  vdw_virial = 0.d0
  eedvir     = 0.d0

! NOTE: vir_rec is the complete reciprocal space virial, including the displacement correction
! This needs to be the first to call since "phi" is used to calculate vir_eeself and e_rec_self
  need_ene_rec = need_pot_enes
  need_vir_rec = need_virials

#ifdef MPI
  if (new_list) then
    pGM_force_img = 0.0d0
    img_frc = 0.d0
    phi = 0.0d0
    do i = 1, my_full_img_cnt
      img = my_full_img_lst(i)
      img_tot_dip(1:3, img) = img_perm_dip(1:3, img) + img_ind_dip(1:3, img)
    end do
  else
    do i = 1, my_full_img_cnt
      img = my_full_img_lst(i)
      pGM_force_img(:, img) = 0.d0
      img_frc(:, img) = 0.d0
      phi(:, img) = 0.d0
      img_tot_dip(1:3, img) = img_perm_dip(1:3, img) + img_ind_dip(1:3, img)
    end do
  end if
  if (i_do_recip) then
    if (block_fft .eq. 0) then
#else
  pGM_force_img = 0.d0
  img_frc = 0.d0
  phi = 0.0d0
  img_tot_dip(1:3, 1:numatoms) = img_perm_dip(1:3,1:numatoms) + img_ind_dip(1:3,1:numatoms)
#endif
      call do_slab_pmesh_kspace_PGM(phi, need_ene_rec, need_vir_rec, need_frc_rec, 3, &
                        img_tot_dip, img_chg, vir_eerecip, vir_eeself, &
                        img_disp, e_recip, e_rec_chg, e_rec_dip, img_perm_dip)
#ifdef MPI
    else
      call do_blk_pmesh_kspace_PGM(phi, need_ene_rec, need_vir_rec, need_frc_rec, 3, &
                        img_tot_dip, img_chg, vir_eerecip, vir_eeself, &
                        img_disp, e_recip, e_rec_chg, e_rec_dip, img_perm_dip)
    end if
    call update_loadbal_timer(elapsed_100usec_recipfrc)
  end if
#endif

! NOTE: phi from get_nb_energy_PGM is indexed as images
  call get_nb_energy_PGM_use_saved_B(ipairs, gbl_img_crd, img_disp, img_chg, img_tot_dip, gbl_eed_cub, tranvec, &
                         img_frc, phi, need_pot_enes, need_virials, &
                         e_dir_chg, e_dir_dip, e_dir_vdw, e_dir_hb, eedvir, vir_eedir, vdw_virial)
#if 0
  call get_nb_energy_PGM(ipairs, gbl_img_crd, img_disp, img_chg, img_tot_dip, gbl_eed_cub, tranvec, &
                         img_frc, phi, need_pot_enes, need_virials, &
                         e_dir_chg, e_dir_dip, e_dir_vdw, e_dir_hb, eedvir, vir_eedir, vdw_virial)
#endif

#ifdef MPI
  call update_loadbal_timer(elapsed_100usec_dirfrc)
  call pGM_SELF_ene_frc(my_img_hi-my_img_lo+1, img_chg(my_img_lo), &
                        img_tot_dip(1, my_img_lo), need_pot_enes, e_self_chg, e_self_dip, &
                        phi(1, my_img_lo))
#else
  call pGM_SELF_ene_frc(numatoms, img_chg, img_tot_dip, need_pot_enes, e_self_chg, e_self_dip, phi)
#endif
! no longer needed
!  phi = phi_rec + phi_dir + phi_self

  call pGM_energy_force_virial(numatoms, crd, phi, img_chg, img_perm_dip, img_tot_dip, &
                         gbl_img_atm_map, gbl_atm_img_map, pGM_force_img, need_pot_enes, &
                         ene_chg, ene_dip)

#ifdef MPI
  call update_loadbal_timer(elapsed_100usec_atmowner)
#endif

! will move scaling to read prmtop, after work out 

!  frc = frc + coulomb_const_kcal_per_mole * pGM_force

#ifdef MPI
  do i = 1, my_full_img_cnt
    img = my_full_img_lst(i)
    img_frc(:, img)     = img_frc(:, img) + coulomb_const_kcal_per_mole * pGM_force_img(:, img)
  end do
#else
  img_frc     = img_frc + coulomb_const_kcal_per_mole * pGM_force_img
#endif

  ene_dip     = coulomb_const_kcal_per_mole * ene_dip     ! The total energy of dipoles.
  ene_chg     = coulomb_const_kcal_per_mole * ene_chg     ! The total energy of monopoles.

  e_recip     = coulomb_const_kcal_per_mole * e_recip     ! The combined monopole and dipole reciprocal energy.

  e_rec_chg   = coulomb_const_kcal_per_mole * e_rec_chg  ! perm charge * reciprocal grad_phi
  e_rec_dip   = coulomb_const_kcal_per_mole * e_rec_dip  ! perm_dip dot * reciprocal grad_phi

  e_self_chg  = coulomb_const_kcal_per_mole * e_self_chg  ! The self energy of dipoles, in direct sum.
  e_self_dip  = coulomb_const_kcal_per_mole * e_self_dip  ! The self energy of monopoles, in direct sum.

  vir_eedir   = coulomb_const_kcal_per_mole * vir_eedir
  vir_eerecip = coulomb_const_kcal_per_mole * vir_eerecip
  vir_eeself  = coulomb_const_kcal_per_mole * vir_eeself 

  ene_vir%ene_tot_chg = ene_chg
  ene_vir%ene_tot_dip = ene_dip
  ene_vir%ene_tot_vdw = e_dir_vdw
  ene_vir%ene_tot_hb  = e_dir_hb 

  ene_vir%ene_dir_chg = e_dir_chg
  ene_vir%ene_dir_dip = e_dir_dip
  ene_vir%ene_dir_vdw = e_dir_vdw
  ene_vir%ene_dir_hb  = e_dir_hb 

  ene_vir%ene_rec_chg = e_rec_chg
  ene_vir%ene_rec_dip = e_rec_dip

  ene_vir%ene_self_chg= e_self_chg
  ene_vir%ene_self_dip= e_self_dip

  e_dir_chg  = ene_chg - e_rec_chg - e_self_chg
  e_dir_dip  = ene_dip - e_rec_dip - e_self_dip

  ene_vir%vireedir  = vir_eedir
  ene_vir%vireerec  = vir_eerecip
  ene_vir%vireeself = vir_eeself
  ene_vir%virvdw    = vdw_virial

  ! RL: per-atom induced dipole output (gated by dipole_print in &pol_gauss namelist)
  saved_nprint = saved_nprint + 1
  if ( dipole_print /= 0 ) then
    if ( .not. opened100 ) then
      open(unit=100, file='ind_dip.txt', status='replace', action='write')
      opened100 = .true.
    end if
    if ( mod(saved_nprint, dipole_print) == 0 ) then
      write(100,'(A,I10)') '# step ', saved_nprint
      do iatom = 1, numatoms
        i = gbl_atm_img_map(iatom)
        write(100,'(I8,3(1x,ES20.12))') iatom, img_ind_dip(1,i), img_ind_dip(2,i), img_ind_dip(3,i)
      end do
      write(100,*)
    end if
  end if

  ! RL: full diagnostic frame dump (operator / b / mu / crd / short / hist) for the recycling study
  if ( pgm_dump_enable /= 0 ) call pgm_dump_frame(numatoms, img_crd, img_ind_dip)

  if ( need_virials .and. pol_gauss_verbose == 2 )then
    write(mdout,'(a,/,4(1x,f14.4))') &
            ' e_recip, e_rec_chg, e_rec_dip = ', &
              e_recip, e_rec_chg, e_rec_dip
    write(mdout,'(a,/,4(1x,f14.4))') &
            ' e_self_chg, e_self_dip, e_dir_chg, e_dir_dip = ', &
              e_self_chg, e_self_dip, e_dir_chg, e_dir_dip
    write(mdout,'(a,/,3(1x,f14.4))') &
            ' ene_chg, ene_dip, e_dir_vdw = ', &
              ene_chg, ene_dip, e_dir_vdw
    write(mdout,'(a,/,3(1x,f14.4))') &
            ' vir_eedir, e_dir_chg, e_dir_dip = ', &
              vir_eedir, e_dir_chg, e_dir_dip
  end if

end subroutine pGM_NonBond_ene_frc
!-------------------------------------------------------------------------------
! assume phi's are images
!-------------------------------------------------------------------------------
subroutine pGM_energy_force_virial(atm_cnt, crd, phi, img_chg, img_perm_dip, img_tot_dip, &
                                   img_atm_map, atm_img_map, pGM_force, need_ene, ene_chg, pGM_energy)

  use pol_gauss_multipoles_mod, only : covalent_ptr, covalent_atm, covalent_dip, &
                                       my_ncovalent, my_covalent_pair, my_covalent_img_pair, &
                                       my_covalent_pair_dip
  use pbc_mod, only : ucell, recip, pbc_box             ! from pbc.F90
  use file_io_dat_mod, only : mdout
!  use pol_gauss_mpole_index_mod
#ifdef MPI
  !use pgm_interface_data_mod, only : my_full_img_cnt, my_full_img_lst
  use pgm_interface_data_mod, only : my_full_img_cnt, my_full_img_lst, gbl_used_img_cnt_phi
  use img_mod, only : gbl_used_img_cnt, gbl_used_img_lst
  use parallel_dat_mod
#endif

  implicit none

!# include "box.h"
# include "pGM/pGM_mpole_index.h"

  integer, intent(in)             :: atm_cnt
  double precision, intent(in)    :: crd(3,*)            ! this is unimaged coordinate
  integer, intent(in)             :: img_atm_map(*)
  integer, intent(in)             :: atm_img_map(*)
  double precision, intent(in)    :: img_chg(atm_cnt)
  double precision, intent(in)    :: img_perm_dip(3,atm_cnt)  ! assume this is the perm dipole
  double precision, intent(in)    :: img_tot_dip(3,atm_cnt)  ! assume this is the total dipole
  double precision, intent(inout) :: phi(10,atm_cnt)     ! combined gradient of phi
  double precision, intent(inout) :: pGM_force(3,atm_cnt)! forces are imaged
  double precision, intent(inout) :: pGM_energy
  double precision, intent(inout) :: ene_chg
  logical, intent(in)             :: need_ene

  integer                      :: i, idx
  integer                      :: j
  integer                      :: k
  integer                      :: atom_first
  integer                      :: atom_last
  integer                      :: covalent_atom
  integer                      :: i_atm, j_atm, img_i, img_j
  double precision             :: distance
  double precision             :: deltr_dot_field
  double precision             :: pGM_energy_stk
  double precision             :: ene_chg_stk
  double precision             :: deltr_dot_field2!, deltr_dot_dipole

!  double precision             :: virial_force(3, atm_cnt)
  double precision             :: delx, dely, delz, dij2
  double precision             :: cov_dipi, cov_dipj, cov_dipk
  double precision             :: deltr_dot_fieldi, deltr_dot_fieldj
  double precision             :: dipphi2, dipphi3, dipphi4, dipdeltij

  double precision             :: dd01, dd02, dd03, dd11, dd12, dd13, dd21, dd22, dd23, dd31, dd32, dd33

  ! Haixin: convert grad of phi to electric field and its derivatives

! to wrap crds
  double precision             :: xij(3)
  double precision             :: box_half(3)
  double precision             :: box_half_min
  double precision             :: box_half_min_sq

  box_half(1:3) = pbc_box(1:3) * 0.5d0
  box_half_min = min(box_half(1), box_half(2), box_half(3))
  box_half_min_sq = box_half_min * box_half_min

  if (need_ene) then
#ifdef MPI
    ene_chg_stk = 0.d0
    pGM_energy_stk = 0.d0
    do idx = 1, my_full_img_cnt
      img_i = my_full_img_lst(idx)
      ene_chg_stk = ene_chg_stk + img_chg(img_i)*phi(1, img_i)
      pGM_energy_stk = pGM_energy_stk + img_perm_dip(1, img_i)*phi(2, img_i) + &
                                        img_perm_dip(2, img_i)*phi(3, img_i) + &
                                        img_perm_dip(3, img_i)*phi(4, img_i)
    end do
    pGM_energy = pGM_energy_stk * 0.5d0
    ene_chg    = ene_chg_stk * 0.5d0
#else
    ene_chg    = 0.5d0 * sum(img_chg(1:atm_cnt)*phi(1,1:atm_cnt))
    pGM_energy = 0.5d0 * sum(img_perm_dip(1:3,1:atm_cnt)*phi(2:4,1:atm_cnt)) ! phi is gradient of electric field
#endif
  end if

#ifdef MPI
  do idx = 1, gbl_used_img_cnt_phi  ! Here, we only need to count the ones with phi() != 0.
    img_i = gbl_used_img_lst(idx)  ! These are the non-zero phi()'s for MPI
#else
  do img_i = 1, atm_cnt
#endif

    ! forces due to covalent monopoles
    ! forces due to total dipoles (covalent dipoles + induced dipoles)
    pGM_force(1,img_i) = pGM_force(1,img_i) -(img_chg(img_i)    * phi(2, img_i) &
                                            + img_tot_dip(1, img_i) * phi(Ind_200, img_i) &
                                            + img_tot_dip(2, img_i) * phi(Ind_110, img_i) &
                                            + img_tot_dip(3, img_i) * phi(Ind_101, img_i))
    pGM_force(2,img_i) = pGM_force(2,img_i) -(img_chg(img_i)    * phi(3,img_i) &
                                            + img_tot_dip(1, img_i) * phi(Ind_110, img_i) &
                                            + img_tot_dip(2, img_i) * phi(Ind_020, img_i) &
                                            + img_tot_dip(3, img_i) * phi(Ind_011, img_i))
    pGM_force(3,img_i) = pGM_force(3,img_i) -(img_chg(img_i)    * phi(4,img_i) &
                                            + img_tot_dip(1, img_i) * phi(Ind_101, img_i) &
                                            + img_tot_dip(2, img_i) * phi(Ind_011, img_i) &
                                            + img_tot_dip(3, img_i) * phi(Ind_002, img_i))

  end do

    ! forces due to changes in covalent dipoles 

  do idx = 1, my_ncovalent
     i_atm = my_covalent_pair(1, idx)
     j_atm = my_covalent_pair(2, idx)
     img_i = my_covalent_img_pair(1, idx) 
     img_j = my_covalent_img_pair(2, idx) 

! need to use crd here,
! If use img_crd, need img_j = atm_img_map(j_atm) and invoke wrap_crds(xij)

      xij(1:3) = crd(1:3,j_atm) - crd(1:3,i_atm)
      dij2     = xij(1)*xij(1) + xij(2)*xij(2) + xij(3)*xij(3)

! This shouldn't be necessary, if crd is used.
      if (dij2 .ge. box_half_min_sq) then
         call wrap_crds(xij)
         dij2     = xij(1)*xij(1) + xij(2)*xij(2) + xij(3)*xij(3)
      end if

      distance = sqrt(dij2)
      if ( distance .gt. 3.0d0 ) then
        write(mdout,*) 'pGM Fatal Error: covalent bond too long!',i_atm,j_atm, distance
        call mexit(mdout,1)
      endif

      deltr_dot_fieldi=   (xij(1) * phi(2, img_i) + xij(2) * phi(3, img_i) + xij(3) * phi(4, img_i)) / dij2
      deltr_dot_fieldj=  -(xij(1) * phi(2, img_j) + xij(2) * phi(3, img_j) + xij(3) * phi(4, img_j)) / dij2

      cov_dipi = my_covalent_pair_dip(1, idx) / distance
      cov_dipj = my_covalent_pair_dip(2, idx) / distance

      dipphi2 = cov_dipi * phi(2, img_i) - cov_dipj * phi(2, img_j)
      dipphi3 = cov_dipi * phi(3, img_i) - cov_dipj * phi(3, img_j)
      dipphi4 = cov_dipi * phi(4, img_i) - cov_dipj * phi(4, img_j)

      dipdeltij=   cov_dipi * deltr_dot_fieldi + cov_dipj * deltr_dot_fieldj

      pGM_force(1,img_i) = pGM_force(1,img_i) + dipphi2 - xij(1) * dipdeltij
      pGM_force(2,img_i) = pGM_force(2,img_i) + dipphi3 - xij(2) * dipdeltij
      pGM_force(3,img_i) = pGM_force(3,img_i) + dipphi4 - xij(3) * dipdeltij

      pGM_force(1,img_j) = pGM_force(1,img_j) - dipphi2 + xij(1) * dipdeltij
      pGM_force(2,img_j) = pGM_force(2,img_j) - dipphi3 + xij(2) * dipdeltij
      pGM_force(3,img_j) = pGM_force(3,img_j) - dipphi4 + xij(3) * dipdeltij

  end do

contains
!*******************************************************************************
!
! Internal Subroutine:  wrap_crds
!
! Description: <TBS>
!              
!*******************************************************************************

subroutine wrap_crds(xpij)

  implicit none

! Formal arguments:

  double precision  xpij(3)

! Local variables:

  integer           m

  do m = 1, 3

    if (xpij(m) .ge. box_half(m)) then

      xpij(m) = xpij(m) - pbc_box(m)

    else if (xpij(m) .lt. -box_half(m)) then

      xpij(m) = xpij(m) + pbc_box(m)

    end if

  end do

  return

end subroutine wrap_crds


end subroutine pGM_energy_force_virial
#if 0
!-------------------------------------------------------------------------------
subroutine map_full_set_imgcrd(atm_cnt, img_atm_map, fraction, img_crd)

  use pbc_mod, only : is_orthog, ucell, pbc_box

  implicit none

  integer, intent(in) :: atm_cnt
  integer, intent(in) :: img_atm_map(atm_cnt)
  double precision, intent(in) :: fraction(3,atm_cnt)
  double precision, intent(out) :: img_crd(3,atm_cnt)

  double precision    :: x_box, y_box, z_box
  double precision    :: f1, f2, f3
  double precision    :: ucell_stk(3, 3)
  integer :: img_idx, atm_idx
  integer :: is_orthog_stk

  x_box = pbc_box(1)
  y_box = pbc_box(2)
  z_box = pbc_box(3)

  ucell_stk(:,:) = ucell(:,:)

  is_orthog_stk = is_orthog

  if (is_orthog_stk .ne. 0) then
    do img_idx = 1, atm_cnt
      atm_idx = img_atm_map(img_idx)
      img_crd(1, img_idx) = fraction(1, atm_idx) * x_box
      img_crd(2, img_idx) = fraction(2, atm_idx) * y_box
      img_crd(3, img_idx) = fraction(3, atm_idx) * z_box
    end do
  else
    do img_idx = 1, atm_cnt
      atm_idx = img_atm_map(img_idx)
      f1 = fraction(1, atm_idx)
      f2 = fraction(2, atm_idx)
      f3 = fraction(3, atm_idx)
      ! ucell(2,1), ucell(3,1), and ucell(3,2) are always 0.d0, so
      ! we can simplify the expression in this critical inner loop
      img_crd(1, img_idx) = f1 * ucell_stk(1, 1) + &
                            f2 * ucell_stk(1, 2) + &
                            f3 * ucell_stk(1, 3)
      img_crd(2, img_idx) = f2 * ucell_stk(2, 2) + &
                            f3 * ucell_stk(2, 3)
      img_crd(3, img_idx) = f3 * ucell_stk(3, 3)

      img_saved_crd(1:3, img_idx) = img_crd(1:3, img_idx)
    end do
  end if
end subroutine map_full_set_imgcrd
!-------------------------------------------------------------------------------
#ifdef MPI
subroutine pGM_adjust_imgcrds(atm_cnt, atm_crd, atm_saved_crd, img_saved_crd, img_crd, img_atm_map)

  use img_mod
  use pbc_mod
  use mdin_ctrl_dat_mod, only : ntp
  use nb_pairlist_mod, only : gbl_saved_box
#ifdef MPI
  use parallel_dat_mod
#endif

  implicit none

  integer, intent(in) :: atm_cnt
  integer, intent(in) :: img_atm_map(atm_cnt)
  double precision, intent(in) :: atm_crd(3, atm_cnt)
  double precision, intent(in) :: atm_saved_crd(3, atm_cnt)
  double precision, intent(in) :: img_saved_crd(3, atm_cnt)
  double precision, intent(out) :: img_crd(3, atm_cnt)

  integer :: img_idx, atm_idx
  double precision :: box_del(3)
 
  if (ntp .eq. 0) then  ! Constant volume

      do img_idx = 1, atm_cnt
        atm_idx = img_atm_map(img_idx)
        img_crd(:,img_idx) = atm_crd(:,atm_idx) + img_saved_crd(:,img_idx) - &
                             atm_saved_crd(:,atm_idx)
!write(171+mytaskid,*)atm_crd(:,atm_idx),img_saved_crd(:,img_idx),atm_saved_crd(:,atm_idx)
      end do
!flush(171+mytaskid)
  else  ! Constant pressure scaling.

    box_del(:) = pbc_box(:) / gbl_saved_box(:)

      do img_idx = 1, atm_cnt
        atm_idx = img_atm_map(img_idx)
        img_crd(:,img_idx) = atm_crd(:, atm_idx) + &
                            (img_saved_crd(:,img_idx) - atm_saved_crd(:,atm_idx)) * box_del(:)
                            
      end do

  end if

end subroutine pGM_adjust_imgcrds
#endif
#endif
!-------------------------------------------------------------------------------
subroutine POL_GAUSS_read_real_list_data(header,nf,dim1,num_list,list)
  implicit none

  character(len=*), intent(in) :: header
  integer, intent(in) :: nf,dim1,num_list
  double precision, intent(out) :: list(dim1,num_list)

  integer :: iok,ionerr,j,k
  character(len=80) :: fmt
  character(len=80) :: fmtin,dtype

  ionerr = 0 !fatal if missing
  fmtin = '(5E16.8)'
  dtype = header//'LIST'
  call nxtsec(nf,  6,  ionerr,fmtin,  dtype,  fmt,  iok)
  read(nf,fmt)((list(j,k),j=1,dim1),k=1,num_list)
end subroutine POL_GAUSS_read_real_list_data

!------------------------------------------------------------------------
subroutine POL_GAUSS_read_list_data(header,nf,dim1,num_list,list)
  implicit none

  character(len=*), intent(in) :: header
  integer, intent(in) :: nf,dim1,num_list
  integer, intent(out) :: list(dim1,num_list)

  integer :: iok,ionerr,j,k
  character(len=80) :: fmt
  character(len=80) :: fmtin,dtype

  ionerr = 0 !fatal if missing
  fmtin = '(10I8)'
  dtype = header//'LIST'
  call nxtsec(nf,  6,  ionerr,fmtin,  dtype,  fmt,  iok)
  read(nf,fmt)((list(j,k),j=1,dim1),k=1,num_list)
end subroutine POL_GAUSS_read_list_data
!--------------------------------------------------------------------
subroutine POL_GAUSS_get_startlist_endlist(num_list,startlist,endlist,siztask)

#ifdef MPI
  use parallel_dat_mod
#endif
  implicit none

  integer,intent(in) :: num_list
  integer,intent(out) :: startlist,endlist,siztask

  integer piece
#ifndef MPI
  integer numtasks, mytaskid

  numtasks=1  
  mytaskid=0
#endif

  if (numtasks > 1) then
    piece = num_list/numtasks
    if (piece * numtasks .lt. num_list) piece = piece + 1
    startlist = mytaskid*piece+1
    endlist   = mytaskid*piece+piece
    if (mytaskid == (numtasks-1)) endlist = num_list
    siztask = endlist-startlist+1
  else
    startlist=1
    endlist = num_list
    siztask = endlist-startlist+1
  end if
end subroutine POL_GAUSS_get_startlist_endlist
!-------------------------------------------------------------
subroutine POL_GAUSS_read_real_scalar(flag,nf,scalar_value)
  implicit none

  character(len=*), intent(in) :: flag
  integer, intent(in) :: nf
  double precision, intent(out) :: scalar_value

  integer :: iok,ionerr
  character(len=80) :: fmt
  character(len=80) :: fmtin

  ionerr = 0 !fatal if missing
  fmtin = '(E16.8)'
  call nxtsec(nf,  6,  ionerr,fmtin,  flag,  fmt,  iok)
  read(nf,fmt)scalar_value
end subroutine POL_GAUSS_read_real_scalar
!--------------------------------------------------------------------

#ifdef MPI
!
! Because pGM calculates molecular virials using a full pairwise method, we need COM and relative atomic
! coordinates, not only the images. To avoid one more coordinate distribution, we trick the crd distribution
! code by adding their images to the used image list.
!
! This should be changed to use pmemd data structure.
!
subroutine pGM_add_my_used_mols(used_img_cnt, used_img_lst, used_img_mask)

  use gbl_datatypes_mod
  use img_mod, only : gbl_atm_img_map
  use pgm_interface_data_mod
  use parallel_dat_mod
  use pol_gauss_mdin_dat_mod, only : nspm_lcl

!  use pgm_interface_data_mod, only : my_used_img_mask

  implicit none

  integer, intent(inout)   :: used_img_cnt
  integer, intent(inout)   :: used_img_lst(*)
  integer(byte), intent(inout)   :: used_img_mask(*)

  integer               :: img
  integer               :: atm
  integer               :: mol
  integer               :: i, j

  if (.not. allocated(my_used_mols_lst)) allocate(my_used_mols_lst(nspm_lcl))
  if (.not. allocated(my_used_mol_mask)) then
     allocate(my_used_mol_mask(nspm_lcl))
     my_used_mol_mask = 0
     my_used_mols_cnt = 0
  end if

  call find_my_used_mols(used_img_cnt, used_img_lst, my_used_mols_cnt, my_used_mols_lst)

! Add the additional images to the full image list. This is needed only at the force/virial calculation
! stage, not the field stage. Therefore, there is no need to add them to the short-range list.

  do i = 1, my_used_mols_cnt
    mol = my_used_mols_lst(i)
    do atm = mol_atm_ptr(mol), mol_atm_ptr(mol+1)-1
      img = gbl_atm_img_map(atm)
      if (used_img_mask(img) .eq. 0) then
        used_img_cnt = used_img_cnt + 1
        used_img_lst(used_img_cnt) = img
        used_img_mask(img) = 1
      end if
    end do
  end do

contains
! 
subroutine find_my_used_mols(img_cnt, img_lst, mols_cnt, mols_lst)

  use img_mod, only : gbl_img_atm_map
  use parallel_dat_mod
  use pgm_interface_data_mod, only : pGM_gbl_atm_mol_map, my_used_mol_mask

  implicit none

  integer, intent(in)   :: img_cnt
  integer, intent(in)   :: img_lst(*)
  integer, intent(out)  :: mols_cnt
  integer, intent(out)  :: mols_lst(*)

  integer               :: img
  integer               :: atm
  integer               :: mol
  integer               :: i, j

!  do i = 1, mols_cnt
!    mol = mols_lst(i)
!    my_used_mol_mask(:) = 0
!  end do

  mols_cnt = 0

  do i = 1, img_cnt
    img = img_lst(i)
    atm = gbl_img_atm_map(img)
    mol = pGM_gbl_atm_mol_map(atm)
    if (my_used_mol_mask(mol) .eq. 0) then
      mols_cnt = mols_cnt + 1
      mols_lst(mols_cnt) = mol
      my_used_mol_mask(mol) = 1
    end if
  end do

  do i = 1, mols_cnt
    mol = mols_lst(i)
    my_used_mol_mask(mol) = 0
  end do

end subroutine find_my_used_mols

end subroutine pGM_add_my_used_mols

#endif

#ifdef MPI
subroutine img_displacement(mols_cnt, mol_atm_ptr, img_disp, amass, atm_crd, mols_lst)
#else
subroutine img_displacement(mols_cnt, mol_atm_ptr, img_disp, amass, atm_crd)
#endif

  use img_mod, only : gbl_atm_img_map
#ifdef MPI
  use parallel_dat_mod
  use prfs_mod
  use mol_list_mod, only : gbl_mol_atms_listdata, gbl_mol_atms_lists, atm_mol_map, &
                           gbl_mol_prfs_listdata, gbl_mol_prfs_lists, gbl_mol_cnt
#else
  use mol_list_mod, only : gbl_mol_atms_listdata, gbl_mol_atms_lists, atm_mol_map, &
                           gbl_mol_cnt
#endif

  implicit none

  integer, intent(in)   :: mols_cnt
#ifdef MPI
  integer, intent(in)   :: mols_lst(mols_cnt)
#endif
  integer, intent(in)   :: mol_atm_ptr(*)
  double precision, intent(in)   :: amass(*)
  double precision, intent(in)   :: atm_crd(3,*)
  double precision, intent(out)  :: img_disp(3,*)

  integer               :: img
  integer               :: atm
  integer               :: i, j
  integer               :: imol
  double precision      :: massmol
  double precision      :: mass
  double precision      :: xmol
  double precision      :: ymol
  double precision      :: zmol

#ifdef MPI
  do j = 1, mols_cnt
    imol = mols_lst(j)
#else
  do imol = 1, mols_cnt
#endif
    massmol = 0.d0
    xmol = 0.d0
    ymol = 0.d0
    zmol = 0.d0
     
    ! get c.o.m. of molecule - TODO: use precalced masses
     
    do atm = mol_atm_ptr(imol), mol_atm_ptr(imol+1) - 1
      mass = amass(atm)
      massmol = massmol + mass
      xmol = xmol + mass*atm_crd(1, atm)
      ymol = ymol + mass*atm_crd(2, atm)
      zmol = zmol + mass*atm_crd(3, atm)
    end do
    xmol = xmol / massmol
    ymol = ymol / massmol
    zmol = zmol / massmol

    do atm = mol_atm_ptr(imol), mol_atm_ptr(imol+1) - 1
      img = gbl_atm_img_map(atm)

      img_disp(1, img) = atm_crd(1, atm) - xmol
      img_disp(2, img) = atm_crd(2, atm) - ymol
      img_disp(3, img) = atm_crd(3, atm) - zmol

    end do
  end do

end subroutine img_displacement

!--------------------------------------------------------------------
subroutine bcast_pGM_parameter

#ifdef MPI
  use pol_gauss_mdin_dat_mod
  use pol_gauss_multipoles_mod
  use pol_gauss_induced
!  use pol_gauss_readparm_mod, only : pGM_MPOLE_get_start_end_lists
!  use pol_gauss_induced, only : pGM_INDUCED_bcast
!  use pol_gauss_multipoles, only : pGM_MPOLE_get_start_end_lists
  use pgm_interface_data_mod

  use prmtop_dat_mod
  use parallel_dat_mod
  use mdin_ctrl_dat_mod, only : ipgm

  implicit none

  integer :: ier
  integer :: ns

  integer          :: ibuf(pgm_ctrl_int_cnt)
  common / pgm_ctrl_int /ibuf
  save :: / pgm_ctrl_int /
  double precision :: buf(pgm_ctrl_dbl_cnt)
  common / pgm_ctrl_dbl /buf
  save :: / pgm_ctrl_dbl /
 
  call mpi_bcast(ibuf, pgm_ctrl_int_cnt, MPI_INTEGER, 0, pmemd_comm, ier)
  call mpi_bcast(buf, pgm_ctrl_dbl_cnt, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)

  call pGM_MPOLE_bcast(natom)

!  call mpi_bcast(nspm_lcl,1,MPI_INTEGER,0,pmemd_comm,ier)

if (.not.master) then

!  num_multipoles = natom                       ! this should have been broadcast already

  allocate(mol_atm_ptr(nspm_lcl+1), &
           mol_atm_cnt(nspm_lcl), &
           pGM_gbl_atm_mol_map(natom), &
           my_used_mol_mask(nspm_lcl), &
           stat=ier)

  REQUIRE(ier==0)
  my_used_mol_mask = 0
end if

  call mpi_bcast(mol_atm_cnt,nspm_lcl,MPI_INTEGER,0,pmemd_comm,ier)
  call mpi_bcast(mol_atm_ptr,nspm_lcl+1,MPI_INTEGER,0,pmemd_comm,ier)
  call mpi_bcast(pGM_gbl_atm_mol_map,natom,MPI_INTEGER,0,pmemd_comm,ier)

  call pGM_INDUCED_bcast(natom)

#endif

end subroutine bcast_pGM_parameter
!----------------------------------------------------------
!-------------------------------------------------------------------------------
subroutine pGM_NonBond_dip_dip_fields_short(atm_cnt, img_crd, img_ind_dip, img_dip_field)

  use pol_gauss_direct_data
#ifdef MPI
  use parallel_dat_mod
  use loadbal_mod
#endif
  use img_mod
  use timers_mod

  implicit none

  integer,intent(in) :: atm_cnt
  double precision,intent(in) :: img_crd(3,*)
  double precision,intent(in) :: img_ind_dip(3,atm_cnt)
  double precision,intent(out) :: img_dip_field(3,atm_cnt)

  integer :: i, ierr, ier

  ! All computed fields are actually grad phi

  call pGM_DIRECT_dip_dip_field_short(img_ind_dip, img_dip_field)

#ifdef MPI
  call update_loadbal_timer(elapsed_100usec_dirfrc)
#endif

  call update_pme_time(dir_frc_sum_timer)

  call update_time(nonbond_time)

end subroutine pGM_NonBond_dip_dip_fields_short
!-------------------------------------------------------------------------------
subroutine pGM_NonBond_dip_dip_fields(atm_cnt, img_crd, img_ind_dip, img_dip_field)
                       
  use pol_gauss_direct_data
  use pol_gauss_self, only : pGM_SELF_dipole_field
  use img_mod
  use pme_slab_recip_PGM_mod
  use pme_blk_recip_PGM_mod
!  use pme_slab_recip_mod
!  use pme_blk_recip_mod
  use timers_mod
  use mdin_ewald_dat_mod, only : block_fft

#ifdef MPI
  use parallel_dat_mod
  use loadbal_mod
  use pme_recip_dat_mod
#endif

  implicit none

  integer, intent(in)           :: atm_cnt
  double precision, intent(in)  :: img_crd(3,*)
  double precision, intent(in)  :: img_ind_dip(3,atm_cnt)
  double precision, intent(out) :: img_dip_field(3,atm_cnt)

  integer, parameter            :: dip_field_select = 2
  integer                       :: i, ierr

  logical, parameter :: need_pot_enes = .false., need_virials = .false., need_frc = .false.

  ! All computed fields are actually grad phi
#ifdef MPI
  if (i_do_recip) then
    if (block_fft .eq. 0) then
#endif
      call do_slab_pmesh_kspace_PGM(img_dip_field, need_pot_enes, need_virials, need_frc, dip_field_select, img_ind_dip)
#ifdef MPI
    else
      call do_blk_pmesh_kspace_PGM(img_dip_field, need_pot_enes, need_virials, need_frc, dip_field_select, img_ind_dip)
    end if
    call update_loadbal_timer(elapsed_100usec_recipfrc)
  end if
#endif

  call pGM_DIRECT_dip_dip_field(img_ind_dip, img_dip_field)

#ifdef MPI
  call update_loadbal_timer(elapsed_100usec_dirfrc)
#endif

  call update_pme_time(dir_frc_sum_timer)

#ifdef MPI
  call pGM_SELF_dipole_field(my_img_hi - my_img_lo + 1, img_ind_dip(1,my_img_lo), &
                                           img_dip_field(1,my_img_lo))
#else
  call pGM_SELF_dipole_field(atm_cnt, img_ind_dip, img_dip_field)
#endif

  call update_pme_time(pme_misc_timer)

#ifdef MPI
  call update_loadbal_timer(elapsed_100usec_atmowner)

!  call mpi_allreduce(field_tmp, img_dip_field, 3*atm_cnt, &
!                     MPI_DOUBLE_PRECISION, MPI_SUM, pmemd_comm, ierr)
#endif

  call update_time(nonbond_time)

end subroutine pGM_NonBond_dip_dip_fields

#ifdef MPI
subroutine pGM_init_used_img_map(used_img_mask, used_img_cnt, used_img_lst)

  use pgm_interface_data_mod, only : my_short_img_lst, my_used_img_mask, my_short_img_cnt
  use img_mod, only : my_img_lo, my_img_hi
  use parallel_dat_mod

  implicit none

  integer, intent(inout)  :: used_img_cnt
  integer, intent(inout)  :: used_img_lst(*)
  integer(byte), intent(inout)  :: used_img_mask(*)
  integer  :: i, img

  do i = 1, used_img_cnt
    img = used_img_lst(i)
    used_img_mask(img) = 0
    my_used_img_mask(img) = 0
    used_img_lst(i) = 0
  end do

  used_img_cnt = 0
  do img = my_img_lo, my_img_hi
    used_img_cnt = used_img_cnt + 1
    my_short_img_lst(used_img_cnt) = img
    my_used_img_mask(img) = 1
    used_img_lst(used_img_cnt)     = img
    used_img_mask(img) = 1
  end do

  my_short_img_cnt = used_img_cnt

end subroutine pGM_init_used_img_map

subroutine pGM_append_long_range_img(used_img_cnt, used_img_lst)

  use pgm_interface_data_mod, only : my_used_img_mask, my_full_img_cnt, my_full_img_lst, &
                                     my_short_img_lst, my_short_img_cnt

  implicit none

  integer, intent(in)  :: used_img_cnt
  integer, intent(in)  :: used_img_lst(*)

  integer  :: idx
  integer  :: img

! my_used_img_mask has been cleared after init_used_img_map() and marked in get_nb_list()

  my_full_img_cnt = my_short_img_cnt
  my_full_img_lst(1:my_short_img_cnt) = my_short_img_lst(1:my_short_img_cnt)

! append remaining used img list to my_full_img_lst

  do idx = 1, used_img_cnt
    img = used_img_lst(idx)
    if (my_used_img_mask(img) .eq. 0) then
      my_used_img_mask(img) = 2
      my_full_img_cnt = my_full_img_cnt + 1
      my_full_img_lst(my_full_img_cnt) = img
    end if
  end do

end subroutine pGM_append_long_range_img

#endif

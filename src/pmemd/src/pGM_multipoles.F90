#include "pGM/copyright_pgm.i"
#include "../include/assert.fh"

!
! Adopted from sander pGM code by YD
!
!---------------------------------------------------------
module pol_gauss_multipoles_mod

   use pgm_interface_data_mod, only : betamax
#ifdef MPI
  use parallel_dat_mod
#endif

  implicit none

  private

!module pol_gauss_multipoles_data
! data from pol_gauss_multipoles module

  integer, save :: num_multipoles
  integer, parameter :: MAXMP=35

!  double precision, parameter         :: coulomb_const_kcal_per_mole = 332.05382d0 !Tinker
  double precision, save, allocatable :: gbl_perm_chg(:)
  double precision, save, allocatable :: gbl_perm_dip(:,:)
  double precision, save, allocatable :: perm_chg_img(:)
  double precision, save, allocatable :: perm_dip_img(:,:)

  !Gaussian radii
!  double precision, parameter         :: betamax = sqrt(2.0d0)/0.7147597d0
!  double precision, save              :: betamax = sqrt(2.0d0)/0.7147597d0
  double precision, save, allocatable :: r_gauss(:)
  double precision, save, allocatable :: b_polar(:)
  double precision, save, allocatable :: r_gauss_img(:)
  double precision, save, allocatable :: b_polar_img(:)

  !Covalent local frame lists
  integer, save                       :: ncovalent
  integer, save, allocatable          :: covalent_ptr(:)
  integer, save, allocatable          :: covalent_atm(:)
  integer, save, allocatable          :: covalent_pair(:,:)
  integer, save, allocatable          :: covalent_pair_lst(:)

  double precision, save, allocatable :: covalent_monop(:)
  double precision, save, allocatable :: covalent_dip(:)
  double precision, save, allocatable :: covalent_pair_dip(:,:)

  integer, save                       :: my_ncovalent = 0
  integer, save, allocatable          :: my_covalent_pair(:,:)
  integer, save, allocatable          :: my_covalent_pair_lst(:)
  double precision, save, allocatable :: my_covalent_pair_dip(:,:)

  integer, save, allocatable          :: my_covalent_img_pair(:,:)
  integer, save, allocatable          :: my_covalent_img_pair_lst(:)
  double precision, save, allocatable :: my_covalent_img_pair_dip(:,:)

!  public coulomb_const_kcal_per_mole, &

  public &
         b_polar, r_gauss, r_gauss_img, &
         perm_chg_img, perm_dip_img, &
         gbl_perm_chg, ncovalent, &
         covalent_monop, covalent_ptr, covalent_atm, covalent_dip, &
         covalent_pair, covalent_pair_lst, covalent_pair_dip, &
         my_ncovalent, my_covalent_pair, my_covalent_pair_lst, &
         my_covalent_img_pair, my_covalent_img_pair_lst, &
         my_covalent_pair_dip, my_covalent_img_pair_dip, &
         pGM_MPOLE_local_to_global, &
         pGM_MPOL_update_my_covalent_lst, pGM_MPOL_setup_covalent_pair, &
         pGM_MPOLE_img, pGM_MPOLE_img_update, pGM_MPOLE_readparm

#ifdef MPI
  public :: pGM_MPOLE_bcast, pGM_add_perm_dip_pairs
#endif

  contains

subroutine pGM_MPOLE_rescale_multipoles()

  integer j,n
  double precision bohr

  ! change units from Bohr to Angstroms
  !bohr = 0.5291772083d0 ! R. Luo: if input uses Bohr as the length unit
  bohr = 1.0d0/4.803239377d0 ! R. Luo: if input dipoles are read in as debyes
  !bohr = 1.0d0 ! R. Luo: if input dipoles are read in as electron*Angstrom

  !the following is changed to scale input global moments only
  do n = 1,num_multipoles
    gbl_perm_dip(1:3, n) = bohr * gbl_perm_dip(1:3, n)
  end do
end subroutine pGM_MPOLE_rescale_multipoles
!---------------------------------------------------------
subroutine pGM_MPOL_setup_covalent_pair()

  use prmtop_dat_mod, only : natom
#ifdef MPI
  use parallel_dat_mod
  use parallel_mod
  use img_mod
#endif

  implicit none

  integer :: ier
  integer :: iatm
  integer :: jatm
  integer :: ibond
  integer :: jbond
  integer :: lst_size
  integer :: lst_idx

  lst_size = ncovalent / 2
  if (.not.allocated(covalent_pair))     allocate(covalent_pair(2, lst_size))
  if (.not.allocated(covalent_pair_dip)) allocate(covalent_pair_dip(2, lst_size))
  if (.not.allocated(covalent_pair_lst)) allocate(covalent_pair_lst(ncovalent))

  lst_idx = 0
  do iatm = 1, natom
    do ibond = covalent_ptr(iatm) + 1, covalent_ptr(iatm+1)
       jatm = covalent_atm(ibond)
       if (jatm .gt. iatm) then
          do jbond = covalent_ptr(jatm) + 1, covalent_ptr(jatm+1)
             if (covalent_atm(jbond) .eq. iatm) then
                lst_idx = lst_idx + 1
                covalent_pair(1, lst_idx) = iatm
                covalent_pair(2, lst_idx) = jatm
                covalent_pair_dip(1, lst_idx) = covalent_dip(ibond)
                covalent_pair_dip(2, lst_idx) = covalent_dip(jbond)
                covalent_pair_lst(ibond) = lst_idx
                covalent_pair_lst(jbond) = lst_idx
                exit
             end if
          end do
       end if
    end do
  end do

end subroutine pGM_MPOL_setup_covalent_pair
!---------------------------------------------------------
#ifdef MPI
subroutine pGM_MPOL_update_my_covalent_lst(img_cnt, img_lst, img_atm_map, atm_img_map)
#else
subroutine pGM_MPOL_update_my_covalent_lst(img_atm_map, atm_img_map)
#endif

  use prmtop_dat_mod, only : natom
#ifdef MPI
  use pgm_interface_data_mod, only : my_used_img_mask
  use parallel_dat_mod
  use parallel_mod
  use img_mod
#endif

  implicit none

#ifdef MPI
  integer, intent(in) :: img_cnt
  integer, intent(in) :: img_lst(img_cnt)
#endif
  integer, intent(in) :: img_atm_map(*)
  integer, intent(in) :: atm_img_map(*)

  integer :: ier
  integer :: iatm
  integer :: jatm
  integer :: ipair
  integer :: pair_idx, idx
  integer :: jbond
  integer :: lst_size
  integer :: img_i
  integer :: img_j
  integer :: i
  integer :: j
  integer, save, allocatable  :: my_used_pair_mask(:)

#ifdef MPI
  integer :: img_idx
#endif

  if (my_ncovalent .eq. 0) then
     call pGM_MPOL_setup_covalent_pair
  end if

  lst_size = ncovalent / 2
  if (.not.allocated(my_covalent_pair))         allocate(my_covalent_pair(2, lst_size))
  if (.not.allocated(my_covalent_pair_dip))     allocate(my_covalent_pair_dip(2, lst_size))
  if (.not.allocated(my_covalent_img_pair))     allocate(my_covalent_img_pair(2, lst_size))
  if (.not.allocated(my_covalent_img_pair_dip)) allocate(my_covalent_img_pair_dip(2, lst_size))

  if (.not.allocated(my_covalent_pair_lst))     allocate(my_covalent_pair_lst(lst_size))
  if (.not.allocated(my_covalent_img_pair_lst)) allocate(my_covalent_img_pair_lst(lst_size))

  if (.not.allocated(my_used_pair_mask)) then
     allocate(my_used_pair_mask(ncovalent))
     my_used_pair_mask = 0
  end if
  do idx = 1, my_ncovalent
     my_used_pair_mask(my_covalent_pair_lst(idx)) = 0
  end do

  my_ncovalent = 0

#ifdef MPI
  do img_idx = 1, img_cnt     ! gbl_used_img_cnt is from img_mod
    img_i = img_lst(img_idx)
    iatm= img_atm_map(img_i)
#else
  do iatm = 1, natom
    img_i = atm_img_map(iatm)
#endif
    do idx = covalent_ptr(iatm) + 1, covalent_ptr(iatm+1)
       pair_idx = covalent_pair_lst(idx)
       if (my_used_pair_mask(pair_idx) .eq. 0) then
          i    = 1
          j    = 2
          if (covalent_pair(2, pair_idx) .eq. iatm) then
             i = 2
             j = 1
          end if
          jatm = covalent_pair(j, pair_idx)
          img_j = atm_img_map(jatm)
#ifdef MPI
          if (my_used_img_mask(img_i) .gt. 0 .and. my_used_img_mask(img_j) .gt. 0) then
#endif
             my_ncovalent = my_ncovalent + 1
             my_covalent_pair(1, my_ncovalent) = iatm
             my_covalent_pair(2, my_ncovalent) = jatm
             my_covalent_pair_dip(1, my_ncovalent) = covalent_pair_dip(i, pair_idx)
             my_covalent_pair_dip(2, my_ncovalent) = covalent_pair_dip(j, pair_idx)
             my_covalent_img_pair(1, my_ncovalent) = img_i
             my_covalent_img_pair(2, my_ncovalent) = img_j
             my_covalent_img_pair_dip(1, my_ncovalent) = covalent_pair_dip(i, pair_idx)
             my_covalent_img_pair_dip(2, my_ncovalent) = covalent_pair_dip(j, pair_idx)
             my_used_pair_mask(pair_idx) = 1
             my_covalent_pair_lst(my_ncovalent) = pair_idx
#ifdef MPI
          end if
#endif
       end if
    end do
  end do

end subroutine pGM_MPOL_update_my_covalent_lst
!---------------------------------------------------------
subroutine pGM_MPOLE_local_to_global(crd)

  use pol_gauss_mdin_dat_mod, only : pol_gauss_verbose
  use constants_module, only : ZERO, ONE
  use file_io_dat_mod, only : mdout
  use prmtop_dat_mod, only : natom
#ifdef MPI
  use pgm_interface_data_mod, only : gbl_used_img_cnt_phi
  use parallel_dat_mod
  use parallel_mod
  use img_mod
#endif

  implicit none

  double precision,intent(in) :: crd(3,*)
!#include "do_flag.h"

  integer :: ier
  integer :: iatm
  integer :: jatm
  integer :: jbond
  integer :: start_bonds
  integer :: end_bonds
  double precision :: vector(3), norm, dipx, dipy, dipz
#ifdef MPI
  integer :: lst_idx
  integer :: img
#endif

! This should be changed to use list of bonds.
#ifdef MPI
  do lst_idx = 1, gbl_used_img_cnt_phi     ! gbl_used_img_cnt_phi is from pGM_interface
    img = gbl_used_img_lst(lst_idx)
    iatm= gbl_img_atm_map(img)
#else
  do iatm = 1, natom
#endif

    dipx = 0.d0
    dipy = 0.d0
    dipz = 0.d0

    start_bonds = covalent_ptr(iatm) + 1
    end_bonds = covalent_ptr(iatm+1)

    ! compute through-bond unit vectors
    do jbond = start_bonds, end_bonds
      jatm = covalent_atm(jbond)
      vector(1:3) = crd(1:3,jatm) - crd(1:3,iatm)
      norm = ONE/sqrt(sum(vector(1:3)**2))
      vector(1:3) = vector(1:3)*norm

    ! recover global dipoles
      dipx = dipx + covalent_dip(jbond)*vector(1)
      dipy = dipy + covalent_dip(jbond)*vector(2)
      dipz = dipz + covalent_dip(jbond)*vector(3)
    enddo
    gbl_perm_dip(1, iatm) = dipx
    gbl_perm_dip(2, iatm) = dipy
    gbl_perm_dip(3, iatm) = dipz
  enddo

  if ( pol_gauss_verbose == 2 ) then
    write(mdout,*) 'converted global multipoles from through-bond dipoles (D)'
    do iatm = 1, num_multipoles
      write(mdout,'(4f15.6)') gbl_perm_chg(iatm), gbl_perm_dip(1:3,iatm)*4.803239377d0
    end do
  end if
end subroutine pGM_MPOLE_local_to_global
!---------------------------------------------------------
subroutine pGM_MPOLE_img(atm_cnt, img_atm_map)

  use pol_gauss_direct_data, only : displacement, displacement_img
  use prmtop_dat_mod, only : natom
  use parallel_dat_mod

  implicit none

  integer, intent(in) :: atm_cnt
  integer, intent(in) :: img_atm_map(*)
  integer :: iatm, i

! non-bonded neighbor list being updated

  if ( .not. allocated(perm_dip_img)) &
     allocate(perm_dip_img(3,natom))
  if ( .not. allocated(r_gauss_img)) allocate(r_gauss_img(natom))
  if ( .not. allocated(b_polar_img)) allocate(b_polar_img(natom))
  if ( .not. allocated(displacement_img)) allocate(displacement_img(3,natom))
  if ( .not. allocated(perm_chg_img)) allocate(perm_chg_img(natom))
  do i = 1, atm_cnt
    iatm = img_atm_map(i)
    perm_chg_img(i)     = gbl_perm_chg(iatm)
    perm_dip_img(1:3,i) = gbl_perm_dip(1:3,iatm)
    r_gauss_img(i) = r_gauss(iatm)
    b_polar_img(i) = b_polar(iatm)
  enddo

end subroutine pGM_MPOLE_img
!---------------------------------------------------------
subroutine pGM_MPOLE_img_update(atm_cnt, img_atm_map)

!  use pol_gauss_direct_data, only  : displacement, displacement_img

  implicit none

  integer, intent(in) :: atm_cnt
  integer, intent(in) :: img_atm_map(*)

  integer :: iatm, i

! non-bonded neighbor list remained the same

!  call distribute_img_frcs(atm_cnt, img_frc, nb_frc)

  do i = 1, atm_cnt
    iatm = img_atm_map(i)
    perm_dip_img(1:3,i) = gbl_perm_dip(1:3,iatm)
  enddo
  
end subroutine pGM_MPOLE_img_update
!-------------------------------------------------------------------------------
function pGM_MPOLE_readparm(nf,num_atoms)

  use pol_gauss_mdin_dat_mod, only : pol_gauss_verbose
  use file_io_dat_mod, only : mdout
!  use pol_gauss_multipoles_data
#ifdef MPI
  use parallel_dat_mod
#endif

  integer :: pGM_MPOLE_readparm
  integer, intent(in) :: nf,num_atoms

  integer :: n, ier, dim1, tmp
!,ncovalent
  double precision :: min_r_gauss
  integer, allocatable :: atmij(:,:), ptrs(:,:)
  double precision, allocatable :: cbv_dip(:,:)
  integer              :: n_cbv

  pGM_MPOLE_readparm = 0

  num_multipoles = num_atoms                       ! this should have been broadcast already

  allocate(covalent_ptr(num_atoms+1),           &  ! pointer to the CBVs of the present atom
           covalent_monop(num_atoms),           &
           b_polar(num_atoms),                  &  ! fifth. read in gaussian multipole radius
           r_gauss(num_atoms),                  &  ! stores b_polar**2
           gbl_perm_chg(num_atoms),             &  ! permanent charges
           gbl_perm_dip(3,num_atoms),           &  ! permanent dipoles in cartesian frame
           stat=ier)

!           global_multipole(MPOLDIM,num_atoms), &  ! allocate global multipoles
  REQUIRE(ier==0)

  gbl_perm_dip = 0.0d0
  gbl_perm_chg = 0.0d0

  ! first. read in covalent atom/local frame pointer
  dim1 = 1
  call POL_GAUSS_read_list_data('POL_GAUSS_COVALENT_POINTERS_',nf, &
                 dim1,num_atoms,covalent_ptr(2))
  ncovalent = sum (covalent_ptr(2:num_atoms+1))
  if (pol_gauss_verbose == 2 ) then
    write(mdout,'(/,A,I8)') 'Finding covalent pointers', num_atoms
    write(mdout,'(10i8)') covalent_ptr(2:num_atoms+1)
  end if
  covalent_ptr(1) = 0
  do n = 1, num_atoms
    covalent_ptr(n+1) = covalent_ptr(n+1) + covalent_ptr(n)
  end do

  allocate(covalent_atm(1:ncovalent), &
           covalent_dip(1:ncovalent),stat=ier)
  REQUIRE(ier==0)

  dim1 = 1
  call POL_GAUSS_read_list_data('POL_GAUSS_COVALENT_ATOMS_',nf, &
                 dim1,ncovalent,covalent_atm)
  if (pol_gauss_verbose == 2 ) then
    write(mdout,'(/,A,I8)') 'Finding covalent atoms', ncovalent
    write(mdout,'(10i8)') covalent_atm(1:ncovalent)
  end if

  ! third. read in throughbond moments (in electron Angstrom)
  dim1 = 1
  call POL_GAUSS_read_real_list_data('POL_GAUSS_COVALENT_DIPOLES_',nf, &
                 dim1,ncovalent,covalent_dip)
  if (pol_gauss_verbose == 2 ) then
    write(mdout,'(/,A,I8)') 'Finding throughbond moments',ncovalent 
    write(mdout,'(5E16.8)') covalent_dip(1:ncovalent)
  end if

  allocate(atmij(2,ncovalent), cbv_dip(2,ncovalent))
  n_cbv = 0

  ! fourth. read in gaussian monopole
  dim1 = 1
  call POL_GAUSS_read_real_list_data('POL_GAUSS_MONOPOLES_',nf, &
                 dim1,num_atoms,covalent_monop)
  if (pol_gauss_verbose == 2 ) then
    write(mdout,'(/,A,I8)') 'Finding gaussian monopoles', num_atoms
    write(mdout,'(5E16.8)') covalent_monop(1:num_atoms)
  end if

  dim1 = 1
  call POL_GAUSS_read_real_list_data('POL_GAUSS_RADII_',nf, &
                 dim1,num_atoms,b_polar)
  if (pol_gauss_verbose == 2 ) then
    write(mdout,'(/,A,I8)') 'Finding gaussian radii', num_atoms
    write(mdout,'(5E16.8)') b_polar(1:num_atoms)
  end if

  ! b_polar is read in as radii
  ! preprocess radius and beta for efficient combination
  r_gauss(1:num_atoms) = 2.0d0*b_polar(1:num_atoms)**2 ! this is to be consistent with pyresp
  !r_gauss(1:num_atoms) = 1.0d0*b_polar(1:num_atoms)**2 ! this is the original condition used for MD testing.
  !r_polar(1:num_atoms) = 1.0d0/b_polar(1:num_atoms)

  min_r_gauss = r_gauss(1)
  do n = 1, num_atoms
    min_r_gauss = min(min_r_gauss, r_gauss(n))
  end do

  betamax = sqrt(2.0d0)/sqrt((2.0d0 * min_r_gauss))

  gbl_perm_chg(1:num_atoms) = covalent_monop(1:num_atoms)

#if 0
  call pGM_MPOLE_get_start_end_lists()
#endif

  pGM_MPOLE_readparm = 1

end function pGM_MPOLE_readparm
#if 0
!---------------------------------------------------------
subroutine pGM_MPOLE_get_start_end_lists()

!  use pol_gauss_multipoles_data, only : num_multipoles, my_mpole_cnt, start_multipoles, end_multipoles
!  use pol_gauss_readparm_mod, only : POL_GAUSS_get_startlist_endlist
  call POL_GAUSS_get_startlist_endlist(num_multipoles,  &
                             start_multipoles, end_multipoles, my_mpole_cnt)

end subroutine pGM_MPOLE_get_start_end_lists
#endif
!---------------------------------------------------------
#ifdef MPI
subroutine pGM_MPOLE_bcast(natom)

  use parallel_dat_mod

  implicit none

  integer :: natom
  integer ier

  call mpi_bcast(betamax, 1, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
  call mpi_bcast(ncovalent, 1, MPI_INTEGER, 0, pmemd_comm, ier)
 
if (.not.master) then

  num_multipoles = natom                       ! this should have been broadcast already
#if 0
  call pGM_MPOLE_get_start_end_lists()
#endif

  allocate(covalent_ptr(natom+1),           &
           covalent_monop(natom),           &  ! charges
           b_polar(natom),                  &  ! gaussian multipole radius
           gbl_perm_chg(natom),             &  ! permanent charges
           gbl_perm_dip(3,natom),           &  ! permanent dipoles in cartesian frame
           r_gauss(natom),                  &  ! stores 2*b_polar**2
           covalent_atm(1:ncovalent), &
           covalent_dip(1:ncovalent), &
           stat=ier)
  REQUIRE(ier==0)
end if
 
  call mpi_bcast(covalent_ptr, natom + 1, MPI_INTEGER, 0, pmemd_comm, ier)
  call mpi_bcast(covalent_monop, natom, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
  call mpi_bcast(b_polar, natom, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
  call mpi_bcast(r_gauss, natom, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
  call mpi_bcast(covalent_atm, ncovalent, MPI_INTEGER, 0, pmemd_comm, ier)
  call mpi_bcast(covalent_dip, ncovalent, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)

  gbl_perm_chg = covalent_monop           ! They remain constants. No need to copy them every time step.

end subroutine pGM_MPOLE_bcast

!---------------------------------------------------------
subroutine pGM_add_perm_dip_pairs(used_img_cnt, used_img_lst, used_img_mask, img_atm_map, atm_img_map)

  implicit none

  integer, intent(inout) :: used_img_cnt
  integer, intent(inout) :: used_img_lst(*)
  integer(byte), intent(inout) :: used_img_mask(*)
  integer, intent(in)    :: img_atm_map(*)
  integer, intent(in)    :: atm_img_map(*)

  integer :: ier
  integer :: i
  integer :: iatm
  integer :: img_i
  integer :: img_j
  integer :: jatm
  integer :: jbond
  integer :: start_bonds
  integer :: end_bonds

  integer :: used_img_cnt_stk

  used_img_cnt_stk = used_img_cnt

  do i = 1, used_img_cnt_stk

    img_i = used_img_lst(i)
    iatm  = img_atm_map(img_i)

    start_bonds = covalent_ptr(iatm) + 1
    end_bonds = covalent_ptr(iatm+1)

    do jbond = start_bonds, end_bonds
      jatm = covalent_atm(jbond)
      img_j = atm_img_map(jatm)
      if (used_img_mask(img_j) .eq. 0) then
        used_img_mask(img_j) = 1
        used_img_cnt = used_img_cnt + 1
        used_img_lst(used_img_cnt) = img_j
      end if
    enddo
  enddo

end subroutine pGM_add_perm_dip_pairs
#endif
!---------------------------------------------------------
end module pol_gauss_multipoles_mod

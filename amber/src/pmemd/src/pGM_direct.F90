#include "pGM/copyright_pgm.i"
!
! Adopted from sander pGM code by YD
!
module pol_gauss_direct_data

  implicit none

  integer, save                       :: num_tensor, num_tensor_short
  integer, save, allocatable          :: dipole_dipole_ptrs(:)
  integer, save, allocatable          :: last_ee_ptrs(:)
  integer, save, allocatable          :: dipole_dipole_list(:)
  integer, save, allocatable          :: dipole_dipole_short_ptrs(:)
  integer, save, allocatable          :: dipole_dipole_list_short(:)
  double precision, save, allocatable :: delvec_save(:,:)   ! delx, dely, delz, delrinv
  double precision, save, allocatable :: Bfact_save(:,:)    ! B(0), B(1), B(2), B(3)
  double precision, save, allocatable :: displacement(:,:)
  double precision, save, allocatable :: displacement_img(:,:)
  double precision, save, allocatable :: dipole_dipole_tensor(:,:)
  double precision, save, allocatable :: dipole_dipole_tensor_short(:,:)
  double precision, parameter         :: safety = 1.25d0, checklist = 0.9d0

contains

!-------------------------------------------------------
subroutine pGM_DIRECT_dip_dip_field(ind_dip,dip_field)

!  use pol_gauss_direct_data, only : num_tensor, dipole_dipole_list, dipole_dipole_tensor, &
!                                    dipole_dipole_ptrs

#ifdef MPI
  use parallel_dat_mod
#endif
  implicit none

  double precision,intent(in) :: ind_dip(3,*)
  double precision,intent(inout) :: dip_field(3,*)

  call pGM_DIRECT_calc_dipdip_field(num_tensor, &
           dipole_dipole_list,dipole_dipole_tensor, &
           ind_dip,dip_field, dipole_dipole_ptrs)

end subroutine pGM_DIRECT_dip_dip_field
!------------------------------------------------------- 
subroutine pGM_DIRECT_dip_dip_field_short(ind_dip,dip_field)

!  use pol_gauss_direct_data, only : num_tensor_short, dipole_dipole_list_short, &
!                                    dipole_dipole_tensor_short, dipole_dipole_short_ptrs
#ifdef MPI
  use parallel_dat_mod
#endif

  implicit none

  double precision,intent(in) :: ind_dip(3,*)
  double precision,intent(inout) :: dip_field(3,*)

  integer n, i, j

  call pGM_DIRECT_calc_dipdip_field(num_tensor_short, &
           dipole_dipole_list_short,dipole_dipole_tensor_short, &
           ind_dip,dip_field, dipole_dipole_short_ptrs)

end subroutine pGM_DIRECT_dip_dip_field_short
!-------------------------------------------------------
subroutine pGM_DIRECT_calc_dipdip_field(num_tensor, &
               dipole_dipole_list,dipole_dipole_tensor, &
               ind_dip,dip_field, dipole_dipole_ptrs)

  use img_mod
#ifdef MPI
  use parallel_dat_mod
#endif

  implicit none

  integer,intent(in)              :: num_tensor
  integer,intent(in)              :: dipole_dipole_list(*)
  double precision,intent(in)     :: dipole_dipole_tensor(6,*)
  double precision,intent(in)     :: ind_dip(3,*)
  double precision,intent(inout)  :: dip_field(3,*)
  integer,intent(in)              :: dipole_dipole_ptrs(*)

  double precision                :: fld_x, fld_y, fld_z
  integer :: i,j,n

  ! again we are computing grad of phi, not field
  ! minus signs due to deriv wrt position of i

#if 0
  do n = 1, num_tensor
    i = dipole_dipole_list(1,n)
#endif
  do i = my_img_lo, my_img_hi
    fld_x = 0.d0
    fld_y = 0.d0
    fld_z = 0.d0
    do n = dipole_dipole_ptrs(i), dipole_dipole_ptrs(i+1) - 1
!      j = dipole_dipole_list(2,n)
      j = dipole_dipole_list(n)
      fld_x = fld_x -  &
         ( dipole_dipole_tensor(1,n)*ind_dip(1,j) + &
           dipole_dipole_tensor(2,n)*ind_dip(2,j) + &
           dipole_dipole_tensor(3,n)*ind_dip(3,j) )
      fld_y = fld_y -  &
         ( dipole_dipole_tensor(2,n)*ind_dip(1,j) + &
           dipole_dipole_tensor(4,n)*ind_dip(2,j) + &
           dipole_dipole_tensor(5,n)*ind_dip(3,j) )
      fld_z = fld_z -  &
         ( dipole_dipole_tensor(3,n)*ind_dip(1,j) + &
           dipole_dipole_tensor(5,n)*ind_dip(2,j) + &
           dipole_dipole_tensor(6,n)*ind_dip(3,j) )
      dip_field(1,j) = dip_field(1,j) -  &
         ( dipole_dipole_tensor(1,n)*ind_dip(1,i) + &
           dipole_dipole_tensor(2,n)*ind_dip(2,i) + &
           dipole_dipole_tensor(3,n)*ind_dip(3,i) )
      dip_field(2,j) = dip_field(2,j) -  &
         ( dipole_dipole_tensor(2,n)*ind_dip(1,i) + &
           dipole_dipole_tensor(4,n)*ind_dip(2,i) + &
           dipole_dipole_tensor(5,n)*ind_dip(3,i) )
      dip_field(3,j) = dip_field(3,j) -  &
         ( dipole_dipole_tensor(3,n)*ind_dip(1,i) + &
           dipole_dipole_tensor(5,n)*ind_dip(2,i) + &
           dipole_dipole_tensor(6,n)*ind_dip(3,i) )
    end do
    dip_field(1,i) = dip_field(1,i) + fld_x
    dip_field(2,i) = dip_field(2,i) + fld_y
    dip_field(3,i) = dip_field(3,i) + fld_z
  end do
end subroutine pGM_DIRECT_calc_dipdip_field
!-------------------------------------------------------------------------------
end module pol_gauss_direct_data

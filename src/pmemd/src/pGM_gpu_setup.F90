!===============================================================================
!
! pGM_gpu_setup.F90 - Wrapper to initialize PGM GPU code
!
! This subroutine is called from pGM_readparm when CUDA is enabled.
! It initializes the GPU context and uploads all PGM parameters.
!
!===============================================================================

subroutine gpu_pgm_setup(ipgm, natom, covalent_ptr, covalent_atm, &
                         covalent_monop, covalent_dip, &
                         b_polar, r_gauss, polarizability)

#ifdef CUDA
   use pGM_gpu_interface_mod
#endif

   implicit none

   ! Arguments
   integer, intent(in) :: ipgm
   integer, intent(in) :: natom
   integer, intent(in) :: covalent_ptr(natom+1)
   integer, intent(in) :: covalent_atm(*)
   real(8), intent(in) :: covalent_monop(natom)
   real(8), intent(in) :: covalent_dip(*)
   real(8), intent(in) :: b_polar(natom)
   real(8), intent(in) :: r_gauss(natom)
   real(8), intent(in) :: polarizability(natom)

#ifdef CUDA
   integer :: num_bonds

   ! Only initialize if PGM is enabled
   if (ipgm /= 1) return

   ! Initialize GPU context
   call pGM_GPU_init(natom)

   ! Upload permanent multipoles (charges, Gaussian radii, polarizabilities)
   call pGM_GPU_upload_multipoles(natom, covalent_monop, r_gauss, polarizability)

   ! Calculate total number of bonds
   num_bonds = covalent_ptr(natom+1) - 1

   ! Upload bond topology for local-frame dipoles
   if (num_bonds > 0) then
      call pGM_GPU_upload_bonds(natom, covalent_ptr, covalent_atm, &
                                 covalent_dip, num_bonds)
   end if

   ! Note: Neighbor list will be uploaded later during runtime
   ! Note: Parameters (ewald coeff, tolerances, etc.) set separately via pGM_GPU_set_params

#endif

end subroutine gpu_pgm_setup

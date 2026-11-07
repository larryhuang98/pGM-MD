!===============================================================================
!
! pGM_gpu_interface.F90 - Fortran interface to PGM CUDA GPU routines
!
! This module provides the interface between Fortran PGM code and CUDA kernels.
! All GPU-related calls go through this module.
!
! Production functions:
!   pGM_GPU_init            - Initialize PGM GPU context
!   pGM_GPU_set_params      - Set SCF solver parameters
!   pGM_GPU_upload_multipoles - Upload charges, r_gauss, polarizability
!   pGM_GPU_upload_bonds    - Upload bond CSR topology for local-frame dipoles
!   pGM_GPU_upload_prmtop   - Store prmtop data in host memory for NL remap
!   pGM_GPU_cleanup         - Free GPU resources
!
!===============================================================================

module pGM_gpu_interface_mod

  use, intrinsic :: iso_c_binding
  implicit none
  private

  ! Public interfaces
  public :: pGM_GPU_init
  public :: pGM_GPU_set_params
  public :: pGM_GPU_set_pcg_params
  public :: pGM_GPU_upload_multipoles
  public :: pGM_GPU_upload_bonds
  public :: pGM_GPU_upload_prmtop
  public :: pGM_GPU_cleanup

  ! Flag to track if GPU is initialized
  logical, save :: gpu_initialized = .false.

  ! C interface declarations
  interface
    subroutine pgm_gpu_init_c(natom) bind(C, name='pgm_gpu_init_')
      use, intrinsic :: iso_c_binding
      integer(c_int), intent(in) :: natom
    end subroutine pgm_gpu_init_c

    subroutine pgm_gpu_set_params_c(ew_coeff, tol, max_iter, solver, precond) &
               bind(C, name='pgm_gpu_set_params_')
      use, intrinsic :: iso_c_binding
      real(c_double), intent(in) :: ew_coeff
      real(c_double), intent(in) :: tol
      integer(c_int), intent(in) :: max_iter
      integer(c_int), intent(in) :: solver
      integer(c_int), intent(in) :: precond
    end subroutine pgm_gpu_set_params_c

    subroutine pgm_gpu_upload_multipoles_c(charge, rGauss, pol) &
               bind(C, name='pgm_gpu_upload_multipoles_')
      use, intrinsic :: iso_c_binding
      real(c_double), intent(in) :: charge(*)
      real(c_double), intent(in) :: rGauss(*)
      real(c_double), intent(in) :: pol(*)
    end subroutine pgm_gpu_upload_multipoles_c

    subroutine pgm_gpu_upload_bonds_c(bondPtr, bondAtom, bondDip, numBonds) &
               bind(C, name='pgm_gpu_upload_bonds_')
      use, intrinsic :: iso_c_binding
      integer(c_int), intent(in) :: bondPtr(*)
      integer(c_int), intent(in) :: bondAtom(*)
      real(c_double), intent(in) :: bondDip(*)
      integer(c_int), intent(in) :: numBonds
    end subroutine pgm_gpu_upload_bonds_c

    subroutine pgm_gpu_upload_prmtop_c(perm_chg, r_gauss, pol, scf_tol, &
                                        scf_maxiter, natom) &
               bind(C, name='pgm_gpu_upload_prmtop_')
      use, intrinsic :: iso_c_binding
      real(c_double), intent(in) :: perm_chg(*), r_gauss(*), pol(*)
      real(c_double), intent(in) :: scf_tol
      integer(c_int), intent(in) :: scf_maxiter, natom
    end subroutine pgm_gpu_upload_prmtop_c

    subroutine pgm_gpu_set_pcg_params_c(solv_opt, local_niter, local_cut, sor_coeff) &
               bind(C, name='pgm_gpu_set_pcg_params_')
      use, intrinsic :: iso_c_binding
      integer(c_int), intent(in) :: solv_opt, local_niter
      real(c_double), intent(in) :: local_cut, sor_coeff
    end subroutine pgm_gpu_set_pcg_params_c

    subroutine pgm_gpu_cleanup_c() bind(C, name='pgm_gpu_cleanup_')
    end subroutine pgm_gpu_cleanup_c
  end interface

contains

!-------------------------------------------------------------------------------
! Initialize PGM GPU context
!-------------------------------------------------------------------------------
subroutine pGM_GPU_init(natom)

  implicit none
  integer, intent(in) :: natom

  call pgm_gpu_init_c(natom)
  gpu_initialized = .true.

end subroutine pGM_GPU_init

!-------------------------------------------------------------------------------
! Set PGM parameters on GPU
!-------------------------------------------------------------------------------
subroutine pGM_GPU_set_params(ew_coeff, dipole_scf_tol, scf_max_iter, &
                               scf_solv_opt, scf_local_niter)

  implicit none
  real(8), intent(in) :: ew_coeff
  real(8), intent(in) :: dipole_scf_tol
  integer, intent(in) :: scf_max_iter
  integer, intent(in) :: scf_solv_opt
  integer, intent(in) :: scf_local_niter

  call pgm_gpu_set_params_c(ew_coeff, dipole_scf_tol, scf_max_iter, &
                            scf_solv_opt, scf_local_niter)

end subroutine pGM_GPU_set_params

!-------------------------------------------------------------------------------
! Set PCG-specific parameters on GPU
!-------------------------------------------------------------------------------
subroutine pGM_GPU_set_pcg_params(solv_opt, local_niter, local_cut, sor_coeff)

  implicit none
  integer, intent(in) :: solv_opt, local_niter
  real(8), intent(in) :: local_cut, sor_coeff

  call pgm_gpu_set_pcg_params_c(solv_opt, local_niter, local_cut, sor_coeff)

end subroutine pGM_GPU_set_pcg_params

!-------------------------------------------------------------------------------
! Upload permanent multipole data to GPU
!-------------------------------------------------------------------------------
subroutine pGM_GPU_upload_multipoles(natom, charge, rGauss, pol)

  implicit none
  integer, intent(in) :: natom
  real(8), intent(in) :: charge(natom)
  real(8), intent(in) :: rGauss(natom)
  real(8), intent(in) :: pol(natom)

  call pgm_gpu_upload_multipoles_c(charge, rGauss, pol)

end subroutine pGM_GPU_upload_multipoles

!-------------------------------------------------------------------------------
! Upload bond topology for local-frame dipoles
!-------------------------------------------------------------------------------
subroutine pGM_GPU_upload_bonds(natom, bondPtr, bondAtom, bondDip, numBonds)

  implicit none
  integer, intent(in) :: natom
  integer, intent(in) :: bondPtr(natom+1)
  integer, intent(in) :: bondAtom(*)
  real(8), intent(in) :: bondDip(*)
  integer, intent(in) :: numBonds

  call pgm_gpu_upload_bonds_c(bondPtr, bondAtom, bondDip, numBonds)

end subroutine pGM_GPU_upload_bonds

!-------------------------------------------------------------------------------
! Upload PGM prmtop data to host storage (called once during init)
!-------------------------------------------------------------------------------
subroutine pGM_GPU_upload_prmtop(natoms, perm_chg, r_gauss, pol, scf_tol, scf_maxiter)

  implicit none
  integer, intent(in) :: natoms, scf_maxiter
  real(8), intent(in) :: perm_chg(natoms)
  real(8), intent(in) :: r_gauss(natoms)
  real(8), intent(in) :: pol(natoms)
  real(8), intent(in) :: scf_tol

  call pgm_gpu_upload_prmtop_c(perm_chg, r_gauss, pol, scf_tol, scf_maxiter, natoms)

end subroutine pGM_GPU_upload_prmtop

!-------------------------------------------------------------------------------
! Cleanup GPU resources
!-------------------------------------------------------------------------------
subroutine pGM_GPU_cleanup()

  implicit none

  if (gpu_initialized) then
    call pgm_gpu_cleanup_c()
    gpu_initialized = .false.
  end if

end subroutine pGM_GPU_cleanup

end module pGM_gpu_interface_mod

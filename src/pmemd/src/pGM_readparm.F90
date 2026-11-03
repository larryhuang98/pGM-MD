#include "pGM/copyright_pgm.i"
! 
! Adopted from sander pGM code by YD
!
!-------------------------------------------------------------------------------
subroutine POL_GAUSS_readparm(nf, natom)

   use mdin_ctrl_dat_mod, only : ipgm
   use pol_gauss_mdin_dat_mod, only : igvdw
   use prmtop_dat_mod, only : nphih, nphia
   use file_io_dat_mod, only : mdout
   use pmemd_lib_mod, only : mexit
   use pol_gauss_induced, only : pGM_INDUCED_readparm, polarizability
   use pol_gauss_multipoles_mod, only : pGM_MPOLE_readparm, covalent_monop, covalent_ptr, &
                                        covalent_atm, covalent_dip, r_gauss, b_polar

#ifdef MPI
   use parallel_dat_mod
#endif

   implicit none

   integer, intent(in) :: nf, natom
   integer :: mpole_valid, polar_valid
   integer :: ier, tmp1, tmp2

#ifdef MPI
if (master) then
#endif

   call POL_GAUSS_check_parm_legal(nf, ipgm)

   ! GVDW is implemented for the nonbonded pair list only; 1-4 pairs would be
   ! evaluated with the Lennard-Jones form in the bonded code, so stop for now.
   if (igvdw .eq. 1 .and. (nphih + nphia) .gt. 0) then
      write(mdout,'(a)') ' pGM: igvdw = 1 (Gaussian VDW) is not yet supported for topologies'
      write(mdout,'(a)') '      with dihedrals (1-4 pairs); use igvdw = 0 (Lennard-Jones).'
      call mexit(mdout, 1)
   end if

   mpole_valid = pGM_MPOLE_readparm(nf, natom)

   polar_valid = pGM_INDUCED_readparm(nf, natom)

! This is now called from init_amber_prmtop_dat() in prmtop_dat.F90
!   call pGM_keep_nsp(nspm, atm_nsp)    ! need this to calculate the true mol-virial

#ifdef MPI
endif
#else
tmp1 = 0
tmp2 = 0
#ifdef GTI
  tmp1 = 1
#endif
#ifdef CUDA
  tmp2 = 1
#endif
! OLD GPU SETUP - commented out; GPU init is handled via pGM_GPU_init in pGM_interface.F90
!#ifdef CUDA
!   call gpu_pgm_setup(ipgm, natom, covalent_ptr, covalent_atm, covalent_monop, covalent_dip, &
!                      b_polar, r_gauss, polarizability)
!#endif
#endif

contains

!-------------------------------------------------------------------------------
subroutine POL_GAUSS_check_parm_legal(nf, ipgm)

   use file_io_dat_mod, only : mdout

  implicit none

  integer,intent(in) :: nf, ipgm

  integer :: iok,ionerr,indicator
  character(len=80) :: fmt,fmtin,dtype

#ifdef MPI
  if (master) then
#endif
  fmtin = '(I5)'
  dtype = 'POL_GAUSS_FORCEFIELD'
  ionerr = 1 ! not fatal if missing
  call nxtsec(nf, mdout, ionerr, fmtin, dtype, fmt, iok)
  if ( iok == 0 )then !this data type found in prmtop
    read(nf,fmt)indicator
    if ( ipgm /= 1 )then
       write(mdout,*)'POL_GAUSS style prmtop, but pgm NOT set to one'
       call mexit(mdout,1)
    endif
    return ! pGM prmtop, pGM set to 1
  else
    if ( ipgm == 1 )then
       write(mdout,*)'NOT an POL_GAUSS style prmtop, but pgm IS set to one'
       call mexit(mdout,1)
    endif
    return ! NON pGM prmtop, pGM NOT set to 1
  endif
#ifdef MPI
  end if
#endif
end subroutine POL_GAUSS_check_parm_legal
!----------------------------------------------------------
end subroutine POL_GAUSS_readparm

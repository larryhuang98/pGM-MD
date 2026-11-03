#include "pGM/copyright_pgm.i"
!-------------------------------------------------------------------------------
subroutine POL_GAUSS_read_mdin(mdin, ipgm0)

  use mdin_ctrl_dat_mod, only : ipgm
  use file_io_mod,     only : nmlsrc
  use file_io_dat_mod, only : mdout
  use pol_gauss_mdin_dat_mod

  implicit none

  integer,intent(in) :: mdin, ipgm0
  integer            :: err, ifind

!  integer :: do_recip=1, do_direct=1, do_self=1, &
!             do_vdw=1,   do_induced=1

  namelist/pol_gauss/do_pol_gauss_nonbond, &
                     use_average, dipole_scf_init, dipole_scf_init_step, dipole_scf_init_order, &
                     scf_solv_opt, scf_sor_niter, scf_cg_niter, scf_local_niter, &
                     pol_gauss_verbose, dipole_scf_use_cg, &
                     dipole_scf_tol, scf_sor_coefficient, scf_local_cut, ee_dsum_cut, ee_damped_cut, &
                     erfc_tol, ee_gauss_cut, dipole_print, densitycontrol, &
                     pgm_dump_enable, pgm_dump_stride, pgm_dump_first, pgm_dump_last, &
                     pgm_dump_b, pgm_dump_mu, pgm_dump_crd, pgm_dump_op, pgm_dump_short, &
                     pgm_dump_hist, pgm_dump_hist_kmax, pgm_dump_frames_n, pgm_dump_frames, &
                     pgm_dump_dir

  err = 0

  do_pol_gauss_nonbond=1
  use_average=0
  dipole_print = 0
  saved_nprint = 0 ! RL: Turning on dipole printing
  ! RL: diagnostic dump defaults
  pgm_dump_enable = 0
  pgm_dump_stride = 1
  pgm_dump_first  = 0
  pgm_dump_last   = 0
  pgm_dump_b = 0
  pgm_dump_mu = 0
  pgm_dump_crd = 0
  pgm_dump_op = 0
  pgm_dump_short = 0
  pgm_dump_hist = 0
  pgm_dump_hist_kmax = 50
  pgm_dump_frames_n = 0
  pgm_dump_frames = 0
  pgm_dump_dir = '.'
  dipole_scf_init=3
  dipole_scf_init_step = 2
  dipole_scf_init_order=3
  scf_solv_opt = 2
  scf_sor_niter = 100
  scf_cg_niter = 25
  scf_local_niter = 3
  pol_gauss_verbose = 0
  dipole_scf_use_cg = 0

  densitycontrol  = -1.0

#if 0
  do_recip=1
  do_direct=1
  do_self=1
  do_vdw=1
  do_induced=1
#endif

  mdin_pol_gauss = .false.

  dipole_scf_tol = 0.0002d0
  scf_sor_coefficient = 0.65d0
  scf_local_cut=3.0d0
  ee_dsum_cut=7.d0
  ee_damped_cut=4.5d0
  erfc_tol = 1.0e-5
  ee_gauss_cut = 3.12342d0 ! corresponding to max error of 10^-5 in erfc

! Need to add a check here

  if (ipgm .ne. 1) return

  ! Read the pol_gauss namelist

  call nmlsrc('pol_gauss', mdin, ifind)

  mdin_pol_gauss = (ifind /= 0)

  if ( mdin_pol_gauss ) then

    read(mdin, nml=pol_gauss)

  else
    write(mdout,*) ' ipgm is set but the &pol_gauss namelist was not found'
    err = err + 1     ! quit for now
  end if

  write(mdout,'(A)')'|'
  write(mdout,'(A)')'| pGM-related parameters:'
  write(mdout,'(A)')'|'
  write(mdout,'(A,I5)')   '| ipgm =                  ', ipgm
  write(mdout,'(A,I5)')   '| pol_gauss_verbose =     ', pol_gauss_verbose
  write(mdout,'(A,I5)')   '| do_pol_gauss_nonbond =  ', do_pol_gauss_nonbond
  write(mdout,'(A,I5)')   '| use_average =           ', use_average
  write(mdout,'(A,I5)')   '| dipole_scf_init =       ', dipole_scf_init
  write(mdout,'(A,I5)')   '| dipole_scf_init_step =  ', dipole_scf_init_step
  write(mdout,'(A,I5)')   '| dipole_scf_init_order = ', dipole_scf_init_order
  write(mdout,'(A,I5)')   '| scf_solv_opt =          ', scf_solv_opt
  write(mdout,'(A,I5)')   '| scf_sor_niter =         ', scf_sor_niter
  write(mdout,'(A,I5)')   '| scf_cg_niter =          ', scf_cg_niter
  write(mdout,'(A,I5)')   '| scf_local_niter =       ', scf_local_niter
  write(mdout,'(A,I5)')   '| dipole_scf_use_cg =     ', dipole_scf_use_cg
  write(mdout,'(A,I5)')   '| dipole_print =          ', dipole_print
  write(mdout,'(A,I5)')   '| saved_nprint =          ', saved_nprint
  write(mdout,'(A)')'| --- pGM diagnostic dumps (serial pmemd.pgm only) ---'
  write(mdout,'(A,I5)')   '| pgm_dump_enable =       ', pgm_dump_enable
  write(mdout,'(A,I5)')   '| pgm_dump_stride =       ', pgm_dump_stride
  write(mdout,'(A,2I7)')  '| pgm_dump_first/last =   ', pgm_dump_first, pgm_dump_last
  write(mdout,'(A,6I4)')  '| pgm_dump b/mu/crd/op/short/hist =', &
                          pgm_dump_b, pgm_dump_mu, pgm_dump_crd, pgm_dump_op, pgm_dump_short, pgm_dump_hist
  write(mdout,'(A,I5)')   '| pgm_dump_hist_kmax =    ', pgm_dump_hist_kmax
  write(mdout,'(A,I5)')   '| pgm_dump_frames_n =     ', pgm_dump_frames_n
  write(mdout,'(A,A)')    '| pgm_dump_dir =          ', trim(pgm_dump_dir)
  write(mdout,'(A,E15.7)')'| ee_dsum_cut =           ', ee_dsum_cut
  write(mdout,'(A,E15.7)')'| ee_damped_cut =         ', ee_damped_cut
  write(mdout,'(A,E15.7)')'| dipole_scf_tol =        ', dipole_scf_tol
  write(mdout,'(A,E15.7)')'| scf_sor_coefficient =   ', scf_sor_coefficient
  write(mdout,'(A,E15.7)')'| scf_local_cut =         ', scf_local_cut
  write(mdout,'(A,E15.7)')'| erfc_tol =              ', erfc_tol
  write(mdout,'(A,E15.7)')'| ee_gauss_cut =          ', ee_gauss_cut
  write(mdout,'(A,E15.7)')'| densitycontrol =        ', densitycontrol
  write(mdout,'(A)')'| '

! Logically, legality check of input data should follow reading

  if ( dipole_scf_init_order > 3 ) then
    write(mdout,'(/,a)') ' pGM_Induced_eval: dipole_scf_init_order too high'
    err = err + 1
  end if

  if ( dipole_scf_init_step > 8 ) then
    write(mdout,'(/,a)') ' pGM_Induced_eval: dipole_scf_init_step too high'
    err = err + 1
  end if

  if ( err .ne. 0 ) then
    call mexit(mdout,1)
  end if

  ! RL: Turning it back
  !if ( dipole_print > 0 ) then
  !  write(mdout,'(/,a)') ' WARNING: dipole_print is ignored'
  !end if

  mdin_pol_gauss = .true.

end subroutine POL_GAUSS_read_mdin
!---------------------------------------------------------

!----------------------------------------------------------------------------------------------
! write_pgm_citation: write the pGM citations so that users can give us press.
!
! Arguments:
!   iout_unit:  unit to write output to (master_setup supplies the value mdout)
!----------------------------------------------------------------------------------------------
subroutine write_pgm_citation(iout_unit)

  implicit none

  ! Formal arguments
  integer, intent(in) :: iout_unit

  write(iout_unit,'(a)') '|----------------- CITATION INFORMATION -----------------'
  write(iout_unit,'(a)') '|'
  write(iout_unit,'(a)') '|    When publishing work that utilized the pGM'
  write(iout_unit,'(a)') '|    polarizable force field option, please cite the'
  write(iout_unit,'(a)') '|    following in addition to the regular AMBER citations:'
  write(iout_unit,'(a)') '|'
  write(iout_unit,'(a)') '|  - Huang, Z.; Wu, Y.; Duan, Y.; Luo, R.'
  write(iout_unit,'(a)') '|    J. Chem. Theory Comput., 2025, 21(2), 847-858.'
  write(iout_unit,'(a)') '|    DOI: 10.1021/acs.jctc.4c01368'
  write(iout_unit,'(a)') '|'
  write(iout_unit,'(a)') '|  - Huang, Z.; Zhao, S.J.; Cieplak, P.; Duan, Y.;'
  write(iout_unit,'(a)') '|    Luo, R.; Wei, H.X.'
  write(iout_unit,'(a)') '|    J. Chem. Theory Comput., 2023, 19(15), 5047-5057.'
  write(iout_unit,'(a)') '|    DOI: 10.1021/acs.jctc.3c00226'
  write(iout_unit,'(a)') '|'
  write(iout_unit,'(a)') '|  - Wei, H.X.; Cieplak, P.; Duan, Y.; Luo, R.'
  write(iout_unit,'(a)') '|    J. Chem. Phys., 2022, 156(11), 114114.'
  write(iout_unit,'(a)') '|    DOI: 10.1063/5.0082548'
  write(iout_unit,'(a)') '|'
  write(iout_unit,'(a)') '|  - Wei, H.X.; Qi, R.X.; Wang, J.M.; Cieplak, P.;'
  write(iout_unit,'(a)') '|    Duan, Y.; Luo, R.'
  write(iout_unit,'(a)') '|    J. Chem. Phys., 2020, 153(11), 114116.'
  write(iout_unit,'(a)') '|    DOI: 10.1063/5.0019560'
  write(iout_unit,'(a)') '|'
  write(iout_unit,'(a)') '|    For inquiries regarding implementation or'
  write(iout_unit,'(a)') '|    adaptation, please contact:'
  write(iout_unit,'(a)') '|    Yong Duan <duan@ucdavis.edu>'
  write(iout_unit,'(a)') '|'
  write(iout_unit,'(a)') '|--------------------------------------------------------'
  write(iout_unit,'(a)') ' '

  return

end subroutine write_pgm_citation


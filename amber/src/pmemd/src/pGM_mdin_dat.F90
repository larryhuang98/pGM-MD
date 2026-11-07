#include "pGM/copyright_pgm.i"
!
! Adopted from sander pGM code by YD
!
module pol_gauss_mdin_dat_mod

  implicit none

  public

! pGM control parameters

! For MPI, all these parameters need to be in common blocks, one for integers and the other for real

  integer                :: do_pol_gauss_nonbond=1
  integer                :: use_average=0
  integer                :: dipole_scf_init=2
  integer                :: dipole_scf_init_step = 2
  integer                :: dipole_scf_init_order=3
  integer                :: scf_solv_opt = 3
  integer                :: scf_sor_niter = 100
  integer                :: scf_cg_niter = 25
  integer                :: scf_local_niter = 3
  integer                :: pol_gauss_verbose = 0
  integer                :: dipole_scf_use_cg = 0

  integer                :: nspm_lcl = 0

  logical, save          :: mdin_pol_gauss = .false.
                      
  double precision       :: dipole_scf_tol = 0.0002d0
  double precision       :: scf_sor_coefficient = 0.65d0
  double precision       :: scf_local_cut=3.0d0
  double precision       :: ee_dsum_cut=7.d0
  double precision       :: ee_damped_cut=4.5d0
  double precision       :: erfc_tol = 1.0e-5
  double precision       :: ee_gauss_cut = 3.12342d0 ! corresponding to max error of 10^-5 in erfc
  double precision       :: densitycontrol = -1.0d0

  ! van der Waals model for pGM: igvdw = 0, Lennard-Jones 12-6 (default);
  ! igvdw = 1, Gaussian VDW with per-type A_rep (topology ACOEF) and C6 (topology BCOEF)
  integer                :: igvdw = 0
  integer                :: gvdw_rep_form = 0      ! 0: A exp(-b kappa^2 r^2); 1: A exp(-b kappa r)
  double precision       :: b_rep_scale = 1.0d0

  ! GPU induction solver (pmemd.pgm.cuda only)
  integer                :: scf_gpu_pred = 1       ! 0 quadratic, 1 cubic + combined field (default), 2 MOE
  integer                :: scf_gpu_check = 3      ! CG iterations between convergence tests (warm start)
  integer                :: scf_gpu_overlap = 1    ! concurrent direct/reciprocal space evaluation
  integer                :: pgm_power = 0          ! >0: print eq-12 induction power every pgm_power steps

  ! RL: Turning on dipole printing
  integer                :: dipole_print = 0
  integer                :: saved_nprint = 0

  ! RL: diagnostic dump flags for the subspace-recycling study.
  ! Serial-only diagnostic feature: NOT placed in the MPI common blocks,
  ! so they are only meaningful for the serial pmemd.pgm build.
  integer                :: pgm_dump_enable   = 0   ! master switch (0 = no dumps)
  integer                :: pgm_dump_stride   = 1   ! dump every N-th *evaluated* step
  integer                :: pgm_dump_first    = 0   ! first nstep to dump (0 = from start)
  integer                :: pgm_dump_last     = 0   ! last  nstep to dump (0 = no upper bound)
  integer                :: pgm_dump_b        = 0   ! dump RHS b (from the b-snapshot)
  integer                :: pgm_dump_mu       = 0   ! dump converged mu* (binary)
  integer                :: pgm_dump_crd      = 0   ! dump image coordinates + box
  integer                :: pgm_dump_op       = 0   ! dump dense operator A_code (expensive)
  integer                :: pgm_dump_short    = 0   ! dump short-range tensor (to rebuild M)
  integer                :: pgm_dump_hist     = 0   ! dump PCG per-iter history
  integer                :: pgm_dump_hist_kmax = 50 ! cap on retained Krylov/search vectors
  integer, parameter     :: pgm_dump_frames_max = 128
  integer                :: pgm_dump_frames_n = 0   ! number of explicit nstep entries below
  integer                :: pgm_dump_frames(pgm_dump_frames_max) = 0
  character(len=256)     :: pgm_dump_dir = '.'      ! output directory for binary dumps
  integer, save          :: pgm_curr_nstep = 0      ! current MD step (set from pme_force)

  common / pgm_ctrl_int /nspm_lcl, do_pol_gauss_nonbond, dipole_scf_use_cg, scf_sor_niter, scf_cg_niter, & !5
            pol_gauss_verbose, scf_solv_opt, scf_local_niter, dipole_scf_init_step, dipole_scf_init, & !10
            dipole_scf_init_order, use_average, dipole_print, saved_nprint, & !14 ! RL
            igvdw, gvdw_rep_form, scf_gpu_pred, scf_gpu_check, scf_gpu_overlap, pgm_power !20

  save :: / pgm_ctrl_int /

  common / pgm_ctrl_dbl /dipole_scf_tol, scf_sor_coefficient, scf_local_cut, ee_dsum_cut, ee_damped_cut, & ! 5
            erfc_tol, ee_gauss_cut, densitycontrol, b_rep_scale   !9

  save :: / pgm_ctrl_dbl /

  integer, parameter    :: pgm_ctrl_int_cnt = 20
  integer, parameter    :: pgm_ctrl_dbl_cnt = 9

!---------------------------------------------------------
end module pol_gauss_mdin_dat_mod


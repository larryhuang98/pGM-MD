#include "pGM/copyright_pgm.i"
#include "../include/assert.fh"
!
! Adopted from sander pGM code by YD
!
!------------------------------------------------------------
module pol_gauss_induced

  use timers_mod
! need this because all arrays are indexed as images.
  use img_mod
#ifdef MPI
  use parallel_dat_mod
#endif
  use pgm_interface_data_mod
  use pol_gauss_direct_data, only : dipole_dipole_ptrs, dipole_dipole_short_ptrs
  use file_io_dat_mod, only : mdout

  implicit none

  private

  integer, save                       :: called_ind_eval = 0

  logical, save, allocatable          :: is_polarizable(:)
  logical, save                       :: allpolarizable = .true.
  integer, save                       :: non_polarizable_cnt = 0
  integer, save, allocatable          :: non_polarizable_lst(:)
  double precision, save, allocatable :: polarizability(:)
  double precision, save, allocatable :: inv_pol(:)
  double precision, save, allocatable :: ind_dip(:,:)
  double precision, save, allocatable :: dip_field(:,:)
  double precision, save, allocatable :: cart_dipole_field(:,:)

! for images

  logical,save, allocatable           :: is_polarizable_img(:)
  double precision, save, allocatable :: pol_img(:)
  double precision, save, allocatable :: inv_pol_img(:)
  double precision, save, allocatable :: ind_dip_img(:,:)
  double precision, save, allocatable :: dip_field_img(:,:)
  double precision, save, allocatable :: cart_dipole_field_img(:,:)

  logical, save                       :: img_atm_map_saved = .false.
  integer, save                       :: old_img_cnt
  integer, save                       :: old_img_lo
  integer, save                       :: old_img_hi
  integer, save, allocatable          :: old_atm_img_map(:)
  integer, save, allocatable          :: old_img_atm_map(:)
  integer, save, allocatable          :: old_gbl_owned_imgs_tbl(:,:)
  integer, save                       :: my_full_img_cnt_old
  integer, save, allocatable          :: my_full_img_lst_old(:)

  logical, save :: initial

  integer, save                       :: recorddim
  double precision, save, allocatable :: record1(:,:,:)
  double precision, save, allocatable :: record2(:,:,:)
  double precision, save, allocatable :: record3(:,:,:)
  double precision, save, allocatable :: record4(:,:,:)
  double precision, save, allocatable :: oldrec(:,:,:)
  double precision, save, allocatable :: olddip(:,:)
  double precision, save, allocatable :: perm_field_img(:,:)
  double precision, save, allocatable :: alpha_perm_field_img(:,:)

  ! RL: diagnostic dump buffers for the subspace-recycling study (serial pmemd.pgm)
  double precision, save, allocatable :: pgm_b_save(:,:)      ! snapshot of RHS b (=alpha_perm_field_img before clobber)
  double precision, save, allocatable :: pgm_hist_p(:,:,:)    ! captured PCG search directions pv      (3,atm_cnt,kmax)
  double precision, save, allocatable :: pgm_hist_Ap(:,:,:)   ! captured A_code*pv = zv                  (3,atm_cnt,kmax)
  double precision, save, allocatable :: pgm_hist_rnorm(:)    ! per-iteration true residual norm
  double precision, save, allocatable :: pgm_hist_alpha(:)    ! CG alpha scalars
  double precision, save, allocatable :: pgm_hist_beta(:)     ! CG beta scalars
  integer, save                       :: pgm_hist_k = 0       ! iterations captured this solve
  integer, save                       :: pgm_hist_cap = 0     ! allocated capacity (= pgm_dump_hist_kmax)
  integer, save                       :: pgm_hist_atmcnt = 0  ! atm_cnt the buffers were sized for

  integer, save                       :: mystart
  integer, save                       :: myend
  integer, save                       :: mycnt
  integer, save                       :: mystart3
  integer, save                       :: myend3
  integer, save                       :: mycnt3
#ifdef MPI
  integer, save                       :: mysendcnt3
  integer, save                       :: max_img_cnt
  integer, save, allocatable          :: recv_taskmap(:)
  integer, save, allocatable          :: send_taskmap(:)

#if 0
! used in an alternative scheme
! In this scheme, all used short range images are gathered to the tasks where they are changed.
! This can avoid a step to distrubte the reduced field and force.
! This scheme requires just one set of communications. But, the message is the entire used images and
! the setup is slightly more demanding.
! might be suitable to short-range which requires real space calculations only.

  integer, save, allocatable          :: recvcnt3(:)
  integer, save, allocatable          :: recvdisp3(:)
  integer, save, allocatable          :: short_recv_cnt(:)
  integer, save, allocatable          :: short_send_cnt(:)
  integer, save, allocatable          :: full_recv_cnt(:)
  integer, save, allocatable          :: full_send_cnt(:)
  integer, save, allocatable          :: gather_recv_cnt(:)
  integer, save, allocatable          :: gather_send_cnt(:)
  integer, save, allocatable          :: gbl_all_used_img_cnt(:,:)
  integer, save, allocatable          :: gbl_img_range(:,:)
#endif

! In this scheme, all changed images are gathered and reduced at the "host" task and the
! results are distributed to those tasks that changed the images. This scheme needs two
! sets back-to-back communications, each is relatively short. The setup is rather simple.

  integer, save, allocatable          :: my_img_lsts(:,:)         ! lists of my images on other nodes
  integer, save, allocatable          :: my_img_cnts_short(:)     ! short-range counts
  integer, save, allocatable          :: my_img_cnts_full(:)      ! full-range counts
  integer, save, allocatable          :: peer_img_lsts(:,:)       ! lists of peer images on my node
  integer, save, allocatable          :: peer_img_cnts_short(:)   ! short-range counts
  integer, save, allocatable          :: peer_img_cnts_full(:)    ! full-range counts
  double precision, save, allocatable :: recv_buf(:,:,:)
  double precision, save, allocatable :: send_buf(:,:,:)

  integer, save                       :: pgm_init_tag = 30
  integer, save                       :: pgm_list_tag = 31
  integer, save                       :: pgm_get_short_fld_tag = 32
  integer, save                       :: pgm_snd_short_fld_tag = 33
  integer, save                       :: pgm_get_full_fld_tag = 34
  integer, save                       :: pgm_snd_full_fld_tag = 35
  integer, save                       :: pgm_gath_fld_tag = 36
#endif

  integer, save, allocatable          :: recidx1(:)
  integer, save, allocatable          :: recidx2(:)
  integer, save, allocatable          :: recidx3(:)
  integer, save, allocatable          :: recidx4(:)

  integer, save                       :: reccnt(4)
  integer, save                       :: currec(4)
  integer, save                       :: saved(4)

  logical, save    :: scf_solv_inited = .false.

  public pGM_INDUCED_eval, pGM_INDUCED_img, pGM_INDUCED_img_update, pGM_INDUCED_readparm, &
         is_polarizable, ind_dip, dip_field, cart_dipole_field, ind_dip_img, dip_field_img, &
         polarizability, &
#ifdef MPI
         pGM_INDUCED_bcast, init_scf_iter_comm, &
         pGM_reduce_full_distr_full_used_img, &
         recv_taskmap, send_taskmap, &
!         pGM_init_used_img_map, pGM_append_long_range_img, &
!         my_short_img_cnt, my_short_img_lst, &
!         my_used_img_mask, &
!         my_full_img_cnt, &
!         my_full_img_lst, &
#endif
         cart_dipole_field_img, &
         pgm_b_save, pgm_hist_p, pgm_hist_Ap, pgm_hist_rnorm, pgm_hist_alpha, &
         pgm_hist_beta, pgm_hist_k, pgm_dump_frame
!         , diprms, dipiter

contains
!------------------------------------------------------------------------
!
! The main driver of induced dipole scf iteration module.
! In this module, the arrays and vectors are indexed as images, be consistent with
! those in reciprocal and direct space sums.
!
!------------------------------------------------------------------------
subroutine pGM_INDUCED_eval(atm_cnt, img_crd, ind_dip_img, ipairs, ene_perm_dip, diprms, dipiter)

  use pol_gauss_mdin_dat_mod, only : dipole_scf_init, dipole_scf_init_order, use_average,&
                                     dipole_scf_init_step, scf_solv_opt, pol_gauss_verbose
  use img_mod
#ifdef MPI
  use parallel_mod, only : mpi_allgathervec
#endif

  integer,intent(in) :: atm_cnt
  integer, intent(in) :: ipairs(*)

  double precision, intent(in)    :: img_crd(3, atm_cnt)
  double precision, intent(inout) :: ind_dip_img(3, atm_cnt)  ! Everything is img, even if the label says otherwise.
  double precision, intent(inout) :: ene_perm_dip
  double precision, intent(inout) :: diprms, dipiter

  double precision :: x(3)

  integer :: n, i, ier, idx

! for evaluation of the methods
  double precision, save, allocatable :: extrpdip(:, :)
  double precision :: rmsd, rmsdx
#ifdef MPI
  double precision, save :: buf(10)
#endif
  integer, save    :: idx_current, idx_last, idx_next
  integer          :: idxmax
  integer          :: iatm, img
  double precision :: tmp(3, atm_cnt)

if ((pol_gauss_verbose == 6 .or. dipole_scf_init .eq. 2) &
     .and. (.not. allocated(olddip))) then
  allocate(olddip(3, atm_cnt))
  olddip = 0.d0
end if

! These for testing purpose, will be removed later
if (pol_gauss_verbose == 6 .and. .not. allocated(extrpdip)) then
   allocate(extrpdip(3, atm_cnt))
   extrpdip = 0.0d0
end if

if ((pol_gauss_verbose == 6 .or. dipole_scf_init .eq. 2) .and. &
                           .not. allocated(olddip)) then
  allocate(olddip(3, atm_cnt))
  olddip = 0.d0
end if

#ifdef MPI
  if (pol_gauss_verbose == 2 .and. master) then
#else
  if (pol_gauss_verbose == 2) then
#endif
    if (use_average == 1) then
      write(mdout, '(a)') ' pGM_Induced_eval: Use average error to check convergence' 
    else
      write(mdout, '(a)') ' pGM_Induced_eval: Use maximum error to check convergence' 
    end if
    write(mdout, '(a)') ' pGM_Induced_eval: Setting up rhs of the linear system' 
  end if

  ! set up the linear system AM * ind_dip_img = perm_field_img, in short as A * x = b
  ! the off-diagonal elements of A (both full and short) are the
  ! dip-dip tensor elements set up in the call, the diagoanl elements
  ! of A is 1/pol, the right hand side perm_field_img() is E_perm, which is
  ! obtained after adding the negative sign to the returned gradient zv

#ifdef MPI
  do idx = 1, my_full_img_cnt
    perm_field_img(:, my_full_img_lst(idx)) = 0.d0
  end do
#else
  perm_field_img = 0.d0
#endif

  call pGM_NonBond_perm_fields(atm_cnt, img_crd, ipairs, perm_field_img, ene_perm_dip)

#ifdef MPI
  call pGM_reduce_full_distr_full_used_img(atm_cnt, perm_field_img)
#endif

! perm_field_img : field
! ind_dip_img : induce dipole
! ap : polarizability
! ip : 1/ap

!  perm_field_img = -zv ! this is E_perm
!  do n = 0, atm_cnt - 1
!    perm_field_img(n*3+1:n*3+3) = -cart_dipole_field_img(1:3,gbl_atm_img_map(n+1))
!  end do

#ifdef MPI
  do idx = 1, my_full_img_cnt
    n = my_full_img_lst(idx)
#else
  do n = 1, atm_cnt
#endif
    alpha_perm_field_img(:,n) = -pol_img(n) * perm_field_img(:,n)  ! this is alpha*E_perm
  end do

  ! RL: snapshot the RHS b here. alpha_perm_field_img is reused as scratch and clobbered
  ! below by the dipole_scf_init==3 extrapolation (pGM_induced.F90:~372), so this is the
  ! only always-valid capture point for the subspace-recycling study.
  if ( .not. allocated(pgm_b_save) ) allocate(pgm_b_save(3, atm_cnt))
  pgm_b_save(:, 1:atm_cnt) = alpha_perm_field_img(:, 1:atm_cnt)

  if ( initial .or. dipole_scf_init .eq. 1) then
#ifdef MPI
  do idx = 1, my_full_img_cnt
    n = my_full_img_lst(idx)
#else
  do n = 1, atm_cnt
#endif
    ind_dip_img(:,n) = alpha_perm_field_img(:,n)  ! this is alpha*E_perm
  end do
    
    initial = .false.
  end if

  ! initialize induced dipoles
  select case ( dipole_scf_init )
    case (1)
      if (pol_gauss_verbose == 2) then
#ifdef MPI
         if (master) &
#endif
           write(mdout,'(/,a)') ' pGM Induced_eval: initial dipole set as pol*E_perm'
      end if

    case (2)
      if (pol_gauss_verbose == 2) then
#ifdef MPI
         if (master ) &
#endif
           write(mdout,'(/,a)') ' pGM Induced_eval: initial dipole set from last step'
      end if
      if ( .not. initial ) then
#ifdef MPI
        do idx = 1, my_full_img_cnt
          n = my_full_img_lst(idx)
#else
        do n = 1, atm_cnt
#endif
          ind_dip_img(1:3, n) = olddip(1:3, n)
        end do
      end if
    case (3)
      if (pol_gauss_verbose == 2) then
#ifdef MPI
         if (master) &
#endif
           write(mdout,'(/,a)') ' pGM Induced_eval: initial dipole extropolated from multiple steps'
      end if

      ! cycling the indices instead of shifting arrays because 
      ! doing all-atom operations so oftern can reduce parallel efficiency.
      call extrap_ind_dipole(atm_cnt, dipole_scf_init_order, dipole_scf_init_step, &
                             pol_img, alpha_perm_field_img, ind_dip_img)
    case default
      ! invalid dipole_scf_init
      ASSERT( .false. )
  end select

  ! NOTE: up to now, ind_dip_img() holds the entire extrapolated induced dipoles, to my_full_img_cnt

  ! solve the linear system
  select case ( scf_solv_opt )
    case ( 1 )
      if (pol_gauss_verbose /= 0) then
#ifdef MPI
         if (master ) &
#endif
           write(mdout,'(/,a)') ' pGM CG selected for induced dipole solution '
      end if
      call pGM_cg(atm_cnt, img_crd, diprms, dipiter, pol_img, alpha_perm_field_img, ind_dip_img)
    case ( 2 )
      if (pol_gauss_verbose /= 0) then
#ifdef MPI
         if (master ) &
#endif
           write(mdout,'(/,a)') ' pGM PCG selected for induced dipole solution '
      end if
      call pGM_pcg(atm_cnt, img_crd, diprms, dipiter, pol_img, alpha_perm_field_img, ind_dip_img)
    case ( 3 )
      if (pol_gauss_verbose /= 0) then
#ifdef MPI
         if (master ) &
#endif
           write(mdout,'(/,a)') ' pGM PCG with Peek selected for induced dipole solution '
      end if
      call pGM_pkcg(atm_cnt, img_crd, diprms, dipiter, pol_img, alpha_perm_field_img, ind_dip_img)
    case ( 4 )
      if (pol_gauss_verbose /= 0) then
#ifdef MPI
         if (master ) &
#endif
           write(mdout,'(/,a)') ' pGM SOR selected for induced dipole solution '
      end if
      call pGM_sor(atm_cnt, img_crd, diprms, dipiter, pol_img, alpha_perm_field_img, ind_dip_img)
    case ( 5 )
      if (pol_gauss_verbose /= 0) then
#ifdef MPI
         if (master ) &
#endif
           write(mdout,'(/,a)') ' pGM Ehanced SOR selected for induced dipole solution '
      end if
      call pGM_esor(atm_cnt, img_crd, diprms, dipiter, pol_img, alpha_perm_field_img, ind_dip_img)
    case default
      ! invalid scf_solv_opt
      ASSERT( .false. )
  end select

  ! save some data for initial guess of induced dipoles
  if ( dipole_scf_init == 3) then
    call update_record(atm_cnt, dipole_scf_init_step, ind_dip_img, &
                       record2, recidx2, currec(2), saved(2), reccnt(2))
    if ( dipole_scf_init_order >= 2 .and. &
                reccnt(2) .gt. dipole_scf_init_step) then
#ifdef MPI
      do idx = 1, my_full_img_cnt
         i = my_full_img_lst(idx)
         alpha_perm_field_img(:, i) = ind_dip_img(:, i) - oldrec(:, i, 1) 
      end do
#else
      alpha_perm_field_img(:, :) = ind_dip_img(:, :) - oldrec(:, :, 1) 
#endif
      call update_record(atm_cnt, dipole_scf_init_step, alpha_perm_field_img, &
                       record3, recidx3, currec(3), saved(3), reccnt(3))
    end if
    if ( dipole_scf_init_order == 3 .and. &
               reccnt(3) .gt. dipole_scf_init_step) then
#ifdef MPI
      do idx = 1, my_full_img_cnt
         i = my_full_img_lst(idx)
         alpha_perm_field_img(:, i) = ind_dip_img(:, i) - oldrec(:, i, 2) 
      end do
#else
      alpha_perm_field_img(:, :) = ind_dip_img(:, :) - oldrec(:, :, 2)
#endif
      call update_record(atm_cnt, dipole_scf_init_step, alpha_perm_field_img, &
                       record4, recidx4, currec(4), saved(4), reccnt(4))
    end if
  end if

  ! return final induced dipoles for energy/force processing

  call update_time(nonbond_time)

if (pol_gauss_verbose == 6) then
#ifdef MPI
buf(1) = sum((ind_dip_img(1:3, my_img_lo:my_img_hi) - olddip(1:3, my_img_lo:my_img_hi))**2)
buf(2)= sum((ind_dip_img(1:3, my_img_lo:my_img_hi) - extrpdip(1:3, my_img_lo:my_img_hi))**2)
call mpi_reduce(buf, buf(3), 2, MPI_DOUBLE_PRECISION, MPI_SUM, 0, pmemd_comm, ier)
rmsd = buf(3)
rmsdx= buf(4)
#else
rmsd  = sum((ind_dip_img(:, :) - olddip(:, :))**2)
rmsdx = sum((ind_dip_img(:, :) - extrpdip(:, :))**2)
#endif

#ifdef MPI
if (master) &
#endif
write(mdout,'(A,3X,5I5,3X,3ES15.5)')'after solve the linear system ind_dip_img', &
dipiter, dipole_scf_init, scf_solv_opt, dipole_scf_init_order, dipole_scf_init_step, &
sqrt(rmsd),sqrt(rmsdx),sqrt(rmsdx/rmsd)
end if

if (dipole_scf_init .eq. 2 .or. pol_gauss_verbose == 6) then 
#ifdef MPI
   do n = my_img_lo, my_img_hi
#else
   do n = 1, atm_cnt
#endif
      olddip(:, n) = ind_dip_img(:, n)
   end do
#ifdef MPI
  call mpi_allgathervec(atm_cnt, olddip)
#endif

endif

contains

!----------------------------------------------------------
! To solve A*x = b with the SOR solver, we intend to utilize the
! iterative approach: x(i+1) = x(i) + wsor*(b - A*x(i))/D
! When wsor = 1, this is just the Gauss-Seidel iteration.
! The code is documented for educational purpopses only and
! for comparison with other faster solvers.
!
! When A can be esaily written as L+D+U, there are additional
! tricks possible, but we can only reasonably use A = T + D,
! D=1/pol, so we have x(i+1) = x(i) + wsor*r(i))*pol,
! with r(i) = b - A*x(i)
! 
! In PB we are using the following, which is to use the latest
! x's to update the subsequent x's as
! x(i+1) = x(i) + wsor*[b - (T + 1/pol)*x(i)]*pol
! x(i+1) = x(i) + wsor*[b - T*x(i)]*pol - wsor*1/pol*x(i)*pol
! x(i+1) = x(i) + wsor*[b - T*x(i)]*pol - wsor*x(i)
! x(i+1) = [1-wsor]*x(i) + wsor*[b - T*x(i)]*pol
! In the above loop, the latest x's are used to speed up convergence.
! since T*x must be realized in one call in pGM, we can't use
! the same trick.
!----------------------------------------------------------
subroutine pGM_sor(atm_cnt, img_crd, r_norm, iter, atm_pol_img, alpha_perm_field_img, ind_dip_img)
!subroutine pGM_sor(numatoms, x, r_norm, iter, perm_field_img, ind_dip_img)
  use pol_gauss_mdin_dat_mod, only : scf_sor_niter,scf_sor_coefficient,&
                             dipole_scf_tol,pol_gauss_verbose
  implicit none

  integer, intent(in) :: atm_cnt
  double precision,  intent(in)    :: img_crd(3,*)
  double precision,  intent(out)   :: r_norm, iter
!  double precision,  intent(in)    :: inv_pol_img(atm_cnt)
  double precision,  intent(in)    :: atm_pol_img(atm_cnt)              ! ap
  double precision,  intent(in)    :: alpha_perm_field_img(3, atm_cnt)  ! bv
  double precision,  intent(inout) :: ind_dip_img(3, atm_cnt)           ! xv

  logical :: unconvg

  double precision :: bnorm, bsum
  double precision :: zv(3, atm_cnt)
  double precision :: rv(3, atm_cnt)
  double precision :: pv(3, atm_cnt)
  double precision :: fld(3, atm_cnt)
  double precision :: dif(3, atm_cnt)
  double precision :: polxp_fld(3, atm_cnt)

  integer :: i, iatm, img, idx

  unconvg = .true.
  iter = 0.0d0
!  call vec_abs_sum(atm_cnt, perm_field_img, bnorm)
  call vec_abs_sum(atm_cnt, alpha_perm_field_img, bsum)
  bnorm = bsum / dble(3*atm_cnt)

  ! main loop
  do while ( unconvg )

    ! Note that A's off-diagonal elements are those of the dipole-dipole tensor * alpha and
    ! its diagonal elements are one.
    ! xv: induced dipole, zv: -dipole field, i.e. dipole-dipole Tensor*xv
    call pGM_NonBond_dip_dip_fields(atm_cnt, img_crd, ind_dip_img, zv)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      rv(1:3, i) = alpha_perm_field_img(1:3, i) - &
                  (atm_pol_img(i) * zv(1:3, i) + ind_dip_img(1:3, i))
    end do

    iter = iter + 1.0d0

    ! check convergence
    call vec_abs_max(atm_cnt, dif, r_norm)
    r_norm = r_norm / bsum

    if ( pol_gauss_verbose == 2 ) then
#ifdef MPI
      if (master) &
#endif
      write(mdout, '(a,i3,a,f14.10)') &
            ' pGM SOR: iter = ', int(iter), ', residue norm: ', r_norm
    end if
    if ( int(iter) >= scf_sor_niter .or. r_norm <= dipole_scf_tol ) then
      unconvg = .false.
      if ( int(iter) >= scf_sor_niter ) then
#ifdef MPI
      if (master) &
#endif
        write(mdout,'(a)') ' pGM SOR: FATAL maxitn exceeded!'
        call mexit(mdout,1)
      end if
    end if

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      ind_dip_img(1:3, i) = ind_dip_img(1:3, i) + scf_sor_coefficient * rv(1:3, i)
    end do

  end do ! do while (unconvg)

  if ( pol_gauss_verbose /= 0 )then
#ifdef MPI
      if (master) &
#endif
    write(mdout,'(/a,i5,1x,f14.9/)') ' pGM SOR: out of loop, iter, norm = ',int(iter),r_norm
  endif

end subroutine pGM_sor
!----------------------------------------------------------
! To solve A*x = b with the ESOR solver, we intend to utilize the
! relation of A = T + D, D=1/pol, and T = T_short + T_long
! so we have T_short*x + T_long*x +D*x = b
! (T_short+D)*x = b - T_long*x
! This shows that we can use the local_preconditoner matrix
! to do the iteration given a revised righ hand side.
!
! Note in this method, A*x can still be computed as T*x has been computed.
! So r = b - A*x is ready available. This allows us to use it to
! check convergence just like other solvers. 
!----------------------------------------------------------
!subroutine pGM_esor(numatoms, x, r_norm, iter, perm_field_img, ind_dip_img)
subroutine pGM_esor(atm_cnt, img_crd, r_norm, iter, atm_pol_img, alpha_perm_field_img, ind_dip_img)

  use pol_gauss_mdin_dat_mod, only : scf_sor_niter,scf_sor_coefficient,&
                             dipole_scf_tol,scf_local_niter,pol_gauss_verbose

  implicit none

  integer, intent(in) :: atm_cnt
  double precision,  intent(in)    :: img_crd(3,*)
  double precision,  intent(out)   :: r_norm, iter
  double precision,  intent(in)    :: atm_pol_img(atm_cnt)              ! ap
  double precision,  intent(in)    :: alpha_perm_field_img(3, atm_cnt)  ! bv
  double precision,  intent(inout) :: ind_dip_img(3, atm_cnt)           ! xv

  double precision                 :: zv(3, atm_cnt)
  double precision                 :: pv(3, atm_cnt)
  double precision                 :: rv(3, atm_cnt)
  double precision                 :: zu(3, atm_cnt)

  logical :: unconvg

  integer :: i, iatm, img, idx
  double precision :: bnorm, bsum

  unconvg = .true.
  iter = 0.0d0

  call vec_abs_sum(atm_cnt, alpha_perm_field_img, bsum)
  bnorm = bsum / dble(3*atm_cnt)

  ! main loop
  do while ( unconvg )

    ! compute T_long*ind_dip_img for local_precondition()
    ! compute residue rv
    ! Note that A's off-diagonal elements are those of the dipole-dipole tensor * alpha and
    ! its diagonal elements are one
    ! xv: induced dipole, zv: -dipole field, i.e. dipole-dipole Tensor*xv
    call pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, ind_dip_img, zv)
    call pGM_pvt_NonBond_dip_dip_fields_short(atm_cnt, img_crd, ind_dip_img, zu, 2)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
       pv(1:3, i) = alpha_perm_field_img(1:3, i) - &
                  atm_pol_img(i)*(zv(1:3, i) - zu(1:3, i)) ! this is alpha*T_long*ind_dip_img
       rv(1:3, i) = alpha_perm_field_img(1:3, i) - &
                 (atm_pol_img(i)*zv(1:3, i) + ind_dip_img(:, i)) ! this is (1 - alpha*T)*ind_dip_img
    end do

    call vec_abs_max(atm_cnt, rv, r_norm)
    r_norm = r_norm / bnorm

    ! check convergence
    if ( pol_gauss_verbose == 2 ) then
#ifdef MPI
      if (master) &
#endif
      write(mdout, '(a,i3,a,f14.10)') &
            ' pGM ESOR: iter = ', int(iter), ', residue norm: ', r_norm
    end if
    if ( int(iter) >= scf_sor_niter .or. r_norm <= dipole_scf_tol ) then
      unconvg = .false.
      if ( int(iter) >= scf_sor_niter ) then
#ifdef MPI
      if (master) &
#endif
        write(mdout,'(a)') ' pGM ESOR: FATAL maxitn exceeded!'
        call mexit(mdout,1)
      end if
    end if

    iter = iter + 1.0d0

    ! Set up rhs as E_perm-T_long*ind_dip_img
    ! pv = bv - pv  ! consolidated to above loop

    ! The preconditioner is solving (alpha*T_short+1)*ind_dip_img = alpha*E_perm-alpha*T_long*ind_dip_img
    call local_precondition(atm_cnt, img_crd, atm_pol_img, pv, ind_dip_img, scf_local_niter)

  end do ! do while (unconvg)

  if ( pol_gauss_verbose /= 0 )then
#ifdef MPI
      if (master) &
#endif
    write(mdout,'(/a,i5,1x,f14.9/)') ' pGM ESOR: out of loop, iter, norm = ',int(iter),r_norm
  end if

end subroutine pGM_esor
!----------------------------------------------------------
! The standard CG for symmetric positive-definite matrix
! will be implemented here as pGM_cg().
!
! Working arrays are all declared as 1-d for vectoerization.
!
! The notation will also follow that of standard practice
! See inline comments on how related vectors are obtained.
!
! The original two dipole systems are no longer necessary so
! only the d system will be retained.
!
! alpha_perm_field_img: alpha * perm_field_img, bv
! ind_dip_img:          induced dipole, xv
!
!----------------------------------------------------------
subroutine pGM_cg(atm_cnt, img_crd, r_norm, iter, atm_pol_img, alpha_perm_field_img, ind_dip_img)

  use pol_gauss_mdin_dat_mod, only : use_average,scf_cg_niter,dipole_scf_tol,pol_gauss_verbose

  implicit none

  integer, intent(in) :: atm_cnt
  double precision,  intent(in)    :: img_crd(3,*)
  double precision,  intent(out)   :: r_norm, iter
  double precision,  intent(in)    :: alpha_perm_field_img(3, atm_cnt)  ! bv
  double precision,  intent(in)    :: atm_pol_img(atm_cnt)    ! ap
  double precision,  intent(inout) :: ind_dip_img(3, atm_cnt) ! xv

  double precision :: zv(3, atm_cnt)
  double precision :: rv(3, atm_cnt)
  double precision :: pv(3, atm_cnt)
  logical :: unconvg
  double precision, save :: alpha, beta, rdotr1, rdotr2, pdotz
  double precision, save :: bnorm, a_norm, bsum
  double precision, save :: buf, bufs(10)

  integer :: n, i, iatm, img, idx

  call vec_abs_sum(atm_cnt, alpha_perm_field_img, bsum)
  bnorm = bsum/dble(3*atm_cnt)
  if ( pol_gauss_verbose /= 0 ) then
    call vec_abs_sum(atm_cnt, ind_dip_img, buf)
#ifdef MPI
    if ( master ) &
#endif
    write(mdout,'(a, f14.9)') 'initial dip norm', buf/dble(3*atm_cnt)
  end if

  ! r(0) = b(0) - A*x(0)
  ! Note that A's off-diagonal elements are those of the dipole-dipole tensor * alpha and
  ! its diagonal elements are one
  ! xv: induced dipole, zv: -dipole field, i.e. dipole-dipole Tensor*xv
  call pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, ind_dip_img, zv) ! zv = induced dipole field here

#ifdef MPI
! pv() is used to calculate the fields zv(), all used img induced dipoles are needed.
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
! 
    rv(1:3, i) = alpha_perm_field_img(1:3, i) -  &
                (atm_pol_img(i) * zv(1:3, i) + ind_dip_img(1:3, i)) ! this is (1 - alpha*T)*xv 
    pv(1:3, i) = rv(1:3, i)
  end do

  ! iteration 0:
  ! compute <r(0),r(0)>
  iter = 0.0d0

!  rdotr1 = sum(rv(:, :) * rv(:, :))
  call vec_sq_sum(atm_cnt, rv, rdotr1)

  ! main loop
  unconvg = .true.

  do while (unconvg)

    ! compute Ap(i) = A * p(i)
    ! compute alpha(i) = <r(i),r(i)>/<p(i),Ap(i)>
    call pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, pv, zv)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      zv(1:3, i) = atm_pol_img(i) * zv(1:3, i) + pv(1:3, i) ! this is ( 1 - alpha*T)*dip
    end do
!    pdotz = sum(pv * zv)
    call vec_mul_sum(atm_cnt, pv, zv, pdotz)
    alpha = rdotr1/pdotz

    ! iteration i:
    iter = iter + 1.0d0

    ! update x(i) = x(i-1) + alpha(i) p(i-1)
    !        r(i) = r(i-1) - alpha(i) Ap(i-1)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      ind_dip_img(1:3, i) = ind_dip_img(1:3, i) + alpha*pv(1:3, i)
      rv(1:3, i)          = rv(1:3, i)          - alpha*zv(1:3, i)  ! rv residual "dipole"
    end do
    if ( use_average == 1 ) then
      call vec_abs_sum(atm_cnt, rv, r_norm)
      r_norm = r_norm/bsum
    else
      call vec_abs_max(atm_cnt, rv, r_norm)
      r_norm = r_norm / bnorm
    end if

    ! check convergence
    if ( pol_gauss_verbose == 3 ) then
#ifdef MPI
       if (master) &
#endif
         write(mdout, '(a,i3,a,f14.10)') &
            ' pGM CG: iter = ', int(iter), ', residue norm: ', r_norm
    endif

    if ( iter >= scf_cg_niter .or. r_norm <= dipole_scf_tol ) then
      unconvg = .false.
      if ( iter >= scf_cg_niter ) then
#ifdef MPI
       if (master) &
#endif
        write(mdout,'(a)') ' pGM CG: FATAL maxitn exceeded!'
        call mexit(mdout,1)
      end if
      exit
    end if

    ! compute beta(i) = <r(i),r(i)>/<r(i-1),r(i-1)>
    !    rdotr2 = dot_product(rv, rv)
    call vec_sq_sum(atm_cnt, rv, rdotr2)
    beta = rdotr2/rdotr1
    rdotr1 = rdotr2

    ! update p(i) = r(i) + beta(i) p(i-1)
#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      pv(1:3,i) = rv(1:3,i) + beta*pv(1:3,i)
    end do

  end do ! do while (unconvg)

  if ( pol_gauss_verbose /= 0 ) then
      call vec_abs_sum(atm_cnt, rv, a_norm)
      a_norm = a_norm / bsum
#ifdef MPI
       if (master) &
#endif
    write(mdout,'(/a,i5,1x,2e20.9/)') ' pGM CG: out of loop, iter, rel norm = ', int(iter), r_norm, a_norm
  end if

end subroutine pGM_cg
!----------------------------------------------------------
! The preconditioned CG for symmetric positive-definite matrix
! will be implemented here as pGM_pcg(). The preconditioner
! is simply the short-range dipole-dipole tensor elements and
! the diagonal elements of the A matrix.
!
! Working arrays are all declared as 1-d for vectoerization.
!
! The notation will also follow that of standard practice
! See inline comments on how related vectors are obtained.
!
! The original two dipole systems are no longer necessary so
! only the d system will be retained.
!----------------------------------------------------------
!subroutine pGM_pcg(atm_cnt, img_crd, r_norm, iter, inv_pol_img, atm_pol_img, perm_field_img, ind_dip_img)
subroutine pGM_pcg(atm_cnt, img_crd, r_norm, iter, atm_pol_img, alpha_perm_field_img, ind_dip_img)

  use pol_gauss_mdin_dat_mod, only : scf_cg_niter,scf_local_niter,dipole_scf_tol,pol_gauss_verbose

  implicit none

  integer, intent(in)              :: atm_cnt
  double precision,  intent(in)    :: img_crd(3,*)
  double precision,  intent(out)   :: r_norm, iter
!  double precision,  intent(in)    :: inv_pol_img(atm_cnt)
  double precision,  intent(in)    :: atm_pol_img(atm_cnt)             ! ap
  double precision,  intent(in)    :: alpha_perm_field_img(3, atm_cnt) ! bv
  double precision,  intent(inout) :: ind_dip_img(3, atm_cnt)          ! xv

  double precision                 :: zv(3, atm_cnt)
  double precision                 :: rv(3, atm_cnt)
  double precision                 :: pv(3, atm_cnt)

  logical :: unconvg

  double precision :: alpha, beta, rdotr1, rdotr2, pdotz
  double precision :: bnorm, a_norm, rmsbv, buf, bsum

  integer :: i, n, ier, img, iatm, idx

  call vec_abs_sum(atm_cnt, alpha_perm_field_img, bsum)
  bnorm = bsum/dble(3*atm_cnt)

  if ( pol_gauss_verbose /= 0 ) then
    call vec_abs_sum(atm_cnt, ind_dip_img, buf)
#ifdef MPI
  if ( master ) &
#endif
    write(mdout,'(a, f14.9)') 'initial dip norm', buf/dble(3*atm_cnt)
  end if

  ! r(0) = b(0) - A*x(0)
  ! Note that A's off-diagonal elements are those of the dipole-dipole tensor * alpha and
  ! its diagonal elements are ONE.

  call pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, ind_dip_img, zv) ! zv = induced dipole field here

#ifdef MPI
  do idx = 1, my_full_img_cnt
    i = my_full_img_lst(idx)
#else
  do i = 1, atm_cnt
#endif

  ! difference, residual dipole

    rv(1:3, i) = alpha_perm_field_img(1:3, i) - &
                   (atm_pol_img(i) * zv(1:3, i) + ind_dip_img(1:3, i)) ! this is A*xv = (1 - alpha*T)*ind_dip_img

  ! apply preconditoner to rv
  ! this means solving a new linear system of M * z = r, i.e. r is the rhs
  ! z(0) = M^-1 * r(0)
  ! z(0) is initialized as Diag(M^-1)*r(0) = r(0) because M's 
  ! diagonal elements are also one as those of A
! zv = rv, replace zv by pv
    pv(1:3, i) = rv(1:3, i)   ! as the "initial guess" in local_precondition()
  end do

  ! improve dip_res by local interactions using residual field
!  call local_precondition(numatoms, x, rv, zv, scf_local_niter)
  call local_precondition(atm_cnt, img_crd, atm_pol_img, rv, pv, scf_local_niter)

  ! p(0) = z(0)
! pv = zv  ! this line is no longer needed because pv has been calculated.

  !  iteration 0:
  !  compute <r(0),z(0)>
  iter = 0.0d0
  call vec_mul_sum(atm_cnt, rv, pv, rdotr1) ! rdotr1 = dot_product(rv, zv), replace zv by pv

  ! main loop
  unconvg = .true.
  do while (unconvg)

    ! compute Ap(i) = A * p(i)
    ! compute alpha(i) = <r(i),z(i)>/<p(i),Ap(i)>
!   call pGM_NonBond_dip_dip_fields(numatoms, x, pv, zv)
    call pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, pv, zv)  ! "zv" is field due to pv

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      zv(1:3, i) = atm_pol_img(i) * zv(1:3, i) + pv(1:3, i) ! "fld" here is combined dipole.
    end do

    call vec_mul_sum(atm_cnt, pv, zv, pdotz) ! pdotz = dot_product(pv, zv)
    alpha = rdotr1/pdotz

    ! iteration i:

    iter = iter + 1.0d0

    ! update x(i) = x(i-1) + alpha(i) p(i-1)
    !        r(i) = r(i-1) - alpha(i) Ap(i-1)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      ind_dip_img(1:3, i) = ind_dip_img(1:3, i) + alpha * pv(1:3, i)
      rv(1:3, i)          = rv(1:3, i)          - alpha * zv(1:3, i)
    end do

    !r_norm = maxval(abs(fld_res/perm_field_img))
    !a_norm = sum(abs(fld_res))/bnorm

    call vec_abs_max(atm_cnt, rv, r_norm)
    r_norm = r_norm / bnorm

    ! check convergence

    if ( pol_gauss_verbose == 3 ) then
#ifdef MPI
       if ( master ) &
#endif
       write(mdout, '(a,i3,a,e14.9)') &
            ' pGM PCG: iter = ', int(iter), ', residue norm: ', r_norm
    end if
    if ( iter >= scf_cg_niter .or. r_norm <= dipole_scf_tol ) then
      unconvg = .false.
      if ( iter >= scf_cg_niter ) then
        write(mdout,'(a)') ' pGM PCG: FATAL maxitn exceeded!'
        call mexit(mdout,1)
      end if
      exit
    end if

    ! apply preconditoner to fld_res
    ! z(i) = M^-1 * r(i)
    ! z(i) is initialized as pol*r(i)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      zv(1:3, i) = rv(1:3, i)
    end do

    ! obtain the induced dipoles that are consistent with fld_res due to short range interactions, stored in "dip"

    call local_precondition(atm_cnt, img_crd, atm_pol_img, rv, zv, scf_local_niter)

    ! compute beta(i) = <r(i),z(i)>/<r(i-1),z(i-1)>

    call vec_mul_sum(atm_cnt, rv, zv, rdotr2)
    beta = rdotr2/rdotr1
    rdotr1 = rdotr2

    ! update p(i) = r(i) + beta(i) p(i-1)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      pv(1:3, i) = zv(1:3, i) + beta*pv(1:3, i)
    end do

  end do ! do while (unconvg)

  if ( pol_gauss_verbose /= 0 ) then
    call vec_abs_sum(atm_cnt, rv, a_norm)
    a_norm = a_norm / bsum
#ifdef MPI
       if (master) &
#endif
    write(mdout,'(/a,i5,1x,2e20.9/)') ' pGM PCG: out of loop, iter, rel norm = ', int(iter), r_norm, a_norm
  endif

end subroutine pGM_pcg
!----------------------------------------------------------
! The preconditioned&peeked CG for symmetric positive-definite matrix
! will be implemented here as pGM_pkcg(). The preconditioner
! is simply the short-range dipole-dipole tensor elements and
! the diagonal elements of the A matrix. Here a peek step is
! implemented as one extra JUR step to save one full CG iteration
! worth of work for majority of the cases.
!
! Working arrays are all declared as 1-d for vectoerization.
!
! The notation will also follow that of standard practice
! See inline comments on how related vectors are obtained.
!
! The original two dipole systems are no longer necessary so
! only the d system will be retained.
!----------------------------------------------------------
subroutine pGM_pkcg(atm_cnt, img_crd, r_norm, iter, atm_pol_img, alpha_perm_field_img, ind_dip_img)

  use pol_gauss_mdin_dat_mod, only : scf_cg_niter,scf_local_niter,scf_sor_coefficient,&
                                     dipole_scf_tol,pol_gauss_verbose, &
                                     pgm_dump_hist, pgm_dump_hist_kmax, pgm_curr_nstep

  implicit none

  integer, intent(in)              :: atm_cnt
  double precision,  intent(in)    :: img_crd(3,*)
  double precision,  intent(out)   :: r_norm, iter
!  double precision,  intent(in)    :: inv_pol_img(atm_cnt)
  double precision,  intent(in)    :: atm_pol_img(atm_cnt)       ! ap
  double precision,  intent(in)    :: alpha_perm_field_img(3, atm_cnt)  ! bv
  double precision,  intent(inout) :: ind_dip_img(3, atm_cnt)    ! xv

  double precision                 :: zv(3, atm_cnt)
  double precision                 :: rv(3, atm_cnt)
  double precision                 :: pv(3, atm_cnt)
#if 0
  double precision                 :: fld(3, atm_cnt)
  double precision                 :: dip(3, atm_cnt)
  double precision                 :: dip_res(3, atm_cnt)
  double precision                 :: fld_res(3, atm_cnt)
  double precision                 :: tmp(3, atm_cnt)
#endif

  logical :: unconvg

  double precision :: alpha, beta, rdotr1, rdotr2, pdotz
  double precision :: bnorm, a_norm, rmsbv, buf, bsum

  integer :: i, n, ier, img, iatm, idx
  integer :: kk
  logical :: capturing

  call vec_abs_sum(atm_cnt, alpha_perm_field_img, bsum)
  bnorm = bsum / dble(3*atm_cnt)

  ! RL: capture PCG history for the subspace-recycling study
  capturing = (pgm_dump_hist /= 0) .and. pgm_is_dump_frame(pgm_curr_nstep)
  if ( capturing ) then
     if ( pgm_hist_cap < pgm_dump_hist_kmax .or. pgm_hist_atmcnt /= atm_cnt ) then
        if ( allocated(pgm_hist_p) )     deallocate(pgm_hist_p)
        if ( allocated(pgm_hist_Ap) )    deallocate(pgm_hist_Ap)
        if ( allocated(pgm_hist_rnorm) ) deallocate(pgm_hist_rnorm)
        if ( allocated(pgm_hist_alpha) ) deallocate(pgm_hist_alpha)
        if ( allocated(pgm_hist_beta) )  deallocate(pgm_hist_beta)
        allocate(pgm_hist_p(3,atm_cnt,pgm_dump_hist_kmax), &
                 pgm_hist_Ap(3,atm_cnt,pgm_dump_hist_kmax), &
                 pgm_hist_rnorm(pgm_dump_hist_kmax), &
                 pgm_hist_alpha(pgm_dump_hist_kmax), &
                 pgm_hist_beta(pgm_dump_hist_kmax))
        pgm_hist_cap = pgm_dump_hist_kmax
        pgm_hist_atmcnt = atm_cnt
     end if
     pgm_hist_k = 0
  end if

  if ( pol_gauss_verbose /= 0 ) then
    call vec_abs_sum(atm_cnt, ind_dip_img, buf)
#ifdef MPI
    if ( master ) &
#endif
    write(mdout,'(a, f14.9)') 'initial dip norm', buf/dble(3*atm_cnt)
  end if

  ! r(0) = b(0) - A*x(0)
  ! Note that A's off-diagonal elements are those of the dipole-dipole
  ! tensor * alpha and its diagonal elements are one
  ! call pGM_NonBond_dip_dip_fields(numatoms, x, xv, zv)
  call pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, ind_dip_img, zv) ! zv = induced dipole field here

#ifdef MPI
  do idx = 1, my_full_img_cnt
    i = my_full_img_lst(idx)
#else
  do i = 1, atm_cnt
#endif
! zv = ap*zv + xv
! rv = bv - zv
    rv(1:3, i) = alpha_perm_field_img(1:3, i) - &
                (atm_pol_img(i) * zv(1:3, i) + ind_dip_img(1:3, i))

  ! apply preconditoner to rv
  ! this means solving a new linear system of M * z = r, i.e. r is the rhs
  ! z(0) = M^-1 * r(0)
  ! z(0) is initialized as diag(M^-1)*r(0) = r(0) because M's diagonals are
  ! also one

  ! replace zv by pv
    pv(1:3, i) = rv(1:3, i)       ! as the "initial guess" in local_precondition()
  end do

  ! improve dip_res by local interactions using residual field

  call local_precondition(atm_cnt, img_crd, atm_pol_img, rv, pv, scf_local_niter)

  ! p(0) = z(0)

  !  iteration 0:
  !  compute <r(0),z(0)>

  iter = 0.0d0

  call vec_mul_sum(atm_cnt, rv, pv, rdotr1)

  ! main loop

  unconvg = .true.
  do while (unconvg)

    ! compute Ap(i) = A * p(i)
    ! compute alpha(i) = <r(i),z(i)>/<p(i),Ap(i)>

    call pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, pv, zv)  ! "zv" is field due to pv

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      zv(1:3, i) = atm_pol_img(i) * zv(1:3, i) + pv(1:3, i)
    end do
    call vec_mul_sum(atm_cnt, pv, zv, pdotz)
    alpha = rdotr1/pdotz

    ! iteration i:
    iter = iter + 1.0d0

    ! update x(i) = x(i-1) + alpha(i) p(i-1)
    !        r(i) = r(i-1) - alpha(i) Ap(i-1)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      ind_dip_img(:, i) = ind_dip_img(:, i) + alpha * pv(:, i)
      rv(:, i)          = rv(:, i)          - alpha * zv(:, i)
    end do

    !r_norm = maxval(abs(fld_res/perm_field_img))
    !a_norm = sum(abs(fld_res))/bnorm

    call vec_abs_max(atm_cnt, rv, r_norm)
    r_norm = r_norm / bnorm

    ! RL: capture this iteration (search dir pv, its matvec zv=A_code*pv which is
    !     still valid here, the residual norm, and CG scalars)
    if ( capturing .and. pgm_hist_k < pgm_hist_cap ) then
       pgm_hist_k = pgm_hist_k + 1
       kk = pgm_hist_k
       pgm_hist_rnorm(kk) = r_norm
       pgm_hist_alpha(kk) = alpha
       pgm_hist_beta(kk)  = beta
       pgm_hist_p(:,:,kk)  = pv(1:3,1:atm_cnt)
       pgm_hist_Ap(:,:,kk) = zv(1:3,1:atm_cnt)
    end if

    ! check convergence

    if ( pol_gauss_verbose == 3 ) then
#ifdef MPI
       if ( master ) &
#endif
       write(mdout, '(a,i3,a,e14.9)') &
            ' pGM PCG: iter = ', int(iter), ', residue norm: ', r_norm
    end if
    if ( iter >= scf_cg_niter .or. r_norm <= dipole_scf_tol ) then
      unconvg = .false.
      if ( iter >= scf_cg_niter ) then
        write(mdout,'(a)') ' pGM PCG: FATAL maxitn exceeded!'
        call mexit(mdout,1)
      end if
      exit
    end if

    ! apply preconditoner to fld_res
    ! z(i) = M^-1 * r(i)
    ! z(i) is initialized as pol*r(i)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      zv(1:3, i) = atm_pol_img(i)*rv(1:3, i)    ! "dip" here is a scratch vector, zv()
    end do

    ! obtain the induced dipoles that are consistent with rv due to short range interactions, stored in "zv"

    call local_precondition(atm_cnt, img_crd, atm_pol_img, rv, zv, scf_local_niter)

    ! compute beta(i) = <r(i),z(i)>/<r(i-1),z(i-1)>
    call vec_mul_sum(atm_cnt, rv, zv, rdotr2)
    beta = rdotr2/rdotr1
    rdotr1 = rdotr2

    ! update p(i) = r(i) + beta(i) p(i-1)

#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      pv(1:3, i) = zv(1:3, i) + beta*pv(1:3, i)
    end do

  end do ! do while (unconvg)

  if ( pol_gauss_verbose /= 0 ) then
    call vec_abs_sum(atm_cnt, rv, a_norm)
    a_norm = a_norm / bsum
#ifdef MPI
  if ( master ) &
#endif
    write(mdout,'(/a,i5,1x,2e20.9/)') ' pGM PKCG: out of loop, iter, rel norm = ', int(iter), r_norm, a_norm
  endif

  ! peek using a SOR step to save some time
  ! a) Haixin's ESOR step
  ! The following call sets zv = -E^pol_short
  ! Note also xv*ip = E^tot = E^perm + E^pol_short + E^pol_long
  ! so xv*ip + zv = E^perm + E^pol_long, which is the rhs of
  ! the linear system in ESOR.
  ! The difference here is that we are trying to solve
  ! A*xv^i = bv + rv^i, which is different from the original
  ! target equation A*xv^i = bv in ESOR, so the rhs here is
  ! E^perm + E^pol_long + rv = zv + xv*ip + rv
  !call pGM_NonBond_dip_dip_fields_short(numatoms, x, xv, zv)
  !zv = zv + xv*ip + rv
  !call local_precondition(numatoms, x, zv, xv, 3)

  ! b) Standard SOR step seems to be good enough
  !xv = xv + scf_sor_coefficient*rv*ap
#ifdef MPI
  do idx = 1, my_full_img_cnt
    i = my_full_img_lst(idx)
#else
  do i = 1, atm_cnt
#endif
    ind_dip_img(1:3, i) = ind_dip_img(1:3, i) + scf_sor_coefficient * rv(1:3, i)
  end do

  ! let's check to see how good it is ...
  if ( pol_gauss_verbose /= 0 ) then
!     call pGM_NonBond_dip_dip_fields(numatoms, x, xv, zv)
     call pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, ind_dip_img, zv) ! zv = induced dipole field here
#ifdef MPI
    do idx = 1, my_full_img_cnt
      i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      zv(1:3, i) = atm_pol_img(i)*zv(1:3, i) + ind_dip_img(1:3, i)
      rv(1:3, i) = alpha_perm_field_img(1:3, i) - zv(1:3, i)
    end do
    call vec_abs_max(atm_cnt, rv, r_norm)
    r_norm = r_norm/bnorm
    call vec_abs_sum(atm_cnt, rv, a_norm)
    a_norm = a_norm / bsum
#ifdef MPI
    if (master) &
#endif
     write(mdout,'(/a,2e20.9/)') ' pGM PKCG: after peek, rel norm = ', r_norm, a_norm
  end if

end subroutine pGM_pkcg
!----------------------------------------------------------
! Preconditioning using short-range dip-dip tensor elements only
! This is simply another CG solver with limited iteration, so
! convergence check is not necessary.
! The job type has been removed by simply passing in the appopriate
! rhs vector field_img and unknown vector ind_dip_img() as in AM * ind_dip_img = field_img, or
! ind_dip_img = AM^-1 * field_img
!
! One output, this produces scf induced dipole consistent with field_img due to short-range interactions
!
!----------------------------------------------------------
!subroutine local_precondition(atm_cnt, img_crd, inv_pol_img, atm_pol_img, field_img, ind_dip_img, niter)
subroutine local_precondition(atm_cnt, img_crd, atm_pol_img, dip_img, ind_dip_img, niter)
! dip_imp:     bv in local_precondition() of sander:pGM_induced.F90
! ind_dip_imp: xv in local_precondition() of sander:pGM_induced.F90

  use pol_gauss_mdin_dat_mod, only : pol_gauss_verbose, scf_sor_coefficient

  implicit none

  integer, intent(in)             :: atm_cnt
  integer, intent(in)             :: niter
  double precision, intent(in)    :: img_crd(*)
!  double precision, intent(in)    :: inv_pol_img(atm_cnt)
  double precision, intent(in)    :: atm_pol_img(atm_cnt)    ! ap
  double precision, intent(in)    :: dip_img(3,atm_cnt)      ! bv
  double precision, intent(inout) :: ind_dip_img(3, atm_cnt) ! xv

! zu, ru, pu are all local vectors, they are now declared locally, not as shared module vectors

  double precision :: zu(3, atm_cnt)
  double precision :: ru(3, atm_cnt)
  double precision :: pu(3, atm_cnt)
  double precision :: alpha, beta, rdotr1, rdotr2, pdotz, r_norm, buf, bufs(10), bsum, bnorm
  integer :: i, n, ier, iatm, img, idx, idx_cnt, iter

! These two lines are moved to if (pol_gauss_verbose == 2) because that's the only place
! the bnorm is used, for better MPI performance.
!  bsum = sum(abs(bv))
!  bnorm = bsum/dble(3*numatoms)
  ! r(0) = b(0) - A*x(0)
  ! Note that A's off-diagonal elements are those of the dipole-dipole
  ! tensor * alpha and its diagonal elements are one
! call pGM_NonBond_dip_dip_fields_short(numatoms, x, xv, zu)
  call pGM_pvt_NonBond_dip_dip_fields_short(atm_cnt, img_crd, ind_dip_img, zu, 1)  ! dip_fld/zu is zeroed inside
! zu = ap*zu + xv
! ru = bv - zu
!
! ! p(0) = r(0) 
! pu = ru

#ifdef MPI
  do idx = 1, my_short_img_cnt
! NOTE: my_short_img_lst(1:my_short_img_cnt) and my_full_img_lst(1:my_short_img_cnt) are identical
!     i = my_short_img_lst(idx)
     i = my_full_img_lst(idx)
#else
  do i = 1, atm_cnt
#endif
    ru(1:3,i) = dip_img(1:3,i) - (atm_pol_img(i)*zu(1:3,i) + ind_dip_img(1:3,i))
    pu(1:3,i) = ru(1:3,i)
  end do

  !  iteration 0:
  !  compute <ru(0),ru(0)>

  call vec_mul_sum(atm_cnt, ru, ru, rdotr1)

  ! main loop
  do iter = 1, niter

    ! compute Ap(i) = A * p(i)
    ! compute alpha(i) = <r(i),r(i)>/<p(i),Ap(i)>
!   call pGM_NonBond_dip_dip_fields_short(numatoms, x, pu, zu) 
    call pGM_pvt_NonBond_dip_dip_fields_short(atm_cnt, img_crd, pu, zu, 1)
!   zu = ap*zu + pu
!   pdotz = dot_product(pu, zu)
!   alpha = rdotr1/pdotz

#ifdef MPI
    do idx = 1, my_short_img_cnt
! NOTE: my_short_img_lst(1:my_short_img_cnt) and my_full_img_lst(1:my_short_img_cnt) are identical
!       i = my_short_img_lst(idx)
       i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      zu(1:3, i) = atm_pol_img(i) * zu(1:3, i) + pu(1:3, i)   ! This is the total dipole.
    end do

    call vec_mul_sum(atm_cnt, pu, zu, pdotz)
    alpha = rdotr1/pdotz

    ! update x(i) = x(i-1) + alpha(i) p(i-1)
    !        r(i) = r(i-1) - alpha(i) Ap(i-1)
#ifdef MPI
    do idx = 1, my_short_img_cnt
! NOTE: my_short_img_lst(1:my_short_img_cnt) and my_full_img_lst(1:my_short_img_cnt) are identical
!       i = my_short_img_lst(idx)
       i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
!     xv = xv + alpha*pu
!     ru = ru - alpha*zu
      ind_dip_img(1:3, i) = ind_dip_img(1:3, i) + alpha*pu(1:3, i)
      ru(1:3, i)          = ru(1:3, i)          - alpha*zu(1:3, i)
    end do

    ! do not check convergence, just for info only
    if ( pol_gauss_verbose == 3 ) then
      call vec_abs_max(atm_cnt, ru, r_norm)
      call vec_abs_sum(atm_cnt, dip_img, bsum)
      bnorm = bsum / dble(3*atm_cnt)
      r_norm = r_norm / bnorm
#ifdef MPI
      if (master) &
#endif
      write(mdout, '(a,i3,a,f14.10)') ' pGM precond: iter = ', iter, ', residue norm: ', r_norm
    end if

  if (iter .lt. niter) then

! these steps are no longer needed for the last step of the loop

    ! compute beta(i) = <r(i),z(i)>/<r(i-1),z(i-1)>
    call vec_mul_sum(atm_cnt, ru, ru, rdotr2)
    beta = rdotr2/rdotr1
    rdotr1 = rdotr2

    ! update p(i) = z(i) + beta(i) p(i-1)

#ifdef MPI
    do idx = 1, my_short_img_cnt
! NOTE: my_short_img_lst(1:my_short_img_cnt) and my_full_img_lst(1:my_short_img_cnt) are identical
!       i = my_short_img_lst(idx)
       i = my_full_img_lst(idx)
#else
    do i = 1, atm_cnt
#endif
      pu(1:3, i) = ru(1:3, i) + beta*pu(1:3, i)
    end do
   end if
  end do ! do iter

  ! Also do a peek step here with the the undamped SOR step

#ifdef MPI
  do idx = 1, my_short_img_cnt
! NOTE: my_short_img_lst(1:my_short_img_cnt) and my_full_img_lst(1:my_short_img_cnt) are identical
!       i = my_short_img_lst(idx)
       i = my_full_img_lst(idx)
#else
  do i = 1, atm_cnt
#endif
    ind_dip_img(1:3, i) = ind_dip_img(1:3, i) + ru(1:3, i)*atm_pol_img(i)
  end do

#ifdef MPI
! by now, only the short-range dipoles are correct on all nodes. Need to distribute to
! the nodes that have long-range contacts.
  call pGM_gather_long_used_img(atm_cnt, ind_dip_img)
#endif

call update_pme_time(pme_misc_timer)

end subroutine local_precondition

!--------------------------------------------------------------------------------------------
end subroutine pGM_INDUCED_eval
!--------------------------------------------------------------------------------------------
subroutine pGM_pvt_NonBond_dip_dip_fields_short(atm_cnt, img_crd, img_dip, img_dip_field, key)

  implicit none

  integer, intent(in)           :: atm_cnt
  double precision, intent(in)  :: img_crd(3,atm_cnt)
  double precision, intent(in)  :: img_dip(3,atm_cnt)
  double precision, intent(out) :: img_dip_field(3,atm_cnt)
  integer,          intent(in)  :: key
  integer                       :: i
#ifdef MPI
  integer                       :: k

  do i = 1, my_short_img_cnt
    img_dip_field(:, my_full_img_lst(i)) = 0.d0
  end do
#else
  img_dip_field(:,:) = 0.d0
#endif

  call pGM_NonBond_dip_dip_fields_short(atm_cnt, img_crd, img_dip, img_dip_field)

#ifdef MPI
  if (key .eq. 2) then
    call pGM_reduce_short_distr_full_used_img(atm_cnt, img_dip_field)
  else
    call pGM_reduce_short_distr_short_used_img(atm_cnt, img_dip_field)
  end if
#endif

end subroutine pGM_pvt_NonBond_dip_dip_fields_short
!------------------------------------------------------------------------
subroutine pGM_pvt_NonBond_dip_dip_fields(atm_cnt, img_crd, img_dip, img_dip_field)

  implicit none

  integer, intent(in)           :: atm_cnt
  double precision, intent(in)  :: img_crd(3,*)
  double precision, intent(in)  :: img_dip(3,atm_cnt)
  double precision, intent(out) :: img_dip_field(3,atm_cnt)

  integer :: i

#ifdef MPI
  do i = 1, my_full_img_cnt
    img_dip_field(:, my_full_img_lst(i)) = 0.d0
  end do
#else
  img_dip_field(:,:) = 0.d0
#endif
  call pGM_NonBond_dip_dip_fields(atm_cnt, img_crd, img_dip, img_dip_field)

#ifdef MPI
  call pGM_reduce_full_distr_full_used_img(atm_cnt, img_dip_field)
#endif

end subroutine pGM_pvt_NonBond_dip_dip_fields
!----------------------------------------------------------
subroutine extrap_ind_dipole(atm_cnt, dipole_scf_init_order, &
                             dipole_scf_init_step, pol, e_perm, ind_dip)

  integer :: atm_cnt, dipole_scf_init_order, dipole_scf_init_step
  double precision :: pol(atm_cnt)
  double precision :: e_perm(3,atm_cnt)
  double precision :: ind_dip(3,atm_cnt)

  double precision :: ind_dip_new(3,atm_cnt)

  integer :: n, i, j, idx
  double precision :: c(8)

  integer, save    :: called=0

  integer :: my_start3, my_end3

#ifdef MPI
  my_start3 = my_img_lo * 3 - 2
  my_end3   = my_img_hi * 3
#else
  my_start3 = 1
  my_end3   = atm_cnt * 3
#endif
  called = called + 1

  ! record1 is used to save pol*E_perm from previous steps to
  ! Prepare for extrapolation by obtaining extrapolation coefficients c(:)

!  ind_dip_new(1:3* atm_cnt,1) = pol(1:3* atm_cnt) * e_perm(1:3* atm_cnt)
  call my_used_img_field_mult_pol13(atm_cnt, pol, e_perm, ind_dip_new)  ! can simply use e_perm instead?

!  if (sum(abs(record1(:,dipole_scf_init_step+1))) > 0d0) then
  if (reccnt(1) .ge. dipole_scf_init_step) then
    call least_square_extrap(atm_cnt, dipole_scf_init_step, record1, c, ind_dip_new)
  else
         ind_dip(:, 1:atm_cnt) = ind_dip_new(:, 1:atm_cnt)
    c = 0.d0
  end if

  call update_record(atm_cnt, dipole_scf_init_step, ind_dip_new, record1, recidx1, &
       currec(1), saved(1), reccnt(1))

  ! record2 is used to save total ind_dip()
  ! First extrap the total ind_dip from previous steps
  if ( dipole_scf_init_order >= 1 ) then
    if (reccnt(2) .ge. dipole_scf_init_step) then
    ! we have enough data, so proceed with extrapolation
#ifdef MPI
      do idx = 1, my_full_img_cnt
         i = my_full_img_lst(idx)
         ind_dip(:, i) = 0.d0
      end do
#else
      ind_dip = 0.d0
#endif
      do j = 1, dipole_scf_init_step
#ifdef MPI
        do idx = 1, my_full_img_cnt
           i = my_full_img_lst(idx)
           ind_dip(:, i) = ind_dip(:, i) + c(j)*record2(:, i, j)
        end do
#else
        ind_dip(:, 1:atm_cnt) = ind_dip(:, 1:atm_cnt) + c(j)*record2(:, 1:atm_cnt, j)
#endif
      end do
!      call correct_mult(atm_cnt, record2, dipole_scf_init_step, c, recorddim, ind_dip)
#ifdef MPI
      do idx = 1, my_full_img_cnt
         i = my_full_img_lst(idx)
         oldrec(:, i, 1) = ind_dip(:, i)
      end do
#else
      oldrec(:,1:atm_cnt,1) = ind_dip(:,1:atm_cnt) ! the predicted ind_dip is saved in oldrec(,1)
#endif
    else 
      if (reccnt(1) .eq. 1 .or. dipole_scf_init_order == 1) then
    ! don't have enough data, just use pol*E_perm or previous ind_dip, which supposes to be kept
#ifdef MPI
         do idx = 1, my_full_img_cnt
            i = my_full_img_lst(idx)
            ind_dip(:, i) = ind_dip_new(:, i)
         end do
#else
         ind_dip(:, 1:atm_cnt) = ind_dip_new(:, 1:atm_cnt)
#endif
      end if

      return

    end if
  end if

  ! record3 is used to save the difference of ind_dip(), or record2()
  ! Second extrap the difference from previous steps and
  ! add to total ind_dip
  if ( dipole_scf_init_order >= 2 ) then
    if (reccnt(3) .ge. dipole_scf_init_step ) then
    ! we have enough data, so proceed with extrapolation
      do j = 1, dipole_scf_init_step
#ifdef MPI
        do idx = 1, my_full_img_cnt
          i = my_full_img_lst(idx)
          ind_dip(:, i) = ind_dip(:, i) + c(j)*record3(:, i, j)
        end do
#else
        ind_dip(:, 1:atm_cnt) = ind_dip(:, 1:atm_cnt) + c(j)*record3(:, 1:atm_cnt, j)
#endif
      end do
      ! the predicted ind_dip is saved in oldrec(,2)
#ifdef MPI
      do idx = 1, my_full_img_cnt
        i = my_full_img_lst(idx)
        oldrec(:, i, 2) = ind_dip(:, i)
      end do
#else
      oldrec(:, 1:atm_cnt, 2) = ind_dip(:, 1:atm_cnt)
#endif
    end if
  end if

  ! record4 is used to save the 2nd-order difference of ind_dip
  ! Third extrap the 2nd-order difference from previous steps and
  ! add to total ind_dip
  if ( dipole_scf_init_order == 3 ) then
    if (reccnt(4) .ge. dipole_scf_init_step) then
    ! we have enough data, so proceed with extrapolation
      do j = 1, dipole_scf_init_step
#ifdef MPI
        do idx = 1, my_full_img_cnt
          i = my_full_img_lst(idx)
          ind_dip(:, i) = ind_dip(:, i) + c(j)*record4(:, i, j)
        end do
#else
        ind_dip(:, 1:atm_cnt) = ind_dip(:, 1:atm_cnt) + c(j)*record4(:, 1:atm_cnt, j)
#endif
      end do
    end if
  end if

  contains
!----------------------------------------------------------
!
!  Here, extrapolate the initial guess of induced dipoles using new pol * E_perm and
!  old iterated induced dipoles and pol * E_perm
!
!----------------------------------------------------------
subroutine least_square_extrap(atm_cnt, init_step, record, c, ind_dip_new)

  implicit none

  integer, intent(in)             :: atm_cnt
  integer, intent(in)             :: init_step
  double precision, intent(inout) :: record(3,atm_cnt,init_step)
  double precision, intent(out)   :: c(init_step)
  double precision, intent(in)    :: ind_dip_new(3,atm_cnt)

#ifdef MPI
  double precision, save, allocatable :: bufsend(:,:)
  double precision, save, allocatable :: bufrecv(:,:)
#endif
  double precision :: matrix(init_step,init_step)
  double precision :: b(init_step)
  double precision :: cg_r(init_step)
  double precision :: cg_p(init_step)
  double precision :: cg_matrix_p(init_step)
  double precision :: alpha, beta, alpha_t

  integer :: i, j, k, n3, ier

#ifdef MPI
  if (.not. allocated(bufsend)) &
     allocate(bufsend(init_step,init_step+1), bufrecv(init_step,init_step+1))

  do i = 1, init_step
    bufsend(i, init_step+1) = sum(record(:, my_img_lo:my_img_hi, i) * ind_dip_new(:,my_img_lo:my_img_hi))
    do j = i, init_step
      bufsend(i, j) = sum(record(:,my_img_lo:my_img_hi, i) * record(:,my_img_lo:my_img_hi, j))
      if (i < j) bufsend(j,i) = bufsend(i,j)
    end do
  end do
  call mpi_allreduce(bufsend, bufrecv, init_step*(init_step+1), MPI_DOUBLE_PRECISION, MPI_SUM, pmemd_comm, ier)
  matrix(1:init_step, 1:init_step) = bufrecv(1:init_step, 1:init_step)
  b(1:init_step) = bufrecv(1:init_step, 1+init_step)
#else
  do i = 1, init_step
    b(i) = sum(record(:, 1:atm_cnt, i) * ind_dip_new(:, 1:atm_cnt))
    do j = i, init_step
      matrix(i, j) = sum(record(:, 1:atm_cnt, i) * record(:, 1:atm_cnt, j))
      if (i < j) matrix(j,i) = matrix(i,j)
    end do
  end do
#endif
! solve least square equation

  c = 0d0
  cg_r = b
  cg_p = cg_r

  do i = 1, init_step
    cg_matrix_p = 0d0
    do j = 1, init_step
      do k = 1, init_step
        cg_matrix_p(j) = cg_matrix_p(j) + matrix(j,k)*cg_p(k)
      end do
    end do
    alpha_t = dot_product(cg_matrix_p,cg_p) 
    if (abs(alpha_t) .gt. 1.0d-50) then
      alpha = dot_product(cg_r,cg_r)/alpha_t
    else
      alpha = 0.0d0
    endif

    c(1:init_step) = c(1:init_step) + alpha*cg_p(1:init_step)
    beta = dot_product(cg_r,cg_r)
    cg_r = cg_r - alpha*cg_matrix_p
    if (beta .gt. 1.0d-50) then
    beta = dot_product(cg_r,cg_r)/beta
 else
    beta = 0.0d0
 end if
    cg_p = cg_r + beta*cg_p
  end do

end subroutine least_square_extrap
!----------------------------------------------------------
end subroutine extrap_ind_dipole
!----------------------------------------------------------
subroutine update_record(atm_cnt, dipole_scf_init_step, dipnew, record, recidx, currec, saved, reccnt)

  implicit none

  integer :: atm_cnt
  integer :: dipole_scf_init_step
  integer :: currec
  integer :: reccnt
  integer :: saved
  integer :: recidx(dipole_scf_init_step)
  double precision :: record(3, atm_cnt, dipole_scf_init_step+1)
  double precision :: dipnew(3, atm_cnt)
  integer :: i, j, idx

#if 0
  do i = dipole_scf_init_step-1, 1, -1
    record(:,i+1) = record(:,i)
  end do
  record(:, 1) = dipnew(:)
#endif

  if (currec .eq. saved) then
    currec = currec + 1
    if (currec .gt. dipole_scf_init_step) currec = 1
  end if
#ifdef MPI
  do idx = 1, my_full_img_cnt
     i = my_full_img_lst(idx)
#else
  do i = 1, atm_cnt
#endif
    record(:, i, currec) = dipnew(:, i)
  end do

  if (saved .eq. 0) then
! reset the index array, synchronize all index arrays.
    j = currec
    do i = 1, dipole_scf_init_step
      recidx(i) = j
      j = j + 1
      if (j .gt. dipole_scf_init_step) j = 1
    end do
  else
! cycle the index array, for book keeping.
  j = recidx(1)
  do i = 2, dipole_scf_init_step
    recidx(i-1) = recidx(i)
  end do
  recidx(dipole_scf_init_step) = j

  end if

  saved = currec

  reccnt = reccnt + 1

end subroutine update_record
!------------------------------------------------------------------------
subroutine my_used_img_field_mult_pol13(atm_cnt, pol, efield, ind_dip)

  implicit none

  integer          :: atm_cnt
  double precision :: pol(atm_cnt)
  double precision :: efield(3,atm_cnt)
  double precision :: ind_dip(3,atm_cnt)
  integer          :: n, i

#ifdef MPI
    do n = 1, my_full_img_cnt
      i = my_full_img_lst(n)
#else
    do i = 1, atm_cnt
#endif
      ind_dip(:, i) = pol(i) * efield(:, i)
    end do

end subroutine my_used_img_field_mult_pol13

subroutine field_mult_pol33(atm_cnt, pol3, efield, ind_dip)

  implicit none

  integer          :: atm_cnt
  double precision :: pol3(3*atm_cnt)
  double precision :: efield(3*atm_cnt)
  double precision :: ind_dip(3*atm_cnt)

    ind_dip(1:3*atm_cnt) = pol3(1:3*atm_cnt) * efield(1:3*atm_cnt)

end subroutine field_mult_pol33

subroutine grad_mult_pol33(atm_cnt, pol3, efield, ind_dip)

  implicit none

  integer          :: atm_cnt
  double precision :: pol3(3*atm_cnt)
  double precision :: efield(3*atm_cnt)
  double precision :: ind_dip(3*atm_cnt)

    ind_dip(1:3*atm_cnt) = -pol3(1:3*atm_cnt) * efield(1:3*atm_cnt)

end subroutine grad_mult_pol33

subroutine field_mult_pol13(atm_cnt, pol, efield, ind_dip)

  implicit none

  integer          :: atm_cnt
  double precision :: pol(atm_cnt)
  double precision :: efield(3, atm_cnt)
  double precision :: ind_dip(3, atm_cnt)

  integer          :: i

  do i = 1, atm_cnt
     ind_dip(1:3,i) = pol(i) * efield(1:3, i)
  end do

end subroutine field_mult_pol13

subroutine grad_mult_pol13(atm_cnt, pol, efield, ind_dip)

  implicit none

  integer          :: atm_cnt
  double precision :: pol(atm_cnt)
  double precision :: efield(3, atm_cnt)
  double precision :: ind_dip(3, atm_cnt)

  integer          :: i

  do i = 1, atm_cnt
     ind_dip(1:3,i) = -pol(i) * efield(1:3, i)
  end do

end subroutine grad_mult_pol13

subroutine correct_mult(atm_cnt, record, nc, c, idim1, ind_dip)

  implicit none

  integer          :: atm_cnt
  integer          :: nc
  integer          :: idim1
  double precision :: c(nc)
  double precision :: record(idim1, nc)
  double precision :: ind_dip(3*atm_cnt)

  integer          :: i

  do i = 1, nc
     ind_dip(1:3*atm_cnt) = ind_dip(1:3*atm_cnt) + c(i) * record(1:3*atm_cnt, i)
  end do

end subroutine correct_mult

!------------------------------------------------------------------------
! This needs to be called immediately after every nonbonded list update
!------------------------------------------------------------------------
subroutine pGM_INDUCED_img(atm_cnt, img_atm_map)

  implicit none

  integer, intent(in)              :: atm_cnt
  integer, intent(in)              :: img_atm_map(*)
  integer :: iatm, i

  if ( .not. allocated(pol_img)) allocate(pol_img(atm_cnt))
  if ( .not. allocated(is_polarizable_img)) allocate(is_polarizable_img(atm_cnt))
  if ( .not. allocated(inv_pol_img)) allocate(inv_pol_img(atm_cnt))
  if ( .not. allocated(ind_dip_img)) allocate(ind_dip_img(3,atm_cnt))

  if ( .not. allocated(old_img_atm_map)) allocate(old_img_atm_map(atm_cnt))
#ifdef MPI
  if ( .not. allocated(old_atm_img_map)) allocate(old_atm_img_map(atm_cnt))
  if ( .not. allocated(old_gbl_owned_imgs_tbl)) allocate(old_gbl_owned_imgs_tbl(2,0:numtasks))
  if ( .not. allocated(my_full_img_lst_old)) allocate(my_full_img_lst_old(atm_cnt))
!  if ( .not. allocated(old_img_owner_map)) allocate(old_img_owner_map(atm_cnt))
#endif

  do i = 1, atm_cnt
    iatm = img_atm_map(i)
    is_polarizable_img(i) = is_polarizable(iatm)
    pol_img(i)     = polarizability(iatm)
    inv_pol_img(i) = inv_pol(iatm)
  enddo
  if (img_atm_map_saved) then
    call shuffle_records(atm_cnt, old_img_atm_map, img_atm_map)
  end if
  old_img_atm_map(1:atm_cnt) = img_atm_map(1:atm_cnt)
#ifdef MPI
  old_atm_img_map(1:atm_cnt) = gbl_atm_img_map(1:atm_cnt)
  old_gbl_owned_imgs_tbl     = gbl_owned_imgs_tbl
  old_img_lo  = my_img_lo
  old_img_hi  = my_img_hi
!  old_img_owner_map = img_owner_map
  my_full_img_cnt_old = my_full_img_cnt
  my_full_img_lst_old(1:my_full_img_cnt) = my_full_img_lst(1:my_full_img_cnt)
#endif
  img_atm_map_saved = .true.

end subroutine pGM_INDUCED_img
!---------------------------------------------------------
subroutine pGM_INDUCED_img_update(atm_cnt, img_atm_map)

!  use pol_gauss_direct_data, only  : displacement, displacement_img

  implicit none

  integer, intent(in) :: atm_cnt
  integer, intent(in) :: img_atm_map(*)

  integer :: iatm, i

end subroutine pGM_INDUCED_img_update
!---------------------------------------------------------
subroutine shuffle_records(atm_cnt, old_img_atm_map, new_img_atm_map)

  use pol_gauss_mdin_dat_mod, only : dipole_scf_init, dipole_scf_init_order, use_average,&
                             dipole_scf_init_step, scf_solv_opt, pol_gauss_verbose

  implicit none

  integer, intent(in) :: atm_cnt

  integer, intent(in) :: old_img_atm_map(atm_cnt)
  integer, intent(in) :: new_img_atm_map(atm_cnt)

  double precision    :: buf(3, atm_cnt)
  integer             :: i, idx, img, iatm, ier
#ifdef MPI
  integer             :: mask(atm_cnt)
  integer             :: lst_need(atm_cnt)
  integer             :: need_cnt
#endif

#ifdef MPI
! all above vectors should be shuffled below
  call mpi_get_all_absent_old_imgs(atm_cnt)
#endif

  call shuffle_one(ind_dip_img, ind_dip)

  do i = 1, dipole_scf_init_step
    call shuffle_one(record1(1,1,i), buf)
    call shuffle_one(record2(1,1,i), buf)
    call shuffle_one(record3(1,1,i), buf)
    call shuffle_one(record4(1,1,i), buf)
  end do
  call shuffle_one(oldrec(1,1,1), buf)
  call shuffle_one(oldrec(1,1,2), buf)

if (dipole_scf_init .eq. 2 .or. pol_gauss_verbose == 6) then
  call shuffle_one(olddip(1,1), buf)
end if

  contains
subroutine shuffle_one(record, buf)

  implicit none

  double precision, intent(inout)  :: buf(3, atm_cnt)
  double precision, intent(inout)  :: record(3, atm_cnt)

  integer                          :: img_i, atm_i, ierr

  do img_i = 1, atm_cnt
    atm_i = old_img_atm_map(img_i)
    buf(1:3, atm_i) = record(1:3, img_i)    ! restore in the order of atom indices
  end do

  do img_i = 1, atm_cnt
    atm_i = new_img_atm_map(img_i)
    record(1:3, img_i) = buf(1:3, atm_i)    ! back to the order of new image indices
  end do
  
end subroutine shuffle_one
!---------------------------------------------------------
end subroutine shuffle_records
!---------------------------------------------------------
function pGM_INDUCED_readparm(nf, atm_cnt)
  use pol_gauss_mdin_dat_mod, only : pol_gauss_verbose
#ifdef MPI
  use parallel_dat_mod
#endif

  implicit none

  integer :: pGM_INDUCED_readparm
  integer,intent(in) :: nf, atm_cnt

  integer :: j,n,ier,dim1,numpolar
  double precision :: polmin

  pGM_INDUCED_readparm = 0
  polmin = 1.d-6

  ! initialize things
  allocate(polarizability( atm_cnt), &
           is_polarizable( atm_cnt), &
           inv_pol( atm_cnt), &
           ind_dip(3, atm_cnt), &
           dip_field(3, atm_cnt), &
           cart_dipole_field(3, atm_cnt), &
           stat=ier)
  REQUIRE(ier==0)

  is_polarizable = .false.
  polarizability = 0.d0
  ind_dip = 0.0d0

  dim1 = 1
  numpolar =  atm_cnt
  call POL_GAUSS_read_real_list_data('POL_GAUSS_POLARIZABILITY_',nf,dim1,numpolar,polarizability)
  if (pol_gauss_verbose == 2) then
    write(117,'(/,A,I8)') 'Finding polarizability',numpolar
    write(117,'(5E16.8)') polarizability(1:numpolar)
  end if
  do n = 1, numpolar
    if ( polarizability(n) > polmin )then
      is_polarizable(n) = .true.
      inv_pol(n) = 1.d0/polarizability(n)
    else
      inv_pol(n) = 0.d0
    endif
  enddo

  call scf_solv_init(atm_cnt)

  pGM_INDUCED_readparm = 1

end function pGM_INDUCED_readparm
!----------------------------------------------------------
subroutine scf_solv_init(atm_cnt)

  use pol_gauss_mdin_dat_mod, only : pol_gauss_verbose,scf_cg_niter, dipole_scf_init_order, dipole_scf_init_step
!  use pol_gauss_multipoles_mod, only : start_multipoles, end_multipoles, my_mpole_cnt

  implicit none

  integer, intent(in) :: atm_cnt

  integer :: ierr, n
#ifdef MPI
  integer :: taskmap_idx, node
#endif

  if (scf_solv_inited) return

  if (pol_gauss_verbose /= 0) write(mdout, '(/a)') ' scf_solv_init(): Allocating working arrays'

  allocate(record1(3, atm_cnt, dipole_scf_init_step), &  ! record[1-4] arrays are 3 x atm_cnt, the full length
           record2(3, atm_cnt, dipole_scf_init_step), &
           record3(3, atm_cnt, dipole_scf_init_step), &
           record4(3, atm_cnt, dipole_scf_init_step), &
           oldrec(3, atm_cnt, 2), &
           perm_field_img(3, atm_cnt), &
           alpha_perm_field_img(3, atm_cnt), &
           stat = ierr)
#if 0
           rv(3,atm_cnt), &
           ru(3,atm_cnt), &
           pv(3,atm_cnt), &
           pu(3,atm_cnt), &
           zv(3,atm_cnt), &
           zu(3,atm_cnt), &
#endif
  REQUIRE(ierr==0)

#ifdef MPI

  my_short_img_cnt = 0
  my_full_img_cnt  = 0

! for scf iteration communcation
! may not need everything here
    allocate(my_short_img_lst(atm_cnt), &
             my_full_img_lst(atm_cnt), &
             my_used_img_mask(atm_cnt), &
             recv_taskmap(numtasks-1), &
             send_taskmap(numtasks-1), &
             my_img_cnts_short(0:numtasks-1), &
             my_img_cnts_full(0:numtasks-1), &
             peer_img_cnts_short(0:numtasks-1), &
             peer_img_cnts_full(0:numtasks-1), &
             stat = ierr)
  REQUIRE(ierr==0)
#if 0
             short_recv_cnt(0:numtasks-1), &
             short_send_cnt(0:numtasks-1), &
             full_recv_cnt(0:numtasks-1), &
             full_send_cnt(0:numtasks-1), &
             gather_recv_cnt(0:numtasks-1), &
             gather_send_cnt(0:numtasks-1), &
             gbl_all_used_img_cnt(2,0:numtasks-1), &
             gbl_img_range(2,0:numtasks-1), &
#endif

  my_used_img_mask = 0

    taskmap_idx = 0
    node = mytaskid + 1
    do
      if (node .ge. numtasks) node = 0
      if (node .eq. mytaskid) exit
      taskmap_idx = taskmap_idx + 1
      recv_taskmap(taskmap_idx) = node
      node = node + 1
    end do

    do taskmap_idx = 1, numtasks - 1
      send_taskmap(taskmap_idx) = recv_taskmap(numtasks - taskmap_idx)
    end do
             
#endif

  recorddim = atm_cnt

  record1 = 0.0d0
  record2 = 0.0d0
  record3 = 0.0d0
  record4 = 0.0d0

  scf_cg_niter = max(2, scf_cg_niter)

  non_polarizable_cnt = 0

  ! set up polarizability arrays for vectorized process
  do n = 1, atm_cnt
    if (is_polarizable(n)) then
      inv_pol(n) = 1.0d0/polarizability(n)
    else
      inv_pol(n) = 0.d0
      allpolarizable = .false.
      if (.not.allocated(non_polarizable_lst)) allocate(non_polarizable_lst(atm_cnt))
      non_polarizable_cnt = non_polarizable_cnt + 1
      non_polarizable_lst(non_polarizable_cnt) = n
    end if
  end do

  allocate(recidx1(dipole_scf_init_step), &
           recidx2(dipole_scf_init_step), &
           recidx3(dipole_scf_init_step), &
           recidx4(dipole_scf_init_step), &
           stat = ierr)

  REQUIRE(ierr==0)

  do n = 1, dipole_scf_init_step
    recidx1(n) = n
    recidx2(n) = n
    recidx3(n) = n
    recidx4(n) = n
  end do

  reccnt = 0
  currec = 1
  saved  = 0

!  mystart = start_multipoles
!  myend   = end_multipoles
!  mycnt   = my_mpole_cnt
!  mycnt3  = mycnt * 3
!  mystart3= mystart * 3 - 2
!  myend3  = myend * 3

  scf_solv_inited = .true.
  initial = .true.
end subroutine scf_solv_init
!------------------------------------------------------------------------
#ifdef MPI

! MPI functions for induced dipole scf iterations

subroutine pGM_INDUCED_bcast(atm_cnt)

  use parallel_dat_mod
!  use pol_gauss_multipoles_mod, only : my_mpole_cnt

   implicit none

   integer atm_cnt,ier,n,j

   if(.not.master) then
      allocate(polarizability(atm_cnt), &
               is_polarizable(atm_cnt), &
               inv_pol(atm_cnt), &
               ind_dip(3,atm_cnt), &
               dip_field(3,atm_cnt), &
               cart_dipole_field(3,atm_cnt), &
               stat=ier)
      REQUIRE(ier==0)
   end if  

   if (.not.master) then
     ind_dip(:,:) = 0.d0
   end if

   call mpi_bcast(polarizability,atm_cnt,MPI_DOUBLE_PRECISION,0,pmemd_comm,ier)
      REQUIRE(ier==0)
   call mpi_bcast(is_polarizable,atm_cnt,MPI_LOGICAL,0,pmemd_comm,ier)
      REQUIRE(ier==0)
   call mpi_bcast(inv_pol,atm_cnt,MPI_LOGICAL,0,pmemd_comm,ier)
      REQUIRE(ier==0)
  
   if(.not.master) then
     call scf_solv_init(atm_cnt)
   end if

#if 0
! set up shifts and counts for scf_solver, assuming my_mpole_cnt has been set
   allocate(recvcnt3(0:numtasks), &
            recvdisp3(0:numtasks), stat=ier)
      REQUIRE(ier==0)

   mysendcnt3 = my_mpole_cnt * 3
   call mpi_allgather(mysendcnt3, 1, MPI_INTEGER, recvcnt3, 1, MPI_INTEGER, pmemd_comm, ier)
      REQUIRE(ier==0)

   recvdisp3(0) = 0
   do n = 1, numtasks-1
     recvdisp3(n) = recvdisp3(n-1) + recvcnt3(n-1)
   end do
#endif

end subroutine pGM_INDUCED_bcast

subroutine init_scf_iter_comm()

  use prmtop_dat_mod, only : natom

  implicit none

  integer               :: irecv_stat(mpi_status_size)
  integer               :: isend_stat(mpi_status_size, numtasks - 1)
  integer               :: recv_req(numtasks - 1)
  integer               :: send_req(numtasks - 1)

  
  integer :: taskmap_idx
  integer :: wait_call
  integer :: task, ier
  integer :: recv_cnt
  integer :: node
  integer :: i, j, cnt_short, cnt_full, idx, img, cnt
  integer :: peer_lo, peer_hi, peer_short, peer_full
  double precision :: ave_img_cnt

  integer :: mask(natom)

  logical :: called = .false.

!  max_img_cnt = maxval(gbl_owned_imgs_tbl(2,:))+2
!  max_img_cnt = max(max_img_cnt, natom)
  max_img_cnt = natom

  if (allocated(recv_buf)) deallocate(recv_buf)
  if (allocated(send_buf)) deallocate(send_buf)
  if (allocated(my_img_lsts)) deallocate(my_img_lsts)
  if (allocated(peer_img_lsts)) deallocate(peer_img_lsts)

  allocate(recv_buf(3,max_img_cnt,0:numtasks-1), &
           send_buf(3,max_img_cnt,0:numtasks-1), &
           my_img_lsts(max_img_cnt,0:numtasks-1), &
           peer_img_lsts(max_img_cnt,0:numtasks-1))

! get used img lists

! separate short distance pairs from long ones

!  do i = 1, my_full_img_cnt
!    j = gbl_used_img_lst(i)
!    my_used_img_mask(j) = 0
!  end do

! mark the short list
!  do i = 1, my_short_img_cnt
!    my_used_img_mask(my_short_img_lst(i)) = 1
!    my_full_img_lst(i) = my_short_img_lst(i)
!  end do

  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(my_img_lsts(1, task), max_img_cnt, MPI_INTEGER, &
                   task, pgm_init_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  ave_img_cnt = dble(natom)/dble(numtasks)

  peer_img_cnts_short = 0

  do idx = 1, my_short_img_cnt
    img = my_full_img_lst(idx)
    if (img .lt. my_img_lo .or. img .gt. my_img_hi) then
      call findtaskid(img, task, gbl_owned_imgs_tbl, numtasks, ave_img_cnt)
      peer_img_cnts_short(task) = peer_img_cnts_short(task) + 1
      peer_img_lsts(peer_img_cnts_short(task),task) = img
    end if
  end do

  call mpi_barrier(mpi_comm_world,err_code_mpi)
  peer_img_cnts_full = peer_img_cnts_short

  do idx = my_short_img_cnt+1, my_full_img_cnt
    img = my_full_img_lst(idx)
    if (img .lt. my_img_lo .or. img .gt. my_img_hi) then
      call findtaskid(img, task, gbl_owned_imgs_tbl, numtasks, ave_img_cnt)
      peer_img_cnts_full(task) = peer_img_cnts_full(task) + 1
      peer_img_lsts(peer_img_cnts_full(task),task) = img
    end if
  end do
  call mpi_barrier(mpi_comm_world,err_code_mpi)

  peer_img_cnts_full(mytaskid) =0
  peer_img_cnts_short(mytaskid)=0

! append the counts to the end

  do taskmap_idx = 1, numtasks - 1
    task = send_taskmap(taskmap_idx)
    cnt  = peer_img_cnts_full(task)
    peer_img_lsts(cnt+1, task) = peer_img_cnts_short(task)
    peer_img_lsts(cnt+2, task) = peer_img_cnts_full(task)
    call mpi_isend(peer_img_lsts(1,task), cnt+2, MPI_INTEGER, &
                   task, pgm_init_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_INTEGER, recv_cnt, ier)
    my_img_cnts_short(task) = my_img_lsts(recv_cnt-1, task)
    my_img_cnts_full(task)  = my_img_lsts(recv_cnt, task)
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 

end subroutine init_scf_iter_comm
!------------------------------------------------------------------------
subroutine pGM_reduce_short_distr_short_used_img(atm_cnt, vec)

  implicit none

  integer, intent(in)             :: atm_cnt
  double precision, intent(inout) :: vec(3, atm_cnt)

  integer               :: irecv_stat(mpi_status_size)
  integer               :: isend_stat(mpi_status_size, numtasks - 1)
  integer               :: recv_req(numtasks - 1)
  integer               :: send_req(numtasks - 1)
  integer               :: recv_cnts(0 : numtasks - 1)

  integer :: taskmap_idx
  integer :: wait_call
  integer :: task, ier
  integer :: recv_cnt
  integer :: i, j, cnt

! gather and reduce partial vectors from peer nodes

  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(recv_buf(1, 1, task), 3*max_img_cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_get_short_fld_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

! only send image vectors owned by peer nodes

    do i = 1, peer_img_cnts_short(task)
      j = peer_img_lsts(i, task)
      send_buf(:, i, task) = vec(:, j)
    end do

    cnt = peer_img_cnts_short(task) * 3
    call mpi_isend(send_buf(1,1,task), cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_get_short_fld_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_DOUBLE_PRECISION, recv_cnt, ier)

! reduce just my own image vectors

    do i = 1, recv_cnt/3
      j = my_img_lsts(i, task)
      vec(:, j) = vec(:, j) + recv_buf(:, i, task)
    end do
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 

! distribute my own image vectors
  
  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(recv_buf(1, 1, task), 3*max_img_cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_snd_short_fld_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

! only send image vectors owned by peer nodes

    do i = 1, my_img_cnts_short(task)
      j = my_img_lsts(i, task)
      send_buf(:, i, task) = vec(:, j)
    end do

    cnt = my_img_cnts_short(task) * 3
    call mpi_isend(send_buf(1,1,task), cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_snd_short_fld_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_DOUBLE_PRECISION, recv_cnt, ier)

! distribute received image vectors

    do i = 1, recv_cnt/3
      j = peer_img_lsts(i, task)
      vec(:, j) = recv_buf(:, i, task)
    end do
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 

end subroutine pGM_reduce_short_distr_short_used_img
!------------------------------------------------------------------------
subroutine pGM_reduce_short_distr_full_used_img(atm_cnt, vec)

  implicit none

  integer, intent(in)             :: atm_cnt
  double precision, intent(inout) :: vec(3, atm_cnt)

  integer               :: irecv_stat(mpi_status_size)
  integer               :: isend_stat(mpi_status_size, numtasks - 1)
  integer               :: recv_req(numtasks - 1)
  integer               :: send_req(numtasks - 1)
  integer               :: recv_cnts(0 : numtasks - 1)

  integer :: taskmap_idx
  integer :: wait_call
  integer :: task, ier
  integer :: recv_cnt
  integer :: i, j, cnt

! gather and reduce partial vectors from peer nodes

  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(recv_buf(1, 1, task), 3*max_img_cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_get_short_fld_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

! only send image vectors owned by peer nodes

    do i = 1, peer_img_cnts_short(task)
      j = peer_img_lsts(i, task)
      send_buf(:, i, task) = vec(:, j)
    end do

    cnt = peer_img_cnts_short(task) * 3
    call mpi_isend(send_buf(1,1,task), cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_get_short_fld_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_DOUBLE_PRECISION, recv_cnt, ier)

! reduce just my own image vectors

    do i = 1, recv_cnt/3
      j = my_img_lsts(i, task)
      vec(:, j) = vec(:, j) + recv_buf(:, i, task)
    end do
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 

! distribute my own image vectors
  
  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(recv_buf(1, 1, task), 3*max_img_cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_snd_full_fld_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

! only send image vectors owned by peer nodes

    do i = 1, my_img_cnts_full(task)
      j = my_img_lsts(i, task)
      send_buf(:, i, task) = vec(:, j)
    end do

    cnt = my_img_cnts_full(task) * 3
    call mpi_isend(send_buf(1,1,task), cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_snd_full_fld_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_DOUBLE_PRECISION, recv_cnt, ier)

! distribute received image vectors

    do i = 1, recv_cnt/3
      j = peer_img_lsts(i, task)
      vec(:, j) = recv_buf(:, i, task)
    end do
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 

end subroutine pGM_reduce_short_distr_full_used_img
!------------------------------------------------------------------------
subroutine pGM_reduce_full_distr_full_used_img(atm_cnt, vec)

  implicit none

  integer, intent(in)             :: atm_cnt
  double precision, intent(inout) :: vec(3, atm_cnt)

  integer               :: irecv_stat(mpi_status_size)
  integer               :: isend_stat(mpi_status_size, numtasks - 1)
  integer               :: recv_req(numtasks - 1)
  integer               :: send_req(numtasks - 1)
  integer               :: recv_cnts(0 : numtasks - 1)

  integer :: taskmap_idx
  integer :: wait_call
  integer :: task, ier
  integer :: recv_cnt
  integer :: i, j, cnt

! gather and reduce partial vectors from peer nodes

  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(recv_buf(1, 1, task), 3*max_img_cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_get_full_fld_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

! only send image vectors owned by peer nodes

    do i = 1, peer_img_cnts_full(task)
      j = peer_img_lsts(i, task)
      send_buf(:, i, task) = vec(:, j)
    end do
    cnt = peer_img_cnts_full(task) * 3
    call mpi_isend(send_buf(1,1,task), cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_get_full_fld_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_DOUBLE_PRECISION, recv_cnt, ier)

! reduce just my own image vectors

    do i = 1, my_img_cnts_full(task)
      j = my_img_lsts(i, task)
      vec(:, j) = vec(:, j) + recv_buf(:, i, task)
    end do
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 

! distribute my own image vectors
  
  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(recv_buf(1, 1, task), 3*max_img_cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_snd_full_fld_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

! only send image vectors owned by peer nodes

    do i = 1, my_img_cnts_full(task)
      j = my_img_lsts(i, task)
      send_buf(:, i, task) = vec(:, j)
    end do

    cnt = my_img_cnts_full(task) * 3
    call mpi_isend(send_buf(1,1,task), cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_snd_full_fld_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_DOUBLE_PRECISION, recv_cnt, ier)

! distribute received image vectors

    do i = 1, peer_img_cnts_full(task)
      j = peer_img_lsts(i, task)
      vec(:, j) = recv_buf(:, i, task)
    end do
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 

end subroutine pGM_reduce_full_distr_full_used_img
!------------------------------------------------------------------------
subroutine pGM_gather_long_used_img(atm_cnt, vec)

  implicit none

  integer, intent(in)             :: atm_cnt
  double precision, intent(inout) :: vec(3, atm_cnt)

  integer               :: irecv_stat(mpi_status_size)
  integer               :: isend_stat(mpi_status_size, numtasks - 1)
  integer               :: recv_req(numtasks - 1)
  integer               :: send_req(numtasks - 1)
  integer               :: recv_cnts(0 : numtasks - 1)

  integer :: taskmap_idx
  integer :: wait_call
  integer :: task, ier
  integer :: recv_cnt
  integer :: i, j, cnt

  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(recv_buf(1, 1, task), 3*max_img_cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_gath_fld_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

    do i = 1, my_img_cnts_full(task) - my_img_cnts_short(task)
      j = my_img_lsts(i+my_img_cnts_short(task), task)
      send_buf(:, i, task) = vec(:, j)
    end do

    cnt = my_img_cnts_full(task) - my_img_cnts_short(task)

    call mpi_isend(send_buf(1,1,task), cnt*3, MPI_DOUBLE_PRECISION, &
                   task, pgm_gath_fld_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_DOUBLE_PRECISION, recv_cnt, ier)
    do i = 1, peer_img_cnts_full(task) - peer_img_cnts_short(task)
      j = peer_img_lsts(i+peer_img_cnts_short(task), task)
      vec(:, j) = recv_buf(:, i, task)
    end do
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 
  
end subroutine pGM_gather_long_used_img

#if 0
subroutine pGM_append_long_range_img(used_img_cnt, used_img_lst)

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

!------------------------------------------------------------------------
subroutine mpi_get_all_absent_old_imgs(atm_cnt)

  use pol_gauss_mdin_dat_mod, only : dipole_scf_init, dipole_scf_init_step, pol_gauss_verbose

  implicit none

  integer, intent(in) :: atm_cnt

  integer             :: mask(atm_cnt)
  integer             :: absent_cnt
  integer             :: absent_in_new_lst(atm_cnt)
  integer             :: taskid_rec(atm_cnt)
  integer             :: need_cnts(0:numtasks)
  integer             :: need_shft(0:numtasks)
  integer             :: recv_cnts(0:numtasks)
  integer             :: send_cnts(0:numtasks)
  integer             :: need_from_old_lst(atm_cnt)
  integer             :: ibufold(atm_cnt)
  integer             :: ibufnew(atm_cnt)

  double precision, allocatable    :: sendbuf(:,:,:)
  double precision, allocatable    :: recvbuf(:,:,:)
  integer             :: sendbufsize
  integer             :: recvbufsize
  integer             :: vect_cnts

  integer, save, allocatable:: request_lsts(:,:)
  integer             :: my_img_cnt
  integer             :: taskmap_idx
  integer             :: task
  integer             :: task_o
  integer             :: wait_call
  integer             :: recev_cnt
  integer             :: irecv_stat(mpi_status_size)
  integer             :: isend_stat(mpi_status_size, numtasks - 1)
  integer             :: recv_req(numtasks - 1)
  integer             :: send_req(numtasks - 1)
  integer             :: recv_req_dat(numtasks - 1)
  integer             :: send_req_dat(numtasks - 1)

  integer             :: idx, i, n, oldtask, cnt, iatm, ier, img, recv_cnt, img_o
  double precision    :: ave_img_cnt

! first, figure out the new images that my_task does not have.

! clear the masks

  do idx = 1, my_full_img_cnt
    img = my_full_img_lst(idx)
    iatm= gbl_img_atm_map(img)
    mask(iatm) = 0
  end do

! mark the masks by old images

  do idx = 1, my_full_img_cnt_old
    img = my_full_img_lst_old(idx)
    iatm= old_img_atm_map(img)
    mask(iatm) = 1
  end do

! count and record the images that are absent in the present list
  absent_cnt  = 0
  need_cnts   = 0

  ave_img_cnt = dble(atm_cnt)/dble(numtasks)

  do idx = 1, my_full_img_cnt
    img = my_full_img_lst(idx)
    iatm= gbl_img_atm_map(img)
    if (mask(iatm) .eq. 0) then
      mask(iatm) = 2
      img_o = old_atm_img_map(iatm)
      call findtaskid(img_o, task_o, old_gbl_owned_imgs_tbl, numtasks, ave_img_cnt)
      absent_cnt = absent_cnt + 1
      ibufold(absent_cnt) = img_o
      ibufnew(absent_cnt) = img
      taskid_rec(absent_cnt) = task_o
      need_cnts(task_o) = need_cnts(task_o) + 1
    end if
  end do

! sort the lists by taskid
  need_shft(0) = 0
  do task = 1, numtasks
    need_shft(task) = need_shft(task-1) + need_cnts(task-1) + 1
  end do

  do task = 0, numtasks - 1
    need_from_old_lst(need_shft(task)+need_cnts(task)+1) = 0
    absent_in_new_lst(need_shft(task)+need_cnts(task)+1) = 0
  end do

  need_cnts = 0

  do idx = 1, absent_cnt
    task = taskid_rec(idx)
    need_cnts(task) = need_cnts(task) + 1
    need_from_old_lst(need_shft(task)+need_cnts(task)) = ibufold(idx)
    absent_in_new_lst(need_shft(task)+need_cnts(task)) = ibufnew(idx)
  end do

  my_img_cnt = old_gbl_owned_imgs_tbl(2, mytaskid)

! wait in the queue to receive the lists of the needed old images

  allocate(request_lsts(my_img_cnt, 0:numtasks-1))

  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(request_lsts(1, task), my_img_cnt, MPI_INTEGER, &
                   task, pgm_list_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1
    task = send_taskmap(taskmap_idx)
      call mpi_isend(need_from_old_lst(need_shft(task)+1), need_cnts(task), MPI_INTEGER, &
                   task, pgm_list_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  send_cnts = 0

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_INTEGER, recv_cnt, ier)
    send_cnts(task) = recv_cnt   ! This is number of images coordinates to send to the peer.
  end do

  vect_cnts   = 1 + 4 * dipole_scf_init_step + 2
  if (dipole_scf_init .eq. 2 .or. pol_gauss_verbose == 6) vect_cnts = vect_cnts + 1

  recvbufsize = (maxval(need_cnts)+1)*vect_cnts

  allocate(recvbuf(3,recvbufsize,0:numtasks-1))

  do taskmap_idx = 1, numtasks - 1

    task = recv_taskmap(taskmap_idx)
  
      call mpi_irecv(recvbuf(1, 1, task), 3*recvbufsize, MPI_DOUBLE_PRECISION, &
                   task, pgm_gath_fld_tag, pmemd_comm, recv_req_dat(taskmap_idx), ier)
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 
  sendbufsize = (maxval(send_cnts)+1)*vect_cnts
  allocate(sendbuf(3,sendbufsize,0:numtasks-1))

! exchange data

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

      call packvects(send_cnts(task), cnt, sendbuf(1,1,task), request_lsts(1,task))

      call mpi_isend(sendbuf(1,1,task), cnt*3, MPI_DOUBLE_PRECISION, &
                   task, pgm_gath_fld_tag, pmemd_comm, send_req_dat(taskmap_idx), ier)
  end do

recv_cnts = 0

  do wait_call = 1, numtasks - 1

      call mpi_waitany(numtasks - 1, recv_req_dat, taskmap_idx, irecv_stat, ier)
      task = recv_taskmap(taskmap_idx)
      call mpi_get_count(irecv_stat, MPI_DOUBLE_PRECISION, recv_cnt, ier)
      recv_cnts(task) = recv_cnt
  end do

  call mpi_waitall(numtasks - 1, send_req_dat, isend_stat, ier) 
  
  do taskmap_idx = 1, numtasks - 1

    task = recv_taskmap(taskmap_idx)

    if (need_cnts(task) .gt. 0) then
      call unpackvects(need_cnts(task),cnt,recvbuf(1,1,task),need_from_old_lst(need_shft(task)+1))
    end if

  end do

  deallocate(sendbuf)
  deallocate(recvbuf)
  deallocate(request_lsts)

end subroutine mpi_get_all_absent_old_imgs

subroutine findtaskid(img, task, tbl, numtasks, ave_img_cnt)

  implicit none

  integer :: img, task, numtasks
  double precision, intent(in) :: ave_img_cnt
  integer :: tbl(2,0:numtasks-1)

! guess the task id
!    task = min(int(dble(img)/dble(ave_img_cnt)), numtasks - 1)

    task = 0
    do while ((img .gt. tbl(1, task)+tbl(2,task)) .and. task .lt. numtasks - 1)
       task = task + 1
    end do

    return

    do while (img .le. tbl(1, task))
      task = task - 1
    end do
    do while (img .gt. tbl(1,task)+tbl(2, task))
      task = task + 1
    end do
    
end subroutine findtaskid

!------------------------------------------------------------------------

subroutine packvects(send_cnt, cnt, buf, imglst)

  use pol_gauss_mdin_dat_mod, only : dipole_scf_init, dipole_scf_init_step, pol_gauss_verbose
                            
  implicit none

  integer, intent(in)    :: send_cnt
  integer, intent(in)    :: imglst(send_cnt)
  integer, intent(inout) :: cnt
  double precision, intent(inout) :: buf(3,*)

  integer                :: i

    cnt  = 0

  call packone(send_cnt, cnt, ind_dip_img, buf, imglst)

  do i = 1, dipole_scf_init_step
    call packone(send_cnt, cnt, record1(1,1,i), buf, imglst)
    call packone(send_cnt, cnt, record2(1,1,i), buf, imglst)
    call packone(send_cnt, cnt, record3(1,1,i), buf, imglst)
    call packone(send_cnt, cnt, record4(1,1,i), buf, imglst)
  end do
  call packone(send_cnt, cnt, oldrec(1,1,1), buf, imglst)
  call packone(send_cnt, cnt, oldrec(1,1,2), buf, imglst)

  if (dipole_scf_init .eq. 2 .or. pol_gauss_verbose == 6) then
    call packone(send_cnt, cnt, olddip(1,1), buf, imglst)
  end if

contains

subroutine packone(send_cnt, cnt, vect, buf, lst)

  implicit none

  integer, intent(in)             :: send_cnt
  integer, intent(in)             :: lst(send_cnt)
  integer, intent(inout)          :: cnt
  double precision, intent(in)    :: vect(3,*)
  double precision, intent(inout) :: buf(3,*)

  integer                         :: i, img, j
  
  do  i = 1, send_cnt
    img = lst(i)
    cnt = cnt + 1
    buf(:, cnt) = vect(:, img)
  end do

end subroutine packone

end subroutine packvects

subroutine unpackvects(recv_cnt, cnt, buf, imglst)

  use pol_gauss_mdin_dat_mod, only : dipole_scf_init, dipole_scf_init_step, pol_gauss_verbose

  implicit none

  integer, intent(in)    :: recv_cnt
  integer, intent(in)    :: imglst(recv_cnt)
  integer, intent(inout) :: cnt
  double precision, intent(inout) :: buf(3,*)

  integer                :: i

    cnt  = 0

  call unpackone(recv_cnt, cnt, ind_dip_img, buf, imglst)

  do i = 1, dipole_scf_init_step
    call unpackone(recv_cnt, cnt, record1(1,1,i), buf, imglst)
    call unpackone(recv_cnt, cnt, record2(1,1,i), buf, imglst)
    call unpackone(recv_cnt, cnt, record3(1,1,i), buf, imglst)
    call unpackone(recv_cnt, cnt, record4(1,1,i), buf, imglst)
  end do
  call unpackone(recv_cnt, cnt, oldrec(1,1,1), buf, imglst)
  call unpackone(recv_cnt, cnt, oldrec(1,1,2), buf, imglst)

  if (dipole_scf_init .eq. 2 .or. pol_gauss_verbose == 6) then
    call unpackone(recv_cnt, cnt, olddip(1,1), buf, imglst)
  end if

contains

subroutine unpackone(recv_cnt, cnt, vect, buf, lst)

  implicit none

  integer, intent(in)             :: recv_cnt
  integer, intent(in)             :: lst(recv_cnt)
  integer, intent(inout)          :: cnt
  double precision, intent(out)   :: vect(3,*)
  double precision, intent(in)    :: buf(3,*)

  integer                         :: i, img, j
  
  do  i = 1, recv_cnt
    img = lst(i)
    cnt = cnt + 1
    vect(:, img) = buf(:, cnt) 
  end do

end subroutine unpackone

end subroutine unpackvects

#if 0
#ifdef MPI
subroutine pGM_init_used_img_map(used_img_mask, used_img_cnt, used_img_lst)

  implicit none

  integer, intent(inout)  :: used_img_cnt
  integer, intent(inout)  :: used_img_lst(*)
  integer, intent(inout)  :: used_img_mask(*)
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

#endif

subroutine pGM_reduce_short_used_img(atm_cnt, vec)

  implicit none

  integer, intent(in)             :: atm_cnt
  double precision, intent(inout) :: vec(3, atm_cnt)

  integer               :: irecv_stat(mpi_status_size)
  integer               :: isend_stat(mpi_status_size, numtasks - 1)
  integer               :: recv_req(numtasks - 1)
  integer               :: send_req(numtasks - 1)
  integer               :: recv_cnts(0 : numtasks - 1)

  integer :: taskmap_idx
  integer :: wait_call
  integer :: task, ier
  integer :: recv_cnt
  integer :: i, j, cnt

  do taskmap_idx = 1, numtasks - 1
    task = recv_taskmap(taskmap_idx)
    call mpi_irecv(recv_buf(1, 1, task), 3*max_img_cnt, MPI_DOUBLE_PRECISION, &
                   task, pgm_short_fld_tag, pmemd_comm, recv_req(taskmap_idx), ier)
  end do

  do taskmap_idx = 1, numtasks - 1

    task = send_taskmap(taskmap_idx)

    do i = 1, short_send_cnt(task)
      j = send_lst(i, task)
      send_buf(:, i, task) = vec(:, j)
    end do

    call mpi_isend(send_buf(1,1,task), short_send_cnt(task)*3, MPI_DOUBLE_PRECISION, &
                   task, pgm_short_fld_tag, pmemd_comm, send_req(taskmap_idx), ier)
  end do

  do wait_call = 1, numtasks - 1
    call mpi_waitany(numtasks - 1, recv_req, taskmap_idx, irecv_stat, ier)
    task = recv_taskmap(taskmap_idx)
    call mpi_get_count(irecv_stat, MPI_INTEGER, recv_cnt, ier)
    recv_cnts(task) = recv_cnt
    do i = 1, short_recv_cnt(task)
      j = recv_lst(i, task)
      vec(:, j) = vec(:, j) + recv_buf(:, i, task)
    end do
  end do

  call mpi_waitall(numtasks - 1, send_req, isend_stat, ier) 
  
end subroutine pGM_reduce_short_used_img

#endif

!----------------------------------------------------------
subroutine clearoldimg3dvec(used_img_cnt, vec)

use img_mod

  implicit none

  integer :: used_img_cnt
  double precision :: vec(3,*)
  integer :: i, j

    do j = 1, used_img_cnt
      i = gbl_used_img_lst(j)
      vec(:, i) = 0.d0
    end do

end subroutine clearoldimg3dvec
#endif /* MPI */
!---------------------------------------------------------
!
!  wrappers of commonly used vector operations, make the code somewhat less
!  painful to read, all require img_mod module and the data in the range
!  of my_img_lo:my_img_hi are reduced data.
! 
!---------------------------------------------------------
subroutine vec_abs_sum(atm_cnt, v, s)

  implicit none

  integer :: atm_cnt
  double precision :: v(3, atm_cnt)
  double precision :: s

  double precision, save :: buf, buf1
  integer :: ier

#ifdef MPI
  buf = sum(abs(v(:,my_img_lo:my_img_hi)))
  call mpi_allreduce(buf, buf1, 1, MPI_DOUBLE_PRECISION, MPI_SUM, pmemd_comm, ier)
  s = buf1
#else
  s = sum(abs(v(:,:)))
#endif

end subroutine vec_abs_sum
!---------------------------------------------------------
subroutine vec_mul_sum(atm_cnt, v1, v2, s)

  implicit none

  integer :: atm_cnt
  double precision :: v1(3, atm_cnt)
  double precision :: v2(3, atm_cnt)
  double precision :: s

  double precision, save :: buf, buf1
  integer :: ier

#ifdef MPI
  buf = sum(v1(:, my_img_lo:my_img_hi) * v2(:, my_img_lo:my_img_hi))
  call mpi_allreduce(buf, buf1, 1, MPI_DOUBLE_PRECISION, MPI_SUM, pmemd_comm, ier)
  s = buf1
#else
  s = sum(v1(:, 1:atm_cnt) * v2(:, 1:atm_cnt))
#endif

end subroutine vec_mul_sum
!---------------------------------------------------------
subroutine vec_sq_sum(atm_cnt, v, s)

  implicit none

  integer :: atm_cnt
  double precision :: v(3, atm_cnt)
  double precision :: s

  double precision, save :: buf, buf1
  integer :: ier

#ifdef MPI
  buf = sum(v(:, my_img_lo:my_img_hi) * v(:, my_img_lo:my_img_hi))
  call mpi_allreduce(buf, buf1, 1, MPI_DOUBLE_PRECISION, MPI_SUM, pmemd_comm, ier)
  s = buf1
#else
  s = sum(v(:, 1:atm_cnt) * v(:, 1:atm_cnt))
#endif

end subroutine vec_sq_sum
!---------------------------------------------------------
subroutine vec_abs_max(atm_cnt, v, s)

  implicit none

  integer :: atm_cnt
  double precision :: v(3, atm_cnt)
  double precision :: s

  double precision, save :: buf, buf1
  integer :: ier

#ifdef MPI
  buf = maxval(abs(v(:, my_img_lo:my_img_hi)))
  call mpi_allreduce(buf, buf1, 1, MPI_DOUBLE_PRECISION, MPI_MAX, pmemd_comm, ier)
  s = buf1
#else
  s = maxval(abs(v(:, :)))
#endif

end subroutine vec_abs_max
!---------------------------------------------------------
!===============================================================================
! RL: diagnostic dump machinery for the subspace-recycling study (serial pmemd.pgm).
!     Not intended for MPI/GPU builds.
!===============================================================================
function pgm_is_dump_frame(nstep_) result(yes)
   ! True if nstep_ is a selected dump frame:
   !   pgm_dump_enable /= 0, within [pgm_dump_first, pgm_dump_last] (0 = unbounded),
   !   and either in the explicit pgm_dump_frames list (if given) or on the stride.
   use pol_gauss_mdin_dat_mod, only : pgm_dump_enable, pgm_dump_stride, &
                                      pgm_dump_first, pgm_dump_last, &
                                      pgm_dump_frames_n, pgm_dump_frames
   implicit none
   integer, intent(in) :: nstep_
   logical :: yes
   integer :: f
   yes = .false.
   if ( pgm_dump_enable == 0 ) return
   if ( pgm_dump_first /= 0 .and. nstep_ < pgm_dump_first ) return
   if ( pgm_dump_last  /= 0 .and. nstep_ > pgm_dump_last  ) return
   if ( pgm_dump_frames_n > 0 ) then
      do f = 1, pgm_dump_frames_n
         if ( pgm_dump_frames(f) == nstep_ ) then
            yes = .true.; return
         end if
      end do
   else
      if ( pgm_dump_stride <= 1 ) then
         yes = .true.
      else
         yes = ( mod(nstep_ - 1, pgm_dump_stride) == 0 )
      end if
   end if
end function pgm_is_dump_frame
!-------------------------------------------------------------------------------
subroutine pgm_export_operator(atm_cnt_, img_crd_, u_)
   ! Writes the dense operator A_code (3*atm_cnt_ x 3*atm_cnt_), one column per
   ! Cartesian basis vector e_j, applying the EXACT production matvec
   ! pGM_pvt_NonBond_dip_dip_fields and the assembly A_code*e_j = e_j + alpha*G(e_j)
   ! (identical to the pGM_pkcg matvec, pGM_induced.F90:1149-1157). The self term
   ! is captured automatically because it is inside the field wrapper.
   implicit none
   integer, intent(in)          :: atm_cnt_, u_
   double precision, intent(in) :: img_crd_(3,*)
   double precision :: e_(3, atm_cnt_), col_(3, atm_cnt_), zv_(3, atm_cnt_)
   integer :: j, a, c, i
   e_  = 0.d0
   do j = 1, 3*atm_cnt_
      a = (j-1)/3 + 1            ! image atom index 1..atm_cnt_
      c = mod(j-1, 3) + 1        ! cartesian component 1..3
      e_ = 0.d0
      e_(c, a) = 1.d0
      call pGM_pvt_NonBond_dip_dip_fields(atm_cnt_, img_crd_, e_, zv_)   ! zv_ = G(e_j) = -T e_j
      do i = 1, atm_cnt_
         col_(1:3, i) = pol_img(i) * zv_(1:3, i)      ! alpha_i * G(e_j)_i
      end do
      col_(c, a) = col_(c, a) + 1.d0                  ! + e_j  (identity diagonal)
      write(u_) col_(1:3, 1:atm_cnt_)                 ! column j of A_code
   end do
end subroutine pgm_export_operator
!-------------------------------------------------------------------------------
subroutine pgm_dump_frame(atm_cnt_, img_crd_, mu_star_)
   ! Writes one binary frame file pgm_frame_<nstep>.bin (Fortran stream access).
   use pol_gauss_mdin_dat_mod, only : pgm_curr_nstep, pgm_dump_b, pgm_dump_mu, &
                                      pgm_dump_crd, pgm_dump_op, pgm_dump_short, &
                                      pgm_dump_hist, pgm_dump_dir
   use pol_gauss_direct_data, only : num_tensor_short, dipole_dipole_short_ptrs, &
                                     dipole_dipole_list_short, dipole_dipole_tensor_short
   use img_mod,           only : gbl_atm_img_map, gbl_img_atm_map
   use pbc_mod,           only : ucell
   implicit none
   integer, intent(in)          :: atm_cnt_
   double precision, intent(in) :: img_crd_(3,*)
   double precision, intent(in) :: mu_star_(3, atm_cnt_)
   character(len=320) :: fname
   integer, parameter :: magic = 20260728, ver = 1
   integer :: u_, nnz, nptrs, k
   double precision :: box(3,3)

   if ( .not. pgm_is_dump_frame(pgm_curr_nstep) ) return

   u_ = 210
   write(fname,'(A,A,I10.10,A)') trim(pgm_dump_dir), '/pgm_frame_', pgm_curr_nstep, '.bin'
   open(unit=u_, file=trim(fname), access='stream', form='unformatted', &
        status='replace', action='write')

   box = ucell
   write(u_) magic
   write(u_) ver
   write(u_) pgm_curr_nstep
   write(u_) atm_cnt_
   write(u_) box
   if ( pgm_dump_crd /= 0 ) write(u_) img_crd_(1:3, 1:atm_cnt_)
   write(u_) pol_img(1:atm_cnt_)                          ! alpha (always; cheap, needed everywhere)
   if ( pgm_dump_b  /= 0 ) write(u_) pgm_b_save(1:3, 1:atm_cnt_)
   if ( pgm_dump_mu /= 0 ) write(u_) mu_star_(1:3, 1:atm_cnt_)
   if ( pgm_dump_short /= 0 ) then
      nnz = num_tensor_short
      nptrs = size(dipole_dipole_short_ptrs)
      write(u_) nnz
      write(u_) nptrs
      write(u_) dipole_dipole_short_ptrs(1:nptrs)
      write(u_) dipole_dipole_list_short(1:nnz)
      write(u_) dipole_dipole_tensor_short(1:6, 1:nnz)
      write(u_) gbl_atm_img_map(1:atm_cnt_)
      write(u_) gbl_img_atm_map(1:atm_cnt_)
   end if
   if ( pgm_dump_hist /= 0 ) then
      k = pgm_hist_k
      write(u_) k
      if ( k > 0 ) then
         write(u_) pgm_hist_rnorm(1:k)
         write(u_) pgm_hist_alpha(1:k)
         write(u_) pgm_hist_beta(1:k)
         write(u_) pgm_hist_p(1:3, 1:atm_cnt_, 1:k)
         write(u_) pgm_hist_Ap(1:3, 1:atm_cnt_, 1:k)
      end if
   end if
   if ( pgm_dump_op /= 0 ) call pgm_export_operator(atm_cnt_, img_crd_, u_)
   close(u_)
end subroutine pgm_dump_frame

end module pol_gauss_induced
!---------------------------------------------------------

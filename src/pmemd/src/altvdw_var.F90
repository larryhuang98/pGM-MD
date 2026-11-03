#include "pGM/copyright_pgm.i"
#include "../include/assert.fh"
! Adopted by YD from degauss code of Zhen Huang@uci
module altvdw_var_mod

  implicit none

  private

  integer :: idevdw, ibuckvdw, igvdw
  double precision, dimension(:), allocatable  :: gbeta
  double precision, dimension(:), allocatable  :: de_a, de_c, de_eps0, de_rmin
  double precision, dimension(:), allocatable  :: de_a_r, de_c_r, de_a_r2, de_c_r2, de_rmin2
  double precision, dimension(:), allocatable  :: bh_a, bh_eps0, bh_rmin, bh_a_r

  common / altvdw_ctrl_cmn /idevdw, ibuckvdw, igvdw
  save :: / altvdw_ctrl_cmn /
  integer, parameter    :: altvdw_ctrl_int_cnt = 3

  public idevdw, ibuckvdw, igvdw, altvdw_ctrl_int_cnt
  public gbeta, de_a, de_c, de_eps0, de_rmin
  public de_a_r, de_c_r, de_a_r2, de_c_r2, de_rmin2
  public bh_a, bh_eps0, bh_rmin, bh_a_r
  public degauss_readparm, devdw_readparm, buckvdw_readparm, gauvdw_readparm
#ifdef MPI
  public :: degauss_bcast, devdw_bcast, buckvdw_bcast, gauvdw_bcast
#endif
  contains

  subroutine degauss_readparm(prmtop, mdout, natom, nttyp, idegauss_lcl, idevdw_lcl, cn1, cn2)

     implicit none

     integer, intent(in) :: prmtop, mdout, natom, nttyp, idegauss_lcl, idevdw_lcl
     double precision, intent(inout) :: cn1(nttyp), cn2(nttyp)
     character(80)       :: type
     character(80)       :: fmt
     character(80)       :: fmtin
     character(80)       :: afmt, ifmt, rfmt, mfmt, nfmt ! New2022
     integer             :: errcode, i

     afmt = '(20A4)'
     ifmt = '(12I6)'
     rfmt = '(5E16.8)'
     mfmt = '(3I8)' ! New2022
     nfmt = '(1E16.8)' ! New2022

     if (idegauss_lcl > 0) then

        if (.not. allocated(gbeta)) then
           allocate(gbeta(natom), stat=errcode)
           REQUIRE(errcode == 0)
        end if

        type = 'GAUSS_RADII_LIST'
        fmtin= rfmt
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)

        if (errcode .eq. 0) then
           read(prmtop, fmt) (gbeta(i), i = 1, natom)
        else
           write(mdout,'(a,f5.1)') "| Gauss Radii not found, use default value", 0.2
           gbeta(:)=2.000d-1 ! if by default, make it as constant value
        end if
        do i=1, natom
           gbeta(i) = 2.0d0 * gbeta(i) ** 2
        end do
     end if

     if (idegauss_lcl > 0 .or. idevdw_lcl > 0) then

        fmtin = rfmt
        type = 'DOUBLE_EXPONENTIAL_A'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           de_a(1: nttyp) = 18.170037d0
           write(mdout, '(a,f5.1)') "| DE vdw parameter alpha not found, use default value", 18.2
        else
           read(prmtop, fmt) (de_a(i),   i = 1, nttyp)
        end if

        fmtin = rfmt
        type = 'DOUBLE_EXPONENTIAL_C'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
            de_c(1: nttyp) =  3.651491d0
            write(mdout, '(a,f5.1)') "| DE vdw parameter beta not found, use default value", 3.6
        else
            read(prmtop, fmt) (de_c(i), i = 1, nttyp)
        end if

      ! initialize eps0 and rmin with LJ potential A and B coefficients
        do i = 1, nttyp
           if (cn1(i) .eq. 0 .or. cn2(i) .eq. 0 ) then
              de_eps0(i) = 0.0
              de_rmin(i) = 1.0
           else
              de_rmin(i) = (2.0d0*cn1(i)/cn2(i))**(1.0d0/6.0d0) ! place holder
              de_eps0(i) = cn2(i)**2 / (4.0d0*cn1(i))
           end if 
        end do

        fmtin = rfmt
        type = 'DOUBLE_EXPONENTIAL_EPS0'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           write(mdout, '(a)') "| DE vdw parameter eps not found, calculated from LJ"
        else
           read(prmtop, fmt) (de_eps0(i), i = 1, nttyp)
        end if

        fmtin = rfmt
        type = 'DOUBLE_EXPONENTIAL_RMIN'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           write(mdout, '(a)') "| DE vdw parameter rmin not found, calculated from LJ"
        else
           read(prmtop, fmt) (de_rmin(i), i = 1, nttyp)
        end if

        if (.not. allocated(de_rmin2)) then
           allocate(de_rmin2(nttyp))
        end if
        if (.not. allocated(de_a_r)) then
           allocate(de_a_r(nttyp))
        end if
        if (.not. allocated(de_c_r)) then
           allocate(de_c_r(nttyp))
        end if
        if (.not. allocated(de_a_r2)) then
           allocate(de_a_r2(nttyp))
        end if
        if (.not. allocated(de_c_r2)) then
           allocate(de_c_r2(nttyp))
        end if

      ! save precalculated parameters back to cn(1) and cn(2) to save calculations
      ! it is only called once, optimization is not that necessary
        do i = 1, nttyp
           cn1(i) = de_eps0(i) * de_c(i) * exp(de_a(i))/(de_a(i)-de_c(i))
           cn2(i) = de_eps0(i) * de_a(i) * exp(de_c(i))/(de_a(i)-de_c(i))
      ! these are potentially useful combinations ...
           de_rmin2(i) = de_rmin(i) * de_rmin(i)
           de_a_r(i)   = de_a(i) / de_rmin(i)
           de_c_r(i)   = de_c(i) / de_rmin(i)
           de_a_r2(i)  = de_a_r(i) * de_a_r(i)
           de_c_r2(i)  = de_c_r(i) * de_c_r(i)
        end do
      
     end if

  end subroutine degauss_readparm

  subroutine devdw_readparm(prmtop, mdout, nttyp, idevdw_lcl, cn1, cn2)

     implicit none

     integer, intent(in) :: prmtop, mdout, nttyp, idevdw_lcl
     double precision, intent(inout) :: cn1(nttyp), cn2(nttyp)
     character(80)       :: type
     character(80)       :: fmt
     character(80)       :: fmtin
     character(80)       :: afmt, ifmt, rfmt, mfmt, nfmt ! New2022
     integer             :: errcode, i

     afmt = '(20A4)'
     ifmt = '(12I6)'
     rfmt = '(5E16.8)'
     mfmt = '(3I8)' ! New2022
     nfmt = '(1E16.8)' ! New2022

     if (idevdw_lcl > 0) then

        fmtin = rfmt
        type = 'DOUBLE_EXPONENTIAL_A'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           de_a(1: nttyp) = 18.28349D0  ! fit to pGM3P-25 VDW
           write(mdout, '(a,f5.1)') "| DE vdw parameter alpha not found, use default value", 18.2
        else
           read(prmtop, fmt) (de_a(i),   i = 1, nttyp)
        end if

        fmtin = rfmt
        type = 'DOUBLE_EXPONENTIAL_C'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
            de_c(1: nttyp) =  3.58185D0  ! fit to pGM3P-25 VDW
            write(mdout, '(a,f5.1)') "| DE vdw parameter beta not found, use default value", 3.6
        else
            read(prmtop, fmt) (de_c(i), i = 1, nttyp)
        end if

      ! initialize eps0 and rmin with LJ potential A and B coefficients
        do i = 1, nttyp
           if (cn1(i) .eq. 0 .or. cn2(i) .eq. 0 ) then
              de_eps0(i) = 0.0
              de_rmin(i) = 0.0
           else
              de_rmin(i) = (2.0d0*cn1(i)/cn2(i))**(1.0d0/6.0d0) ! place holder
              de_eps0(i) = cn2(i)**2 / (4.0d0*cn1(i))
           end if 
        end do

        fmtin = rfmt
        type = 'DOUBLE_EXPONENTIAL_EPS0'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           write(mdout, '(a)') "| DE vdw parameter eps not found, calculated from LJ"
        else
           read(prmtop, fmt) (de_eps0(i), i = 1, nttyp)
        end if

        fmtin = rfmt
        type = 'DOUBLE_EXPONENTIAL_RMIN'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           write(mdout, '(a)') "| DE vdw parameter rmin not found, calculated from LJ"
        else
           read(prmtop, fmt) (de_rmin(i), i = 1, nttyp)
        end if

        if (.not. allocated(de_rmin2)) then
           allocate(de_rmin2(nttyp))
        end if
        if (.not. allocated(de_a_r)) then
           allocate(de_a_r(nttyp))
        end if
        if (.not. allocated(de_c_r)) then
           allocate(de_c_r(nttyp))
        end if
        if (.not. allocated(de_a_r2)) then
           allocate(de_a_r2(nttyp))
        end if
        if (.not. allocated(de_c_r2)) then
           allocate(de_c_r2(nttyp))
        end if

      ! save precalculated parameters back to cn(1) and cn(2) to save calculations
      ! it is only called once, optimization is not that necessary
        do i = 1, nttyp
           if (de_eps0(i) .eq. 0.0d0) then
              cn1(i) = 0.0d0
              cn2(i) = 0.0d0
           else 
              if (de_c(i) .eq. 0.0d0) then
                 cn1(i) = 0.0d0
              else
                 cn1(i) = de_eps0(i) * de_c(i) * exp(de_a(i))/(de_a(i)-de_c(i))
              end if
              if (de_a(i) .eq. 0.0d0) then
                 cn2(i) = 0.0d0
              else
                 cn2(i) = de_eps0(i) * de_a(i) * exp(de_c(i))/(de_a(i)-de_c(i))
              end if
           end if
#if 0
           cn1(i) = de_eps0(i) * de_c(i) * exp(de_a(i))/(de_a(i)-de_c(i))
           cn2(i) = de_eps0(i) * de_a(i) * exp(de_c(i))/(de_a(i)-de_c(i))
#endif
      ! these are potentially useful combinations ...
           if (de_rmin(i) .eq. 0.0d0) de_rmin(i) = 1.0d0
           de_rmin2(i) = de_rmin(i) * de_rmin(i)
           de_a_r(i)   = de_a(i) / de_rmin(i)
           de_c_r(i)   = de_c(i) / de_rmin(i)
           if (de_a(i) .eq. 0.0d0) de_a_r(i) = 1.0d0
           if (de_c(i) .eq. 0.0d0) de_c_r(i) = 1.0d0
           de_a_r2(i)  = de_a_r(i) * de_a_r(i)
           de_c_r2(i)  = de_c_r(i) * de_c_r(i)
        end do
      
     end if

  end subroutine devdw_readparm

  subroutine buckvdw_readparm(prmtop, mdout, nttyp, ibuckvdw_lcl, cn1, cn2)

     implicit none

     integer, intent(in) :: prmtop, mdout, nttyp, ibuckvdw_lcl
     double precision, intent(inout) :: cn1(nttyp), cn2(nttyp)
     character(80)       :: type
     character(80)       :: fmt
     character(80)       :: fmtin
     character(80)       :: afmt, ifmt, rfmt, mfmt, nfmt ! New2022
     integer             :: errcode, i
     double precision    :: epsa

     afmt = '(20A4)'
     ifmt = '(12I6)'
     rfmt = '(5E16.8)'
     mfmt = '(3I8)' ! New2022
     nfmt = '(1E16.8)' ! New2022

     if (ibuckvdw_lcl > 0) then

        if (.not. allocated(bh_a))    allocate(bh_a(nttyp))
        if (.not. allocated(bh_rmin)) allocate(bh_rmin(nttyp))
        if (.not. allocated(bh_eps0)) allocate(bh_eps0(nttyp))
        if (.not. allocated(bh_a_r))  allocate(bh_a_r(nttyp))

      ! initialize eps0 and rmin from LJ potential A and B coefficients
        do i = 1, nttyp
           if (cn1(i) .eq. 0 .or. cn2(i) .eq. 0 ) then
              bh_rmin(i) = 0.0
              bh_eps0(i) = 0.0
           else
      ! bh_rmin() is the distance at which vdw energy is lowest.
              bh_rmin(i) = (2.0d0*cn1(i)/cn2(i))**(1.0d0/6.0d0) ! place holder
              bh_eps0(i) = cn2(i)**2 / (4.0d0*cn1(i))
           end if 
        end do

        fmtin = rfmt
        type = 'BUCKINGHAM_ALPHA'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           bh_a(1: nttyp) = 15.5529D0  ! fit to TIP3P LJ profile x r**2
           write(mdout, '(a,f5.1)') "| Buchingham vdw parameter alpha not found, use default value", 18.2
        else
           read(prmtop, fmt) (bh_a(i),   i = 1, nttyp)
        end if

        fmtin = rfmt
        type = 'BUCKINGHAM_EPS0'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           write(mdout, '(a)') "| Buchingham vdw parameter eps not found, calculated from LJ"
        else
           read(prmtop, fmt) (bh_eps0(i), i = 1, nttyp)
        end if

        fmtin = rfmt
        type = 'BUCKINGHAM_RMIN'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           write(mdout, '(a)') "| Buckingham vdw parameter rmin not found, calculated from LJ"
        else
           read(prmtop, fmt) (bh_rmin(i), i = 1, nttyp)
        end if

      ! the modified Buckingham (Exp-Six) form
      ! save precalculated parameters back to cn(1) and cn(2) to save calculations
        do i = 1, nttyp
           if (bh_eps0(i) .eq. 0.0d0) then
              cn1(i) = 0.0d0
              cn2(i) = 0.0d0
           else
              cn1(i) = bh_eps0(i) * exp(bh_a(i)) * 6.0d0 / (bh_a(i) - 6.0d0)
              cn2(i) = bh_eps0(i) * bh_a(i)/(bh_a(i) - 6.0d0) * bh_rmin(i)**6
           end if
           if (bh_rmin(i) .le. 0.0d0) bh_rmin(i) = 1.0d0
           bh_a_r(i) = bh_a(i)/bh_rmin(i)
        end do
     end if

  end subroutine buckvdw_readparm

  subroutine gauvdw_readparm(prmtop, mdout, nttyp, ibuckvdw_lcl, cn1, cn2)

     implicit none

     integer, intent(in) :: prmtop, mdout, nttyp, ibuckvdw_lcl
     double precision, intent(inout) :: cn1(nttyp), cn2(nttyp)
     character(80)       :: type
     character(80)       :: fmt
     character(80)       :: fmtin
     character(80)       :: afmt, ifmt, rfmt, mfmt, nfmt ! New2022
     integer             :: errcode, i
     double precision    :: epsa

     afmt = '(20A4)'
     ifmt = '(12I6)'
     rfmt = '(5E16.8)'
     mfmt = '(3I8)' ! New2022
     nfmt = '(1E16.8)' ! New2022

     if (ibuckvdw_lcl > 0) then

        if (.not. allocated(bh_a)) allocate(bh_a(nttyp))
        if (.not. allocated(bh_rmin)) allocate(bh_rmin(nttyp))
        if (.not. allocated(bh_eps0)) allocate(bh_eps0(nttyp))
        if (.not. allocated(bh_a_r)) allocate(bh_a_r(nttyp))

      ! initialize eps0 and rmin from LJ potential A and B coefficients
        do i = 1, nttyp
           if (cn1(i) .eq. 0 .or. cn2(i) .eq. 0 ) then
              bh_rmin(i) = 0.0
              bh_eps0(i) = 0.0
           else
      ! bh_rmin() is the distance at which vdw energy is lowest.
              bh_rmin(i) = (2.0d0*cn1(i)/cn2(i))**(1.0d0/6.0d0) ! place holder
              bh_eps0(i) = cn2(i)**2 / (4.0d0*cn1(i))
           end if 
        end do

        fmtin = rfmt
        type = 'BUCKINGHAM_ALPHA'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           bh_a(1: nttyp) = 18.170037d0
           write(mdout, '(a,f5.1)') "| Buchingham vdw parameter alpha not found, use default value", 18.2
        else
           read(prmtop, fmt) (bh_a(i),   i = 1, nttyp)
        end if

        fmtin = rfmt
        type = 'BUCKINGHAM_EPS0'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           write(mdout, '(a)') "| Buchingham vdw parameter eps not found, calculated from LJ"
        else
           read(prmtop, fmt) (bh_eps0(i), i = 1, nttyp)
        end if

        fmtin = rfmt
        type = 'BUCKINGHAM_RMIN'
        call nxtsec(prmtop, mdout, 1, fmtin, type, fmt, errcode)
        if (errcode == -2) then
           write(mdout, '(a)') "| Buckingham vdw parameter rmin not found, calculated from LJ"
        else
           read(prmtop, fmt) (bh_rmin(i), i = 1, nttyp)
        end if

      ! the modified Buckingham (Exp-Six) form
      ! save precalculated parameters back to cn(1) and cn(2) to save calculations
        do i = 1, nttyp
           cn1(i) = bh_eps0(i) * exp(bh_a(i)) * 6.0d0 / (bh_a(i) - 6.0d0)
           cn2(i) = bh_eps0(i) * bh_a(i)/(bh_a(i) - 6.0d0) * bh_rmin(i)**6
           if (bh_rmin(i) .le. 0.0d0) bh_rmin(i) = 1.0d0
           bh_a_r(i) = bh_a(i)/bh_rmin(i)
        end do
     end if

  end subroutine gauvdw_readparm

#ifdef MPI
  subroutine degauss_bcast(natom, nttyp, idegauss_lcl)
!  use parms, only :  nttyp

  use parallel_dat_mod

!  include 'mpif.h'
!# include "extra.h"
!# include "parallel.h"
  integer, intent(in) :: idegauss_lcl
  integer, intent(in) :: natom
  integer, intent(in) :: nttyp
  integer :: ier

    if (idegauss_lcl .le. 0) return

    if (.not. master) then
      allocate(gbeta(natom),             &
               de_a(nttyp),              &  
               de_c(nttyp),              &  
               de_eps0(nttyp),           &  
               de_rmin(nttyp),           &
               de_rmin2(nttyp),          &
               de_a_r(nttyp),            &
               de_c_r(nttyp),            &
               de_a_r2(nttyp),           &
               de_c_r2(nttyp),           &
               stat=ier)
      REQUIRE(ier==0)
    end if
    call mpi_bcast(gbeta, natom, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_a, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_c, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_eps0, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_rmin, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_rmin2, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_a_r, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_c_r, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_a_r2, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_c_r2, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)

  end subroutine degauss_bcast

  subroutine devdw_bcast(nttyp, idevdw_lcl)
!  use parms, only :  nttyp

  use parallel_dat_mod

!  include 'mpif.h'
!# include "extra.h"
!# include "parallel.h"
  integer, intent(in) :: idevdw_lcl
  integer, intent(in) :: nttyp
  integer :: ier

    if (idevdw_lcl .le. 0) return

    if (.not. master) then
      allocate(de_a(nttyp),              &  
               de_c(nttyp),              &  
               de_eps0(nttyp),           &  
               de_rmin(nttyp),           &
               de_rmin2(nttyp),          &
               de_a_r(nttyp),            &
               de_c_r(nttyp),            &
               de_a_r2(nttyp),           &
               de_c_r2(nttyp),           &
               stat=ier)
      REQUIRE(ier==0)
    end if
    call mpi_bcast(de_a, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_c, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_eps0, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_rmin, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_rmin2, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_a_r, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_c_r, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_a_r2, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(de_c_r2, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)

  end subroutine devdw_bcast

  subroutine buckvdw_bcast(nttyp, ibuckvdw_lcl)
!  use parms, only :  nttyp

  use parallel_dat_mod

!  include 'mpif.h'
!# include "extra.h"
!# include "parallel.h"
  integer, intent(in) :: nttyp
  integer, intent(in) :: ibuckvdw_lcl
  integer :: ier

    if (ibuckvdw_lcl .le. 0) return

    if (.not. master) then
      allocate(bh_a(nttyp),              &  
               bh_eps0(nttyp),           &  
               bh_rmin(nttyp),           &
               bh_a_r(nttyp),           &
               stat=ier)
      REQUIRE(ier==0)
    end if

    call mpi_bcast(bh_a, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(bh_eps0, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(bh_rmin, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(bh_a_r, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)

  end subroutine buckvdw_bcast

  subroutine gauvdw_bcast(nttyp, igvdw_lcl)

  use parallel_dat_mod

  integer, intent(in) :: nttyp
  integer, intent(in) :: igvdw_lcl
  integer :: ier

    if (igvdw_lcl .le. 0) return

    if (.not. master) then
      allocate(bh_a(nttyp),              &  
               bh_eps0(nttyp),           &  
               bh_rmin(nttyp),           &
               bh_a_r(nttyp),           &
               stat=ier)
      REQUIRE(ier==0)
    end if

    call mpi_bcast(bh_a, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(bh_eps0, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(bh_rmin, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)
    call mpi_bcast(bh_a_r, nttyp, MPI_DOUBLE_PRECISION, 0, pmemd_comm, ier)

  end subroutine gauvdw_bcast
#endif
end module altvdw_var_mod


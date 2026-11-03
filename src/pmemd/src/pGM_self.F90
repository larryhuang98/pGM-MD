#include "pGM/copyright_pgm.i"
#include "../include/assert.fh"
!
! Adopted from sander pGM code by YD
!
!------------------------------------------------------------
module pol_gauss_self

!  use pol_gauss_mpole_index_mod

  implicit none
  private

# include "pGM/pGM_mpole_index.h"

  integer,save :: do_flag

  public pGM_SELF_dipole_field, &
         pGM_SELF_ene_frc, &
         pGM_SELF_set_user_bit
#if 0
#ifdef MPI
  public pGM_SELF_bcast
#endif
#endif

  contains
!-------------------------------------------------------
#if 0
#ifdef MPI
subroutine pGM_SELF_bcast
  implicit none
  integer ierr

  include 'mpif.h'
# include "parallel.h"
  call mpi_bcast(do_flag,1,MPI_INTEGER,0,commsander,ierr)
end subroutine pGM_SELF_bcast
#endif
#endif
!-------------------------------------------------------
subroutine pGM_SELF_set_user_bit(do_this)

   use file_io_dat_mod, only : mdout

  integer,intent(in) :: do_this
#if 0

  ! set the valid bit---this part always since no parmread needed
  do_flag = ibset(do_flag,VALID_BIT)

  if ( do_this == 1 )then ! do in all cases
    do_flag = ibset(do_flag,USER_INDUCE_BIT)
    do_flag = ibset(do_flag,USER_POSTINDUCE_BIT)
  elseif ( do_this == 2 )then ! do the induction, not the post-induction
    do_flag = ibset(do_flag,USER_INDUCE_BIT)
    do_flag = ibclr(do_flag,USER_POSTINDUCE_BIT)
  elseif ( do_this == 3 )then ! do the post-induction, not the induction
    do_flag = ibclr(do_flag,USER_INDUCE_BIT)
    do_flag = ibset(do_flag,USER_POSTINDUCE_BIT)
  elseif ( do_this == 0 )then 
    do_flag = ibclr(do_flag,USER_INDUCE_BIT)
    do_flag = ibclr(do_flag,USER_POSTINDUCE_BIT)
  else
    write(mdout,*)'pGM_SELF_set_user_bit: bad value of user do_this'
!    call mexit(mdout,1)
    stop
  endif
#endif

end subroutine pGM_SELF_set_user_bit
!-------------------------------------------------------
subroutine pGM_SELF_dipole_field(atm_cnt, dip, dip_field)

  use constants_module, only : three,four,pi
  use mdin_ewald_dat_mod, only : ew_coeff

  use img_mod

  implicit none

  integer, intent(in)             :: atm_cnt
  double precision, intent(in)    :: dip(*)
  double precision, intent(inout) :: dip_field(*)
  
  double precision :: factor

  factor = four * ew_coeff**3 / (three*sqrt(pi))

  dip_field(1:3*atm_cnt) = dip_field(1:3*atm_cnt) - factor*dip(1:3*atm_cnt)

end subroutine pGM_SELF_dipole_field
!-------------------------------------------------------
subroutine pGM_SELF_ene_frc(atm_cnt, chg, dip, need_ene, ene_chg, ene_dip, phi)

  use constants_module, only : coulomb_const_kcal_per_mole, zero,half,two,pi
  use mdin_ewald_dat_mod, only : ew_coeff

  use img_mod

  integer,intent(in) :: atm_cnt
  double precision,intent(in) :: chg(*)
  double precision,intent(in) :: dip(3,*)   ! here, assume these are ind_dip + perm_dip
  logical, intent(in)         :: need_ene
  double precision,intent(inout) :: ene_chg
  double precision,intent(inout) :: ene_dip
  double precision,intent(inout) :: phi(10,atm_cnt)

  double precision :: delx,dely,delz,B(0:4),fac,fact,gmi(10),gphi(10),i_di(3), &
            i_mi(3),e_pp,e_ind
  double precision :: Rn(1),Rn_1(4),Rn_2(10),Rn_3(20),Rn_4(35)
  double precision :: e_chg_stk
  double precision :: e_dip_stk
  integer i,j,n

!# include "do_flag.h"
!# include "ew_pme_recip.h"

!  if ( iand(do_flag,PROCEED_POSTINDUCE) /= PROCEED_POSTINDUCE )return

  e_chg_stk = 0.d0
  e_dip_stk = 0.d0

  fact = two*ew_coeff / sqrt(PI)
  fac = -two*ew_coeff*ew_coeff
  do j = 0,4
    B(j) = fact/(two*j+1)
    fact = fac*fact
  end do
#if 0
  delx = zero
  dely = zero
  delz = zero

  n = 4
  Rn(Ind_000) = B(n)
  Rn_1(Ind_000) = B(n-1)
  Rn_1(Ind_100) = delx*Rn(Ind_000)
  Rn_1(Ind_010) = dely*Rn(Ind_000)
  Rn_1(Ind_001) = delz*Rn(Ind_000)
  Rn_2(Ind_000) = B(n-2)
  Rn_2(Ind_100) = delx*Rn_1(Ind_000)
  Rn_2(Ind_010) = dely*Rn_1(Ind_000)
  Rn_2(Ind_001) = delz*Rn_1(Ind_000)
  Rn_2(Ind_200) = Rn_1(Ind_000) + delx*Rn_1(Ind_100)
  Rn_2(Ind_020) = Rn_1(Ind_000) + dely*Rn_1(Ind_010)
  Rn_2(Ind_002) = Rn_1(Ind_000) + delz*Rn_1(Ind_001)
  Rn_2(Ind_110) = delx*Rn_1(Ind_010)
  Rn_2(Ind_101) = delx*Rn_1(Ind_001)
  Rn_2(Ind_011) = dely*Rn_1(Ind_001)
  Rn_3(Ind_000) = B(n-3) 
  Rn_3(Ind_100) = delx*Rn_2(Ind_000)
  Rn_3(Ind_010) = dely*Rn_2(Ind_000)
  Rn_3(Ind_001) = delz*Rn_2(Ind_000)
  Rn_3(Ind_200) = Rn_2(Ind_000) + delx*Rn_2(Ind_100)
  Rn_3(Ind_020) = Rn_2(Ind_000) + dely*Rn_2(Ind_010)
  Rn_3(Ind_002) = Rn_2(Ind_000) + delz*Rn_2(Ind_001)
  Rn_3(Ind_110) = delx*Rn_2(Ind_010)
  Rn_3(Ind_101) = delx*Rn_2(Ind_001)
  Rn_3(Ind_011) = dely*Rn_2(Ind_001)
  Rn_3(Ind_300) = 2.d0*Rn_2(Ind_100) + delx*Rn_2(Ind_200)
  Rn_3(Ind_030) = 2.d0*Rn_2(Ind_010) + dely*Rn_2(Ind_020)
  Rn_3(Ind_003) = 2.d0*Rn_2(Ind_001) + delz*Rn_2(Ind_002)
  Rn_3(Ind_210) = dely*Rn_2(Ind_200)
  Rn_3(Ind_201) = delz*Rn_2(Ind_200)
  Rn_3(Ind_120) = delx*Rn_2(Ind_020)
  Rn_3(Ind_021) = delz*Rn_2(Ind_020)
  Rn_3(Ind_102) = delx*Rn_2(Ind_002)
  Rn_3(Ind_012) = dely*Rn_2(Ind_002)
  Rn_3(Ind_111) = delx*Rn_2(Ind_011)
  Rn_4(Ind_000) = B(n-4) 
  Rn_4(Ind_100) = delx*Rn_3(Ind_000)
  Rn_4(Ind_010) = dely*Rn_3(Ind_000)
  Rn_4(Ind_001) = delz*Rn_3(Ind_000)
  Rn_4(Ind_200) = Rn_3(Ind_000) + delx*Rn_3(Ind_100)
  Rn_4(Ind_020) = Rn_3(Ind_000) + dely*Rn_3(Ind_010)
  Rn_4(Ind_002) = Rn_3(Ind_000) + delz*Rn_3(Ind_001)
  Rn_4(Ind_110) = delx*Rn_3(Ind_010)
  Rn_4(Ind_101) = delx*Rn_3(Ind_001)
  Rn_4(Ind_011) = dely*Rn_3(Ind_001)
  Rn_4(Ind_300) = 2.d0*Rn_3(Ind_100) + delx*Rn_3(Ind_200)
  Rn_4(Ind_030) = 2.d0*Rn_3(Ind_010) + dely*Rn_3(Ind_020)
  Rn_4(Ind_003) = 2.d0*Rn_3(Ind_001) + delz*Rn_3(Ind_002)
  Rn_4(Ind_210) = dely*Rn_3(Ind_200)
  Rn_4(Ind_201) = delz*Rn_3(Ind_200)
  Rn_4(Ind_120) = delx*Rn_3(Ind_020)
  Rn_4(Ind_021) = delz*Rn_3(Ind_020)
  Rn_4(Ind_102) = delx*Rn_3(Ind_002)
  Rn_4(Ind_012) = dely*Rn_3(Ind_002)
  Rn_4(Ind_111) = delx*Rn_3(Ind_011)
  Rn_4(Ind_400) = 3.d0*Rn_3(Ind_200) + delx*Rn_3(Ind_300)
  Rn_4(Ind_040) = 3.d0*Rn_3(Ind_020) + dely*Rn_3(Ind_030)
  Rn_4(Ind_004) = 3.d0*Rn_3(Ind_002) + delz*Rn_3(Ind_003)
  Rn_4(Ind_310) = dely*Rn_3(Ind_300)
  Rn_4(Ind_301) = delz*Rn_3(Ind_300)
  Rn_4(Ind_130) = delx*Rn_3(Ind_030)
  Rn_4(Ind_031) = delz*Rn_3(Ind_030)
  Rn_4(Ind_103) = delx*Rn_3(Ind_003)
  Rn_4(Ind_013) = dely*Rn_3(Ind_003)
  Rn_4(Ind_220) = Rn_3(Ind_020) + delx*Rn_3(Ind_120)
  Rn_4(Ind_202) = Rn_3(Ind_002) + delx*Rn_3(Ind_102)
  Rn_4(Ind_022) = Rn_3(Ind_002) + dely*Rn_3(Ind_012)
  Rn_4(Ind_211) = dely*Rn_3(Ind_201)
  Rn_4(Ind_121) = delx*Rn_3(Ind_021)
  Rn_4(Ind_112) = delx*Rn_3(Ind_012)

! these are the only non-zero terms
  Rn(Ind_000)   = B(4)
  Rn_1(Ind_000) = B(3)
  Rn_2(Ind_000) = B(2)
  Rn_2(Ind_200) = Rn_1(Ind_000)  ! B(3)
  Rn_2(Ind_020) = Rn_1(Ind_000)  ! B(3)
  Rn_2(Ind_002) = Rn_1(Ind_000)  ! B(3)
  Rn_3(Ind_000) = B(1) 
  Rn_3(Ind_200) = Rn_2(Ind_000)  ! B(2)
  Rn_3(Ind_020) = Rn_2(Ind_000)  ! B(2)
  Rn_3(Ind_002) = Rn_2(Ind_000)  ! B(2)
  Rn_4(Ind_000) = B(0) 
  Rn_4(Ind_200) = Rn_3(Ind_000)  ! B(1)
  Rn_4(Ind_020) = Rn_3(Ind_000)  ! B(1)
  Rn_4(Ind_002) = Rn_3(Ind_000)  ! B(1)
#endif
! these are the only non-zero terms needed
  Rn_4(Ind_000) = B(0) 
  Rn_4(Ind_200) = B(1)
  Rn_4(Ind_020) = B(1)
  Rn_4(Ind_002) = B(1)

if (need_ene) then

  do i = 1, atm_cnt

! these are the only non-zero terms

    gphi(Ind_000)=  Rn_4(Ind_000)*chg(i)
    gphi(Ind_100)= -Rn_4(Ind_200)*dip(1,i)
    gphi(Ind_010)= -Rn_4(Ind_020)*dip(2,i)
    gphi(Ind_001)= -Rn_4(Ind_002)*dip(3,i)
    gphi(Ind_200)=  Rn_4(Ind_200)*chg(i)
    gphi(Ind_020)=  Rn_4(Ind_020)*chg(i)
    gphi(Ind_002)=  Rn_4(Ind_002)*chg(i)
    phi(1:7,i) = phi(1:7,i) - gphi(1:7)

    e_chg_stk = e_chg_stk + gphi(Ind_000) * chg(i)
    e_dip_stk = e_dip_stk + gphi(Ind_100) * dip(1, i) + gphi(Ind_010) * dip(2, i) + gphi(Ind_001) * dip(3, i)

  end do

  ene_chg = 0.5d0 * e_chg_stk
  ene_dip =-0.5d0 * e_dip_stk

else

  do i = 1, atm_cnt

! these are the only non-zero terms

    gphi(Ind_000)=  Rn_4(Ind_000)*chg(i)
    gphi(Ind_100)= -Rn_4(Ind_200)*dip(1,i)
    gphi(Ind_010)= -Rn_4(Ind_020)*dip(2,i)
    gphi(Ind_001)= -Rn_4(Ind_002)*dip(3,i)
    gphi(Ind_200)=  Rn_4(Ind_200)*chg(i)
    gphi(Ind_020)=  Rn_4(Ind_020)*chg(i)
    gphi(Ind_002)=  Rn_4(Ind_002)*chg(i)
    phi(1:7,i) = phi(1:7,i) - gphi(1:7)

  end do

end if
#if 0
  do i = 1, atm_cnt
!    do j = 1,10
!      gmi(j) = global_multipole(j,i)
!    end do
!    do j = 1,3
!      i_di(j) = ind_dip(j,i)
!    end do
!    gmi(2:4) = gmi(2:4) + i_di(1:3)

    ! self-field due to permanent mpoles at i and derivs wrt r_j-r_i (at 0)
!    gphi(Ind_000)=  Rn_4(Ind_000)*chg(i)
!    gphi(Ind_000)=  Rn_4(Ind_000)*gmi(Ind_000)+Rn_4(Ind_100)*gmi(Ind_100)+ &
!                    Rn_4(Ind_010)*gmi(Ind_010)+Rn_4(Ind_001)*gmi(Ind_001)+ &
!                    Rn_4(Ind_200)*gmi(Ind_200)+Rn_4(Ind_020)*gmi(Ind_020)+ &
!                    Rn_4(Ind_002)*gmi(Ind_002)+Rn_4(Ind_110)*gmi(Ind_110)+ &
!                    Rn_4(Ind_101)*gmi(Ind_101)+Rn_4(Ind_011)*gmi(Ind_011)
!    gphi(Ind_100)= -Rn_4(Ind_200)*dip(1,i)
!    gphi(Ind_100)=-(Rn_4(Ind_100)*gmi(Ind_000)+Rn_4(Ind_200)*gmi(Ind_100)+ &
!                    Rn_4(Ind_110)*gmi(Ind_010)+Rn_4(Ind_101)*gmi(Ind_001)+ &
!                    Rn_4(Ind_300)*gmi(Ind_200)+Rn_4(Ind_120)*gmi(Ind_020)+ &
!                    Rn_4(Ind_102)*gmi(Ind_002)+Rn_4(Ind_210)*gmi(Ind_110)+ &
!                    Rn_4(Ind_201)*gmi(Ind_101)+Rn_4(Ind_111)*gmi(Ind_011))
!    gphi(Ind_010)= -Rn_4(Ind_020)*dip(2,i)
!    gphi(Ind_010)=-(Rn_4(Ind_010)*gmi(Ind_000)+Rn_4(Ind_110)*gmi(Ind_100)+ &
!                    Rn_4(Ind_020)*gmi(Ind_010)+Rn_4(Ind_011)*gmi(Ind_001)+ &
!                    Rn_4(Ind_210)*gmi(Ind_200)+Rn_4(Ind_030)*gmi(Ind_020)+ &
!                    Rn_4(Ind_012)*gmi(Ind_002)+Rn_4(Ind_120)*gmi(Ind_110)+ &
!                    Rn_4(Ind_111)*gmi(Ind_101)+Rn_4(Ind_021)*gmi(Ind_011))
!    gphi(Ind_001)= -Rn_4(Ind_002)*dip(3,i)
!    gphi(Ind_001)=-(Rn_4(Ind_001)*gmi(Ind_000)+Rn_4(Ind_101)*gmi(Ind_100)+ &
!                    Rn_4(Ind_011)*gmi(Ind_010)+Rn_4(Ind_002)*gmi(Ind_001)+ &
!                    Rn_4(Ind_201)*gmi(Ind_200)+Rn_4(Ind_021)*gmi(Ind_020)+ &
!                    Rn_4(Ind_003)*gmi(Ind_002)+Rn_4(Ind_111)*gmi(Ind_110)+ &
!                    Rn_4(Ind_102)*gmi(Ind_101)+Rn_4(Ind_012)*gmi(Ind_011))
!    gphi(Ind_200)= Rn_4(Ind_200)*chg(i)
!    gphi(Ind_200)=  Rn_4(Ind_200)*gmi(Ind_000)+Rn_4(Ind_300)*gmi(Ind_100)+ &
!                    Rn_4(Ind_210)*gmi(Ind_010)+Rn_4(Ind_201)*gmi(Ind_001)+ &
!                    Rn_4(Ind_400)*gmi(Ind_200)+Rn_4(Ind_220)*gmi(Ind_020)+ &
!                    Rn_4(Ind_202)*gmi(Ind_002)+Rn_4(Ind_310)*gmi(Ind_110)+ &
!                    Rn_4(Ind_301)*gmi(Ind_101)+Rn_4(Ind_211)*gmi(Ind_011)
!    gphi(Ind_020)=  Rn_4(Ind_020)*chg(i)
!    gphi(Ind_020)=  Rn_4(Ind_020)*gmi(Ind_000)+Rn_4(Ind_120)*gmi(Ind_100)+ &
!                    Rn_4(Ind_030)*gmi(Ind_010)+Rn_4(Ind_021)*gmi(Ind_001)+ &
!                    Rn_4(Ind_220)*gmi(Ind_200)+Rn_4(Ind_040)*gmi(Ind_020)+ &
!                    Rn_4(Ind_022)*gmi(Ind_002)+Rn_4(Ind_130)*gmi(Ind_110)+ &
!                    Rn_4(Ind_121)*gmi(Ind_101)+Rn_4(Ind_031)*gmi(Ind_011)
!    gphi(Ind_002)=  Rn_4(Ind_002)*chg(i)
!    gphi(Ind_002)=  Rn_4(Ind_002)*gmi(Ind_000)+Rn_4(Ind_102)*gmi(Ind_100)+ &
!                    Rn_4(Ind_012)*gmi(Ind_010)+Rn_4(Ind_003)*gmi(Ind_001)+ &
!                    Rn_4(Ind_202)*gmi(Ind_200)+Rn_4(Ind_022)*gmi(Ind_020)+ &
!                    Rn_4(Ind_004)*gmi(Ind_002)+Rn_4(Ind_112)*gmi(Ind_110)+ &
!                    Rn_4(Ind_103)*gmi(Ind_101)+Rn_4(Ind_013)*gmi(Ind_011)

! These terms are zero.

!    gphi(Ind_110)=  Rn_4(Ind_110)*gmi(Ind_000)+Rn_4(Ind_210)*gmi(Ind_100)+ &
!                    Rn_4(Ind_120)*gmi(Ind_010)+Rn_4(Ind_111)*gmi(Ind_001)+ &
!                    Rn_4(Ind_310)*gmi(Ind_200)+Rn_4(Ind_130)*gmi(Ind_020)+ &
!                    Rn_4(Ind_112)*gmi(Ind_002)+Rn_4(Ind_220)*gmi(Ind_110)+ &
!                    Rn_4(Ind_211)*gmi(Ind_101)+Rn_4(Ind_121)*gmi(Ind_011)
!    gphi(Ind_101)=  Rn_4(Ind_101)*gmi(Ind_000)+Rn_4(Ind_201)*gmi(Ind_100)+ &
!                    Rn_4(Ind_111)*gmi(Ind_010)+Rn_4(Ind_102)*gmi(Ind_001)+ &
!                    Rn_4(Ind_301)*gmi(Ind_200)+Rn_4(Ind_121)*gmi(Ind_020)+ &
!                    Rn_4(Ind_103)*gmi(Ind_002)+Rn_4(Ind_211)*gmi(Ind_110)+ &
!                    Rn_4(Ind_202)*gmi(Ind_101)+Rn_4(Ind_112)*gmi(Ind_011)
!    gphi(Ind_011)=  Rn_4(Ind_011)*gmi(Ind_000)+Rn_4(Ind_111)*gmi(Ind_100)+ &
!                    Rn_4(Ind_021)*gmi(Ind_010)+Rn_4(Ind_012)*gmi(Ind_001)+ &
!                    Rn_4(Ind_211)*gmi(Ind_200)+Rn_4(Ind_031)*gmi(Ind_020)+ &
!                    Rn_4(Ind_013)*gmi(Ind_002)+Rn_4(Ind_121)*gmi(Ind_110)+ &
!                    Rn_4(Ind_112)*gmi(Ind_101)+Rn_4(Ind_022)*gmi(Ind_011)
!    phi(1:10,i) = phi(1:10,i) - gphi(1:10)
!    phi(1:7,i) = phi(1:7,i) - gphi(1:7)
  end do
#endif /* 0 */
end subroutine pGM_SELF_ene_frc
!-------------------------------------------------------
end module pol_gauss_self
!-------------------------------------------------------
#if 0
subroutine pGM_SELF_dip_field_test(numatoms, dip)

   use  pol_gauss_self, only : pGM_SELF_dipole_field
   use  pol_gauss_mdin_dat_mod, only : pol_gauss_verbose

   implicit none

   integer :: numatoms
   double precision              :: dip(3,*)
   double precision, allocatable :: field(:,:)

   if (pol_gauss_verbose == 3 ) then
   allocate(field(3,numatoms))

   field = 0.0d0

   write(117,*)
   write(117,*)'pGM_SELF_dipole_field', numatoms
   call pGM_SELF_dipole_field(numatoms, dip, field)

   write(117,*)'Field'
   write(117,'(6ES16.7)')field(1:3,1:10)
   write(117,*)
   end if

end subroutine pGM_SELF_dip_field_test
#endif
!-------------------------------------------------------

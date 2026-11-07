#include "copyright_pgm.i"
!*******************************************************************************
!
! Adopted from non-pGM pmemd code by YD
!
! Subroutine:  scalar_sumrc
!
! Description: Branched, optimized version of scalar_sumrc by DS Cerutti.  Note
!              that the VIRIAL pre-processor directive can only be set to 1 if
!              POT_ENES is also set to 1.
!              
!*******************************************************************************
#if USE_SAVED_ETERM
#if VIRIAL
subroutine scalar_sumrc_use_saved_eterm_uv(q, ewaldcof, vol, prefac1, prefac2, prefac3, &
                        nfft1, nfft2, nfft3, x_dim, y_dim, z_dim, G, eer, vir)
#elif POT_ENES
subroutine scalar_sumrc_use_saved_eterm_u(q, ewaldcof, vol, prefac1, prefac2, prefac3, &
                        nfft1, nfft2, nfft3, x_dim, y_dim, z_dim, G, eer)
#else
#define WEIGHT_ONLY 1
subroutine scalar_sumrc_use_saved_eterm(q, ewaldcof, vol, prefac1, prefac2, prefac3, &
                        nfft1, nfft2, nfft3, x_dim, y_dim, z_dim, G)
#endif
#else
#define NOT_USE_SAVED_ETERM 1
#if SAVE_ETERM
subroutine scalar_sumrc_save_eterm(q, ewaldcof, vol, prefac1, prefac2, prefac3, &
                        nfft1, nfft2, nfft3, x_dim, y_dim, z_dim, G)
#elif VIRIAL
subroutine scalar_sumrc_uv(q, ewaldcof, vol, prefac1, prefac2, prefac3, &
                           nfft1, nfft2, nfft3, x_dim, y_dim, z_dim, &
                           eer, vir)
#elif POT_ENES
subroutine scalar_sumrc_u(q, ewaldcof, vol, prefac1, prefac2, prefac3, &
                          nfft1, nfft2, nfft3, x_dim, y_dim, z_dim, &
                          eer)
#else
subroutine scalar_sumrc(q, ewaldcof, vol, prefac1, prefac2, prefac3, &
                        nfft1, nfft2, nfft3, x_dim, y_dim, z_dim)
#endif
#endif

  use parallel_dat_mod
  use pbc_mod
  use pme_slab_fft_mod

  implicit none

! Formal arguments:

  integer               :: nfft1, nfft2, nfft3
  integer               :: x_dim, y_dim, z_dim
  double precision      :: q(2,z_dim,x_dim,y_dim)
  double precision      :: ewaldcof
  double precision      :: vol
  double precision      :: prefac1(nfft1), prefac2(nfft2), prefac3(nfft3)
#if POT_ENES
  double precision      :: eer
#endif
#if VIRIAL
  double precision      :: vir(3,3)
#endif

! Local variables:

  double precision      :: fac_2, pi_vol_inv, qterm
#if POT_ENES
  double precision      :: energy
#endif
  double precision      :: eterm, eterm12, eterm_struc2, vterm
  double precision      :: eterms, eterms12, eterms_struc2s, vterms
  double precision      :: mhat1, mhat2, mhat3, msq_inv
#if POT_ENES
  double precision      :: mhat1s, mhat2s, mhat3s, msqs_inv
#endif
  double precision      :: recip_11, recip_22, recip_33
#if VIRIAL
  double precision      :: vir_11, vir_22, vir_33, vir_21, vir_31, vir_32
#endif
  integer               :: k1, k2, k2q, k3, k3_start, m1, m2, m3
#if POT_ENES
  integer               :: k1s, k2s, k3s, m1s, m2s, m3s
  integer               :: k3s_tbl(nfft3)
  integer               :: m3s_tbl(nfft3)
#endif
  integer               :: nf1, nf2, nf3
  integer               :: m3_tbl(nfft3)
#if SAVE_ETERM || USE_SAVED_ETERM
  double precision      :: G(z_dim,x_dim,y_dim)
#endif

  recip_11 = recip(1, 1)
  recip_22 = recip(2, 2)
  recip_33 = recip(3, 3)

  fac_2 = (2.d0 * PI * PI) / (ewaldcof * ewaldcof)

  pi_vol_inv = 1.d0 / (PI * vol)
  nf1 = nfft1 / 2
  ! There is an assumption that nfft1 must be even!
  nf2 = nfft2 / 2
  if (2 * nf2 .lt. nfft2) nf2 = nf2 + 1
  nf3 = nfft3 / 2
  if (2 * nf3 .lt. nfft3) nf3 = nf3 + 1

#if POT_ENES
  energy = 0.d0
#endif
#if VIRIAL
  vir_11 = 0.d0
  vir_22 = 0.d0
  vir_33 = 0.d0
  vir_21 = 0.d0
  vir_31 = 0.d0
  vir_32 = 0.d0
#endif

! Tables used to avoid branching in the innermost loop:

#ifndef WEIGHT_ONLY
  do k3 = 1, nfft3
    if (k3 .le. nf3) then
      m3_tbl(k3) = k3 - 1
    else
      m3_tbl(k3)= k3 - 1 - nfft3
    end if
#if POT_ENES
    k3s = mod(nfft3 - k3 + 1, nfft3) + 1
    k3s_tbl(k3) = k3s
    if (k3s .le. nf3) then
      m3s_tbl(k3) = k3s - 1
    else
      m3s_tbl(k3) = k3s - 1 - nfft3
    end if
#endif
  end do
#endif /* WEIGHT_ONLY */

! Insist that q(1,1,1,1) is set to 0.d0 (true already for neutral)
! All results using these elements are calculated, but add 0.d0, so
! it is like they are not used.

  if(my_zx_slab_start .eq. 0) then
    q(1, 1, 1, 1) = 0.d0
    q(2, 1, 1, 1) = 0.d0
  endif

! Big loop:

! k2q is the z index into the actual q array in this process;
! k2 is the index that would be used if the entire q array, which only exists
! for the uniprocessor case.

  do k2q = 1, my_zx_slab_cnt
    k2 = k2q + my_zx_slab_start
#ifndef WEIGHT_ONLY
    if (k2 .le. nf2) then
      m2 = k2 - 1
    else
      m2 = k2 - 1 - nfft2
    end if
    mhat2 = recip_22 * m2
#if POT_ENES
    k2s = mod(nfft2 - k2 + 1, nfft2) + 1
    if (k2s .le. nf2) then
      m2s = k2s - 1
    else
      m2s = k2s - 1 - nfft2
    end if
    mhat2s = recip_22 * m2s
#endif
#endif /* WEIGHT_ONLY */
    do k1 = 1, nf1 + 1
#ifndef WEIGHT_ONLY
      if (k1 .le. nf1) then
        m1 = k1 - 1
      else
        m1 = k1 - 1 - nfft1
      end if
      mhat1 = recip_11 * m1
      eterm12 = m1_exp_tbl(m1) * m2_exp_tbl(m2) * &
                prefac1(k1) * prefac2(k2) * pi_vol_inv
#endif /* WEIGHT_ONLY */
      k3_start = 1
      if ((my_zx_slab_start .eq. 0) .and. (k2 .eq. 1) .and. (k1 .eq. 1)) then
        k3_start = 2
      end if
#ifndef WEIGHT_ONLY
      if (k1 .gt. 1 .and. k1 .le. nfft1) then
#if POT_ENES
        k1s = nfft1 - k1 + 2
        if (k1s .le. nf1) then
          m1s = k1s - 1
        else
          m1s = k1s - 1 - nfft1
        end if
        mhat1s = recip_11 * m1s
        eterms12 = m1_exp_tbl(m1s) * m2_exp_tbl(m2s) * &
                   prefac1(k1s) * prefac2(k2s) * pi_vol_inv
#endif
#endif /* WEIGHT_ONLY */
        do k3 = k3_start, nfft3
#ifndef WEIGHT_ONLY
          m3 = m3_tbl(k3)
          mhat3 = recip_33 * m3
          msq_inv = 1.d0 / (mhat1 * mhat1 + mhat2 * mhat2 + mhat3 * mhat3)
#endif /* WEIGHT_ONLY */
          ! The product of the following 3 table lookups is exp(-fac * msq):
          ! (two of the lookups occurred in calculating eterm12)
#if USE_SAVED_ETERM
          eterm = G(k3, k1, k2q)
#else
          eterm = eterm12 * m3_exp_tbl(m3) * prefac3(k3) * msq_inv
#ifdef SAVE_ETERM
          G(k3, k1, k2q)    = eterm
#endif
#endif
#if POT_ENES
          qterm = (q(1, k3, k1, k2q) * q(1, k3, k1, k2q) + &
                   q(2, k3, k1, k2q) * q(2, k3, k1, k2q))
#endif
          q(1, k3, k1, k2q) = eterm * q(1, k3, k1, k2q)
          q(2, k3, k1, k2q) = eterm * q(2, k3, k1, k2q)
#ifndef WEIGHT_ONLY
#if POT_ENES
          k3s = k3s_tbl(k3)
          m3s = m3s_tbl(k3)
          mhat3s = recip_33 * m3s
          eterm_struc2 = eterm * qterm
          ! The product of the following 3 table lookups is exp(-fac * msqs):
          ! (two of the lookups occurred in calculating eterms12)
          msqs_inv = 1.d0 / (mhat1s*mhat1s + mhat2s*mhat2s + mhat3s*mhat3s)
          eterms = eterms12 * m3_exp_tbl(m3s) * prefac3(k3s) * msqs_inv
          eterms_struc2s = eterms * qterm
          energy = energy + eterm_struc2 + eterms_struc2s
#endif
#if VIRIAL
          vterm = (fac_2 + 2.d0 * msq_inv) * eterm_struc2
          vterms = (fac_2 + 2.d0 * msqs_inv) * eterms_struc2s
          vir_21 = vir_21 + vterm * mhat1 * mhat2 + vterms * mhat1s * mhat2s
          vir_31 = vir_31 + vterm * mhat1 * mhat3 + vterms * mhat1s * mhat3s
          vir_32 = vir_32 + vterm * mhat2 * mhat3 + vterms * mhat2s * mhat3s
          vir_11 = vir_11 + vterm * mhat1 * mhat1 - eterm_struc2 + &
                            vterms * mhat1s * mhat1s - eterms_struc2s
          vir_22 = vir_22 + vterm * mhat2 * mhat2 - eterm_struc2 + &
                            vterms * mhat2s * mhat2s - eterms_struc2s
          vir_33 = vir_33 + vterm * mhat3 * mhat3 - eterm_struc2 + &
                            vterms * mhat3s * mhat3s - eterms_struc2s
#endif
        end do
      else
        do k3 = k3_start, nfft3
          m3 = m3_tbl(k3)
          mhat3 = recip_33 * m3
          msq_inv = 1.d0 / (mhat1 * mhat1 + mhat2 * mhat2 + mhat3 * mhat3)
          ! The product of the following 3 table lookups is exp(-fac * msq):
          ! (two of the lookups occurred in calculating eterm12)
#if USE_SAVED_ETERM
          eterm = G(k3, k1, k2q)
#else
          eterm = eterm12 * m3_exp_tbl(m3) * prefac3(k3) * msq_inv
#ifdef SAVE_ETERM
          G(k3, k1, k2q)    = eterm
#endif
#endif
#if POT_ENES
          qterm = (q(1, k3, k1, k2q) * q(1, k3, k1, k2q) + &
                   q(2, k3, k1, k2q) * q(2, k3, k1, k2q))
#endif
          q(1, k3, k1, k2q) = eterm * q(1, k3, k1, k2q)
          q(2, k3, k1, k2q) = eterm * q(2, k3, k1, k2q)
#if POT_ENES
          eterm_struc2 = eterm * qterm
          energy = energy + eterm_struc2
#endif
#if VIRIAL
          vterm = (fac_2 + 2.d0 * msq_inv) * eterm_struc2
          vir_21 = vir_21 + vterm * mhat1 * mhat2
          vir_31 = vir_31 + vterm * mhat1 * mhat3
          vir_32 = vir_32 + vterm * mhat2 * mhat3
          vir_11 = vir_11 + vterm * mhat1 * mhat1 - eterm_struc2
          vir_22 = vir_22 + vterm * mhat2 * mhat2 - eterm_struc2
          vir_33 = vir_33 + vterm * mhat3 * mhat3 - eterm_struc2
#endif
#endif /* WEIGHT_ONLY */
        end do
#ifndef WEIGHT_ONLY
      end if
#endif /* WEIGHT_ONLY */
    end do
  end do

#if POT_ENES
  eer = 0.5d0 * energy
#endif
#if VIRIAL
  vir(1, 1) = 0.5d0 * vir_11
  vir(2, 1) = 0.5d0 * vir_21
  vir(3, 1) = 0.5d0 * vir_31
  vir(1, 2) = 0.5d0 * vir_21
  vir(2, 2) = 0.5d0 * vir_22
  vir(3, 2) = 0.5d0 * vir_32
  vir(1, 3) = 0.5d0 * vir_31
  vir(2, 3) = 0.5d0 * vir_32
  vir(3, 3) = 0.5d0 * vir_33
#endif
  return
#if USE_SAVED_ETERM
#if VIRIAL && POT_ENES
end subroutine scalar_sumrc_use_saved_eterm_uv
#elif POT_ENES
end subroutine scalar_sumrc_use_saved_eterm_u
#else
#undef WEIGHT_ONLY
end subroutine scalar_sumrc_use_saved_eterm
#endif
#else
#undef NOT_USE_SAVED_ETERM
#if SAVE_ETERM
end subroutine scalar_sumrc_save_eterm
#elif VIRIAL && POT_ENES
end subroutine scalar_sumrc_uv
#elif POT_ENES
end subroutine scalar_sumrc_u
#else
end subroutine scalar_sumrc
#endif
#endif

!*******************************************************************************
!
! Subroutine:  scalar_sumrc_nonorthog
!
! Description: Branched, optimized version of scalar_sumrc_nonorthog by DS
!              Cerutti.  Brings the cost of simulations of triclinic cells in
!              line with rectilinear cells on an atom-by-atom basis.  Note
!              that the VIRIAL pre-processor directive can only be set to 1 if
!              POT_ENES is also set to 1.
!              
!*******************************************************************************
#if USE_SAVED_ETERM
#if VIRIAL
subroutine scalar_sumrc_nonorthog_use_saved_eterm_uv(q, ewaldcof, vol, &
                                  prefac1, prefac2, prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  x_dim, y_dim, z_dim, G, eer, vir)
#elif POT_ENES
subroutine scalar_sumrc_nonorthog_use_saved_eterm_u(q, ewaldcof, vol, &
                                  prefac1, prefac2, prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  x_dim, y_dim, z_dim, G, eer)
#else
#define WEIGHT_ONLY 1
subroutine scalar_sumrc_nonorthog_use_saved_eterm(q, ewaldcof, vol, &
                                  prefac1, prefac2, prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  x_dim, y_dim, z_dim, G)
#endif
#else
#define NOT_USE_SAVED_ETERM 1
#if SAVE_ETERM
subroutine scalar_sumrc_nonorthog_save_eterm(q, ewaldcof, vol, &
                                  prefac1, prefac2, prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  x_dim, y_dim, z_dim, G)
#elif VIRIAL
subroutine scalar_sumrc_nonorthog_uv(q, ewaldcof, vol, &
                                     prefac1, prefac2, prefac3, &
                                     nfft1, nfft2, nfft3, &
                                     x_dim, y_dim, z_dim, &
                                     eer, vir)
#elif POT_ENES
subroutine scalar_sumrc_nonorthog_u(q, ewaldcof, vol, &
                                    prefac1, prefac2, prefac3, &
                                    nfft1, nfft2, nfft3, &
                                    x_dim, y_dim, z_dim, &
                                    eer)
#else
subroutine scalar_sumrc_nonorthog(q, ewaldcof, vol, &
                                  prefac1, prefac2, prefac3, &
                                  nfft1, nfft2, nfft3, &
                                  x_dim, y_dim, z_dim)
#endif
#endif

  use parallel_dat_mod
  use pbc_mod
  use pme_slab_fft_mod

  implicit none

! Formal arguments:

  integer               :: nfft1, nfft2, nfft3
  integer               :: x_dim, y_dim, z_dim
  double precision      :: q(2,z_dim,x_dim,y_dim)
  double precision      :: ewaldcof
  double precision      :: vol
  double precision      :: prefac1(nfft1), prefac2(nfft2), prefac3(nfft3)
#if POT_ENES
  double precision      :: eer
#endif
#if VIRIAL
  double precision      :: vir(3,3)
#endif
#if SAVE_ETERM || USE_SAVED_ETERM
  double precision      :: G(z_dim,x_dim,y_dim)
#endif

! Local variables:

#if POT_ENES
  double precision      :: energy
  double precision      :: mhat1s, mhat2s, mhat3s, msqs, msqs_inv
#endif
  double precision      :: fac, fac_2, pi_vol_inv
  double precision      :: eterm, eterm_struc2, vterm
  double precision      :: eterms, eterms_struc2s, vterms
  double precision      :: mhat1, mhat2, mhat3, msq, msq_inv
  double precision      :: mhat1yz, mhat2yz, mhat3yz, pfacyz
  double precision      :: e2ab_term, e2ab_arg, e2ab_inc, e2ab_shift, eb2_term
  double precision      :: myexp
  double precision      :: ea2_term(nfft1)
  double precision      :: recip_stk(3,3)
#if VIRIAL
  double precision      :: vir_11, vir_22, vir_33, vir_21, vir_31, vir_32
#endif
  integer               :: k, k1, k2, k2q, k3, k1_start, m1, m2, m3
#if POT_ENES
  integer               :: k1s, k2s, k3s, m1s, m2s, m3s
#endif
  integer               :: nf1, nf2, nf3

  recip_stk(:,:) = recip(:,:)

  fac = (PI * PI) / (ewaldcof * ewaldcof)

  fac_2 = 2.d0 * fac

  pi_vol_inv = 1.d0 / (PI * vol)

  nf1 = nfft1 / 2
  if (2 * nf1 .lt. nfft1) nf1 = nf1 + 1
  nf2 = nfft2 / 2
  if (2 * nf2 .lt. nfft2) nf2 = nf2 + 1
  nf3 = nfft3 / 2
  if (2 * nf3 .lt. nfft3) nf3 = nf3 + 1

#ifndef WEIGHT_ONLY
  eb2_term = -fac * ( recip_stk(1,1) * recip_stk(1,1) + &
                      recip_stk(2,1) * recip_stk(2,1) + &
                      recip_stk(3,1) * recip_stk(3,1) )
  do m1 = 0, nf1-1
    ea2_term(m1+1) = exp( eb2_term * m1 * m1 ) * prefac1(m1+1)
  end do
  ea2_term(nf1+1) = exp( eb2_term * (nf1 - nfft1) * (nf1 - nfft1) ) * &
                    prefac1(nf1+1)

#endif /* WEIGHT_ONLY */

#if POT_ENES
  energy = 0.d0
#endif
#if VIRIAL
  vir_11 = 0.d0
  vir_22 = 0.d0
  vir_33 = 0.d0
  vir_21 = 0.d0
  vir_31 = 0.d0
  vir_32 = 0.d0
#endif

! Insist that q(1,1,1,1) is set to 0 (true already for neutral)
  if(my_zx_slab_start .eq. 0) then
    q(1, 1, 1, 1) = 0.d0
    q(2, 1, 1, 1) = 0.d0
  endif

! Big loop:

  do k2q = 1, my_zx_slab_cnt
    k2 = k2q + my_zx_slab_start
#ifndef WEIGHT_ONLY
    if (k2 .le. nf2) then
      m2 = k2 - 1
    else
      m2 = k2 - 1 - nfft2
    end if

#if POT_ENES
    k2s = mod(nfft2 - k2 + 1, nfft2) + 1
    if (k2s .le. nf2) then
      m2s = k2s - 1
    else
      m2s = k2s - 1 - nfft2
    end if
#endif
#endif /* WEIGHT_ONLY */

    do k3 = 1, nfft3
#ifndef WEIGHT_ONLY
      if (k3 .le. nf3) then
        m3 = k3 - 1
      else
        m3 = k3 - 1 - nfft3
      end if
#if POT_ENES
      k3s = mod(nfft3 - k3 + 1, nfft3) + 1
      if (k3s .le. nf3) then
        m3s = k3s - 1
      else
        m3s = k3s - 1 - nfft3
      end if
#endif

      ! Pre-compute some constants out here (really just cleans up the code
      ! in the inner loop--compiler optimization renders it moot for timings)
      mhat1yz = recip_stk(1,2) * m2 + recip_stk(1,3) * m3
      mhat2yz = recip_stk(2,2) * m2 + recip_stk(2,3) * m3
      mhat3yz = recip_stk(3,2) * m2 + recip_stk(3,3) * m3
      e2ab_arg = -fac * 2.0 * (recip_stk(1,1) * mhat1yz + &
                               recip_stk(2,1) * mhat2yz + &
                               recip_stk(3,1) * mhat3yz)
      e2ab_inc = exp( e2ab_arg )
      e2ab_shift = exp( -nfft1 * e2ab_arg )
      eb2_term = exp( -fac * (mhat1yz * mhat1yz + mhat2yz * mhat2yz + &
                              mhat3yz * mhat3yz) )
      pfacyz = prefac2(k2) * prefac3(k3) * pi_vol_inv
#endif /* WEIGHT_ONLY */
 
      ! Determine the starting index of the inner loop and
      ! set e2ab_term appropriately
      if ((my_zx_slab_start .eq. 0) .and. (k3 + k2 .eq. 2)) then
        k1_start = 2
#ifndef WEIGHT_ONLY
        e2ab_term = e2ab_inc * pfacyz * eb2_term
#endif /* WEIGHT_ONLY */
      else
        k1_start = 1
#ifndef WEIGHT_ONLY
        e2ab_term = pfacyz * eb2_term
#endif  /* WEIGHT_ONLY */
      end if
      do k1 = k1_start, nf1 + 1
#ifndef WEIGHT_ONLY
        if (k1 .le. nf1) then
          m1 = k1 - 1
        else
          m1 = k1 - 1 - nfft1
          e2ab_term = e2ab_term * e2ab_shift
        end if
        mhat1 = recip_stk(1,1) * m1 + mhat1yz
        mhat2 = recip_stk(2,1) * m1 + mhat2yz
        mhat3 = recip_stk(3,1) * m1 + mhat3yz
        msq = mhat1 * mhat1 + mhat2 * mhat2 + mhat3 * mhat3
        msq_inv = 1.d0 / msq

#ifdef USE_SAVED_ETERM
        eterm = G(k3, k1, k2q)
#else
        ! We now get this exponential by table lookup
        eterm = ea2_term(k1) * e2ab_term * msq_inv
#if SAVE_ETERM
        G(k3, k1, k2q)    = eterm
#endif
#endif
        e2ab_term = e2ab_term * e2ab_inc
#if POT_ENES
        eterm_struc2 = eterm * (q(1, k3, k1, k2q) * q(1, k3, k1, k2q) + &
                                q(2, k3, k1, k2q) * q(2, k3, k1, k2q))
        energy = energy + eterm_struc2
#endif
#if VIRIAL
        vterm = fac_2 + 2.d0 * msq_inv
        vir_21 = vir_21 + eterm_struc2 * (vterm * mhat1 * mhat2)
        vir_31 = vir_31 + eterm_struc2 * (vterm * mhat1 * mhat3)
        vir_32 = vir_32 + eterm_struc2 * (vterm * mhat2 * mhat3)
        vir_11 = vir_11 + eterm_struc2 * (vterm * mhat1 * mhat1 - 1.d0)
        vir_22 = vir_22 + eterm_struc2 * (vterm * mhat2 * mhat2 - 1.d0)
        vir_33 = vir_33 + eterm_struc2 * (vterm * mhat3 * mhat3 - 1.d0)
#endif
#if POT_ENES || VIRIAL
        if (k1 .gt. 1 .and. k1 .le. nfft1) then
#endif
#if POT_ENES
          k1s = nfft1 - k1 + 2
          if (k1s .le. nf1) then
            m1s = k1s - 1
          else
            m1s = k1s - 1 - nfft1
          end if
          mhat1s = recip_stk(1,1) * m1s + &
                   recip_stk(1,2) * m2s + &
                   recip_stk(1,3) * m3s
          mhat2s = recip_stk(2,1) * m1s + &
                   recip_stk(2,2) * m2s + &
                   recip_stk(2,3) * m3s
          mhat3s = recip_stk(3,1) * m1s + &
                   recip_stk(3,2) * m2s + &
                   recip_stk(3,3) * m3s
          msqs = mhat1s * mhat1s + mhat2s * mhat2s + mhat3s * mhat3s
          msqs_inv = 1.d0 / msqs

          ! Getting the exponential via table lookup is currently not done
          ! for nonorthogonal unit cells.
          eterms = exp(-fac * msqs) * &
                   prefac1(k1s) * prefac2(k2s) * prefac3(k3s) * &
                   pi_vol_inv * msqs_inv
          eterms_struc2s = eterms * (q(1, k3, k1, k2q) * q(1, k3, k1, k2q) + &
                                     q(2, k3, k1, k2q) * q(2, k3, k1, k2q))
          energy = energy + eterms_struc2s
#endif
#if VIRIAL
          vterms = fac_2 + 2.d0 * msqs_inv
          vir_21 = vir_21 + eterms_struc2s * (vterms * mhat1s * mhat2s)
          vir_31 = vir_31 + eterms_struc2s * (vterms * mhat1s * mhat3s)
          vir_32 = vir_32 + eterms_struc2s * (vterms * mhat2s * mhat3s)
          vir_11 = vir_11 + eterms_struc2s * (vterms * mhat1s * mhat1s - 1.d0)
          vir_22 = vir_22 + eterms_struc2s * (vterms * mhat2s * mhat2s - 1.d0)
          vir_33 = vir_33 + eterms_struc2s * (vterms * mhat3s * mhat3s - 1.d0)
#endif
#if POT_ENES || VIRIAL
        endif
#endif
#else /* WEIGHT_ONLY */
        eterm = G(k3, k1, k2q)
#endif /* WEIGHT_ONLY */
        q(1, k3, k1, k2q) = eterm * q(1, k3, k1, k2q)
        q(2, k3, k1, k2q) = eterm * q(2, k3, k1, k2q)
      end do
    end do
  end do
#if POT_ENES
  eer = 0.5d0 * energy
#endif
#if VIRIAL
  vir(1, 1) = 0.5d0 * vir_11
  vir(2, 1) = 0.5d0 * vir_21
  vir(3, 1) = 0.5d0 * vir_31
  vir(1, 2) = 0.5d0 * vir_21
  vir(2, 2) = 0.5d0 * vir_22
  vir(3, 2) = 0.5d0 * vir_32
  vir(1, 3) = 0.5d0 * vir_31
  vir(2, 3) = 0.5d0 * vir_32
  vir(3, 3) = 0.5d0 * vir_33
#endif
  return
#if USE_SAVED_ETERM
#if VIRIAL && POT_ENES
end subroutine scalar_sumrc_nonorthog_use_saved_eterm_uv
#elif POT_ENES
end subroutine scalar_sumrc_nonorthog_use_saved_eterm_u
#else
#undef WEIGHT_ONLY
end subroutine scalar_sumrc_nonorthog_use_saved_eterm
#endif
#else
#undef NOT_USE_SAVED_ETERM
#if SAVE_ETERM
end subroutine scalar_sumrc_nonorthog_save_eterm
#elif VIRIAL && POT_ENES
end subroutine scalar_sumrc_nonorthog_uv
#elif POT_ENES
end subroutine scalar_sumrc_nonorthog_u
#else
end subroutine scalar_sumrc_nonorthog
#endif
#endif

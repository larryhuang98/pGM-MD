#include "copyright_pgm.i"
#ifdef _SAVE_BFACT
#undef _SAVE_BFACT
#endif
#ifdef _USE_SAVED_BFACT
#undef _USE_SAVED_BFACT
#endif
!
! Adopted from non-pGM pmemd code by YD
!
!*******************************************************************************
! 
! Subroutine:  get_nb_energy_PGM and get_direct_fld_PGM
!
! Description:
!              
! The main routine for non bond energy (vdw and hbond) and field as well as
! direct part of ewald sum.  It is structured for parallelism.
!
!*******************************************************************************

#ifdef FIELD_ONLY
#ifdef SAVE_BFACT
subroutine get_direct_fld_PGM_save_B(ipairs, img_crd, img_chg, img_dip, eed_cub, tranvec, gradphi)
#define _SAVE_BFACT
#else
subroutine get_direct_fld_PGM(ipairs, img_crd, img_chg, img_dip, eed_cub, tranvec, gradphi)
#endif
#else /* FIELD_ONLY */
#ifdef SAVE_BFACT
#define _USE_SAVED_BFACT
subroutine get_nb_energy_PGM_use_saved_B(ipairs, img_crd, img_disp, img_chg, img_dip, eed_cub, tranvec, &
                          img_frc, phi, need_pot_enes, need_virials, &
                          eed, epolar, evdw, ehb, eedvir, eed_virial, vdw_virial)
#else
subroutine get_nb_energy_PGM(ipairs, img_crd, img_disp, img_chg, img_dip, eed_cub, tranvec, &
                          img_frc, phi, need_pot_enes, need_virials, &
                          eed, epolar, evdw, ehb, eedvir, eed_virial, vdw_virial)
#endif
#endif /* FIELD_ONLY */

  use img_mod
  use timers_mod
  use mdin_ctrl_dat_mod
  use mdin_ewald_dat_mod
  use parallel_mod
  use parallel_dat_mod
  use prmtop_dat_mod
  use ene_frc_splines_mod
!#ifdef PGM
  use pol_gauss_direct_data
  use pol_gauss_multipoles_mod, only : r_gauss, r_gauss_img
#ifdef FIELD_ONLY
  use pol_gauss_mdin_dat_mod, only    : scf_local_cut
#else
!#ifdef NEED_VIR
  use pol_gauss_direct_data, only : displacement_img
  use pol_gauss_mdin_dat_mod, only : igvdw, gvdw_rep_form, b_rep_scale
#endif
!#endif

  implicit none

! Formal arguments:

#ifdef PGM
#include "pGM_mpole_index.h"
#endif
  double precision, intent(in)     :: img_chg(*)
  double precision, intent(in)     :: img_dip(3, *)
  double precision, intent(in)     :: img_crd(3, *)
  double precision, intent(in)     :: eed_cub(*)
  integer                          :: ipairs(*)
  double precision, intent(in)     :: tranvec(1:3, 0:17)
#ifdef FIELD_ONLY
  double precision, intent(in)     :: gradphi(3,*)
!  double precision, intent(out)    :: dipole_dipole_tensor(6,*)
!  double precision, intent(out)    :: dipole_dipole_tensor_short(6,*)
#ifdef _SAVE_BFACT
!  double precision, intent(out)    :: delvec_save(4,*)
!  double precision, intent(out)    :: Bfact_save(4,*)
!  logical, intent(out)             :: has_VDW_save(*)
#endif
!  integer, intent(out)             :: dipole_dipole_list(*)
!  integer, intent(out)             :: dipole_dipole_list_short(*)
!  integer, intent(out)             :: num_tensor
!  integer, intent(out)             :: num_tensor_short
#else
  double precision                 :: phi(10,*)
  double precision                 :: img_frc(3,*)
  double precision                 :: img_disp(3,*)
#ifdef _USE_SAVED_BFACT
!  integer, intent(in)              :: num_tensor
!  integer, intent(in)              :: dipole_dipole_list(*)
!  double precision, intent(in)     :: gradphi(3,*)
!  double precision, intent(in)     :: dipole_dipole_tensor(6,*)
!  double precision, intent(in)     :: delvec_save(4,*)
!  double precision, intent(in)     :: Bfact_save(4,*)
#endif
  logical, intent(in)              :: need_pot_enes
  logical, intent(in)              :: need_virials
  double precision, intent(out)    :: eed
  double precision, intent(out)    :: epolar
  double precision, intent(out)    :: evdw
  double precision, intent(out)    :: ehb
  double precision, intent(out)    :: eedvir
  double precision, intent(out)    :: eed_virial(3, 3)
  double precision, intent(out)    :: vdw_virial(3, 3)
#endif

! Local variables and parameters:
  double precision                  del
  double precision                  dxdr
  double precision                  max_nb_cut2, es_cut2, es_cut
  double precision                  x_i, y_i, z_i
  double precision                  x_tran(1:3, 0:17)
  double precision      eedtbdns_stk
  double precision      eedvir_stk, eed_stk, evdw_stk, ehb_stk
!  double precision      vxx, vxy, vxz, vyy, vyz, vzz
  double precision      vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz
  integer               i
  integer               ipairs_idx
  integer               ntypes_stk
  integer               img_i
  integer               ee_eval_cnt
  integer               full_eval_cnt
  integer               common_tran    ! flag - 1 if translation not needed
  logical               cutoffs_equal

!  integer, allocatable :: itmp(:,:)
  integer  :: idx1,idx2
#ifdef FIELD_ONLY
!  integer               :: dip_dip_cnt
!  integer               :: dip_dip_cnt_short
!  integer               :: dip_dip_lst_imgi(natom)
!  integer               :: dip_dip_lst_short_imgi(natom)
#endif

! All these should be moved to outside
#ifdef FIELD_ONLY
  if (ipgm .gt. 0 .or. ipol .eq. 5) then
    num_tensor=0
    num_tensor_short=0
    dipole_dipole_ptrs(my_img_lo) = 1
    dipole_dipole_short_ptrs(my_img_lo) = 1
  end if
#endif /* FIELD_ONLY */
!  allocate(itmp(2,my_img_hi+1))

#ifdef MPI
  if (my_img_lo .gt. my_img_hi) return
#endif /* MPI */

#ifdef CUDA
  call gpu_get_nb_energy()
#else

  dxdr = ew_coeff
  eedtbdns_stk = eedtbdns
  del = 1.d0 / eedtbdns_stk
  max_nb_cut2 = vdw_cutoff * vdw_cutoff
  es_cut = es_cutoff
  es_cut2 = es_cut * es_cut
  cutoffs_equal = (vdw_cutoff .eq. es_cutoff)

#ifndef FIELD_ONLY
  ntypes_stk = ntypes

  eedvir_stk = 0.d0
  eed_stk = 0.d0
  evdw_stk = 0.d0
  ehb_stk = 0.d0

  vxx = 0.d0
  vxy = 0.d0
  vxz = 0.d0
  vyx = 0.d0
  vyy = 0.d0
  vyz = 0.d0
  vzx = 0.d0
  vzy = 0.d0
  vzz = 0.d0

  eed_virial(1:3,1:3) = 0.d0

#endif  /* FIELD_ONLY */
  ipairs_idx =1
#ifndef FIELD_ONLY
  if (need_pot_enes) then
#endif
    do img_i = my_img_lo, my_img_hi
#ifdef _USE_SAVED_BFACT
! if saved b-factors are used, only img_i index is needed, nothing else.
#else /* _USE_SAVED_BFACT */

      ! Common translation (ie. no translation) flag is packed at
      ! the front of each sublist followed by the count(s) of sublist
      ! image pair entries.

      common_tran = ipairs(ipairs_idx)
      ipairs_idx = ipairs_idx + 1
    
      ! Electrostatic evaluation-only count followed by
      ! full evaluation count packed at the front of each pair sublist.

      ee_eval_cnt = ipairs(ipairs_idx)
      full_eval_cnt = ipairs(ipairs_idx + 1)
      ipairs_idx = ipairs_idx + 2

      if (ee_eval_cnt + full_eval_cnt .gt. 0) then
        x_i = img_crd(1, img_i)
        y_i = img_crd(2, img_i)
        z_i = img_crd(3, img_i)
        if (common_tran .eq. 0) then
          ! We need all the translation vectors:
          do i = 0, 17
            x_tran(1, i) = tranvec(1, i) - x_i
            x_tran(2, i) = tranvec(2, i) - y_i
            x_tran(3, i) = tranvec(3, i) - z_i
          end do
        else
          ! Just put the x,y,z values in the middle cell
          x_tran(1, 13) = - x_i
          x_tran(2, 13) = - y_i
          x_tran(3, 13) = - z_i
        end if
#endif /* _USE_SAVED_BFACT */
#ifdef FIELD_ONLY
! _USE_SAVED_BFACT and FIELD_ONLY are mutually exclusive.
#ifndef _USE_SAVED_BFACT

#ifndef _SAVE_BFACT
        if (cutoffs_equal) then
            if(use_pme .eq. 0) then
        call pairs_calc_PGM_fld_vac(img_crd, img_chg, img_dip, efs_tbl, &
                          eed_cub, ipairs(ipairs_idx), x_tran, &
                          gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                          dipole_dipole_list_short, dipole_dipole_tensor_short)
            else
        call pairs_calc_PGM_fld(img_crd, img_chg, img_dip, efs_tbl, &
                          eed_cub, ipairs(ipairs_idx), x_tran, &
                          gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                          dipole_dipole_list_short, dipole_dipole_tensor_short)
            end if
        else
            if(use_pme .eq. 0) then
        call pairs_calc_PGM_fld_2cut_vac(img_crd, img_chg, img_dip, efs_tbl, &
                          eed_cub, ipairs(ipairs_idx), x_tran, &
                          gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                          dipole_dipole_list_short, dipole_dipole_tensor_short)
            else
        call pairs_calc_PGM_fld_2cut(img_crd, img_chg, img_dip, efs_tbl, &
                          eed_cub, ipairs(ipairs_idx), x_tran, &
                          gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                          dipole_dipole_list_short, dipole_dipole_tensor_short)
            end if
        end if
#else
        if (cutoffs_equal) then
            if(use_pme .eq. 0) then
        call pairs_calc_PGM_fld_vac_save_B(img_crd, img_chg, img_dip, efs_tbl, &
                          eed_cub, ipairs(ipairs_idx), x_tran, &
                          gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                          dipole_dipole_list_short, dipole_dipole_tensor_short)
            else
        call pairs_calc_PGM_fld_save_B(img_crd, img_chg, img_dip, efs_tbl, &
                          eed_cub, ipairs(ipairs_idx), x_tran, &
                          gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                          dipole_dipole_list_short, dipole_dipole_tensor_short)
            end if
        else
            if(use_pme .eq. 0) then
        call pairs_calc_PGM_fld_2cut_vac_save_B(img_crd, img_chg, img_dip, efs_tbl, &
                          eed_cub, ipairs(ipairs_idx), x_tran, &
                          gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                          dipole_dipole_list_short, dipole_dipole_tensor_short)
            else
        call pairs_calc_PGM_fld_2cut_save_B(img_crd, img_chg, img_dip, efs_tbl, &
                          eed_cub, ipairs(ipairs_idx), x_tran, &
                          gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
                          dipole_dipole_list_short, dipole_dipole_tensor_short)
            end if
        end if
#endif /* _SAVE_BFACT */
#endif /* _USE_SAVED_BFACT */
#else  /* FIELD_ONLY */
#ifndef _USE_SAVED_BFACT
        ! We always need virials from this routine if we need energies,
        ! because the virials are used in estimating the pme error.

        if (cutoffs_equal) then
          if(fswitch .le. 0) then
            if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
              call pairs_calc_PGM_efv_gvdw_vac(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
              call pairs_calc_PGM_efv_vac(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               endif
            else
               if (igvdw .gt. 0) then
              call pairs_calc_PGM_efv_gvdw(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
              call pairs_calc_PGM_efv(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                  eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                  gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               endif
            end if
          else
            call pairs_calc_PGM_efv_fs(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                              eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                              gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
          end if
        else
          if(fswitch .le. 0) then
            if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
              call pairs_calc_PGM_efv_gvdw_2cut_vac(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
              call pairs_calc_PGM_efv_2cut_vac(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                   eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                  gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
            else
               if (igvdw .gt. 0) then
              call pairs_calc_PGM_efv_gvdw_2cut(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
              call pairs_calc_PGM_efv_2cut(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                   eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                  gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
            end if
          else
            call pairs_calc_PGM_efv_2cut_fs(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                   eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                   gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
          end if
        end if
#else
        ! pGM no-longer estimates the pme error which was useful for development and validation of pme.
        ! pGM cutoffs in _use_saved_B are handled by _save_B. The 2-cut options have not been implemented in
        ! either.
        if (need_virials) then
          if(fswitch .le. 0) then
            if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
              call pairs_calc_PGM_efv_gvdw_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
              call pairs_calc_PGM_efv_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
            else
               if (igvdw .gt. 0) then
              call pairs_calc_PGM_efv_gvdw_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
              call pairs_calc_PGM_efv_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                  eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                  gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
            end if
          else
            call pairs_calc_PGM_efv_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                              eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                              gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
          end if
        else
          if(fswitch .le. 0) then
            if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
              call pairs_calc_PGM_ef_gvdw_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
              call pairs_calc_PGM_ef_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
            else
               if (igvdw .gt. 0) then
              call pairs_calc_PGM_ef_gvdw_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
              call pairs_calc_PGM_ef_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                                  eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                  gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
            end if
          else
            call pairs_calc_PGM_ef_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, efs_tbl, &
                              eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                              gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
          end if
        end if
#endif
#endif  /* FIELD_ONLY */
#ifndef _USE_SAVED_BFACT
         ipairs_idx = ipairs_idx + ee_eval_cnt + full_eval_cnt

      end if
#endif

#ifdef FIELD_ONLY
  dipole_dipole_ptrs(img_i+1) = num_tensor + 1
  dipole_dipole_short_ptrs(img_i+1) = num_tensor_short + 1
#endif
    end do
#ifndef FIELD_ONLY
  else ! do not need energies...

#ifndef _USE_SAVED_BFACT
  ipairs_idx = 1
#endif
    do img_i = my_img_lo, my_img_hi

#ifndef _USE_SAVED_BFACT
! if saved b-factors are used, only img_i index is needed, nothing else.
      ! Common translation (ie. no translation) flag is packed at
      ! the front of each sublist followed by the count(s) of sublist
      ! image pair entries.

      common_tran = ipairs(ipairs_idx)
      ipairs_idx = ipairs_idx + 1
    
      ! Electrostatic evaluation-only count followed by
      ! full evaluation count packed at the front of each pair sublist.

      ee_eval_cnt = ipairs(ipairs_idx)
      full_eval_cnt = ipairs(ipairs_idx + 1)
      ipairs_idx = ipairs_idx + 2

      if (ee_eval_cnt + full_eval_cnt .gt. 0) then
        x_i = img_crd(1, img_i)
        y_i = img_crd(2, img_i)
        z_i = img_crd(3, img_i)
        if (common_tran .eq. 0) then
          ! We need all the translation vectors:
          do i = 0, 17
            x_tran(1, i) = tranvec(1, i) - x_i
            x_tran(2, i) = tranvec(2, i) - y_i
            x_tran(3, i) = tranvec(3, i) - z_i
          end do
        else
          ! Just put the x,y,z values in the middle cell
          x_tran(1, 13) = - x_i
          x_tran(2, 13) = - y_i
          x_tran(3, 13) = - z_i
        end if
        if (need_virials) then

          if (cutoffs_equal) then
            if(fswitch .le. 0) then
              if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_fv_gvdw_vac(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_fv_vac(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                   eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                   gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)!modified _sp by Ashraf
               end if
              else
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_fv_gvdw(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_fv(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                   eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                   gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)!modified _sp by Ashraf
               end if
              end if
            else
              call pairs_calc_PGM_fv_fs(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                               eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                               gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
            end if
          else
            if(fswitch .le. 0) then
              if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_fv_gvdw_2cut_vac(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_fv_2cut_vac(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                        eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                        gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
              else
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_fv_gvdw_2cut(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_fv_2cut(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                        eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                        gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
              end if
            else
              call pairs_calc_PGM_fv_2cut_fs(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                    eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                    gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
            end if
          end if

        else

          if (cutoffs_equal) then
            if(fswitch .le. 0) then
              if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_f_gvdw_vac(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_f_vac(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                      gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
              else
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_f_gvdw(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_f(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                  eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                  gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
              end if
            else
              call pairs_calc_PGM_f_fs(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                              eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                              gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
            end if
          else
            if(fswitch .le. 0) then
              if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_f_gvdw_2cut_vac(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_f_2cut_vac(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                       eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                       gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)!modified _sp by Ashraf
               end if
              else
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_f_gvdw_2cut(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_f_2cut(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                       eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                       gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)!modified _sp by Ashraf
               end if
              end if
            else
              call pairs_calc_PGM_f_2cut_fs(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                   eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                   gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
            end if
          end if

        end if

        ipairs_idx = ipairs_idx + ee_eval_cnt + full_eval_cnt

      end if
#else  /* _USE_SAVED_BFACT */
        if (need_virials) then
            if(fswitch .le. 0) then
              if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_fv_gvdw_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_fv_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                   eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                   gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)!modified _sp by Ashraf
               end if
              else
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_fv_gvdw_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_fv_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                   eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                   gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)!modified _sp by Ashraf
               end if
              end if
            else
              call pairs_calc_PGM_fv_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                               eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                               gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
            end if
        else
            if(fswitch .le. 0) then
              if(use_pme .eq. 0) then
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_f_gvdw_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_f_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                      gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
              else
               if (igvdw .gt. 0) then
                call pairs_calc_PGM_f_gvdw_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                      eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                     gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               else
                call pairs_calc_PGM_f_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                                  eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                                  gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
               end if
              end if
            else
              call pairs_calc_PGM_f_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, fs_tbl, &
                              eed_cub, phi, typ_ico, ipairs(ipairs_idx), &
                              gbl_img_iac, gbl_cn1, gbl_cn2, x_tran)
            end if
        end if
#endif /* _USE_SAVED_BFACT */

#ifdef FIELD_ONLY
  dipole_dipole_ptrs(img_i+1) = num_tensor + 1
  dipole_dipole_short_ptrs(img_i+1) = num_tensor_short + 1
#endif
    end do

  end if

  ! Save the energies:
                                                                                
  eedvir = eedvir_stk
  eed = eed_stk
  evdw = evdw_stk
  ehb = ehb_stk

  ! Save the virials.

  vdw_virial(1, 1) = vxx
  vdw_virial(1, 2) = vxy
  vdw_virial(1, 3) = vxz
!  vdw_virial(2, 1) = vxy
  vdw_virial(2, 1) = vyx
  vdw_virial(2, 2) = vyy
  vdw_virial(2, 3) = vyz
!  vdw_virial(3, 1) = vxz
  vdw_virial(3, 1) = vzx
!  vdw_virial(3, 2) = vyz
  vdw_virial(3, 2) = vzy
  vdw_virial(3, 3) = vzz

  eed_virial(1:3, 1:3) = eed_virial(1:3, 1:3) * 0.5d0
  eedvir = eed_virial(1,1) + eed_virial(2,2) + eed_virial(3,3)

#endif /* FIELD_ONLY */
#endif /* CUDA */
  return

contains

#ifdef _FIELD_ONLY_
#undef _FIELD_ONLY_
#endif

#ifdef _USE_SAVED_BFACT_
#undef _USE_SAVED_BFACT_
#endif

#ifdef _SAVE_BFACT_
#undef _SAVE_BFACT_
#endif

#ifndef FIELD_ONLY

#ifdef _USE_SAVED_BFACT
#define _USE_SAVED_BFACT_
#endif

! energy + force combos
#define BUILD_PAIRS_CALC_EF
#define NEED_GVDW
!subroutine pairs_calc_PGM_ef_gvdw(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_gvdw_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_ef(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"
#undef BUILD_PAIRS_CALC_EF

#define BUILD_PAIRS_CALC_EF
#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define NEED_GVDW
!subroutine pairs_calc_PGM_ef_gvdw_2cut(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_gvdw_2cut_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_ef_2cut(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_2cut_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"
#undef BUILD_PAIRS_CALC_NOVEC_2CUT
#undef BUILD_PAIRS_CALC_EF


#define BUILD_PAIRS_CALC_EF
#define FSWITCH
!subroutine pairs_calc_PGM_ef_fs(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
!subroutine pairs_calc_PGM_ef_2cut_fs(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_2cut_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"
#undef BUILD_PAIRS_CALC_NOVEC_2CUT
#undef FSWITCH
#undef BUILD_PAIRS_CALC_EF

#define BUILD_PAIRS_CALC_EF
#define VACUUM
#define NEED_GVDW
!subroutine pairs_calc_PGM_ef_gvdw_vac(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_gvdw_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW

!subroutine pairs_calc_PGM_ef_vac(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define NEED_GVDW
!subroutine pairs_calc_PGM_ef_gvdw_2cut_vac(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_gvdw_2cut_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_ef_2cut_vac(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_ef_2cut_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"
#undef VACUUM
#undef BUILD_PAIRS_CALC_NOVEC_2CUT
#undef BUILD_PAIRS_CALC_EF

! energy + force + virial combos
#define BUILD_PAIRS_CALC_EFV
#define NEED_GVDW
!subroutine pairs_calc_PGM_efv_gvdw(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_gvdw_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_efv(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define NEED_GVDW
!subroutine pairs_calc_PGM_efv_gvdw_2cut(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_gvdw_2cut_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_efv_2cut(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_2cut_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                               phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define FSWITCH
!subroutine pairs_calc_PGM_efv_2cut_fs(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_2cut_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef BUILD_PAIRS_CALC_NOVEC_2CUT
!subroutine pairs_calc_PGM_efv_fs(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"
#undef FSWITCH

#define VACUUM
#define NEED_GVDW
!subroutine pairs_calc_PGM_efv_gvdw_vac(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_gvdw_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_efv_vac(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define NEED_GVDW
!subroutine pairs_calc_PGM_efv_gvdw_2cut_vac(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_gvdw_2cut_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_efv_2cut_vac(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_efv_2cut_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, ef_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef BUILD_PAIRS_CALC_NOVEC_2CUT
#undef VACUUM
#undef BUILD_PAIRS_CALC_EFV

! force + virial combos
#define BUILD_PAIRS_CALC_FV
#define NEED_GVDW
!subroutine pairs_calc_PGM_fv_gvdw(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_gvdw_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_fv(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define NEED_GVDW
!subroutine pairs_calc_PGM_fv_gvdw_2cut(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_gvdw_2cut_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_fv_2cut(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_2cut_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)

#include "pairs_calc_PGM.i"
#undef BUILD_PAIRS_CALC_NOVEC_2CUT

#define FSWITCH
!subroutine pairs_calc_PGM_fv_fs(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
!subroutine pairs_calc_PGM_fv_2cut_fs(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_2cut_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"
#undef BUILD_PAIRS_CALC_NOVEC_2CUT
#undef FSWITCH

#define VACUUM
#define NEED_GVDW
!subroutine pairs_calc_PGM_fv_gvdw_vac(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_gvdw_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_fv_vac(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define NEED_GVDW
!subroutine pairs_calc_PGM_fv_gvdw_2cut_vac(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_gvdw_2cut_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_fv_2cut_vac(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fv_2cut_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef BUILD_PAIRS_CALC_NOVEC_2CUT
#undef VACUUM
#undef BUILD_PAIRS_CALC_FV

! force combos
#define BUILD_PAIRS_CALC_F
#define NEED_GVDW
!subroutine pairs_calc_PGM_f_gvdw(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_gvdw_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_f(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define NEED_GVDW
!subroutine pairs_calc_PGM_f_gvdw_2cut(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_fdevdw__2cut_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_f_2cut(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_2cut_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define FSWITCH
!subroutine pairs_calc_PGM_f_2cut_fs(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_2cut_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"
#undef BUILD_PAIRS_CALC_NOVEC_2CUT

!subroutine pairs_calc_PGM_f_fs(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_fs_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"
#undef FSWITCH

#define VACUUM
#define NEED_GVDW
!subroutine pairs_calc_PGM_f_gvdw_vac(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_gvdw_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_f_vac(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define NEED_GVDW
!subroutine pairs_calc_PGM_f_gvdw_2cut_vac(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                             phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_gvdw_2cut_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef NEED_GVDW
!subroutine pairs_calc_PGM_f_2cut_vac(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                             phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
!subroutine pairs_calc_PGM_f_2cut_vac_use_saved_B(img_frc, img_crd, img_chg, img_dip, f_tbl, eed_cub, &
!                        phi, ico, ipairs_sublst, img_iac, cn1, cn2, x_tran)
#include "pairs_calc_PGM.i"

#undef BUILD_PAIRS_CALC_NOVEC_2CUT
#undef VACUUM
#undef BUILD_PAIRS_CALC_F

#ifdef _USE_SAVED_BFACT

#undef _USE_SAVED_BFACT_

end subroutine get_nb_energy_PGM_use_saved_B

#else

end subroutine get_nb_energy_PGM

#endif  /* _USE_SAVED_BFACT */

#else  /* FIELD_ONLY */

#define _FIELD_ONLY_

#ifdef _SAVE_BFACT
#define _SAVE_BFACT_
#endif

#define BUILD_PAIRS_CALC_NOVEC_2CUT
#define VACUUM
!subroutine pairs_calc_PGM_fld_2cut_vac(img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
!                        gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
!                        dipole_dipole_list_short, dipole_dipole_tensor_short)
!subroutine pairs_calc_PGM_fld_2cut_vac_save_B(img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
!                        gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
!                        dipole_dipole_list_short, dipole_dipole_tensor_short)
#include "pairs_calc_PGM.i"    /* vacuum dual cutoffs field */
#undef VACUUM

!subroutine pairs_calc_PGM_fld_2cut(img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
!                        gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
!                        dipole_dipole_list_short, dipole_dipole_tensor_short)
!subroutine pairs_calc_PGM_fld_2cut_save_B(img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
!                        gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
!                        dipole_dipole_list_short, dipole_dipole_tensor_short)
#include "pairs_calc_PGM.i"    /* dual cutoffs field */
#undef BUILD_PAIRS_CALC_NOVEC_2CUT

#define VACUUM
!subroutine pairs_calc_PGM_fld_vac(img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
!                        gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
!                        dipole_dipole_list_short, dipole_dipole_tensor_short)
!subroutine pairs_calc_PGM_fld_vac_save_B(img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
!                        gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
!                        dipole_dipole_list_short, dipole_dipole_tensor_short)
#include "pairs_calc_PGM.i"    /* vacuum single cutoff field */
#undef VACUUM

!subroutine pairs_calc_PGM_fld(img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
!                        gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
!                        dipole_dipole_list_short, dipole_dipole_tensor_short)
!subroutine pairs_calc_PGM_fld_save_B(img_crd, img_chg, img_dip, ef_tbl, eed_cub, ipairs_sublst, x_tran, &
!                        gradphi, num_tensor, dipole_dipole_list, dipole_dipole_tensor, num_tensor_short, &
!                        dipole_dipole_list_short, dipole_dipole_tensor_short)
#include "pairs_calc_PGM.i"    /* single cutoff field */

#ifdef _SAVE_BFACT
end subroutine get_direct_fld_PGM_save_B
#undef _SAVE_BFACT
#undef _SAVE_BFACT_
#else
end subroutine get_direct_fld_PGM
#endif

#undef _FIELD_ONLY_

#endif   /* FIELD_ONLY */


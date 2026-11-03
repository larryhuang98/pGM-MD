#include "pGM/copyright_pgm.i"
module pgm_interface_data_mod

   use gbl_datatypes_mod

  implicit none

  logical, save :: need_virials_pgm
  integer, save, allocatable :: mol_atm_ptr(:)
  integer, save, allocatable :: mol_atm_cnt(:)
#ifdef MPI
  integer, save              :: my_used_mols_cnt
  integer, save, allocatable :: my_used_mols_lst(:)
  integer, save, allocatable :: pGM_gbl_atm_mol_map(:)
  integer, save, allocatable :: my_used_mol_mask(:)
  integer, save              :: my_short_img_cnt
  integer, save, allocatable :: my_short_img_lst(:)
  integer(byte), save, allocatable    :: my_used_img_mask(:) ! to be compatible with gbl_used_img_map
  integer, save              :: gbl_used_img_cnt_phi   ! *_cnt_phi record length of phi array
                                                       ! for img > gbl_used_img_cnt_phi,
                                                       ! phi() is zero. The force loops only on *_cnt_phi.
  integer, save              :: my_full_img_cnt
  integer, save, allocatable :: my_full_img_lst(:)
#endif
  
  type nb_uv_rec
    sequence
    double precision  :: ene_tot_chg
    double precision  :: ene_tot_dip
    double precision  :: ene_tot_vdw
    double precision  :: ene_tot_hb
    double precision  :: ene_dir_chg
    double precision  :: ene_dir_dip
    double precision  :: ene_dir_vdw
    double precision  :: ene_dir_hb
    double precision  :: ene_rec_chg
    double precision  :: ene_rec_dip
    double precision  :: ene_self_chg
    double precision  :: ene_self_dip
    double precision  :: vireedir(3, 3)
    double precision  :: vireerec(3, 3)
    double precision  :: vireeself(3, 3)
    double precision  :: virvdw(3, 3)
  end type nb_uv_rec

  type(nb_uv_rec), parameter  :: null_nb_uv_rec = &
     nb_uv_rec(0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, &
               0.d0, 0.d0, 0.d0, 0.d0, &
               9*0.d0, 9*0.d0, 9*0.d0, 9*0.d0)

  double precision, save  :: betamax = sqrt(2.0d0)/0.7147597d0
  double precision, save  :: diprms, dipiter

end module pgm_interface_data_mod

